## eval script
import argparse
import json
import os
import re

# os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
from datasets import Dataset, IterableDataset, Audio
from transformers import Wav2Vec2ForCTC
from tqdm import tqdm

from scripts.loader import load_ft_data, load_whisper_res, load_multipa_res, read_sanna
from scripts.whipa_utils import parse_langs, prep_dataset, conjoin_ds
from scripts.metrics import STIPA_METRICS, retokenize_ipa


parser = argparse.ArgumentParser(description="Specify config or languages, options per data source")
parser.add_argument("-ckpt", "--model_checkpoint",
                    default=False,
                    help="Specify model checkpoint to evaluate")
parser.add_argument("-corp", "--corpora", nargs="*",
                    help="Specify corpora to include for finetuning.")
parser.add_argument("-lg", "--languages",  nargs="*", default='all',
                    help="Specify language code (split by space) or 'all'."
                    +"to specify different languages per corpus, separate arguments with 'x', e.g. 'ja ta x ara")
parser.add_argument("-nost", "--no_streaming", action="store_true",
                    help="Turn of data streaming.")
parser.add_argument("-n", "--n_samples", nargs="*", default=False,
                    help="Specify the number of samples to be used as the test data for each language." \
                    "For example, if you want to use 1000, 2000, 3000 test samples for Japanese, Polish," \
                    "and Maltese, then specify as -l ja pl mt -te 1000 2000 3000. When multiple languages are specified,"\
                    "but only one sample number, the sample number will be applied to all languages." \
                    "Pass 0 to pick the maximum value for a given language.")
parser.add_argument("-td", "--to_disk", action="store_true",
                    help="Specify whether to work with IterableDatasets or Datasets (default IterableDS).")
parser.add_argument("-maxs", "--max_seconds", type=int, default=6,
                    help="Specify maximum length of audio clips to use,")  # TBD enable for all corpora? only included to be comaprable with MultIPA
parser.add_argument("-np", "--num_proc", type=int, default=1,
                    help="Specify number of workers") 
parser.add_argument("-norm", "--norm_str", action="store_false",
                    help="Specify whether to normalise predicted characters or not (to comply with MultIPA)")  
parser.add_argument("-sa", "--save_as", type=str, default=False,
                    help="Specify how to save results") 
parser.add_argument("-sv", "--save", action="store_true",
                    help="Specify whether to save results") 
parser.add_argument("-micro", "--incl_micro", action="store_false",
                    help="Specify whether to include per-prediction results") 
parser.add_argument("-dn", "--device_nr", default=('cuda:0' if torch.cuda.is_available() else 'cpu'),
                    help="Specify cuda device") 
parser.add_argument("-peft", "--peft", action="store_false",
                    help="Specify whether model to be loaded is PEFT LoRA") 
parser.add_argument("-zt", "--zero_shot", action="store_true",
                    help="Specify whether to use zero-shot evaluation")
parser.add_argument("-how", "--how", type=str, default="fallback",
                    help="Specify whether to use fallback for outlier guessing")
parser.add_argument("-dev", "--dev", action="store_true",
                    help="Use validation split")

args, _ = parser.parse_known_args()


def whipa_predict(model, tokenizer, arr, rate_limit, og_beams, how, fallback):
    out = tokenizer.decode(model.generate(arr, num_beams=og_beams)[0])
    if how!="vanilla":
        # check validity of prediciton
        if len(retokenize_ipa(out)) > rate_limit:  # max articulatory phone/secs rate
            print("fb")
            print(len(retokenize_ipa(out)), out )
            # 1: beam backoff
            if not fallback:
                n_beams = (1,3,5,7)
                backoff_bank = [x for x in n_beams if x>og_beams] + [x for x in n_beams if x<og_beams]
            else:
                # print("using", fallback)
                backoff_bank = fallback
                backoff = backoff_bank.copy()

            # try to alter beam size and yield a non-overshot transcription
            while (len(retokenize_ipa(out)) > rate_limit) and backoff:
                beams = list(backoff)[0]
                backoff.remove(beams)
                out = tokenizer.decode(model.generate(arr, num_beams=beams)[0])

        # 2: Beam backoff failed; try backoff with repetition_penalty
        if (len(retokenize_ipa(out))) > rate_limit:
            # reset decoding backoff
            backoff = backoff_bank.copy()
            # first try og beam size again with penatly, then repeat backoff
            beams = og_beams

            while (len(retokenize_ipa(out)) > rate_limit) and backoff:
                out = tokenizer.decode(model.generate(arr, num_beams=beams,
                                                    repetition_penalty=1.15)[0])
                # update beam size if loop is not excited
                beams = list(backoff)[0]
                backoff.remove(beams)

        # 3: Beam backoff with repetition penalty also failed;
        # truncate prediciton with exponential weight decay penalty
        if (len(retokenize_ipa(out))) > rate_limit:
            penalty = 2.0
            while (len(retokenize_ipa(out)) > rate_limit) and penalty <= 5:
                out = tokenizer.decode(model.generate(arr, num_beams=og_beams,
                            exponential_decay_length_penalty=(int(rate_limit*0.8), penalty))[0])
                penalty += 1.5
        # if this still fails somehow (positive logs for example); force truncate
        if (len(retokenize_ipa(out))) > rate_limit:
            out = tokenizer.decode(model.generate(arr, num_beams=og_beams)[0])[:rate_limit]

    return out 


def get_preds(model, data: Dataset|IterableDataset, tokenizer,
              lg_rates={"el": 13, "fi": 17, "hu": 15, "ja": 16, "mt": 14, "pl": 16, "ta": 17},
              gen_args={"num_beams": 1},
              how="vanilla",
              fallback=False) -> dict:
    eval_data = dict()
    for batch in tqdm(data):
        lg = batch["locale"]
        eval_data[lg] = eval_data.get(lg, {"gold": list(), "pred": list()})
        eval_data[lg]["gold"].append(batch["ipa"])

        # generate & decode prediction
        og_beams = gen_args["num_beams"]
        arr = torch.tensor(batch["input_features"]).unsqueeze(0).to(model.device)

        if type(model)!=Wav2Vec2ForCTC:
            if ZERO_SHOT:
                model.generation_config.language = "<|{}|>".format(
                    {"cnh": "bo", "hsb": "pl", "lg": "sw", "tt": "tr"}[lg])
                        # set max phone/sec rate
            max_rate = (lg_rates[lg] if lg in lg_rates else 20.0)  
            rate_limit = round((len(batch['audio']['array'])/16_000)*max_rate)

            out = whipa_predict(model, tokenizer, arr, rate_limit, og_beams, how, fallback)
        else:
            # inference as detailed in: https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/README.md#use-wav2vec-20-with-transformers
            predicted_ids = torch.argmax(model(torch.tensor(arr)).logits, dim=-1).squeeze()
            out = tokenizer.decode(predicted_ids, skip_special_tokens=True)

        # save prediction
        eval_data[batch["locale"]]["pred"].append(out)

    return eval_data


def calc_metrics(eval_metrics: STIPA_METRICS, preds: list[str] = False,
                 golds: list[str] = False, mode: str = "macro") -> dict:
    assert (preds and golds) and (type(preds)!=str and type(golds)!=str), "Missing required arguments of iterable type: preds, golds"
    if mode not in ["macro", "micro"]:
        print("Unknown mode", mode, "defaulting to 'macro'")
        mode = "macro"
    n_samples = len(preds)
    res = dict()

    for i, (p, g) in tqdm(enumerate(zip(preds, golds))):
        for metr, val in eval_metrics.compute_all(pred=p, gold=g, char_based=False).items():
            if mode=="macro":
                res[metr] = res.get(metr, 0) + val/n_samples
            else:
                res[g] = res.get(g, {'pred': p})
                res[g][metr] = val
    return res


def get_metrics(data, eval_metrics: STIPA_METRICS=False, normalize=False,
                include_micro=False) -> dict:
    # calculate overall metrics & per language
    if not eval_metrics:
        eval_metrics = STIPA_METRICS(normalize_str=normalize)

    results = {lg: dict() for lg in set(data.keys())}

    for lg, data in tqdm(data.items()):
        # calc metrics per pred:gold sample
        micros = calc_metrics(eval_metrics, preds=data["pred"], golds=data["gold"],
                              mode="micro")
        # get averages within lg
        results[lg]["macro"] = {m: (sum([sample[m] for sample in micros.values()])
                                    / len(micros.values()))
                                for m in list(micros.values())[0].keys() if m!="pred"}

        if include_micro: results[lg]["micro"] = micros

    # get averages over entire data
    results["macro"] = {m: (sum([results[lg]["macro"][m] for lg in results])
                            / len(results.keys()))
                        for m in list(results.values())[0]["macro"].keys()}

    return results

namemap = {
    "a_bf1k": 
        {"ckpt": "a_bf1k10/checkpoint-440",
         "fname": "a_bf_1k-4.txt"},
    "a_bl1k": 
        {"ckpt": "a_bl1k10/checkpoint-440",
         "fname": "a_bl_1k-4.txt"},
    "a_lf1k": 
        {"ckpt": "a_lf1k10/checkpoint-660",
         "fname": "a_lf_1k-6.txt"},
    "a_ll1k": 
        {"ckpt": "a_ll1k10/checkpoint-880",
         "fname": "a_ll_1k-8.txt"},
    "a_llall5": 
        {"ckpt": "a_llall5/checkpoint-1749",
         "fname": "a_ll_all-3.txt"},
        
    "b_bl_asc1k":
        {"ckpt": "b_bl_asc1k/checkpoint-378",
         "fname": "b_bl_asc1k-6.txt"},
    "b_bl_ascall":
        {"ckpt": "b_bl_ascall/checkpoint-1020",
         "fname": "b_bl_ascall-10.txt"}, 
    "b_ll_asc1k":
        {"ckpt": "b_ll_asc1k/checkpoint-378",
         "fname": "b_ll_asc1k-6.txt"},
    "b_ll_ascall":
        {"ckpt": "b_ll_ascall/checkpoint-1020",
         "fname": "b_ll_ascall-10.txt"},



    "b_bl_thchs1k":
        {"ckpt": "b_bl_thchs1k/checkpoint-252",
         "fname": "b_bl_thchs1k-4.txt"},
    "b_bl_thchsall":
        {"ckpt": "b_bl_thchsall/checkpoint-1570",
         "fname": "b_bl_thchsall-10.txt"},
    "b_ll_thchs1k":
        {"ckpt": "b_ll_thchs1k/checkpoint-630",
         "fname": "b_ll_thchs1k-10.txt"},
    "b_ll_thchsall":
        {"ckpt": "b_ll_thchsall/checkpoint-1570",
         "fname": "b_ll_thchsall-10.txt"},
        
    "c_bl_comb91k":
        {"ckpt": "c_bl_comb91k/checkpoint-564",
         "fname": "c_bl_comb91k-4.txt"},

    "c_ll_comb91k":
        {"ckpt": "c_ll_comb91k/checkpoint-846",
         "fname": "c_ll_comb91k-6.txt"},
    "d_ll_araelmt1k":
        {"ckpt": "d_ll_araelmt1k/checkpoint-940",
         "fname": "d_ll_araelmt1k-10.txt"},
    "d_bl_araelmt1k":
        {"ckpt": "d_bl_araelmt1k/checkpoint-376",
         "fname": "d_bl_araelmt1k-4.txt"},

        
    "multipa": {"fname": "multipa1k.txt"}
    
}

if __name__=="__main__":

    STREAMING = (False if args.no_streaming else True)
    TO_DISK = args.to_disk
    MAX_SECONDS = args.max_seconds
    NUM_PROC = args.num_proc
    LIMIT = args.n_samples
    DEVICE = args.device_nr
    PEFT = args.peft
    ZERO_SHOT = args.zero_shot
    LANGS = list()

    if "x" in args.languages:  # multiple lg ids passed for 1+ corpora
        corp_lgs = list()
        for lg in args.languages:
            if lg != "x":
                corp_lgs.append(lg)
            else:
                LANGS.append(corp_lgs)
                corp_lgs = list()
        LANGS.append(corp_lgs)
    else:
        LANGS = [args.languages]


    if not (args.model_checkpoint.startswith("openai") or args.model_checkpoint=="multipa"):
        CHECKPOINT = os.path.join("../models/", args.model_checkpoint)
        with open(os.path.join(os.path.dirname(CHECKPOINT), "ft_config.json"), "r") as f:
            ft_config = json.load(f)
            gen_args = ft_config["gen_args"]
            fallback = ft_config["fallback"]

    else:
        CHECKPOINT = args.model_checkpoint
        gen_args = {"num_beams": 1}
        fallback = False

    # load resources
    model, tokenizer, processor = (
        load_whisper_res(CHECKPOINT, get_tokenizer=True, get_processor=True,
                         is_whipa=(True if not CHECKPOINT.startswith("openai") else False),
                         peft=PEFT,
                         zero_shot=ZERO_SHOT
                         ) 
        if CHECKPOINT!="multipa" else load_multipa_res())

    model = model.eval().to(DEVICE)

    # load data
    datasets = list()
    for i, corp in enumerate(args.corpora):
        if not re.findall(r"multipa-test|sanna", corp):
            lgs = parse_langs(LANGS[i], corp)[corp]
            dsplit = ("dev" if args.dev else (
                "test" if corp in ["multipa", "tusom", "asc", "thchs"] else  None))
            datasets.append(load_ft_data(corp, dsplit=dsplit, 
                                         lgs=lgs, limit=LIMIT, stream=STREAMING,
                                         max_seconds=MAX_SECONDS, to_disk=TO_DISK,
                                         num_proc=NUM_PROC))
        elif corp=="multipa-test":
            # multipa out-domain test set (local files)
            datasets.append(Dataset.from_json("../data/multipa/test/test_set.json").cast_column(
                                        "audio", Audio(sampling_rate=16_000)))
        else:
            datasets.append(read_sanna(id="0"+corp[-1]))

        # Align types of Dataset/IterableDataset and concat ds from all sources.
        data = conjoin_ds(datasets, TO_DISK)


    # Prep data for prediction: Extract input features+labels, drop
    # samples with overlength.
    data = prep_dataset(data, processor, tokenizer, seed=False,
                        max_len=(model.generation_config.max_length if type(model)!=Wav2Vec2ForCTC else False))

    # generate predictions
    eval_data = get_preds(model, data, tokenizer, gen_args=gen_args, how=args.how,
                          fallback=fallback)

    # evaluate predictions
    results = get_metrics(eval_data, include_micro=args.incl_micro, normalize=args.norm_str)

    if args.save or args.save_as:
        outdir = os.path.join("../results/", '-'.join(args.corpora))
        if not os.path.isdir(outdir): os.mkdir(outdir)
        fname = os.path.join(outdir, (args.save_as if args.save_as
                                      else args.model_checkpoint))
        i=1
        while os.path.isfile(fname):
            fname = fname[:-4] + str(i) + ".txt"
            i+=1
        with open(fname, "w") as f:
            f.write(str(results))
    else:
        for lg, x in results.items():
            print(lg, x)
            print()

    print("-"*35, "COMPLETE", "-"*35)

## Whisper fine-tuning script ##
import argparse
import ast
import json
import os
import time

# os.environ["CUDA_VISIBLE_DEVICES"]="0"

import pandas as pd
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import Dataset

# PEFT related imports
# https://github.com/huggingface/peft/blob/main/examples/int8_training/peft_bnb_whisper_large_v2_training.ipynb
# https://github.com/Vaibhavs10/fast-whisper-finetuning/blob/main/README.md
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from scripts.loader import load_ft_data, load_whisper_res
from scripts.whipa_utils import parse_langs, prep_dataset, get_dirname, conjoin_ds
from eval import calc_metrics


def fetch(argname: str, gpu_default, cpu_default):
    if args.use_config and argname in HYPERPARAMS:
        return (HYPERPARAMS[argname] if not HYPERPARAMS[argname] in ["True", "False"]
                else ast.literal_eval(HYPERPARAMS[argname]))
    elif torch.cuda.is_available():
        return gpu_default
    else:
        return cpu_default


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    # source: https://huggingface.co/blog/fine-tune-whisper
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        return control


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids


    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # calculate average across all custom metrics
    metrics = calc_metrics(preds=pred_str, golds=label_str, normalize_str=True)

    end = time.time()
    metrics["time"] = end-start
    METRIC_LOG.append(metrics)

    ## If checkpoints were saved: add FT params file & eval metrics
    if training_args.save_steps and not PEFT:
        pd.DataFrame(METRIC_LOG).to_csv(os.path.join(trainer._get_output_dir(""), "metric_log.csv"))

    return metrics


if __name__=="__main__":
    # Arguments, adapted from multipa/main.py
    parser = argparse.ArgumentParser(description="Specify config or languages, options per data source")
    parser.add_argument("-lg", "--languages",  nargs="*",
                        help="Specify language code (split by space) or 'all'."
                        +"to specify different languages per corpus, separate arguments with 'x', e.g. 'ja ta x ara")
    parser.add_argument("-mo", "--modelname", type=str, default='base', nargs=1,
                        help="Specify model to use for finetuning.")
    parser.add_argument("-corp", "--corpora", nargs="*",
                        help="Specify corpora to include for finetuning.")
    parser.add_argument("-st", "--streaming", action="store_true",
                        help="Specify how to load (multipa) data.")
    parser.add_argument("-td", "--to_disk", action="store_true",
                        help="Specify whether to work with IterableDatasets or Datasets (default IterableDS).")
    parser.add_argument("-config", "--use_config", type=str, default=False,
                        help="Specify config setting to use for FT params,"
                        + "e.g. '--use_config baseline'.")
    parser.add_argument("-maxs", "--max_seconds", type=int, default=6,
                        help="Specify maximum length of audio clips to use,")  # TBD enable for all corpora? only included to be comaprable with MultIPA
    parser.add_argument("-peft", "--peft", action="store_true",
                        help="Set to use LoRA PEFT.")

    # tr, te, num_proc adapted from multipa/main.py; added using 0 for all
    parser.add_argument("-tr", "--train_samples", nargs="*", default=False,
                        help="Specify the number of samples to be used as the training data for each language." \
                        "For example, if you want to use 1000, 2000, 3000 training samples for Japanese, Polish," \
                        "and Maltese, then specify as -l ja pl mt -tr 1000 2000 3000. When multiple languages are specified,"\
                        "but only one sample number, the sample number will be applied to all languages." \
                        "Pass 0 to pick the maximum value for a given language.")
    parser.add_argument("-te", "--test_samples", nargs="*", default=False,
                        help="Specify the number of samples to be used as the test data for each language." \
                        "For example, if you want to use 1000, 2000, 3000 test samples for Japanese, Polish," \
                        "and Maltese, then specify as -l ja pl mt -te 1000 2000 3000. When multiple languages are specified,"\
                        "but only one sample number, the sample number will be applied to all languages." \
                        "Pass 0 to pick the maximum value for a given language.")
    parser.add_argument("--num_proc", type=int, default=1,
                        help="Specify the number of CPUs for preprocessing. Default set to 1.")

    args = parser.parse_args()
    PARAMS = {"train": dict(), "dev": dict()}

    with open("ft_config.json", "r") as f:
        config = json.load(f) # default for dirpaths
    global_config = config["loading_params"]

    if args.use_config:
        # global settings (constant)
        STREAMING = ast.literal_eval(global_config["streaming"])
        print("Streaming:", STREAMING)
        TO_DISK = ast.literal_eval(global_config["to_disk"])
        NUM_PROC = int(global_config["num_proc"])
        # config-specific settings
        print("Using {} config...".format(args.use_config))
        model_config = config[args.use_config]
        MODELNAME = model_config["modelname"]
        HYPERPARAMS = model_config["hyperparams"]
        MAX_SECONDS = int(HYPERPARAMS.get("max_seconds", 999_999))
        PEFT = ast.literal_eval(HYPERPARAMS["peft"])

        for split in ["train", "dev"]:
            # get corpora names, languages to use and sample limits per split
            PARAMS[split]["corpora"] = {
                corp: {"languages": model_config["corpora"][split][corp]["languages"],
                       "limit": (model_config["corpora"][split][corp]["limit"]
                                 if "limit" in model_config["corpora"][split][corp] else False)}
                for corp in model_config["corpora"][split].keys()}
    else:
        # Use cmd line args (except for data dirs, default to ../data/)
        # non config mode does not differentiate between train/dev corpora/languages
        modelmap = {"base": "openai/whisper-base",
                    "large-v2": "openai/whisper-large-v2"}

        STREAMING = args.streaming
        TO_DISK = args.to_disk
        MODELNAME =  (modelmap[args.modelname]
                      if args.modelname in modelmap else args.modelname)
        MAX_SECONDS = args.max_seconds
        NUM_PROC = args.num_proc
        PEFT = args.peft

        corpora = (args.corpora if type(args.corpora)==list
                   else ([args.corpora,] if not " " in args.corpora 
                         else args.corpora.split()))
        lgs = parse_langs(args.languages, corpora)

        for split in ["train", "dev"]:
            sh = PARAMS[split]  # set shorthand
            sh["corpora"] = sh.get("corpora", dict())
            for i, corp in enumerate(corpora):
                skip_n = sum([len(lgs[corpora[j-1]]) for j in range(i)]) if i else 0
                sh["corpora"][corp] = {
                    "languages": lgs[corp], 
                    "limit": (args.train_samples[skip_n:skip_n+len(lgs[corp])] if split=="train"
                              else args.test_samples[skip_n:skip_n+len(lgs[corp])])}

    ## Load model resources ##
    model, tokenizer, processor = load_whisper_res(MODELNAME, get_tokenizer=True,
                                    get_processor=True, peft=PEFT, trainable=True)
    model = model.train()
    ## Load data ##
    data_bin = {"train": list(), "dev": list()}

    for split in ["train", "dev"]:
        # Load specifided train and validation data
        print(PARAMS[split]["corpora"])
        data_bin[split] = [load_ft_data(
                                corp, split, specs["languages"],
                                dirpath=(global_config["dirpaths"][corp]
                                         if args.use_config else False),  # use defaults in loader.py
                                limit=specs["limit"],     # only load n samples for language
                                stream=STREAMING,         # determine how to preprocess multipa/thchs, stream recommended to avoid processing all data (slow)
                                max_seconds=MAX_SECONDS,  # currently only effective for multipa to emulate Taguchi et al 2023
                                to_disk=TO_DISK,          # decide whether to use iterableDS or DS
                                num_proc=NUM_PROC
                                )
                           for corp, specs in PARAMS[split]["corpora"].items()]
        # Align types of Dataset/IterableDataset and concat ds from all sources.
        data_bin[split] = conjoin_ds(data_bin[split], TO_DISK)

        # Prep data for finetuning: Extract input features+labels, drop
        # samples with overlength, shuffle.
        data_bin[split] = prep_dataset(data_bin[split], processor, tokenizer,
                            max_len=model.generation_config.max_length,
                            seed=(42 if split=="train" else 35), # tbd variable seed?
                            num_proc=NUM_PROC)

    ## Define data_collator and training args
    # cf. https://huggingface.co/blog/fine-tune-whisper
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id)

    # get unique directory name for run, create dir for logs/model
    out_dir = get_dirname(HYPERPARAMS["output_dir"] if args.use_config
                          else "../models/"+args.modelname+"-args")

    # define traning args: priority: config > GPU defaults (arg2) > CPU defaults (arg3) | peft
    training_args = Seq2SeqTrainingArguments(
        output_dir=out_dir,
        overwrite_output_dir=fetch("overwrite_output_dir", True, True),
        per_device_train_batch_size=fetch("per_device_train_batch_size", 16, 8),
        gradient_accumulation_steps=fetch("gradient_accumulation_steps", 1, 1),
        learning_rate=fetch("learning_rate", 1e-5, 5e-5),
        warmup_steps=fetch("warmup_steps", 100, 10),
        warmup_ratio=fetch("warmup_ratio", 0.0, 0.0),
        max_steps=fetch("max_steps", 1000, 100),
        logging_first_step=fetch("logging_first_step", True, True),
        gradient_checkpointing=fetch("gradient_checkpointing", True, True),  # slightly slower training but saves mem
        num_train_epochs=(fetch("num_train_epochs", 5, 1) if type(data_bin["train"])==Dataset else 0),  # incompatible with IterableDS
        fp16=fetch("fp16", True, False),  # only saves mem on large batch size, load two models in mem, slower training
        eval_strategy=fetch("eval_strategy", "steps", "steps"),
        per_device_eval_batch_size=fetch("per_device_eval_batch_size", 8, 4),
        predict_with_generate=(fetch("predict_with_generate", True, True) if not PEFT else False),  # incompatible with PEFT
        generation_max_length=fetch("generation_max_length", 225, 225),
        save_steps=fetch("save_steps", 200, 200),
        eval_steps=fetch("eval_steps", 200, 200),
        logging_steps=fetch("logging_steps", 200, 50),
        report_to=["tensorboard"],
        load_best_model_at_end=fetch("load_best_model_at_end", True, True),
        metric_for_best_model=(fetch("metric_for_best_model", "cer", "cer") if not PEFT else None),  # incompatible with PEFT
        greater_is_better=fetch("greater_is_better", False, False),
        torch_empty_cache_steps=fetch("torch_empty_cache_steps", 5, 5),
        hub_private_repo=fetch("hub_private_repo", True, True),
        push_to_hub=fetch("push_to_hub", False, False),
        remove_unused_columns=(True if not PEFT else False),  # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
        label_names=["labels"],  # required for PEFT
        )

    ## Fine-tune model.
    # Define trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=data_bin["train"],
        eval_dataset=data_bin["dev"],
        data_collator=data_collator,
        compute_metrics=(compute_metrics if not PEFT else None),
        callbacks=(None if not PEFT else [SavePeftModelCallback]),
        processing_class=tokenizer
        )

    # Collect eval metrics
    METRIC_LOG = list()

    # Fine-tune model
    print("-"*80)
    # save config setting
    if training_args.save_steps:
        with open(os.path.join(out_dir, "ft_config.json"), "w") as f:
            json.dump((model_config if args.use_config else training_args.to_dict().to_json_string()), f)

    # fine tune
    print("Start fine-tuning...")
    start = time.time()
    trainer.train()

    if fetch("push_to_hub", False, False):
        trainer.push_to_hub()

    if not fetch("save_steps", False, False):
        if not PEFT:
            trainer.save_model()
            model.save_pretrained(training_args.output_dir)
            tokenizer.save_pretrained(training_args.output_dir)
            processor.save_pretrained(training_args.output_dir)
        else:
            merged_model = model.merge_and_unload()
            merged_model.save_pretrained(
                out_dir,save_adapters=True, save_embedding_layers=True)

    print("-"*35, "COMPLETE", time.time()-start, "-"*35)

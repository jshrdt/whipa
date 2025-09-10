# dataloaders & pre-processing #
import ast
import os
import yaml
import json

from datasets import load_dataset, concatenate_datasets, Dataset, IterableDataset, Audio, Value
from transformers import WhisperProcessor, WhisperTokenizer, WhisperForConditionalGeneration, BitsAndBytesConfig, TRANSFORMERS_CACHE
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer

from functools import partial
from peft import prepare_model_for_kbit_training, LoraConfig, LoraConfig, get_peft_model, PeftModel

from scripts.whipa_utils import parse_langs, get_mapping, transliterate

def lookup_g2p(sample, g2p_file, dsplit=None):
    with open(g2p_file, "r") as f:
        g2p = eval(f.read())[dsplit]

    sample["ipa"] = g2p[sample["sentence"]]
    return sample

def add_ipa(model, tokenizer):
    # add special ipa token; https://discuss.huggingface.co/t/how-to-add-all-standard-special-tokens-to-my-tokenizer-and-model/21529
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|ip|>"] + tokenizer.all_special_tokens})

    # add custom IPA language id token to lang_id
    model.generation_config.lang_to_id["<|ip|>"] = tokenizer.convert_tokens_to_ids(["<|ip|>"])[0]
    # resize token embeddings to account for new special token
    model.resize_token_embeddings(len(tokenizer))


def load_whisper_res(modelname, get_tokenizer=False, get_processor=False,
                     is_whipa=False, peft=False, trainable=False,
                     zero_shot=False) -> WhisperForConditionalGeneration:
    resources = list()

    if is_whipa==False or (peft and is_whipa==False):
        basename = modelname
    elif is_whipa==True or zero_shot:
        with open(os.path.join(os.path.dirname(modelname), "ft_config.json"), "r") as f:
            basename = json.load(f)["modelname"]
    else:
        basename = modelname

    tokenizer = WhisperTokenizer.from_pretrained(basename, task="transcribe")

    print("Loading {} model {} ({})...".format(("new" if not is_whipa else "fine-tuned"), 
                modelname, ("full size" if not peft else "in 8bit for PEFT")))

    if peft: # and torch.cuda.is_available():
        # Get new PEFT model
        if not is_whipa:
            # https://github.com/huggingface/peft/blob/main/examples/int8_training/peft_bnb_whisper_large_v2_training.ipynb
            # https://github.com/Vaibhavs10/fast-whisper-finetuning/blob/main/Whisper_w_PEFT.ipynb
            model = WhisperForConditionalGeneration.from_pretrained(modelname,
                        quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map="auto")
            # select decoder modules only
            # https://github.com/openai/whisper/discussions/1707
            target_modules = []
            for id, (name, param) in enumerate(model.named_modules()):
                if 'model.decoder' in name and ('q_proj' in name or 'v_proj' in name):
                    target_modules.append(name)

            lora_config = LoraConfig(r=32, lora_alpha=64, target_modules=target_modules,
                                     lora_dropout=0.05, bias="none")

            model = get_peft_model(model, lora_config)
            model.enable_input_require_grads() 

        else:
            # Load pre-trained PEFT model
            # get prefix from lora config to load base model
            if not modelname.startswith("openai"):
                with open(os.path.join(os.path.dirname(modelname), "ft_config.json"), "r") as f:
                    basename = json.load(f)["modelname"]
            else:
                basename = modelname.split("/")[1]
            # load a base whisper model, resize embeddings
            print(basename)
            basemodel = WhisperForConditionalGeneration.from_pretrained(basename)  # for PEFT models, the base model was not altered
            add_ipa(basemodel, tokenizer)

            if trainable:
                # prep for training if contd training from peft ckpt
                basemodel = prepare_model_for_kbit_training(basemodel)
                basemodel.enable_input_require_grads() 
            # add fine-tuned lora config to base model
            model = PeftModel.from_pretrained(basemodel, modelname, is_trainable=trainable)  # modelname should be a PEFT checkpoint

        model.print_trainable_parameters()

    else:
        # Load full model for fine-tuning
        model = WhisperForConditionalGeneration.from_pretrained(modelname)
        # Freeze encoder
        for param in model.model.encoder.parameters():
            param.requires_grad = False

    if not (is_whipa or zero_shot): add_ipa(model, tokenizer)

    if not zero_shot: model.generation_config.language = "<|ip|>"
    model.generation_config.task = "transcribe"

    resources.append(model)

    if get_tokenizer:
        resources.append(tokenizer)

    if get_processor:
        resources.append(WhisperProcessor.from_pretrained(basename, task="transcribe"))

    return (resources if len(resources)>1 else resources[0])


def load_multipa_res():
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns")

    tokenizer._added_tokens_encoder = {}
    tokenizer._added_tokens_decoder = {}

    tokenizer._special_tokens_map['bos_token'] = None
    tokenizer._special_tokens_map['eos_token'] = None

    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1,
                                                sampling_rate=16_000,
                                                padding_value=0.0,
                                                do_normalize=True,
                                                return_attention_mask=True)

    processor = Wav2Vec2Processor(feature_extractor=feature_extractor,
                                  tokenizer=tokenizer)

    multipa = Wav2Vec2ForCTC.from_pretrained(
                                "ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns",
                                attention_dropout=0.1,
                                hidden_dropout=0.1,
                                feat_proj_dropout=0.0,
                                mask_time_prob=0.05,
                                layerdrop=0.1,
                                ctc_loss_reduction="mean",
                                pad_token_id=processor.tokenizer.pad_token_id,
                                vocab_size=(processor.tokenizer.vocab_size),
                                ).eval()
    
    return multipa, tokenizer, processor


def gen_from_iterable_dataset(iterable_ds):
    # suggested by https://stackoverflow.com/questions/76227219/can-i-convert-an-iterabledataset-to-dataset
    yield from iterable_ds


def read_asc_files(asc_dir: str = "../data/arabic-speech-corpus/") -> dict:
    data = {"audio": list(), "sentence": list(), "locale": list()}
    # transcript in phonetic-transcipt.txt from ASC source
    # source: https://en.arabicspeechcorpus.com
    script_name = "phonetic-transcipt.txt" if asc_dir.endswith("corpus/") else  "phonetic-transcript.txt"
    with open(asc_dir+script_name) as f:
        data["sentence"] = [(x.split('" "')[0][1:], x.replace("'", " '").split('" "')[1][:-2]) for x in f.readlines()]
        assert len(list(set([len(x) for x in data["sentence"]])))==1, "Data format not parallel"

    # wavs in /wav
    for root, dirs, files in os.walk(os.path.join(asc_dir, "wav")):
        for fname in files:
            if fname.endswith(".wav"):
                data["audio"].append(os.path.join(root, fname))
                data["locale"].append("ara")

    # Align wav and transcription
    # sort filename & transcript lists by id-name, if id names match on
    # all lines, apply sorting and drop filename id's from transcript tuples
    if all(x[0].endswith(x[1][0]) for x in zip(sorted(data["audio"]), sorted(data["sentence"]))):
        data["audio"] = sorted(data["audio"])
        data["sentence"] = [x[1] for x in sorted(data["sentence"])]
    else:
        raise RuntimeError("Misalignment of files")

    return data


def load_multipa(langs: list, dsplit: str, limit: bool|list = False,
                 max_seconds: bool|int = False, stream: bool = True,
                 to_disk: bool = False, num_proc: int = 8) -> list[Dataset|IterableDataset]:
    # Created in crossreference to both preprocess.py and main.py, but with
    # substantial changes as the code in the source repository was nonfunctional
    # and split across files.
    # Data loading, preprocessing (including transliteration), and preparation
    # now happends on the fly, but preprocessing in advance is recommended

    # Collect language splits
    data_list = list()

    for i, lang in enumerate(langs):
        # Fetch CommonVoice data split for lang in streaming mode
        cv_data = load_dataset("mozilla-foundation/common_voice_11_0",
                               lang, split=dsplit, streaming=stream,  #! stream pass here vs to local later
                               num_proc=(None if stream else num_proc))

        # Remove unnecessary(?) metadata
        cv_data = cv_data.remove_columns(["accent", "age", "client_id", "path",
                                "down_votes", "gender", "segment", "up_votes",
                                ])

        if lang == "ta":
            # MultIPA applies this filter since pronunciation of "ச" is deemed inconsistent
            # Try to avoid processig entire dataset if limit is passed: just up
            # the limit to hopefully include sufficient samples (multipa did not account for this, filtering AFTER clipping)
            if limit and limit[i] and limit[i]<999_999 and metadata:
                ta_limit = limit[i]
                if round(ta_limit + (ta_limit*100/ta_limit)) < metadata["locales"]["ta"]["buckets"][(dsplit if dsplit != "validation" else "dev")]:
                    cv_data = cv_data.take(int(ta_limit + (ta_limit*100/ta_limit)))
            # Filter out samples
            cv_data = cv_data.filter(lambda sent: "ச" not in sent, input_columns="sentence")

        if max_seconds:
            # In multipa/main.py, remove_long_data happens during preprocessing and
            # main.py; however it is the last step executed and as such their
            # train/test samples are actually less than specified.
            # To 1) stay true to our number of samples, and 2) avoid having to store
            # these values until the end; we prune here.
            # Note: if this line breaks due to a NoneType not iterable (Feature's id=None) error, try downgrading datasets==3.2.0
            cv_data = cv_data.filter(lambda audio: len(audio["array"])<audio["sampling_rate"]*max_seconds,
                                     input_columns="audio",)

        # Clip data if limit was passed
        if limit and limit[i]:
            cv_data = cv_data.take(limit[i])

        # get IPa transcriptions from g2p file or on the fly
        g2p_file = "../data/multipa/ipa/cv_{}_ipa.txt".format(lang)
        if os.path.isfile(g2p_file):
            cv_data = (cv_data.map(partial(lookup_g2p, g2p_file=g2p_file, dsplit=dsplit))
                       if type(cv_data)==IterableDataset
                       else cv_data.map(partial(lookup_g2p, g2p_file=g2p_file, dsplit=dsplit),
                                        num_proc=num_proc))
        else:
            print("File not found: {}. Converting G2P on the fly; run scripts/g2p_multipa.py first to preprocess dataset".format(g2p_file))
            cv_data = (cv_data.map(transliterate) if type(cv_data)==IterableDataset
                       else cv_data.map(transliterate, num_proc=num_proc))

        # Add current split to DS list
        data_list.append(cv_data if not (stream and to_disk)
                         else Dataset.from_generator(partial(
                             gen_from_iterable_dataset, cv_data), features=cv_data.features))
    return data_list


def read_thchs_files(dsplit: list,
                data_dir: str = "../data/data_thchs30/",
                grid_dir: str = "../data/data_thchs30/grids-sdp/",
                mapping_path: str = "../data/data_thchs30/grids-sdp/filename-mapping.json"):

    if not os.path.isdir(grid_dir):
        raise FileNotFoundError("Directory for phonetics transcripts not found at '../data/data_thchs30/grids-sdp/'"
                                "Download grids-sdp from https://zenodo.org/records/7528596 and place them into data/data_thchs30 or pass correct path to grid_dir")
    # phonetic transcription grids have new filenames & cannot be linked to 
    # the train/dev/test splits as such. However, using the speech-dataset-parser
    # for thchs with the sdp-grids creates one such file.
    mapping = get_mapping(mapping_path)

    data = {"audio": list(), "sentence": list(), "locale": list()}
    for root, dirs, files in os.walk(os.path.join(data_dir, dsplit)):
        # data split folders contain .wav audios (desired), and .wav.trn orthographic
        # transcripts: not needed. Names of the .wav files are used to identify
        # the corresponding phonetic transcripts in grid-sdp via the mapping file.
        for fname in files:
            if fname.endswith(".wav"):
                data["audio"].append(os.path.join(root, fname))
                # technically not a sentence but named for consistency with converter funcs
                data["sentence"].append(os.path.join(grid_dir, mapping[fname][:-4]+".TextGrid"))
                data["locale"].append("cmn")
            else:
                # orthographic transcripts
                pass
    return data


def read_tusom2021_files(dsplit: str, dirpath: str = "../data/tusom2021-main/data") -> dict:
    data = {"audio": list(), "ipa": list(), "locale": list()}
    # load wavfile to transcription mapping
    with open(os.path.join(dirpath, "{}.yml".format(dsplit)), "r") as f:
        mapping = yaml.load(f, Loader=yaml.SafeLoader)

    # combine with wav files
    for root, dirs, files in os.walk(os.path.join(dirpath, "wav")):
        for fname in files:
            if fname.endswith(".wav") and fname in mapping.keys():
                data["audio"].append(os.path.join(root, fname))
                data["locale"].append("tus")
                data["ipa"].append(mapping[fname]["no_tones"])

    return data

def s2frame(secs, sampling_rate=16_000):
    return round(int(secs)*sampling_rate + (secs-int(secs))*sampling_rate)

def read_sanna(id="01"):
    import textgrid
    praatfile = "sanna{}.TextGrid".format(id)
    audio = "sanna{}.wav".format(id)

    wav = Dataset.from_dict({"audio": [os.path.join("../data/sanna/", audio)]}
                            ).cast_column("audio", Audio(sampling_rate=16_000))["audio"][0]["array"]

    tg = textgrid.TextGrid.fromFile(os.path.join("../data/sanna/", praatfile))
    ds = {"audio": list(), "ipa": list(), "locale": list()}

    phonetier = {"01": "S1","02": "S1",
                 "03": "YS", "05": "YS", "06": "YS",
                 "04": "Women1-1973"}[id]
    for i, tag in enumerate(tg.getList(phonetier)[0]):
        if i==0:
            offset = s2frame(tag.bounds()[0]) if s2frame(tag.bounds()[0])>0 else 0
            if phonetier=='YS': offset=0

        if tag.mark and tag.mark not in ["eh", "éh", "eh eh"]:  # only "proper" sanna samples have a transcription
            ds["ipa"].append(tag.mark)
            stamp = tag.bounds()
            start, stop = (s2frame(stamp[0])-offset-10, s2frame(stamp[1])-offset+10)
            if start <0: start=0
            ds["audio"].append({"array": wav[start:stop], "sampling_rate": 16_000} )
            ds["locale"].append("sanna")

    return Dataset.from_dict(ds)


## shared ##
def load_ft_data(source: str, dsplit: str = None, lgs: str|list = "all", dirpath: str = False,
                 limit: bool = False, max_seconds: int|bool = 6, stream: bool = True,
                 to_disk: bool = False, num_proc: int = 8) -> tuple[Dataset|IterableDataset, list[int]]:
    # Collection of all currently valid corpora names
    known_sources = ["asc", "multipa", "thchs", "tusom", "voxangeles", "multipa_test"]
    assert source in known_sources, ("Dataset name unknown. Choose one of", known_sources)
    # Ensure corpus-lg matches and list format to avoid accidental string iteration
    if lgs:
        lgs = parse_langs(lgs, source)[source]
    data_stats = 0
    if limit:
        # replace format limit multipa: (lim.split() if " " in lim else ([lim, ] if type(lim)!=list else lim))
        if type(limit)!= list: limit = [limit, ]
        if type(lgs)==list and len(limit)<len(lgs): limit = limit*len(lgs)
        limit = [int(x) for x in limit]


    print("Loading {} (limit: {}) from {}-{}.".format(
        lgs, ([x if x!=0 else "-" for x in limit] if limit else "-"), source, dsplit))

    if source=="multipa":
        if dsplit == "dev": dsplit = "validation"
        # Read commonvoice release metadata to retrieve size of datasets
        # (in case of streaming size cannot be estimted from IterableDataset,
        # but is needed to set the sizes for tamil filtering and clipping)
        cv_cache_path = os.path.join(TRANSFORMERS_CACHE,
                                     "datasets--mozilla-foundation--common_voice_11_0/snapshots")
        global metadata
        metadata = False
        if os.path.isfile(cv_cache_path):
            for root, dirs, files in os.walk(os.path.join(cv_cache_path)):
                for fname in files:
                    if fname.startswith("release"):
                        with open(os.path.join(root, fname), "r") as f:
                            metadata = ast.literal_eval(f.read().split("STATS = ")[1])

        # Cast data cutoff to int or get max size of language's cv split from metadata 
        # if 0 was passed as partial max value, if no metadata: irrationally large n instead.
        if limit:
            limit = [val if val!=0
                     else (metadata["locales"][lgs[i]]["buckets"][dsplit]
                           if metadata else 999_999)
                     for i, val in enumerate(limit)]

        data = load_multipa(lgs, dsplit, limit=limit, max_seconds=max_seconds,
                            stream=stream, to_disk=to_disk, num_proc=num_proc)
        data_stats = limit if limit else [0]

    elif source=="voxangeles":
        data = load_dataset("speech31/voxangeles", streaming=False)["test"] # only 'test' split defined
        # filter out languages
        if not "all" in lgs:
            data = (data.filter(lambda lg: lg in lgs, num_proc=num_proc, input_columns="lang") if type(data)==Dataset
                    else data.filter(lambda lg: lg in lgs, input_columns="lang"))
        # create data splits 
        if dsplit:
            # default: 0.8, 0.1, 0.1
            data = data.class_encode_column("lang")
            splits = data.train_test_split(train_size=0.8, stratify_by_column="lang")
            if dsplit=="train":
                data = splits["train"]
            else:
                # dev/test splits correspoing to 0.1 of og data
                dev_test = splits["test"].train_test_split(train_size=0.5, stratify_by_column="lang")
                data = dev_test["train"] if dsplit=="dev" else dev_test["test"]
        # clip data, you can do this, but it will lead not lead to a balanced clip
        if limit:
            limit = [int(limit[0]) if int(limit[0])!=0 else (len(data) if type(data)==Dataset else 999_999),]
            data = data.take(limit[0])
        # add locale and ipa columns for consistency
        data = data.add_column("locale", (data["lang"] if not dsplit else 
                        [data.features["lang"].int2str(i) for i in data["lang"]]))
        data = data.rename_column("word", "ipa") 
        data = data.remove_columns(["file", "phn", "lang"])

    elif source=="asc":
        # Arabic Speech Corpus only has train/test splits, consists of medium
        # length utterance (multi word/sentence) & has orthographic and 
        # phonetic Buckwalter transcriptions.
        if not dirpath: dirpath = "../data/arabic-speech-corpus/"
        dirpath = (dirpath if dsplit in ["train", "dev"]
                   else os.path.join(dirpath, "test set/"))
        # Read files, align audio files and transcripts, cast to dataset
        data = Dataset.from_dict(read_asc_files(dirpath))
        if dsplit!="test":
            data = data.train_test_split(train_size=0.9)[("train" if dsplit=="train" else "test")]
        # Clip if limit was passed
        if limit: data = data.take(limit[0])
        # Transliterate phonetic Buckwalter transcription to IPA
        data = data.map(transliterate, num_proc=num_proc)
        data_stats = [len(data), ]

    elif source=="thchs":
        if not dirpath: dirpath = "../data/data_thchs30"
        # Read files, align audio files and transcripts, cast to dataset
        # size of train is 10k samples -> process as iterable DS
        # train, dev, test splits predefined
        data_raw = Dataset.from_dict(read_thchs_files(dsplit, dirpath))
        data_stats = (limit if limit else len(data_raw["audio"]))

        # Clip data prior to mapping to reduce workload
        if limit: data_raw = data_raw.take(limit[0])
        if stream:
            # .map and on custom IterableDS loses feature info  -> breaks audio casting; cf. https://github.com/huggingface/datasets/issues/5284, https://github.com/huggingface/datasets/issues/5752
            # but can be passed manually since they are known
            features = data_raw.features.copy()
            features["ipa"] = Value(dtype="string", id=None)
        # get phonetic transcriptions from grids-sdp textgrid files
        data = (data_raw.to_iterable_dataset().map(transliterate, features=features) 
                if stream else data_raw.map(transliterate, num_proc=num_proc))
        data = data.remove_columns(["sentence"])

    elif source=="multipa_test":
        data = Dataset.from_json("../data/multipa/test/test_set.json")

    # Concat and audios cast_column, must happen AFTER the loading functions but BEFORE
    # attempting to combine data from different corpora (cannot combine multi-dim arrays).
    # Combine language splits
    if type(data)==list: data = concatenate_datasets(data)

    # Resample audio
    print("Resampling audio...")
    data = data.cast_column("audio", Audio(sampling_rate=16_000))

    if type(data)==Dataset: data_stats = [len(data), ]

    # Transform to IterableDS if specified
    if type(data)==Dataset and not to_disk: 
        if not data_stats: data_stats = [limit,] if (limit and limit<len(data)) else [len(data),]
        data = data.to_iterable_dataset()

    # and vice versa
    if type(data)==IterableDataset and to_disk:
        data = Dataset.from_generator(partial(gen_from_iterable_dataset, data),
                                       features=data.features)
        if not data_stats: data_stats = [limit,] if (limit and limit<len(data)) else [len(data),]

    print("Samples {} {}: {}".format(source, dsplit, sum(data_stats)))

    return data

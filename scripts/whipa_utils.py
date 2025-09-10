import json
import os
import re
import shutil
import sys

from epitran import Epitran
from datasets import IterableDataset, Dataset, concatenate_datasets
from functools import partial
from transformers import WhisperProcessor, WhisperTokenizer
import textgrid

# Combined script from MultIPA G2P tool and romkan release (https://github.com/soimort/python-romkan/blob/master/src/romkan/common.py, 22.02.2025)
from scripts.japanese_to_ipa_romkan_merge import Japanese2IPA

sys.path.insert(0, "../data/multipa") # multipa dir, https://github.com/ctaguchi/multipa
from converter.maltese_to_ipa import Maltese2IPA
from converter.finnish_to_ipa import Finnish2IPA
from converter.greek_to_ipa import Greek2IPA
from converter.tamil_to_ipa import Tamil2IPA


def parse_langs(lgs: str|list, corpora: list) -> dict[str: list|str]:
    # Format as iterable list
    if type(lgs) == str: lgs = lgs.split() if " " in lgs else [lgs,]
    if type(corpora) == str: corpora = corpora.split() if " " in corpora else [corpora,]
    # Hard-code: valid corpus-lg combinations
    all_lgs = {"asc": "ara",     # monolingual south levantine arabic
               "multipa": ["ja", "pl", "mt", "hu", "fi", "el", "ta"],
               "thchs": "cmn",  # monolingual mandarin chinese
               "tusom": "tus",  # monolingual tusom
               "voxangeles": ["abk", "ace", "ady", "aeb", "afn", "afr", "agx", "ajp", 
                                "aka", "apc", "ape", "apw", "asm", "azb", "bam", "bem",
                                "ben", "bfd", "bfq", "bhk", "bin", "brv", "bsq", "bwr",
                                "cbv", "ces", "cha", "cji", "col", "cpn", "dag", "dan",
                                "deg", "dyo", "efi", "ell", "ema", "eus", "ewe", "ffm",
                                "fin", "fub", "gaa", "gla", "guj", "gwx", "hak", "hau",
                                "haw", "heb", "hil", "hin", "hni", "hrv", "hun", "hye",
                                "ibb", "ibo", "idu", "ilo", "isl", "its", "kan", "kea",
                                "khm", "klu", "knn", "kri", "kub", "kye", "lad", "lar",
                                "lav", "led", "lgq", "lit", "lkt", "lug", "mak", "mal",
                                "mlt", "mya", "nan", "njm", "nld", "ozm", "pam", "pes",
                                "prs", "run", "sbc", "tsw", "tzm", "wuu", "yue"],  # 95 total
               }

    use_lgs = {corp: list() for corp in corpora}
    i = 0
    for l in lgs:
        if l in all_lgs[corpora[i]]:
            use_lgs[corpora[i]].append(l)
        elif l=="all":
            use_lgs[corpora[i]] = all_lgs[corpora[i]]
        else: 
            raise ValueError("Unknown language {} for corpus {}. ".format(l, corpora[i])
                            +"Verify that corpora and languages are passed in the same order"
                            +" or choose one of known language keys (or 'all) for {}: {}".format(
                                corpora[i], all_lgs[corpora[i]]))
    return use_lgs

def transliterate(sample: dict):
    ##! adapted from multipa/preprocess.py ##
    ##! sections changed marked with #! ##

    #! if "chapter_id" in sample.keys():  #! prev: sample.column_names, throws AttributeError for Laz<Rows object from train.map(transliterate)
    #!     lang = "en"
    #! else:
    ## ! TBD: pass lang as kw arg, reduce to input column sentence, then rename to ipa outside
    lang = sample["locale"]
    sent = sample["sentence"]
    if lang == "ja":
        converter = Japanese2IPA()
        ipa = converter.remove_ja_punct(sent)
        ipa = converter.convert_sentence_to_ipa(ipa)
    elif lang == "mt":
        ipa = Maltese2IPA.maltese_generate_ipa(sent)
    elif lang == "fi":
        ipa = Finnish2IPA.finnish_generate_ipa(sent)
    elif lang == "el":
        ipa = Greek2IPA.greek_generate_ipa(sent)
    elif lang == "hu":
        ipa = re.findall(r"[\s\w]", sent.lower(), re.MULTILINE)
        ipa = "".join(ipa)
        epi = Epitran("hun-Latn")
        ipa = epi.transliterate(ipa)
    elif lang == "pl":
        ipa = re.findall(r"[\s\w]", sent.lower(), re.MULTILINE)
        ipa = "".join(ipa)
        epi = Epitran("pol-Latn")
        ipa = epi.transliterate(ipa)
    elif lang == "ta":
        ipa = Tamil2IPA.tamil_generate_ipa(sent)
    #! elif lang == "en":
    #!     ipa = English2IPA.english_generate_ipa(sent)
    elif lang == "ara":  #! added ara for Arabic Speech Corpus (Halabi 2016)
        #! transliterate buckwalter phonetic transcript to IPA
        converter = Buckwalter2IPA()  #! 
        ipa = converter.convert_buckwalter_to_ipa(sent)  #!
    elif lang == "cmn":  #! added cmn for THCHS-30
        converter = THCHS2IPA()  #! 
        ipa = converter.convert_thchs_to_ipa(sent)  #! sent is a TextGrid file path here
    else:
        raise Exception("Unknown locale (language) found")
    sample["ipa"] = "".join(ipa.split())
    return sample

def gen_from_iterable_dataset(iterable_ds):
    # suggested by https://stackoverflow.com/questions/76227219/can-i-convert-an-iterabledataset-to-dataset
    yield from iterable_ds

def conjoin_ds(datasets, to_disk=False):
    data = concatenate_datasets([
            ds if to_disk==False and type(ds)==IterableDataset else
            (ds.to_iterable_dataset() if to_disk==False and type(ds)==Dataset
             else Dataset.from_generator(partial(gen_from_iterable_dataset, ds),
                                         features=ds.features))
            for ds in datasets
            ])
    return data

def prepare_dataset_ipa(batch: dict, processor: WhisperProcessor,
                        tokenizer: WhisperTokenizer) -> dict: #! added processor/tokeniser as args
    ##! adapted from multipa/main.py to be used with other datasets too ##
    ##! sections changed marked with #! ##
    audio = batch["audio"] if "audio" in batch.keys() else {"array": batch["array"],
                                                            "sampling_rate": 16_000}

    # batched output is unbatched
    batch["input_features"] = getattr(
        processor(audio["array"], sampling_rate=audio["sampling_rate"]),
        ("input_features" if type(processor)==WhisperProcessor else "input_values")
        )[0]
    # with processor.as_target_processor(): #!
    #! edited to use WhisperProcessor & tokeniser
    batch["labels"] = tokenizer.encode_plus(batch["ipa"], add_special_tokens=True).input_ids
    return batch


def prep_dataset(data: Dataset|IterableDataset, processor: WhisperProcessor,
                 tokenizer: WhisperTokenizer, seed: int = 11,
                 max_len: bool|int = False, num_proc: int = 8) ->  Dataset|IterableDataset:
    # Prep audio data with Whisper tools, extracting input values & labels
    print("Creating input values and labels...")
    data = (data.map(partial(prepare_dataset_ipa, processor=processor, tokenizer=tokenizer)) 
            if type(data)==IterableDataset
            else data.map(partial(prepare_dataset_ipa, processor=processor,
                                  tokenizer=tokenizer), num_proc=num_proc))

    # Rm sampels that exceed Whisper's max input sequence length
    if max_len:
        if type(data)==Dataset:
            data = data.select([i for i, b in enumerate(data["labels"]) if len(b)<max_len])
        else:
            data = data.filter(lambda batch: len(batch)<max_len, input_columns=["labels"]) # passing input col ca 3500x times faster than without (300s->0,08s)

    # Shuffle data
    if seed:
        data = data.shuffle(seed=seed)

    return data

def get_dirname(outdir):
    if os.path.isdir(outdir):
        i = 2
        dirn = outdir+"-"+str(i)
        while os.path.isdir(dirn):
            i += 1
            dirn = outdir+"-"+str(i)
        outdir = dirn
        os.mkdir(outdir)
    return outdir


class Buckwalter2IPA():
    # Lookup table synthesised from:
    # (1) Halabi, N. (2016). (dissertation). Modern standard Arabic phonetics for speech synthesis. 
    # University of Southampton University of Southampton.; tables 4-7 on pp.46f. (=corpus source)
    # lines marked with # deviate from (1), reason stated after
    # (2) https://en.wikipedia.org/wiki/Buckwalter_transliteration
    # (3) https://en.wikipedia.org/wiki/Romanization_of_Arabic

    rev_BW2IPA = {
        # in IPA fashion we use ː not :
        # "sil" indicates speaker silence, is skipped in transliteration
        # back-to-back appearance of same vowel is transcriped as a long vowel, e.g. aa -> aː
        "'": "", # word boundary
        "b": "b", # assumed as B does not appear in corpus
        "T": "tˤ",
        "^": "θ",
        "j": "ʒ", # assumed, as J is listed twice 
        "r": "r",
        "z": "z",
        "s": "s",
        "$": "ʃ",
        "S": "sˤ",
        "G": "ɣ",
        "g": "ɣ", # differentiation unclear; G only appears 3 times in corpus
        "f": "f",  # assumed as F does not appear in corpus
        "q": "q", # assumed as Q does not appear in corpus
        "k": "k",  # assumed as K does not appear in corpus
        "l": "l", # assumed as L does not appear in corpus
        "y": "j",
        "v": "v",
        "p": "p",
        "J": "d͡ʒ",
        "aa": "æː",
        "i": "i", #assumed, only appears 6 times in corpus
        "i0": "i",
        "u1": "o̞",
        "U1": "o̞", # assumed, same corrresponding grapheme
        "i1": "ɪ",
        "AA": "ɑː",
        "A": "ɑ",
        "H": "ħ",
        "x": "x",  # assumed as X does not appear in corpus
        "d": "d", # assumed as D appears twice; capital letters often for shaddah, in line w wiki entry
        "*": "ð",
        "D": "dˤ",
        "t": "t", # assumed as T appears twice; capital letters often for shaddah
        "Z": "ðˤ",
        "E": "ʕ",
        "m": "m", # assumed as M does not appear in corpus
        "n": "n", # assumed as N does not appear in corpus
        "h": "h", # assumed as H appears twice; capital letters often for shaddah
        "W": "w",
        "w": "w", #
        "uu0": "uː",
        "u": "u", # assumed, unclear, only appears 3 times in corpus
        "ii0": "iː",
        "a": "a",
        "u0": "æ",
        "uu1": "o̞ː",
        "UU1": "o̞ː", # assumed, same grapheme
        "ii1": "ɪː",
        "sil": "",
        " ": "",
        # "j": "g", # assumed, duplicate?
        "<":  "ʔ", # assumed
        "II0": "i", #assumed from 0650 kasrah
        "U0": "a", # assumed from 064E fatḥah
        "UU0": "w", # assumed from 0648	wāw
        "I0": "ɪ", # assumed, same corresponding grapheme
        "I1": "ɪ", # assumed, same corresponding grapheme
        "-": "ə", # retrieved from asc readme info
        "dist": "" # corrputed phone
        }

    def convert_buckwalter_to_ipa(self, sent: str) -> str:
        ipa_line = list()

        for char in sent.split():
            if char in self.rev_BW2IPA: ipa_line.append(self.rev_BW2IPA[char])
            elif len(char)==2 and char[0] in self.rev_BW2IPA:
                ipa_line.append(self.rev_BW2IPA[char[0]]+"ː")
            else:
                raise ValueError("Unknown character", char)
        return " ".join(ipa_line)


class THCHS2IPA():
    def convert_thchs_to_ipa(self, tg_path: str, normalize=False) -> str:
        # read textgrid with phonetic transcription from grids-sdp
        tg = textgrid.TextGrid.fromFile(tg_path)
        ipa = "".join([tag.mark for tag in tg.getList("transcription")[0]])
        # remove silence markers and punctuation
        for mark in ["SIL"+a for a in ["X", "0", "1", "2", "3"]] + ["。",]:
            ipa = ipa.replace(mark, "")
        return ipa


def get_mapping(mapping_path, data_dir = "../data/data_thchs30/"):
    if not os.path.isfile(mapping_path):
        tmp_dir = os.path.join(data_dir, "thchs30_tmp/")
        # creates temporary folder with structure speaker > {.wav, .wav.trn}
        # .trn are orthographic transcriptions, to be ignored
        # Since we only need the file mapping and no redundant copies of data
        # onyl the filename mapping is kept & moved to grid_dir. tmp_dir is 
        # removed.
        print("Processing thchs dir to get file id mapping for grids-sdp")

        os.system("dataset-converter-cli convert-thchs {}} {}".format(data_dir, tmp_dir))
        shutil.move(os.path.join(tmp_dir, "filename-mapping.json"), os.path.join(data_dir, "grids-sdp/"))

    with open(mapping_path) as f:
        # maps new names to old ones eg. {'A11;2;chi/001.TextGrid': 'data/A11_0.TextGrid'} (also for wavs)
        # revert mapping to recreate data splits
        mapping = {val.split("/")[1]: idx for idx, val in json.load(f).items()}

    return mapping

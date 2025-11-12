# whipa
Code repository of "Towards Language-Agnostic STIPA: Universal Phonetic Transcription to Support Language Documentation at Scale."

Abstract:  
This paper explores the use of existing state-of-the-art speech recognition models (ASR) for the task of transcribing speech with narrow phonetic transcriptions using the International Phonetic Alphabet (Speech-to-IPA, STIPA). Unlike conventional ASR systems focused on orthographic output for high-resource languages, STIPA can be used as a language-agnostic interface valuable for documenting under-resourced and unwritten languages. We introduce a new STIPA dataset for South Levantine Arabic and present a large-scale evaluation of STIPA models across 21 language families. Additionally, we provide a use case on Sanna, a severely endangered language. Our findings show that fine-tuned ASR models can produce accurate IPA transcriptions with limited supervision, significantly reducing phonetic error rates even in extremely low-resource settings. The results highlight the potential of STIPA for scalable language documentation and the relevance of training data composition.


## usage

Refer to `code/deploy.py` for loading of fine-tuned (Lo)WhIPA models and inference with fallback heuristics during decoding.

#### 1) Loading the model
```
from deploy import WHIPA  
whipa = WHIPA(model_path="path/to/whipa-model")  
# OR:
lowhipa = WHIPA(model_path="path/to/lowhipa-model", lora=True)
```
* specifying the model base is recommended, typically `base_model="openai/whisper-base"` or `base_model="openai/whisper-large-v2"`
* WHIPA class will also load WhisperTokenizer and WhisperPreprocessor as attributes

#### 2) Loading the data (see code/scripts/whipa_utils.prep_dataset)

* expected format: class `datasets.arrow_dataset.Dataset` with `dict_keys(['audio', 'input_features'])`
* `Note:` `sample['audio']['array']` is used to calculate the length of the sample (in seconds)
 
#### 3) Inference

Use

    ipa_prediction = whipa.transcribe_ipa(sample)

Experimentation with parameters during `transcribe_ipa()` is encouraged:  

* `n_beams`: default decoding beam size (int)
* `fallback_beams`: backoff beam sizes & order (list)
* `max_phones_per_sec_rate`: maximum phone/second rate (int/float)
* repetition penalty (float)
* exponential_decay_length_penalty (float)

See also : [WhisperForConditionalGeneration.generate](https://huggingface.co/docs/transformers/en/model_doc/whisper#transformers.WhisperForConditionalGeneration.generate)

#### 4) Evaluation metrics

The `STIPA_METRICS` class in code/scripts/metrics builds on PanPhon and [ctaguchi/multipa](https://github.com/ctaguchi/multipa) and includes the following:

* optional string normalisation (NFD + rule-based replacement for e.g. Chao tone letters and outdated IPA characters + removal of orthographic characters (e.g. punctuation)), default: yes
* optional removal of Chao tones and suprasegmental, default: no
* Levenshtein metric (do_lvnsthn)
* Character Error Rate (CER, do_cer)
* Phone Edit Distance (PED, do_ped)
* Phonetic Feature Edit Distance (PFED, do_pfed)
* Phone Error Rate (PER, do_per)
* normalised Phone Error Rate (PER_norm, do_per_norm)
* Phonetic Feature Error Rate (PFER, do_pfer_)
* normalised Phonetic Feature Error Rate (PFER_norm, do_pfer_norm)
* (old) PER/PFER implementation as in [ctaguchi/multipa](https://github.com/ctaguchi/multipa) (inactive)

Basic usage:
```
from metrics import STIPA_METRICS
eval_metrics = STIPA_METRICS()
eval_metrics.compute_all(pred=predicted_ipa, gold=gold_ipa, char_based=False)
# OR, e.g.:
eval_metrics.do_pfer(pred=predicted_ipa, gold=gold_ipa, char_based=False)
```
The WhIPA re-implementations account for some more phonetic detail than those referenced from PanPhon and MultIPA. While all implementation build on the PanPhon feature table, unknwon phone combinations are handled differently. E.g.: The combination of [ä] + [:] to [ä:] happens to be missing from PanPhon, effects are as follows:
```
PanPhon-PFER(hyp="ä:", ref="a") = 0.0%; retrieves no information for [ä:]
MultIPA-PFER(hyp="ä:", ref="a") = 200%; assigns zero-vectors to both [ä] and [:], overshooting target length
WhIPA-PFER(hyp="ä:", ref="a") = 4.17%; retrieves feature vectors from the longest match ([a]), and adds information from the PanPhon diacritics rulebook (here: [-long] -> [+ long] feature).
```

Still, metrics leave room for improvement. Especially regarding length mismatch sensitivity:
```
Reference: [s t ø u l ə]  
Hypothesis A: [s t ø: l ɛ], PED 3, PFER 50%, PFED 1.08%, PFER 16.06%  
Hypothesis B: [s t ø p l ɛ], PED 2, PFER 33.33% PFED 0.42, PFER 6.94%
```
Hypothesis A is punished more severly due to a length mismatch despite the phonetic surface form of affected segments [ø:] (long vowel) being closer to the reference [øu] (two vowels, diphthong or hiatus) than Hypothesis B [øp] (vowel + voiceless plosive).


### code/

* fine-tune.py: Fine-tuning script, to use with cmd line args OR ft_config.json
```	
$ python fine_tune.py -config [key-in-config]
```
* eval.py: prediction/evaluation script
```
$ python eval.py -ckpt [model-folder-name] -corp asc -norm -micro -how fallback -peft -n 5
```

####  code/scripts :  

* metrics.py: PER/PFER computation, MultIPA PER/PFER included for reproduction
* loader.py: loading for cv train and test data, voxangeles, acs, thchs30, sanna
* whipa_utils.py: various functions to declutter main scripts, incl buckwalter2IPA transliteration
* japanese_to_ipa_romkan_merge.py: merger of multipa/conveter/japanese_to_ipa and romkan/src/romkan/common.py kana-hepburn conversion function/tables romkan.to_roma() (incompatible with >Python3.11 due to imp module)
* preproc_multipa_test_set.py: use to prep CV transfer test set


## data

* see data/README.md

## models (to be updated, sep20 2025)
| Name | Model | Data | Checkpoint | Link |
| --- | --- | --- | --- | --- | 
| whipa-base-cv | Whisper-base, full fine-tuning | 1k samples each of CommonVoice Greek, Finnish, Japanese, Hungarian, Maltese, Polish, Tamil | 4 | [jshrdt/whipa-base-cv](https://huggingface.co/jshrdt/whipa-base-cv) |
| whipa-large-cv | Whisper-large-v2 | 1k samples each of CommonVoice Greek, Finnish, Japanese, Hungarian, Maltese, Polish, Tamil | 6 | [jshrdt/whipa-large-cv](https://huggingface.co/jshrdt/whipa-large-cv) |
| lowhipa-base-cv | Whisper-base, PEFT LoRA adapter | 1k samples each of CommonVoice Greek, Finnish, Japanese, Hungarian, Maltese, Polish, Tamil | 4 | [jshrdt/lowhipa-base-cv](https://huggingface.co/jshrdt/lowhipa-base-cv) |
| lowhipa-large-cv | Whisper-large-v2, PEFT LoRA adapter | 1k samples each of CommonVoice Greek, Finnish, Japanese, Hungarian, Maltese, Polish, Tamil | 8 | [jshrdt/lowhipa-large-cv](https://huggingface.co/jshrdt/lowhipa-large-cv) |
| lowhipa-base-asc| Whisper-base, PEFT LoRA adapter | 1k samples from Arabic Speech Corpus (ASC) | 6 | [jshrdt/lowhipa-base-asc](https://huggingface.co/jshrdt/lowhipa-base-asc) |
| lowhipa-large-asc| Whisper-large-v2, PEFT LoRA adapter | 1k samples South Levantine Arabic from Arabic Speech Corpus (ASC) | 6 | [jshrdt/lowhipa-large-asc](https://huggingface.co/jshrdt/lowhipa-large-asc) |
| lowhipa-base-thchs30| Whisper-base, PEFT LoRA adapter | 1k samples Mandarin Chinese from THCHS-30 | 4 | [jshrdt/lowhipa-base-thchs30](https://huggingface.co/jshrdt/lowhipa-base-thchs30) |
| lowhipa-large-thchs30| Whisper-large-v2, PEFT LoRA adapter | 1k samples Mandarin Chinese from THCHS-30 | 10 | [jshrdt/lowhipa-large-thchs30](https://huggingface.co/jshrdt/lowhipa-large-thchs30) |
| lowhipa-base-comb| Whisper-base, PEFT LoRA adapter | 1k samples Mandarin Chinese from THCHS-30 | 4 | [jshrdt/lowhipa-base-comb](https://huggingface.co/jshrdt/lowhipa-base-comb) |
| lowhipa-large-comb| Whisper-large-v2, PEFT LoRA adapter | 1k samples Mandarin Chinese from THCHS-30 | 6 | [jshrdt/lowhipa-large-comb](https://huggingface.co/jshrdt/lowhipa-large-comb) |
| lowhipa-large-SR | Whisper-large-v2, PEFT LoRA adapter | 1k samples each from CV Greek, CV Maltese, ASC South Levantine Arabic | 10 | [jshrdt/lowhipa-large-sr](https://huggingface.co/jshrdt/lowhipa-large-sr)


## citation

WhIPA paper:  
Jacob Lee Suchardt, Hana El-Shazli, and Pierluigi Cassotti. 2025. Towards Language-Agnostic STIPA: Universal Phonetic Transcription to Support Language Documentation at Scale. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 31411–31427, Suzhou, China. Association for Computational Linguistics. https://doi.org/10.18653/v1/2025.emnlp-main.1600

```
@inproceedings{suchardt-etal-2025-towards,
    title = "Towards Language-Agnostic {STIPA}: Universal Phonetic Transcription to Support Language Documentation at Scale",
    author = "Suchardt, Jacob Lee  and
      El-Shazli, Hana  and
      Cassotti, Pierluigi",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.1600/",
    doi = "10.18653/v1/2025.emnlp-main.1600",
    pages = "31411--31427",
    ISBN = "979-8-89176-332-6",
    abstract = "This paper explores the use of existing state-of-the-art speech recognition models (ASR) for the task of generating narrow phonetic transcriptions using the International Phonetic Alphabet (STIPA). Unlike conventional ASR systems focused on orthographic output for high-resource languages, STIPA can be used as a language-agnostic interface valuable for documenting under-resourced and unwritten languages. We introduce a new dataset for South Levantine Arabic and present the first large-scale evaluation of STIPA models across 51 language families. Additionally, we provide a use case on Sanna, a severely endangered language. Our findings show that fine-tuned ASR models can produce accurate IPA transcriptions with limited supervision, significantly reducing phonetic error rates even in extremely low-resource settings. The results highlight the potential of STIPA for scalable language documentation."
}
```

Master's thesis:  
```
@mastersthesis{suchardt25-universalphonerecognitionthesis,
    author = {Suchardt, Jacob Lee},
    title = {Training for the Unexpected: Approaching Universal Phone Recognition for Computer-Assisted IPA Transcription of Low-Resource Languages},
    school = {University of Gothenburg},
    year = {2025},
    type = {Master's thesis},
    note= {\url{https://hdl.handle.net/2077/87916}}
}
```

## Acknowledgements

### PanPhon

* https://github.com/dmort27/panphon
```
@inproceedings{Mortensen-et-al:2016,
  author    = {David R. Mortensen and
               Patrick Littell and
               Akash Bharadwaj and
               Kartik Goyal and
               Chris Dyer and
               Lori S. Levin},
  title     = {PanPhon: {A} Resource for Mapping {IPA} Segments to Articulatory Feature Vectors},
  booktitle = {Proceedings of {COLING} 2016, the 26th International Conference on Computational Linguistics: Technical Papers},
  pages     = {3475--3484},
  publisher = {{ACL}},
  year      = {2016}
}
```

### MultIPA repository and paper

* https://github.com/ctaguchi/multipa [ver: jun 06, 2024; 69bd23fcaf2270d8b87d0e7255f74de821c52986]

```
@inproceedings{taguchi23_interspeech,
	title = {Universal Automatic Phonetic Transcription into the International Phonetic Alphabet},
	author = {Chihiro Taguchi and Yusuke Sakai and Parisa Haghani and David Chiang},
  	year = {2023},
  	booktitle = {Interspeech 2023},
  	pages = {2548--2552},
  	doi = {10.21437/Interspeech.2023-2584},
  	issn = {2958-1796},
}
```


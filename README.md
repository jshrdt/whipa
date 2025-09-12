# whipa
Towards Language-Agnostic STIPA: Universal Phonetic Transcription to Support Language Documentation at Scale

## usage

### code

* fine-tune.py: Fine-tuning script, to use with cmd line args OR ft-config.json
	
> $ python fine_tune.py -config lowhipa-base-cv

* eval.py: prediction/evaluation script

> $ python eval.py -ckpt lowhipa-base-cv -corp asc -norm -micro -how fallback -peft -n 5


#### support scripts :  

* metrics.py: PER/PFER computation, MultIPA PER/PFER included for reproduction
* loader.py: loading for cv train and test data, voxangeles, acs, thchs30, sanna
* whipa_utils.py: various functions to declutter main scripts, incl buckwalter2IPA transliteration
* japanese_to_ipa_romkan_merge.py: merger of multipa/conveter/japanese_to_ipa and romkan/src/romkan/common.py kana-hepburn conversion function/tables romkan.to_roma() (incompatible with >Python3.11 due to imp module)
* preproc_multipa_test_set.py: use to prep CV transfer test set


## data

* see data/README.md

## models
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

## Acknowledgements

### Multipa

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

## citation

[PAPER TBA]

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


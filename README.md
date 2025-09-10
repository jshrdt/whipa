# whipa
Towards Language-Agnostic STIPA: Universal Phonetic Transcription to Support Language Documentation at Scale

## usage

### code

* fine-tune.py: Fine-tuning script, to use with cmd line args OR ft-config.json
	
> $ python fine_tune.py -config cv_whipa_base

* eval.py: prediction/evaluation script

> $ python eval.py -ckpt b_thchs_bl1k/checkpoint-252 -corp voxangeles -norm -micro -sa b_thchs_bl1k-4.txt -how fallback -peft

#### support scripts :  

* metrics.py: PER/PFER computation, MultIPA PER/PFER included for reproduction
* loader.py: loading for cv train and test data, voxangeles, acs, thchs30, sanna
* whipa_utils.py: various functions to declutter main scripts, incl buckwalter2IPA transliteration
* japanese_to_ipa_romkan_merge.py: merger of multipa/conveter/japanese_to_ipa and romkan/src/romkan/common.py kana-hepburn conversion function/tables romkan.to_roma() (incompatible with >Python3.11 due to imp module)
* preproc_multipa_test_set.py: use to prep CV transfer test set


## data

* see data/README.md

## models
| Model | Data | Link |
| --- | --- | --- |  
| TBD |  |  |

## citation

[PAPER TBA]

Master's thesis:  
```
@mastersthesis{suchardt25-universalphonerecognitionthesis,
    author = {Suchardt, J. L.},
    title = {Training for the Unexpected: Approaching Universal Phone Recognition for Computer-Assisted IPA Transcription of Low-Resource Languages},
    school = {University of Gothenburg},
    year = {2025},
    type = {Master's thesis},
    note= {\url{https://hdl.handle.net/2077/87916}}
}
```


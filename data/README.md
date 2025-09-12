# ./data

Unzip & place corpora here

Compatible with:

### Arabic Speech Corpus

* https://en.arabicspeechcorpus.com [CC BY 4.0, accessed: Feb 19, 2025]

```
@phdthesis{halabi-thesis-arabic-speech-corpus,
  author  = "Halabi, Nawar",
  title   = "Modern Standard Arabic Phonetics for Speech Synthesis",
  school  = "University of Southampton",
  address = "Stanford, CA",
  year    = 1956,
  month   = jun
}
```

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

### THCHS-30 and aligned transcriptions

* Speech data https://openslr.org/18/ (Apache License v.2.0, accessed Feb 13, 2025)

```
@misc{wang2015-thchs30,
      title={THCHS-30 : A Free Chinese Speech Corpus}, 
      author={Dong Wang and Xuewei Zhang},
      year={2015},
      eprint={1512.01882},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/1512.01882}, 
}
```

* Transcriptions "grids-sdp.zip" https://zenodo.org/records/7528596 (MIT License, DOI https://doi.org/10.5281/zenodo.7528595, ver: 0.0.1, January 12, 2023)
* ! transcriptions should be placed into data/data_thchs30/

```
@misc{taubert23-thchs30ipa,
  author       = {Taubert, Stefan},
  title        = {THCHS-30 - Aligned IPA transcriptions},
  month        = jan,
  year         = "2023",
  publisher    = {Zenodo},
  version      = {0.0.1},
  doi          = {10.5281/zenodo.7528596},
  url          = {https://doi.org/10.5281/zenodo.7528596},
}
```

### VoxAngeles

* on [GitHub](https://github.com/pacscilab/voxangeles) and [HuggingFace](https://huggingface.co/datasets/speech31/voxangeles), [CC BY-NC 4.0, Mar 11, 2025]

```
@inproceedings{chodroff-etal-2024-phonetic,
    title = "Phonetic Segmentation of the {UCLA} Phonetics Lab Archive",
    author = "Chodroff, Eleanor  and Pa{\v{z}}on, Bla{\v{z}}  and Baker, Annie  and Moran, Steven",
    editor = "Calzolari, Nicoletta  and Kan, Min-Yen  and Hoste, Veronique  and Lenci, Alessandro  and Sakti, Sakriani  and Xue, Nianwen",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.1114/",
    pages = "12724--12733",
    abstract = "Research in speech technologies and comparative linguistics depends on access to diverse and accessible speech data. The UCLA Phonetics Lab Archive is one of the earliest multilingual speech corpora, with long-form audio recordings and phonetic transcriptions for 314 languages (Ladefoged et al., 2009). Recently, 95 of these languages were time-aligned with word-level phonetic transcriptions (Li et al., 2021). Here we present VoxAngeles, a corpus of audited phonetic transcriptions and phone-level alignments of the UCLA Phonetics Lab Archive, which uses the 95-language CMU re-release as our starting point. VoxAngeles also includes word- and phone-level segmentations from the original UCLA corpus, as well as phonetic measurements of word and phone durations, vowel formants, and vowel f0. This corpus enhances the usability of the original data, particularly for quantitative phonetic typology, as demonstrated through a case study of vowel intrinsic f0. We also discuss the utility of the VoxAngeles corpus for general research and pedagogy in crosslinguistic phonetics, as well as for low-resource and multilingual speech technologies. VoxAngeles is free to download and use under a CC-BY-NC 4.0 license."
}
```
2007. The UCLA Phonetics Lab Archive. Los Angeles, CA: UCLA Department of Linguistics. http://archive.phonetics.ucla.edu/.


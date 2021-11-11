# &nbsp; ![Joey-NMT](joey-small.png) SpeechJoey


This is an extension of JoeyNMT to Speech-to-Text tasks.


## New Features

- speech data augmentation: SpecAugment and CMVN
- subsampling: 1d convolutional layer in encoder
- objectives: jointly minimize negative log likelihood and CTC loss
- on-the-fly data loading
- tokenizer integration
- evaluation metrics:
  - accuracy in one-hot-tensor level
  - WER in decoded string surface
  

## Dependencies
- pytorch v1.9.0
- torchaudio v0.9.0
Note that we removed torchtext (which was necessary in the original JoeyNMT.)


## Getting Started

- train ASR from scratch: demo with OPENSLR 25 (afrikaans) dataset
- fine tune: demo with MuST-C pretrained model on tatoeba corpus


## Benchmarks
Benchmark results on WMT and IWSLT datasets are reported [here](benchmarks-s2t.md).

## Pre-trained Models
Pre-trained models from reported benchmarks for download (contains config, vocabularies, tokenizer, best checkpoint and dev/test hypotheses):

### LibriSpeech en ASR
Assume the data are downloaded and placed under `$HOME/LibriSpeech`.
```
$ python scripts/prepare_librispeech.py --data_root $HOME/LibriSpeech
```

- [ASR 100h]()
- [ASR 960h]()

### MuST-C en-de ST
Assume that the data are downloaded and placed under `$HOME/MUSTC`.
```
$ python scripts/prepare_mustc.py --data_root $HOME/MUSTC --trg_lang de
```

- [ASR en]() pretrained to initialize encoder
- [MT en-de]() pretrained to initialize decoder
- [ST en-de]() end-to-end speech translation model

### CoVoST2 de-en ST
Assume that the data are downloaded and placed under `$HOME/COVOST`.
```
$ python scripts/prepare_covost.py --data_root $HOME/COVOST --src_lang en --trg_lang de
```

- [ASR de]() pretrained to initialize encoder
- [MT de-en]() pretrained to initialize decoder
- [ST de-en]() end-to-end speech translation model

## Contact

Mayumi Ohta `ohta@cl.uni-heidelberg.de` (Heidelberg University)

## Reference
- [SpeechJoey]():
```
@inproceedings{placeholder,
    title = "{SpeechJoey}:  Minimalistic Speech-to-Text Modeling with {JoeyNMT}",
}
```

- [Joey NMT](https://arxiv.org/abs/1907.12484):

```
@inproceedings{kreutzer-etal-2019-joey,
    title = "Joey {NMT}: A Minimalist {NMT} Toolkit for Novices",
    author = "Kreutzer, Julia  and
      Bastings, Jasmijn  and
      Riezler, Stefan",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP): System Demonstrations",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D19-3019",
    doi = "10.18653/v1/D19-3019",
    pages = "109--114",
}
```


#!/usr/bin/env python3
"""
Prepare MUSTC

Adapted from https://github.com/pytorch/fairseq/blob/master/examples/speech_to_text/prep_mustc_data.py

    expected dir structure:
        MUSTC_ROOT
        └── en-de
            ├── data
            │   ├── dev
            │   │   ├── txt
            │   │   │   └── dev.yaml
            │   │   └── wav
            │   │       ├── ted_767.wav
            │   │       ├── [...]
            │   │       └── ted_837.wav
            │   ├── train
            │   │   ├── txt/
            │   │   └── wav/
            │   ├── tst-COMMON
            │   │   ├── txt/
            │   │   └── wav/
            │   └── tst-HE
            │       ├── txt/
            │       └── wav/
            ├── fbank80
            │   ├── ted_767_1.npy
            │   ├── [...]
            │   └── ted_767_1.npy
            ├── fbank80.zip
            └── clips
"""

import argparse
from pathlib import Path
from itertools import groupby
from typing import Tuple
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

import yaml
import pandas as pd

from torch import Tensor
from torch.utils.data import Dataset
import torchaudio

from audiodata_utils import build_sp_model, create_zip, get_zip_manifest, \
    load_tsv, Normalizer, save_tsv

from joeynmt.helpers import write_list_to_file
from joeynmt.helpers_for_audio import extract_fbank_features, get_n_frames


COLUMNS = ["id", "src", "n_frames", "trg", "speaker"]

N_MEL_FILTERS = 80
N_WORKERS = 4 #cpu_count()
SP_MODEL_TYPE = "bpe" # one of ["bpe", "unigram", "char"]
VOCAB_SIZE = 7000 #joint vocab
LOWERCASE = {'en': True, 'de': False}

class MUSTC(Dataset):
    """
    Create a Dataset for MuST-C. Each item is a tuple of the form:
    waveform, num of frames, source utterance, target utterance, speaker_id,
    utterance_id
    """

    SPLITS = ["dev", "tst-COMMON", "tst-HE", "train"]
    LANGUAGES = ["de", "es", "fr", "it", "nl", "pt", "ro", "ru"]

    def __init__(self, root: Path, lang: str, split: str) -> None:
        assert split in self.SPLITS and lang in self.LANGUAGES
        _root = Path(root) / f"en-{lang}" / "data" / split
        wav_root, txt_root = _root / "wav", _root / "txt"
        assert _root.is_dir() and wav_root.is_dir() and txt_root.is_dir()
        # Load audio segments
        with (txt_root / f"{split}.yaml").open("r") as f:
            segments = yaml.load(f, Loader=yaml.BaseLoader)
        # Load source and target utterances
        for _lang in ["en", lang]:
            with (txt_root / f"{split}.{_lang}").open("r") as f:
                utterances = [r.strip() for r in f]
            assert len(segments) == len(utterances)
            for i, u in enumerate(utterances):
                segments[i][_lang] = u
        # Gather info
        self.data = []
        for wav_filename, _seg_group in groupby(segments, lambda x: x["wav"]):
            wav_path = wav_root / wav_filename
            sample_rate = torchaudio.info(wav_path.as_posix()).sample_rate
            seg_group = sorted(_seg_group, key=lambda x: float(x["offset"]))
            for i, segment in enumerate(seg_group):
                offset = int(float(segment["offset"]) * sample_rate)
                duration = int(float(segment["duration"]) * sample_rate)
                _id = f"{wav_path.stem}_{i}"
                self.data.append({
                    "wav_path": wav_path.as_posix(),
                    "offset": offset,
                    "duration": duration,
                    "src_utt": segment["en"],
                    "trg_utt": segment[lang],
                    "spk_id": segment["speaker_id"],
                    "utt_id": _id
                })

        self.return_wav = True

    def __getitem__(self, n: int) -> Tuple[Tensor, int, int, str, str, str, str]:
        #wav_path, offset, duration, src_utt, tgt_utt, spk_id, utt_id = self.data[n]
        data = self.data[n]
        if self.return_wav:
            waveform, sr = torchaudio.load(data["wav_path"],
                                           frame_offset=data["offset"],
                                           num_frames=data["duration"])
            assert data["duration"] == waveform.size(1), (data["duration"],
                                                          waveform.size(1))
            n_frames = get_n_frames(waveform.size(1), sr)
            self.data[n]['n_frames'] = n_frames
        else:
            waveform, sr = None, None
            assert 'n_frames' in data
            n_frames = int(data['n_frames'])
        return waveform, sr, n_frames, data["src_utt"], data["trg_utt"], \
               data["spk_id"], data["utt_id"]

    def __len__(self) -> int:
        return len(self.data)


def process(data_root, languages):
    root = Path(data_root).absolute()
    for lang in languages:
        assert lang in MUSTC.LANGUAGES
        cur_root = root / f"en-{lang}"

        feature_root = cur_root / f"fbank{N_MEL_FILTERS}"
        feature_root.mkdir(exist_ok=True)

        # normalizer
        mapping_path = Path(__file__).resolve().parent / "mapping_en.txt"
        normalizer = {'en': Normalizer(lang='en', lowercase=LOWERCASE['en'],
                                       remove_punc=True, normalize_num=True,
                                       mapping_path=mapping_path,
                                       escape=True),
                      lang: Normalizer(lang=lang, lowercase=LOWERCASE['de'],
                                       remove_punc=False, normalize_num=False,
                                       escape=True)}

        # Extract features
        print(f"Create en-{lang} dataset.")
        datasets = {}
        for split in MUSTC.SPLITS:
            print(f"Fetching split {split}...")
            datasets[split] = MUSTC(root, lang, split)
            print(f"Extracting log mel filter bank features ...")
            for i, (wav, sr, n_frames, _, _, spk_id, utt_id) \
                    in enumerate(tqdm(datasets[split])):
                try:
                    extract_fbank_features(wav, sr, n_frames, utt_id,
                                           feature_root=feature_root,
                                           n_mel_bins=N_MEL_FILTERS,
                                           overwrite=False)
                except Exception as e:
                    print(i, e)
        
        # Pack features into ZIP
        print("ZIPing features...")
        create_zip(feature_root, feature_root.with_suffix(".zip"))
        print("Fetching ZIP manifest...")
        zip_manifest = get_zip_manifest(feature_root.with_suffix(".zip"))
        # Generate TSV manifest
        print("Generating manifest...")
        all_data = []
        with tqdm(total=len(zip_manifest)) as pbar:
            for split, dataset in datasets.items():
                dataset.return_wav = False  # a bit faster...
                for _, _, n_frames, src_utt, trg_utt, spk_id, utt_id in dataset:
                    all_data.append({
                        "id": utt_id,
                        "src": zip_manifest[utt_id],
                        "n_frames": n_frames,
                        "en_orig": src_utt,
                        f"{lang}_orig": trg_utt,
                        "en_utt": normalizer['en'](src_utt),
                        f"{lang}_utt": normalizer[lang](trg_utt),
                        "speaker": spk_id,
                        "split": split
                    })
                    pbar.update(1)
            all_df = pd.DataFrame.from_records(all_data)
            save_tsv(all_df, (cur_root/'all_data_tmp.tsv'))


        # Generate joint vocab
        print("Building joint vocab...")
        raw_textfile = cur_root / f"train.en{lang}"
        train_df = all_df[all_df.split == "train"]
        train = pd.concat([train_df["en_utt"], train_df[f"{lang}_orig"]])
        write_list_to_file(raw_textfile, train.to_list())

        spm_filename = cur_root / f"spm_{SP_MODEL_TYPE}{VOCAB_SIZE}"
        symbols = set([x[1] for x in (normalizer["en"].escape
                                      + normalizer[lang].escape)])
        kwargs = {'model_type': SP_MODEL_TYPE,
                  'vocab_size': VOCAB_SIZE,
                  'character_coverage': 1.0,
                  'num_workers': N_WORKERS,
                  'user_defined_symbols': ','.join(symbols)}
        spm = build_sp_model(raw_textfile, spm_filename, **kwargs)

        print(f"Saving in tsv ...")
        for split in MUSTC.SPLITS:
            split_df = all_df[all_df.split == split]
            for _task, _lang in [("asr", "en"), ("st", lang)]:
                # apply sentencepiece
                #split_df[f"{_lang}_tok"] = split_df[f"{_lang}_utt"].apply(
                #    lambda x: ' '.join(spm.encode(x, out_type=str)))
                # save tsv file
                save_tsv(split_df.rename(columns={f'{_lang}_utt': 'trg'})[COLUMNS],
                         cur_root / f"{split}_{_task}.tsv")
                # save text file (for mt pretraining)
                write_list_to_file(cur_root / f"{split}.{_lang}",
                                   split_df[f"{_lang}_utt"].to_list())
        print("Done!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", "-d", required=True, type=str)
    parser.add_argument("--trg_lang", default="de", required=True, type=str, nargs='+')
    args = parser.parse_args()

    process(args.data_root, args.trg_lang)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
prepare LibriSpeech dataset
"""
# Adapted from https://github.com/pytorch/fairseq/blob/master/examples/speech_to_text/prep_librispeech_data.py

import argparse
from tqdm import tqdm
from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np
from torch import Tensor
import torchaudio
from torchaudio.datasets import LIBRISPEECH as LIBRISPEECH_torchaudio
from torchaudio.datasets.librispeech import FOLDER_IN_ARCHIVE

from audiodata_utils import build_sp_model, create_zip, get_zip_manifest, \
    load_tsv, save_tsv

from joeynmt.helpers import write_list_to_file
from joeynmt.helpers_for_audio import extract_fbank_features, get_n_frames, \
    remove_punc


COLUMNS = ["id", "src", "n_frames", "trg", "speaker"]

# pre-defined parameters
N_MEL_FILTERS = 80
N_WORKERS = 16
SP_MODEL_TYPE = "unigram" # one of ["bpe", "unigram", "char"]
VOCAB_SIZE = {"train-clean-100": 5000, "train-960": 10000}
LOWERCASE = True

class LIBRISPEECH(LIBRISPEECH_torchaudio):
    SPLITS = [
        "train-clean-100",
        "train-clean-360",
        "train-other-500",
        "dev-clean",
        "dev-other",
        "test-clean",
        "test-other",
    ]

    def __init__(self,
                 root: Path,
                 split: str = "train-clean-100",
                 download: bool = False):
        super().__init__(root, split, FOLDER_IN_ARCHIVE, download)
        self.return_wav = True
        self.return_utt = True

    def _load_text(self, text_file, file_id):
        with text_file.open("r") as ft:
            for line in ft:
                text_id, utterance = line.strip().split(" ", 1)
                if file_id == text_id:
                    break
            else:
                # Translation not found
                raise FileNotFoundError("Translation not found for " + file_id)
        return utterance

    def __getitem__(self, n: int) -> Tuple[Tensor, int, int, str, str, str]:
        file_id = self._walker[n]
        spk_id, cpt_id, utt_no = file_id.split("-")
        utt_id = f"{spk_id}-{cpt_id}-{int(utt_no)}"
        path = Path(self._path)

        if self.return_utt:
            text_file = (path / spk_id / cpt_id
                         / f"{spk_id}-{cpt_id}").with_suffix(self._ext_txt)
            utt = self._load_text(text_file, file_id)
        else:
            utt = None

        audio_file = (path / spk_id / cpt_id
                      / file_id).with_suffix(self._ext_audio)
        if self.return_wav:
            waveform, sample_rate = torchaudio.load(audio_file.as_posix())
            n_frames = get_n_frames(waveform.size(1), sample_rate)
        else:
            waveform = None
            meta = torchaudio.load(audio_file.as_posix())
            sample_rate = meta.sample_rate
            n_frames = get_n_frames(meta.num_frames - 1152, sample_rate)

        return waveform, sample_rate, n_frames, utt, spk_id, utt_id


def process(output_root):
    out_root = Path(output_root).absolute()
    out_root.mkdir(exist_ok=True)

    # feature_root across splits
    feature_root = out_root / f"fbank{N_MEL_FILTERS}"
    feature_root.mkdir(exist_ok=True)

    # Extract features
    datasets = {}
    for split in LIBRISPEECH.SPLITS:
        print(f"Fetching split {split}...")
        datasets[split] = LIBRISPEECH(out_root, split=split, download=True)
        datasets[split].return_utt = False  # a bit faster...
        print(f"Extracting log mel filter bank features...")
        for i, (waveform, sample_rate, n_frames, _, _, utt_id) \
                in enumerate(tqdm(datasets[split])):
            try:
                extract_fbank_features(waveform, sample_rate, n_frames, utt_id,
                                       feature_root=feature_root,
                                       n_mel_bins=N_MEL_FILTERS,
                                       overwrite=False)
            except Exception as e:
                print(i, e)
    
    # Pack features into ZIP
    print("ZIPing features...")
    create_zip(feature_root, feature_root.with_suffix(".zip"))
    print("Fetching ZIP manifest...")
    #zip_manifest = get_zip_manifest(feature_root.with_suffix(".zip"))
    zip_manifest = get_zip_manifest(feature_root.with_suffix(".zip"),
                                    npy_root=feature_root)
    
    # Generate TSV manifest
    print("Generating manifest...")
    all_data = []
    with tqdm(total=len(zip_manifest)) as pbar:
        for split, dataset in datasets.items():
            dataset.return_wav = False  # a bit faster...
            for _, _, n_frames, utt, spk_id, utt_id in dataset:
                record = {"id": utt_id,
                          "src": zip_manifest[utt_id],
                          "n_frames": n_frames,
                          "trg": utt.lower() if LOWERCASE else utt,
                          "split": split,
                          "speaker": spk_id}
                all_data.append(record)
                pbar.update(1)
        all_df = pd.DataFrame.from_records(all_data)
        save_tsv(all_df, out_root / f'joey_all_data_tmp.tsv')
        del all_data

    print("Saving in tsv...")
    trainsets = {
        "train-clean-100": ["train-clean-100"],
        "train-960": ["train-clean-100", "train-clean-360", "train-other-500"]}

    for trainset, train_splits in trainsets.items():
        raw_textfile = out_root / f"{trainset}.en"
        train_df = all_df[all_df.split.isin(train_splits)]
        write_list_to_file(raw_textfile, train_df.trg.to_list())

        print(f"\tBuilding vocab of {trainset}...")
        vocab_size = VOCAB_SIZE[trainset]
        spm_filename = out_root / f"spm_{trainset}_{SP_MODEL_TYPE}{vocab_size}"
        kwargs = {'model_type': SP_MODEL_TYPE,
                  'vocab_size': vocab_size,
                  'character_coverage': 1.0,
                  'num_workers': N_WORKERS}
        spm = build_sp_model(raw_textfile, spm_filename, **kwargs)

        print(f"\tSaving {trainset} in tsv ...")
        for split in [train_splits, "dev-clean", "dev-other",
                                    "test-clean", "test-other"]:
            if isinstance(split, list):
                df = all_df[all_df.split.isin(split)]
                out_filename = trainset
            elif isinstance(split, str):
                df = all_df[all_df.split == split]
                out_filename = split
            #df["trg"] = df["trg"].apply(
            #    lambda x: ' '.join(spm.encode(x.lower(), out_type=str)))
            tsv_file = out_root / f"{out_filename}_spm{vocab_size}.tsv"
            save_tsv(df[COLUMNS], tsv_file)
            print(f'\t{tsv_file} saved.')
    print('done!')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", "-d", required=True, type=str)
    args = parser.parse_args()

    process(args.data_root)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Prepare CoVoST2

Adapted from https://github.com/pytorch/fairseq/blob/master/examples/speech_to_text/prep_mustc_data.py

    expected dir structure:
        CoVoST2_ROOT
        ├── en.tar.gz
        ├── en          # src_lang
        │   ├── clips
        │   │   ├── common_voice_en_100000.mp3
        │   │   ├── [...]
        │   │   └── common_voice_en_9999944.mp3
        │   ├── covost_v2.en_de.dev.tsv
        │   ├── covost_v2.en_de.test.tsv
        │   ├── covost_v2.en_de.train.tsv
        │   ├── covost_v2.en_de.tsv
        │   ├── covost_v2.en_de.tsv.tar.gz
        │   ├── dev.tsv
        │   ├── test.tsv
        │   ├── train.tsv
        │   └── validated.tsv
        ├── tatoeba     # out-of-domain test data
        │   ├── clips
        │   │   ├── 1000503.mp3
        │   │   ├── [...]
        │   │   └── 998651.mp3
        │   ├── tatoeba.zip
        │   └── data
        │       └── tt
        │           └── tatoeba20191004.s2t.de_en.tsv
        └── [... other scripts]
"""

import argparse
from pathlib import Path
from typing import Tuple, Callable
from functools import partial
from tqdm import tqdm
import urllib.request
import math
from multiprocessing import  Pool, cpu_count

import pandas as pd
import numpy as np

from torch import Tensor
from torch.utils.data import Dataset
import torchaudio
from torchaudio.datasets.utils import download_url, extract_archive

from audiodata_utils import build_sp_model, create_zip, get_zip_manifest, \
    load_tsv, Normalizer, save_tsv

from joeynmt.helpers import write_list_to_file
from joeynmt.helpers_for_audio import extract_fbank_features, get_n_frames, \
    remove_punc


COLUMNS = ["id", "src", "n_frames", "trg", "speaker"]

N_MEL_FILTERS = 80
N_WORKERS = 16 #cpu_count()
SP_MODEL_TYPE = "bpe" # one of ["bpe", "unigram", "char"]
VOCAB_SIZE = 7000 #joint vocab
LOWERCASE = {'en': True, 'de': False}
REMOVE_PUNC = {'en': True, 'de': False}


def _check_audio_meta(df: pd.DataFrame, root: Path):
    def _validate(path):
        n_frames = float("nan")
        try:
            meta = torchaudio.info(path.as_posix())
            assert hasattr(meta, 'num_frames')
            # -1152 for sox compliance
            n_frames = get_n_frames(meta.num_frames - 1152, meta.sample_rate)
        except Exception as e:
            #print(e)
            pass
        return n_frames

    df['num_frames'] = df['path'].apply(lambda x: _validate(root / x))
    return df


def _parallel_df_apply(df: pd.DataFrame, func: Callable,
                       n_cores: int = N_WORKERS) -> pd.DataFrame:
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


class CoVoST2(Dataset):
    """
    Create a Dataset for CoVoST2 (https://github.com/facebookresearch/covost).

    :param: root
    :param: src_lang
    :param: trg_lang
    :param: split
    """

    SPLITS = ["dev", "test", "train"]
    URL_TEMPLATE = ("https://dl.fbaipublicfiles.com/covost/"
                    "covost_v2.{src_lang}_{trg_lang}.tsv.tar.gz")
    XX_EN_LANGUAGES = ["fr", "de", "es", "ca", "it", "ru", "zh-CN", "pt", "fa",
                       "et", "mn", "nl", "tr", "ar", "sv-SE", "lv", "sl", "ta",
                       "ja", "id", "cy"]
    EN_XX_LANGUAGES = ["de", "tr", "fa", "sv-SE", "mn", "zh-CN", "cy", "ca",
                       "sl", "et", "id", "ar", "ta", "lv", "ja"]
    FEATURE_ROOT = f"fbank{N_MEL_FILTERS}"
    MP3_ROOT = "clips"

    def __init__(self, root: Path, src_lang: str, trg_lang: str, split: str):
        assert split in self.SPLITS
        assert src_lang is not None
        self.has_translation = trg_lang is not None # False -> asr; True -> ast
        if self.has_translation:
            assert "en" in {src_lang, trg_lang}
            if src_lang == "en":
                assert trg_lang in self.EN_XX_LANGUAGES
            else:
                assert src_lang in self.XX_EN_LANGUAGES
        else:
            # Hack here so that we can get "split" column from CoVoST TSV.
            # Note that we use CoVoST train split for ASR which is an extension
            # to Common Voice train split.
            trg_lang = "de" if src_lang == "en" else "en"

        self.root = Path(root) / src_lang
        if not self.root.is_dir():
            raise NotADirectoryError(f"{self.root} does not exist.")

        data_url = self.URL_TEMPLATE.format(
            src_lang=src_lang, trg_lang=trg_lang
        )
        data_archive = self.root / Path(data_url).name
        if not data_archive.is_file():
            download_url(data_url, self.root.as_posix(), hash_value=None)
        data_extracted = extract_archive(data_archive.as_posix())

        cv_tsv_path = self.root / "validated.tsv"
        assert cv_tsv_path.is_file()
        cv_tsv = load_tsv(cv_tsv_path)
        covost_tsv = load_tsv(Path(data_extracted[0]))

        df = pd.merge(left=cv_tsv[["path", "sentence", "client_id"]],
                      right=covost_tsv[["path", "translation", "split"]],
                      how="inner", on="path")
        if split == "train":
            self.df = df[(df["split"] == split) | (df["split"] == f"{split}_covost")]
        else:
            self.df = df[df["split"] == split]

        # check validity
        self._drop_invalid(mp3_path=(self.root / self.MP3_ROOT))

        # return flags
        self.return_wav = True  # whether to call torchaudio.load()
        self.return_npy = False # whether to call np.load()

    def _drop_invalid(self, mp3_path: Path) -> None:
        """check AudioMetaData"""
        func = partial(_check_audio_meta, root=mp3_path)
        self.df = _parallel_df_apply(self.df, func, n_cores=N_WORKERS)
        for idx, row in self.df[self.df['num_frames'].isna()].iterrows():
            print(f'Skip {idx}-th instance in {mp3_path / row["path"]}.')
        self.df.dropna(subset=['num_frames'], inplace=True)
        self.df.reset_index(drop=True, inplace=True)

    def __getitem__(self, n: int) -> Tuple[Tensor, int, np.ndarray, str, str, str, str]:
        """Load the n-th sample from the dataset.

        :param n: The index of the sample to be loaded
        :returns: tuple of ``(waveform, num_frames, sample_rate, transcription,
                              translation, speaker_id, utterance_id)``
        """
        data = self.df.iloc[n]
        assert not math.isnan(data['num_frames']), data

        transcription = data["sentence"]
        translation = data["translation"] if self.has_translation else None
        speaker_id = data["client_id"]
        utterance_id = data["path"].replace(".mp3", "")

        if self.return_npy:
            npy_path = self.root / self.FEATURE_ROOT / data["path"].replace(".mp3", ".npy")
            assert npy_path.is_file()
            npy = np.load(npy_path.as_posix())
            if "num_frames" in data:
                assert abs(data["num_frames"] - npy.shape[0]) <= 1, \
                    (data["num_frames"], npy.shape[0])
        else:
            npy = None

        if self.return_wav:
            mp3_path = self.root / self.MP3_ROOT / data["path"]
            waveform, sample_rate = torchaudio.load(mp3_path)
        else:
            waveform, sample_rate = None, None
        return (waveform, sample_rate, npy, transcription, translation,
                speaker_id, utterance_id)

    def __len__(self) -> int:
        return len(self.df)


class Tatoeba(CoVoST2):
    SPLITS = ['test']
    URL_TEMPLATE = "https://dl.fbaipublicfiles.com/covost/tatoeba.zip"
    LANG_CODE_2_TO_3 = {
        'fr': 'fra', 'de': 'deu', 'nl': 'nld', 'ru': 'rus', 'en': 'eng', 'es': 'spa'
    }
    def __init__(self, root: Path, src_lang: str, trg_lang: str, split: str):
        assert src_lang is not None
        self.has_translation = trg_lang is not None # False -> asr; True -> ast

        assert trg_lang == "en" and src_lang in self.LANG_CODE_2_TO_3
        self.root = Path(root) / "tatoeba"
        if not self.root.is_dir():
            raise NotADirectoryError(f"{self.root} does not exist.")

        data_url = self.URL_TEMPLATE
        data_archive = self.root / Path(data_url).name
        if not data_archive.is_file():
            download_url(data_url, self.root.as_posix(), hash_value=None)
        data_extracted = extract_archive(data_archive.as_posix())

        for tsv_path in data_extracted:
            if tsv_path.endswith(f"{src_lang}_{trg_lang}.tsv"):
                cv_tsv_path = self.root / tsv_path
                break
        assert cv_tsv_path.is_file()
        self.df = load_tsv(cv_tsv_path).rename(
            columns={"en_sentence": "translation", "speaker": "client_id"})
        self.df["split"] = split #"tatoeba"
        self.df["path"] = self.df["id"].apply(lambda x: f"{x}.mp3")
        #save_tsv(df, self.root / cv_tsv_path.name)

        # download mp3
        mp3_path = self.root / self.MP3_ROOT
        mp3_path.mkdir(exist_ok=True)
        self._download_mp3(lang=src_lang, mp3_path=mp3_path, overwrite=False)

        # check validity
        self._drop_invalid(mp3_path=mp3_path)

        # whether to call torchaudio.load()
        self.return_wav = True
        self.return_npy = False

    def _download_mp3(self, lang: str, mp3_path: Path, overwrite: bool = False):
        lang_3 = self.LANG_CODE_2_TO_3[lang]
        for name, row in tqdm(self.df.iterrows(), total=len(self.df)):
            s_id = row['id']
            if overwrite or not (mp3_path / f'{s_id}.mp3').is_file():
                url = f'https://audio.tatoeba.org/sentences/{lang_3}/{s_id}.mp3'
                try:
                    urllib.request.urlretrieve(url, (mp3_path / f'{s_id}.mp3'))
                except Exception as e:
                    #raise Exception(e)
                    continue


def process(data_root: str, src_lang: str, trg_lang: str, tatoeba: bool):
    data_class = CoVoST2 if not tatoeba else Tatoeba

    root = Path(data_root).absolute()
    curr_root = (root / src_lang) if not tatoeba else (root / 'tatoeba')

    # filterbank dir (shared across splits)
    feature_root = curr_root / data_class.FEATURE_ROOT
    feature_root.mkdir(exist_ok=True)

    # Extract features
    print(f"Create {data_class.__name__} {src_lang}-{trg_lang} dataset.")
    datasets = {}
    for split in data_class.SPLITS:
        print(f"Fetching split {split}...")
        datasets[split] = data_class(root, src_lang, trg_lang, split)
        
        print(f"Extracting log mel filter bank features...")
        assert datasets[split].return_wav is True
        for i, (wav, sample_rate, _, _, _, spk_id, utt_id) \
                in enumerate(tqdm(datasets[split])):
            try:
                extract_fbank_features(waveform=wav,
                                       sample_rate=sample_rate,
                                       output_path=(feature_root / f'{utt_id}.npy'),
                                       n_mel_bins=N_MEL_FILTERS,
                                       overwrite=False)
            except Exception as e:
                print(f'Skip {i}-th instance: {utt_id}.mp3.', e)
                continue
        
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
            dataset.return_npy = True
            dataset.return_wav = False  # a bit faster...
            for i, (_, _, npy, src_utt, trg_utt, spk_id, utt_id) \
                    in enumerate(dataset):
                # cleanup text
                if LOWERCASE[src_lang]:
                    src_utt = src_utt.lower()
                if LOWERCASE[trg_lang]:
                    trg_utt = trg_utt.lower()
                if REMOVE_PUNC[src_lang]:
                    src_utt = remove_punc(src_utt)
                if REMOVE_PUNC[trg_lang]:
                    trg_utt = remove_punc(trg_utt)
                # construct entry
                record = {
                    "id": utt_id,
                    "src": zip_manifest[utt_id],
                    "n_frames": npy.shape[0],
                    f"{src_lang}_utt": src_utt,
                    f"{trg_lang}_utt": trg_utt,
                    "speaker": spk_id,
                    "split": split
                }
                all_data.append(record)
                pbar.update(1)
        all_df = pd.DataFrame.from_records(all_data)
        save_tsv(all_df, (curr_root / 'joey_all_data_tmp.tsv'))
        del all_data

    # Generate joint vocab
    if not tatoeba:
        print("Building joint vocab...")
        raw_textfile = curr_root / f"train.{src_lang}{trg_lang}"
        train_df = all_df[all_df.split == "train"]
        train = pd.concat([train_df[f'{src_lang}_utt'], train_df[f'{trg_lang}_utt']])
        write_list_to_file(raw_textfile, train.to_list())

        spm_filename = curr_root / f"spm_{SP_MODEL_TYPE}{VOCAB_SIZE}"
        kwargs = {'model_type': SP_MODEL_TYPE,
                  'vocab_size': VOCAB_SIZE,
                  'character_coverage': 1.0,
                  'num_workers': N_WORKERS}
        spm = build_sp_model(raw_textfile, spm_filename, **kwargs)

    print("Saving data in tsv ...")
    for split in data_class.SPLITS:
        split_df = all_df[all_df.split == split]
        for _task, _lang in [("asr", src_lang), ("st", trg_lang)]:
            # apply joint vocab
            #split_df[f'{_lang}_utt'] = split_df[f'{_lang}_utt'].apply(
            #    lambda x: ' '.join(spm.encode(x, out_type=str)))
            save_tsv(split_df.rename(columns={f'{_lang}_utt': 'trg'})[COLUMNS],
                     curr_root / f"joey_{_task}_{split}.tsv")
            write_list_to_file(curr_root / f"{split}.{_lang}",
                               split_df[f'{_lang}_utt'].to_list())
        print(f'\t{split} tsv saved.')
    print("Done!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", "-d", required=True, type=str)
    parser.add_argument("--src_lang", default="en", required=True, type=str)
    parser.add_argument("--trg_lang", default="de", required=True, type=str)
    parser.add_argument("--tatoeba", action="store_true")
    args = parser.parse_args()

    process(args.data_root, args.src_lang, args.trg_lang, args.tatoeba)


if __name__ == "__main__":
    main()

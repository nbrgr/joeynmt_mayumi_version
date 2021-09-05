# coding: utf-8
"""
Collection of helper functions for audio processing
"""

import io
import os
from pathlib import Path
import sys
from typing import List, Tuple, Union
import unicodedata

import numpy as np

from joeynmt.constants import PAD_ID


_REMOVE_PUNC_MAP = {i: None for i in range(sys.maxunicode)
                    if unicodedata.category(chr(i)).startswith('P')}
def remove_punc(sent: str) -> str:
    """Remove punctuation based on Unicode category.
    Note: punctuations in audio transcription are often removed.

    :param sent: sentence string
    """
    return sent.translate(_REMOVE_PUNC_MAP)


class SpeechInstance:
    def __init__(self, fbank_path: str, n_frames: int, ind: Union[int, str]):
        """Speech Instance

        :param fbank_path: (str) Feature file path in the format of
            "<zip path>:<byte offset>:<byte length>".
        :param n_frames: (int) number of frames
        :param ind: index
        """
        self.fbank_path = fbank_path
        self.n_frames = n_frames
        self.id = ind

    def __len__(self):
        return self.n_frames


# from fairseq
def _is_npy_data(data: bytes) -> bool:
    return data[0] == 147 and data[1] == 78

# from fairseq
def _get_features_from_zip(path, byte_offset, byte_size):
    with path.open("rb") as f:
        f.seek(byte_offset)
        data = f.read(byte_size)
    byte_features = io.BytesIO(data)
    if len(data) > 1 and _is_npy_data(data):
        features = np.load(byte_features)
    else:
        raise ValueError(f'Unknown file format for "{path}"')
    return features

# from fairseq
def get_features(root_path: Path, fbank_path: str) -> np.ndarray:
    """Get speech features from ZIP file
       accessed via byte offset and length

    :return: (np.ndarray) speech features in shape of (num_frames, num_freq)
    """
    _path, *extra = fbank_path.split(":")
    _path = root_path / _path
    if not os.path.exists(_path):
        raise FileNotFoundError(f"File not found: {_path}")

    if len(extra) == 0:
        if _path.suffix == ".npy":
            features = np.load(_path.as_posix())
        else:
            raise ValueError(f"Invalid file type: {_path}")
    elif len(extra) == 2:
        assert _path.suffix == ".zip"
        extra = [int(i) for i in extra]
        features = _get_features_from_zip(_path, extra[0], extra[1])
    else:
        raise ValueError(f"Invalid path: {fbank_path}")
    return features


def pad_features(feat_list: List[SpeechInstance], root_path: Path,
                 embed_size: int = 80, pad_index: int = PAD_ID) \
        -> Tuple[np.ndarray, List[int]]:
    """
    Pad continuous feature representation in batch.
    called in batch construction (not in data loading)

    :param feat_list: list of SpeechInstance
    :param root_path: (Path) data root path
    :param embed_size: (int) number of frequencies
    :param pad_index: pad index
    :returns:
      - features np.ndarray, (batch_size, src_len, embed_size)
      - lengths List[int], (batch_size)
    """
    max_len = max([len(f) for f in feat_list])
    batch_size = len(feat_list)

    # encoder input has shape of (batch_size, src_len, embed_size)
    # (see encoder.forward())
    features = np.zeros((batch_size, max_len, embed_size), dtype=float)
    features.fill(pad_index)
    lengths = []

    for i, b in enumerate(feat_list):
        f = get_features(root_path, b.fbank_path)
        length = min(int(f.shape[0]), max_len)
        features[i, :length, :] = f[:length, :]
        lengths.append(length)

    m = max(lengths)
    if m < features.shape[1]:
        features = features[:, :m, :]

    # validation
    assert len(lengths) == features.shape[0]
    #assert max(lengths) == features.shape[1]
    assert embed_size == features.shape[2]

    return features, lengths

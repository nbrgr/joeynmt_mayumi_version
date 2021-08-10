# coding: utf-8
"""
Vocabulary module
"""
from collections import Counter
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from joeynmt.constants import BOS_ID, BOS_TOKEN, EOS_ID, EOS_TOKEN, PAD_ID, \
    PAD_TOKEN, UNK_ID, UNK_TOKEN
from joeynmt.helpers import ConfigurationError, flatten, read_list_from_file, \
    write_list_to_file

logger = logging.getLogger(__name__)


class Vocabulary:
    """ Vocabulary represents mapping between tokens and indices. """

    def __init__(self, tokens: List[str]) -> None:
        """
        Create vocabulary from list of tokens.
        Special tokens are added if not already in list.

        :param tokens: list of tokens
        """
        # warning: stoi grows with unknown tokens, don't use for saving or size

        # special symbols
        self.specials = [UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]

        # don't allow to access _stoi and _itos outside of this class
        self._stoi: Dict[str, int] = {}     # string to index
        self._itos: List[str] = []          # index to string

        # construct
        self.add_tokens(tokens=self.specials + tokens)
        assert len(self._stoi) == len(self._itos)

        # assign after stoi is built
        self.pad_index = self.lookup(PAD_TOKEN)
        self.bos_index = self.lookup(BOS_TOKEN)
        self.eos_index = self.lookup(EOS_TOKEN)
        assert self.pad_index == PAD_ID
        assert self.bos_index == BOS_ID
        assert self.eos_index == EOS_ID
        assert self._itos[UNK_ID] == UNK_TOKEN

    def __str__(self) -> str:
        return self._stoi.__str__()

    def add_tokens(self, tokens: List[str]) -> None:
        """
        Add list of tokens to vocabulary

        :param tokens: list of tokens to add to the vocabulary
        """
        for t in tokens:
            new_index = len(self._itos)
            # add to vocab if not already there
            if t not in self._itos:
                self._itos.append(t)
                self._stoi[t] = new_index

    def to_file(self, file: Path) -> None:
        """
        Save the vocabulary to a file, by writing token with index i in line i.

        :param file: path to file where the vocabulary is written
        """
        write_list_to_file(file, self._itos)

    def is_unk(self, token: str) -> bool:
        """
        Check whether a token is covered by the vocabulary

        :param token:
        :return: True if covered, False otherwise
        """
        return self.lookup(token) == UNK_ID

    def lookup(self, token: str) -> int:
        """
        look up the encoding dictionary.
        (needed for multiprocessing)
        :param token: surface str
        :return: token id
        """
        return self._stoi.get(token, UNK_ID)

    def __len__(self) -> int:
        return len(self._itos)

    def __eq__(self, other) -> bool:
        if isinstance(other, Vocabulary):
            return self._itos == other._itos
        return False

    def array_to_sentence(self, array: np.ndarray, cut_at_eos: bool = True,
                          skip_pad: bool = True) -> List[str]:
        """
        Converts an array of IDs to a sentence, optionally cutting the result
        off at the end-of-sequence token.

        :param array: 1D array containing indices
        :param cut_at_eos: cut the decoded sentences at the first <eos>
        :param skip_pad: skip generated <pad> tokens
        :return: list of strings (tokens)
        """
        sentence = []
        for i in array:
            s = self._itos[i]
            if cut_at_eos and s == EOS_TOKEN:
                break
            if skip_pad and s == PAD_TOKEN:
                continue
            sentence.append(s)
        return sentence

    def arrays_to_sentences(self, arrays: np.ndarray, cut_at_eos: bool = True,
                            skip_pad: bool = True) -> List[List[str]]:
        """
        Convert multiple arrays containing sequences of token IDs to their
        sentences, optionally cutting them off at the end-of-sequence token.

        :param arrays: 2D array containing indices
        :param cut_at_eos: cut the decoded sentences at the first <eos>
        :param skip_pad: skip generated <pad> tokens
        :return: list of list of strings (tokens)
        """
        sentences = []
        for array in arrays:
            sentences.append(
                self.array_to_sentence(array=array, cut_at_eos=cut_at_eos,
                                       skip_pad=skip_pad))
        return sentences

    def sentences_to_ids(self, sentences: List[List[str]], bos: bool = True,
                         eos: bool = True) -> Tuple[List[List[int]], List[int]]:
        """
        Encode sentences to indices and pad sequences
        :param sentences: list of tokenized sentences
        :return: padded ids and lengths
        """
        max_len = max([len(sent) for sent in sentences])
        if bos:
            max_len += 1
        if eos:
            max_len += 1
        padded, lengths = [], []
        for sent in sentences:
            encoded = [self.lookup(s) for s in sent]
            if bos:
                encoded = [self.bos_index] + encoded
            if eos:
                encoded = encoded + [self.eos_index]
            offset = max(0, max_len - len(encoded))
            padded.append(encoded + [self.pad_index] * offset)
            lengths.append(len(encoded))
        return padded, lengths

    def log_vocab(self, k: int) -> str:
        """first k vocab entities"""
        return " ".join(f'({i}) {t}' for i, t in enumerate(self._itos[:k]))


def build_vocab(max_size: int, min_freq: int, tokens: List[List[str]] = None,
                vocab_file: Path = None) -> Vocabulary:
    """
    Builds vocabulary from given `tokens` list or `vocab_file`.

    :param max_size: maximum size of vocabulary
    :param min_freq: minimum frequency for an item to be included
    :param tokens: list of tokenized sentences (raw dataset)
    :param vocab_file: file to store the vocabulary,
        if not None, load vocabulary from here
    :return: Vocabulary created from either `tokens` or `vocab_file`
    """
    def sort_and_cut(counter: Counter, max_size: int, min_freq: int):
        """
        Cut counter to most frequent, sorted numerically and alphabetically
        :param counter:
        """
        # filter counter by min frequency
        if min_freq > -1:
            counter = Counter({t: c for t, c in counter.items()
                               if c >= min_freq})

        # sort by frequency, then alphabetically
        tokens_and_frequencies = sorted(counter.items(),
                                        key=lambda tup: tup[0])
        tokens_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)
        vocab_tokens = [i[0] for i in tokens_and_frequencies[:max_size]]
        assert len(vocab_tokens) <= max_size, (len(vocab_tokens), max_size)
        return vocab_tokens

    if vocab_file is not None:
        # load it from file
        if vocab_file.suffix == ".tsv":
            tokens = {}
            for line in read_list_from_file(vocab_file):
                surface, freq = line.split("\t")
                tokens[surface] = int(freq)

            unique_tokens = sort_and_cut(Counter(tokens), max_size, min_freq)
        else:
            unique_tokens = read_list_from_file(vocab_file)

    elif tokens is not None:
        # create newly
        counter = Counter(flatten(tokens))
        unique_tokens = sort_and_cut(counter, max_size, min_freq)
    else:
        raise ConfigurationError("Please provide a vocab file path or "
                                 "sentence list.", logger)

    vocab = Vocabulary(unique_tokens)
    assert len(vocab) <= max_size + len(vocab.specials), (len(vocab), max_size)

    # check for all except for UNK token whether they are OOVs
    for s in vocab.specials[1:]:
        assert not vocab.is_unk(s)

    return vocab

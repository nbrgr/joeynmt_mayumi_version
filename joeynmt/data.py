# coding: utf-8
"""
Data module
"""
from __future__ import annotations

from functools import partial
import logging
from pathlib import Path
import sys
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import sentencepiece as spm

import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset, \
    RandomSampler, Sampler, SequentialSampler

from joeynmt.batch import Batch
from joeynmt.constants import PAD_ID
from joeynmt.helpers import log_data_info, write_list_to_file
from joeynmt.vocabulary import Vocabulary, build_vocab

logger = logging.getLogger(__name__)


# multiprocessing
try:
    torch.multiprocessing.set_start_method('spawn')
except RuntimeError as e:
    logger.debug("torch.multiprocessing.set_start_method('spawn') failed. %s",
                 e)


def load_data(data_cfg: dict, datasets: list = None, num_workers: int = 0) \
        -> Tuple[Vocabulary, Vocabulary, Optional[Dataset], Optional[Dataset],
                 Optional[Dataset]]:
    """
    Load train, dev and optionally test data as specified in configuration.
    Vocabularies are created from the training set with a limit of `voc_limit`
    tokens and a minimum token frequency of `voc_min_freq`
    (specified in the configuration dictionary).

    The training data is filtered to include sentences up to `max_sent_length`
    on source and target side.

    If you set ``random_train_subset``, a random selection of this size is used
    from the training set instead of the full training set.

    :param data_cfg: configuration dictionary for data
        ("data" part of configuration file)
    :param datasets: list of dataset names to load
    :param num_workers:
    :returns:
        - src_vocab: source vocabulary extracted from training data
        - trg_vocab: target vocabulary extracted from training data
        - train_data: training dataset
        - dev_data: development dataset
        - test_data: test dataset if given, otherwise None
    """
    if datasets is None:
        datasets = ["train", "dev", "test"]

    # load data from files
    src_lang = data_cfg["src"]
    trg_lang = data_cfg["trg"]
    train_path = data_cfg.get("train", None)
    dev_path = data_cfg.get("dev", None)
    test_path = data_cfg.get("test", None)

    if train_path is None and dev_path is None and test_path is None:
        raise ValueError('Please specify at least one data source path.')

    lowercase = data_cfg.get("lowercase", False)
    max_len = data_cfg.get("max_sent_length", -1)

    train_data = None
    if "train" in datasets and train_path is not None:
        logger.info("Loading training data...")
        train_data = TranslationDataset(path=Path(train_path),
                                        exts=(src_lang, trg_lang),
                                        lowercase=lowercase,
                                        max_len=max_len, is_train=True)

    src_max_size = data_cfg.get("src_voc_limit", sys.maxsize)
    src_min_freq = data_cfg.get("src_voc_min_freq", 1)
    trg_max_size = data_cfg.get("trg_voc_limit", sys.maxsize)
    trg_min_freq = data_cfg.get("trg_voc_min_freq", 1)

    src_vocab_file = data_cfg.get("src_vocab", None)
    trg_vocab_file = data_cfg.get("trg_vocab", None)
    src_vocab_file = None if src_vocab_file is None else Path(src_vocab_file)
    trg_vocab_file = None if trg_vocab_file is None else Path(trg_vocab_file)

    assert src_vocab_file is not None
    assert trg_vocab_file is not None

    logger.info("Building vocabulary...")
    src_vocab = build_vocab(min_freq=src_min_freq, max_size=src_max_size,
                            vocab_file=src_vocab_file)
    trg_vocab = build_vocab(min_freq=trg_min_freq, max_size=trg_max_size,
                            vocab_file=trg_vocab_file)
    assert src_vocab.pad_index == trg_vocab.pad_index
    assert src_vocab.bos_index == trg_vocab.bos_index
    assert src_vocab.eos_index == trg_vocab.eos_index

    # build sentencepiece tokenizer
    src_tokenizer = Tokenizer(src_vocab, **data_cfg["src_spm"])
    trg_tokenizer = Tokenizer(trg_vocab, **data_cfg["trg_spm"])

    if train_data is not None:
        train_data.tokenizer = {'src': src_tokenizer, 'trg': trg_tokenizer}
        train_data.padding = {'src': src_vocab.sentences_to_ids,
                              'trg': trg_vocab.sentences_to_ids}

    dev_data = None
    if "dev" in datasets and dev_path is not None:
        logger.info("Loading dev data...")
        dev_data = TranslationDataset(path=Path(dev_path),
                                      exts=(src_lang, trg_lang),
                                      lowercase=lowercase,
                                      is_train=False,
                                      src_tokenizer=src_tokenizer,
                                      trg_tokenizer=trg_tokenizer,
                                      src_padding=src_vocab.sentences_to_ids,
                                      trg_padding=trg_vocab.sentences_to_ids)

    test_data = None
    if "test" in datasets and test_path is not None:
        logger.info("Loading test data...")
        # check if target exists
        if not Path(f'{test_path}.{trg_lang}').is_file():
            # no target is given -> create dataset from src only
            trg_lang = None
        test_data = TranslationDataset(path=Path(test_path),
                                       exts=(src_lang, trg_lang),
                                       lowercase=lowercase,
                                       is_train=False,
                                       src_tokenizer=src_tokenizer,
                                       trg_tokenizer=trg_tokenizer,
                                       src_padding=src_vocab.sentences_to_ids,
                                       trg_padding=trg_vocab.sentences_to_ids)

    logger.info("Data loaded.")
    log_data_info(src_vocab, trg_vocab, train_data, dev_data, test_data)
    return src_vocab, trg_vocab, train_data, dev_data, test_data


def collate_fn(batch, src_process, trg_process, pad_index, device) -> Batch:
    """
    custom collate function
    Note: you might need another custom collate_fn()
        if you switch to a different batch class.
        Please override the batch class here. (not in TrainManager)
    :param batch:
    :param src_process:
    :param trg_process:
    :param pad_index:
    :param device:
    :return: joeynmt batch object
    """
    src_list, trg_list = [], []
    for s, t in batch:
        if s is not None:
            src_list.append(s)
            trg_list.append(t)
    assert len(batch) == len(src_list), (len(batch), len(src_list))
    src, src_length = src_process(src_list, bos=False, eos=True)

    if all(t is None for t in trg_list):
        trg, trg_length = None, None
    else:
        assert all(t is not None for t in trg_list), trg_list
        trg, trg_length = trg_process(trg_list, bos=True, eos=True)

    return Batch(torch.LongTensor(src),
                 torch.LongTensor(src_length),
                 torch.LongTensor(trg) if trg else None,
                 torch.LongTensor(trg_length) if trg_length else None,
                 pad_index, device)


def make_data_iter(dataset: TranslationDataset,
                   batch_size: int,
                   batch_type: str = "sentence",
                   seed: int = 42,
                   shuffle: bool = False,
                   num_workers: int = 0,
                   pad_index: int = PAD_ID,
                   device: torch.device = torch.device("cpu")) -> DataLoader:
    """
    Returns a torch DataLoader for a torch Dataset. (no bucketing)

    :param dataset: torch dataset containing src and optionally trg
    :param batch_size: size of the batches the iterator prepares
    :param batch_type: measure batch size by sentence count or by token count
    :param seed: random seed for shuffling
    :param shuffle: whether to shuffle the data before each epoch
        (no effect if set to True for testing)
    :param num_workers: number of cpus for multiprocessing
    :param pad_index:
    :param device:
    :return: torch DataLoader
    """
    # sampler
    sampler: Sampler[int]   # (type annotation)
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)
        sampler = RandomSampler(dataset, generator=generator)
    else:
        sampler = SequentialSampler(dataset)

    # batch generator
    if batch_type == "sentence":
        batch_sampler = SentenceBatchSampler(sampler,
                                             batch_size=batch_size,
                                             drop_last=False)
    elif batch_type == "token":
        batch_sampler = TokenBatchSampler(sampler,
                                          batch_size=batch_size,
                                          drop_last=False)

    assert dataset.padding['src'] is not None
    assert dataset.padding['trg'] is not None

    # data iterator
    return DataLoader(dataset,
                      batch_sampler=batch_sampler,
                      collate_fn=partial(collate_fn,
                                         src_process=dataset.padding['src'],
                                         trg_process=dataset.padding['trg'],
                                         pad_index=pad_index,
                                         device=device),
                      num_workers=num_workers)


class TranslationDataset(Dataset):
    """
    TranslationDataset which stores raw sentence pairs.
    used for file data.

    :param path: file name (w/o ext)
    :param exts: file ext (language code pair)
    :param lowercase: whether to lowercase
    :param max_len: max length of an instance
    :param is_train: bool indicator for train set or not
    :param src_tokenizer: tokenizer for src
    :param trg_tokenizer: tokenizer for trg
    :param src_padding: padding function for src
    :param trg_padding: padding function for trg
    """
    def __init__(self, path: Path, exts: Tuple[str, Union[str, None]],
                 lowercase: bool = False, max_len: max_len = -1, is_train: bool = False,
                 src_tokenizer: Tokenizer = None, trg_tokenizer: Tokenizer = None,
                 src_padding: Callable = None, trg_padding: Callable = None):
        src_lang, trg_lang = exts
        self.has_trg = False if trg_lang is None else True

        # tokenizers
        self.tokenizer = {'src': src_tokenizer, 'trg': trg_tokenizer}

        # load data and store offsets of line breaks
        def _read_offsets(file_path: Path) -> List[int]:
            with file_path.open("r", encoding="utf-8") as f:
                num = 0
                offsets = [0]  # offsets list
                line = f.readline()
                while line and line != "\n":
                    num += 1
                    offsets.append(f.tell())
                    line = f.readline()
            return offsets[:num]

        self.file_path = {'src': path.with_suffix(f'{path.suffix}.{src_lang}')}
        self.offsets = {'src': _read_offsets(self.file_path['src'])}

        if self.has_trg:
            self.file_path['trg'] = path.with_suffix(f'{path.suffix}.{trg_lang}')
            self.offsets['trg'] = _read_offsets(self.file_path['trg'])
            assert len(self.offsets['src']) == len(self.offsets['trg'])

        # preprocessing
        self.lowercase = lowercase
        self.max_len = max_len
        self.is_train = is_train
        if self.is_train:
            assert self.has_trg

        # padding func: will be assigned after vocab is built
        self.padding = {'src': src_padding, 'trg': trg_padding}

        # file io objects
        self.file_objects = {'src': open(
            self.file_path['src'], "r", encoding="utf-8")}
        if self.has_trg:
            self.file_objects['trg'] = open(
                self.file_path['trg'], "r", encoding="utf-8")

        # place holder
        self.cache = {}

    def get_item(self, idx: int, side: str, sample: bool = False,
                 filter_by_length: bool = False) -> List[str]:
        self.file_objects[side].seek(self.offsets[side][idx]) # seek line break
        line = self.file_objects[side].readline().rstrip('\n')
        if self.lowercase:
            line = line.lower()
        item = self.tokenizer[side](line, sample=sample)
        if filter_by_length and len(item) > self.max_len:
            item = None
        return item

    def cache_item_pair(self, idx: int) -> int:
        """cache item pair of given index. called by BatchSampler."""
        src, trg = None, None
        src = self.get_item(idx=idx, side='src', sample=self.is_train,
                            filter_by_length=self.is_train and self.max_len > 0)
        if self.has_trg:
            trg = self.get_item(idx=idx, side='trg', sample=self.is_train,
                            filter_by_length=self.is_train and self.max_len > 0)
            if trg is None:
                src = None
        n_tokens = 0 if src is None else max(0 if src is None else len(src),
                                             0 if trg is None else len(trg))
        self.cache[idx] = (src, trg)
        return n_tokens

    def __getitem__(self, idx: int) -> Tuple[List[str], Optional[List[str]]]:
        # pass through
        assert idx in self.cache
        src, trg = self.cache[idx]
        del self.cache[idx]
        return src, trg

    def get_raw_texts(self) -> Tuple[List[str], List[str]]:
        if len(self.offsets['src']) > 100000:
            logger.warning("This might raise a memory error.")

        src_list = self.file_path['src'].read_text().splitlines()
        if self.lowercase:
            src_list = [item.lower() for item in src_list]

        trg_list = []
        if self.has_trg:
            trg_list = self.file_path['trg'].read_text().splitlines()
            if self.lowercase:
                trg_list = [item.lower() for item in trg_list]
        return src_list, trg_list

    def open_file(self) -> None:
        for side, file_io in self.file_objects.items():
            if file_io.closed:
                self.file_objects[side] = open(
                    self.file_path[side], "r", encoding="utf-8")
            assert self.file_objects[side].closed is False

    def close_file(self) -> None:
        for side, file_io in self.file_objects.items():
            if not file_io.closed:
                self.file_objects[side].close()
            assert self.file_objects[side].closed is True

    def __len__(self) -> int:
        return len(self.offsets['src'])

    def __repr__(self) -> str:
        return "%s(len(src)=%s, len(trg)=%s, is_train=%r, lowercase=%s, " \
               "max_len=%d)" % (self.__class__.__name__, len(self.offsets['src']),
                                len(self.offsets['trg']) if self.has_trg else 0,
                                self.is_train, self.lowercase, self.max_len)


class MonoDataset(Dataset):
    """
    MonoDataset which stores raw input sentences.
    used for stream data.

    :param lowercase: whether to lowercase
    :param src_tokenizer: tokenizer for src
    :param src_padding: padding function for src
    """
    def __init__(self, lowercase: bool = False, src_tokenizer: Tokenizer = None,
                 src_padding: Callable = None):
        self.has_trg = False

        # tokenizer
        self.tokenizer = {'src': src_tokenizer}

        # preprocessing
        self.lowercase = lowercase
        self.max_len = -1
        self.is_train = False

        # padding func: will be assigned after vocab is built
        self.padding = {'src': src_padding}

        # place holder
        self.cache = {}

    def set_item(self, line: str):
        idx = len(self.cache)
        if self.lowercase:
            line = line.lower()
        item = self.tokenizer['src'](line, sample=False)
        self.cache[idx] = item

    def cache_item_pair(self, idx: int) -> int:
        assert idx in self.cache
        return len(self.cache[idx])

    def __getitem__(self, idx: int) -> Tuple[List[str], Optional[List[str]]]:
        # pass through
        assert idx in self.cache
        return self.cache[idx], None


class SentenceBatchSampler(BatchSampler):
    """
    Wraps another sampler to yield a mini-batch of indices based on num of
    instances. An instance longer than dataset.max_len will be filtered out.

    :param sampler: Base sampler. Can be any iterable object
    :param batch_size: Size of mini-batch.
    :param drop_last: If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
    """
    def __init__(self, sampler: Sampler, batch_size: int, drop_last: bool):
        super().__init__(sampler, batch_size, drop_last)

    def __iter__(self):
        batch = []
        d = self.sampler.data_source
        for idx in self.sampler:
            n_tokens = d.cache_item_pair(idx)
            if n_tokens > 0: # otherwise drop instance
                batch.append(idx)
                if len(batch) >= self.batch_size:
                    yield batch
                    batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch


class TokenBatchSampler(BatchSampler):
    """
    Wraps another sampler to yield a mini-batch of indices based on num of
    tokens (incl. padding). An instance longer than dataset.max_len will be
    filtered out.
    * no bucketing implemented

    :param sampler: Base sampler. Can be any iterable object
    :param batch_size: Size of mini-batch.
    :param drop_last: If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
    """
    def __init__(self, sampler: Sampler, batch_size: int, drop_last: bool):
        super().__init__(sampler, batch_size, drop_last)

    def __iter__(self):
        batch = []
        max_tokens = 0
        d = self.sampler.data_source
        for idx in self.sampler:
            n_tokens = d.cache_item_pair(idx)
            if n_tokens > 0: # otherwise drop instance
                batch.append(idx)
                max_tokens = max(max_tokens, n_tokens)
                if max_tokens * len(batch) >= self.batch_size:
                    yield batch
                    batch = []
                    max_tokens = 0
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        raise NotImplementedError


class Tokenizer:
    def __init__(self, vocab: Vocabulary, model_file: str,
                 enable_sampling: bool = True, alpha: float = 0.1,
                 nbest_size: int = -1):
        self.sp = spm.SentencePieceProcessor(model_file=model_file)
        self.sp.SetVocabulary(vocab._itos)

        self.enable_sampling = enable_sampling
        self.alpha = alpha
        self.nbest_size = nbest_size

    def __call__(self, raw_input: str, sample: bool = False) -> List[str]:
        if sample:
            tokenized = self.sp.encode(raw_input, out_type=str,
                                       enable_sampling=self.enable_sampling,
                                       alpha=self.alpha,
                                       nbest_size=self.nbest_size)
        else:
            tokenized = self.sp.encode(raw_input, out_type=str)
        return tokenized

    def post_process(self, output: List[str]) -> str:
        # return "".join(output).replace(" ", "").replace("▁", " ").strip()
        return self.sp.decode(output)

    def __repr__(self):
        return "%s(sp=%r)" % (self.__class__.__name__,
                              self.sp.__class__.__name__)

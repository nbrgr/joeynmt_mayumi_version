# coding: utf-8
"""
Data module
"""
from functools import partial
import logging
from pathlib import Path
import sys
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset, \
    RandomSampler, Sampler, SequentialSampler
import torchaudio

from joeynmt.batch import Batch, SpeechBatch
from joeynmt.constants import PAD_ID
from joeynmt.data_augmentation import SpecAugment, CMVN
from joeynmt.helpers import ConfigurationError, log_data_info, \
    read_list_from_file, write_list_to_file
from joeynmt.helpers_for_audio import SpeechInstance, get_n_frames, \
    pad_features, remove_punc
from joeynmt.tokenizers import BasicTokenizer, SentencePieceTokenizer, \
    SubwordNMTTokenizer, build_tokenizer
from joeynmt.vocabulary import Vocabulary, build_vocab

logger = logging.getLogger(__name__)


def load_data(data_cfg: dict, datasets: list = None) \
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
    :returns:
        - src_vocab: source vocabulary extracted from training data
        - trg_vocab: target vocabulary extracted from training data
        - train_data: training dataset
        - dev_data: development dataset
        - test_data: test dataset if given, otherwise None
    """
    if datasets is None:
        datasets = ["train", "dev", "test"]

    task = data_cfg.get("task", "MT")
    assert task in {"MT", "s2t"}

    src_cfg = data_cfg["src"]
    trg_cfg = data_cfg["trg"]

    # load data from files
    if task == "MT":
        src_lang = src_cfg["lang"]
        trg_lang = trg_cfg["lang"]
    elif task == "s2t":
        root_path = data_cfg["root_path"]
    train_path = data_cfg.get("train", None)
    dev_path = data_cfg.get("dev", None)
    test_path = data_cfg.get("test", None)

    if train_path is None and dev_path is None and test_path is None:
        raise ValueError('Please specify at least one data source path.')

    if task == "s2t":
        num_freq = src_cfg.get("num_freq", 80) # frequency dimension

    # data augmentation
    kwargs = {}
    if task == "s2t":
        if "specaugment" in data_cfg.keys():
            kwargs["specaugment"] = SpecAugment(**data_cfg["specaugment"])
            logger.info(kwargs["specaugment"])

        if "cmvn" in data_cfg.keys():
            kwargs["cmvn"] = CMVN(**data_cfg["cmvn"])
            logger.info(kwargs["cmvn"])

    train_data = None
    if "train" in datasets and train_path is not None:
        logger.info("Loading training data...")
        kwargs["random_subset"] = data_cfg.get("random_train_subset", -1)

        _max_sent_len = data_cfg.get("max_sent_length", -1)  # backward compatibility
        kwargs["min_len"] = {"src": src_cfg.get("min_length", -1),
                             "trg": trg_cfg.get("min_length", -1)}
        kwargs["max_len"] = {"src": src_cfg.get("max_length", _max_sent_len),
                             "trg": trg_cfg.get("max_length", _max_sent_len)}
        if task == "s2t":
            train_data = TsvDataset(path=(Path(root_path)/train_path),
                                    task=task,
                                    num_freq=num_freq,
                                    is_train=True, **kwargs)
        elif task == "MT":
            train_data = PlaintextDataset(path=Path(train_path),
                                          exts=(src_lang, trg_lang),
                                          task=task,
                                          is_train=True, **kwargs)

    # load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = {}
    if task == "MT":
        tokenizer['src'] = build_tokenizer(data_cfg, "src")
    tokenizer['trg'] = build_tokenizer(data_cfg, "trg")

    # build vocab
    logger.info("Building vocabulary...")

    _src_vocab = src_cfg.get("voc_file", None)
    src_vocab_file = None if _src_vocab is None else Path(_src_vocab)
    src_voc_size = src_cfg.get("voc_limit", sys.maxsize)
    src_min_freq = src_cfg.get("voc_min_freq", 1)

    _trg_vocab = trg_cfg.get("voc_file", None)
    trg_vocab_file = None if _trg_vocab is None else Path(_trg_vocab)
    trg_voc_size = trg_cfg.get("voc_limit", sys.maxsize)
    trg_min_freq = trg_cfg.get("voc_min_freq", 1)

    # set tokenizer
    if train_data is not None:
        train_data.tokenizer = tokenizer

    src_list, trg_list = None, None
    if (src_vocab_file is None and task == "MT") \
            or (trg_vocab_file is None and task == "s2t"):
        assert train_data is not None
        src_list, trg_list = train_data.get_raw_texts()
        if task == "MT":
            src_list = [tokenizer['src'](sent) for sent in src_list]
        trg_list = [tokenizer['trg'](sent) for sent in trg_list]

    trg_vocab = build_vocab(min_freq=trg_min_freq, max_size=trg_voc_size,
                            tokens=trg_list, vocab_file=trg_vocab_file)
    src_vocab = None
    if task == "MT":
        src_vocab = build_vocab(min_freq=src_min_freq, max_size=src_voc_size,
                                tokens=src_list, vocab_file=src_vocab_file)
        assert src_vocab.pad_index == trg_vocab.pad_index
        assert src_vocab.bos_index == trg_vocab.bos_index
        assert src_vocab.eos_index == trg_vocab.eos_index

    # set vocab to spm
    if 'src' in tokenizer and isinstance(tokenizer['src'], SentencePieceTokenizer):
        tokenizer['src'].spm.SetVocabulary(src_vocab._itos)
    if isinstance(tokenizer['trg'], SentencePieceTokenizer):
        tokenizer['trg'].spm.SetVocabulary(trg_vocab._itos)

    # padding func
    src_padding = partial(src_vocab.sentences_to_ids, bos=False, eos=True) \
        if task == "MT" else partial(pad_features,
                                     root_path=Path(root_path),
                                     embed_size=num_freq,
                                     pad_index=trg_vocab.pad_index) # no src_vocab
    trg_padding = partial(trg_vocab.sentences_to_ids, bos=True, eos=True)

    # set padding func
    if train_data is not None:
        train_data.padding = {'src': src_padding, 'trg': trg_padding}

    # dev data
    dev_data = None
    if "dev" in datasets and dev_path is not None:
        logger.info("Loading dev data...")
        kwargs["random_subset"] = data_cfg.get("random_dev_subset", -1)
        if task == "s2t":
            dev_data = TsvDataset(path=(Path(root_path)/dev_path),
                                  task=task,
                                  is_train=False,
                                  trg_tokenizer=tokenizer['trg'],
                                  src_padding=src_padding,
                                  trg_padding=trg_padding, **kwargs)
        elif task == "MT":
            dev_data = PlaintextDataset(path=Path(dev_path),
                                        exts=(src_lang, trg_lang),
                                        task=task,
                                        is_train=False,
                                        src_tokenizer=tokenizer['src'],
                                        trg_tokenizer=tokenizer['trg'],
                                        src_padding=src_padding,
                                        trg_padding=trg_padding, **kwargs)

    # test data
    test_data = None
    if "test" in datasets and test_path is not None:
        logger.info("Loading test data...")
        kwargs["random_subset"] = data_cfg.get("random_test_subset", -1)
        if task == "s2t":
            test_data = TsvDataset(path=(Path(root_path)/test_path),
                                   task=task,
                                   is_train=False,
                                   trg_tokenizer=tokenizer['trg'],
                                   src_padding=src_padding,
                                   trg_padding=trg_padding, **kwargs)
        elif task == "MT":
            # check if target exists
            if not Path(f'{test_path}.{trg_lang}').is_file():
                # no target is given -> create dataset from src only
                trg_lang = None
            test_data = PlaintextDataset(path=Path(test_path),
                                         exts=(src_lang, trg_lang),
                                         task=task,
                                         is_train=False,
                                         src_tokenizer=tokenizer['src'],
                                         trg_tokenizer=tokenizer['trg'],
                                         src_padding=src_padding,
                                         trg_padding=trg_padding, **kwargs)

    logger.info("Data loaded.")
    log_data_info(src_vocab, trg_vocab, train_data, dev_data, test_data)
    return src_vocab, trg_vocab, train_data, dev_data, test_data


def collate_fn(batch, task, src_process, trg_process, pad_index, device,
               normalization, specaugment, cmvn, is_train) -> Batch:
    """
    custom collate function
    Note: you might need another custom collate_fn()
        if you switch to a different batch class.
        Please override the batch class here. (not in TrainManager)
    :param batch:
    :param task:
    :param src_process:
    :param trg_process:
    :param pad_index:
    :param device:
    :param normalization:
    :param specaugment:
    :param cmvn:
    :param is_train:
    :return: joeynmt batch object
    """
    src_list, trg_list = [], []
    for s, t in batch:
        if s is not None:
            src_list.append(s)
            trg_list.append(t)
    assert len(batch) == len(src_list), (len(batch), len(src_list))
    src, src_length = src_process(src_list)

    if all(t is None for t in trg_list):
        trg, trg_length = None, None
    else:
        assert all(t is not None for t in trg_list), trg_list
        assert trg_process is not None
        trg, trg_length = trg_process(trg_list, bos=True, eos=True)

    if task == "MT":
        batch = Batch(
            src=torch.LongTensor(src),
            src_length=torch.LongTensor(src_length),
            trg=torch.LongTensor(trg) if trg else None,
            trg_length=torch.LongTensor(trg_length) if trg_length else None,
            device=device,
            pad_index=pad_index,
            normalization=normalization
        )
    elif task == "s2t":
        batch = SpeechBatch(
            src=src,
            src_length=torch.LongTensor(src_length),
            trg=torch.LongTensor(trg) if trg else None,
            trg_length=torch.LongTensor(trg_length) if trg_length else None,
            device=device,
            pad_index=pad_index,
            normalization=normalization,
            specaugment=specaugment,
            cmvn=cmvn,
            is_train=is_train
        )
    return batch


def make_data_iter(dataset: Dataset,
                   batch_size: int,
                   batch_type: str = "sentence",
                   seed: int = 42,
                   shuffle: bool = False,
                   #num_workers: int = 0,
                   pad_index: int = PAD_ID,
                   normalization: str = "batch",
                   device: torch.device = torch.device("cpu")) -> DataLoader:
    """
    Returns a torch DataLoader for a torch Dataset. (no bucketing)

    :param dataset: torch dataset containing src and optionally trg
    :param batch_size: size of the batches the iterator prepares
    :param batch_type: measure batch size by sentence count or by token count
    :param seed: random seed for shuffling
    :param shuffle: whether to shuffle the data before each epoch
        (no effect if set to True for testing)
    #:param num_workers: number of cpus for multiprocessing
    :param pad_index:
    :param device:
    :param normalization:
    :return: torch DataLoader
    """
    assert isinstance(dataset, Dataset), dataset
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
    if dataset.has_trg:
        assert dataset.padding['trg'] is not None

    # data iterator
    return DataLoader(dataset,
                      batch_sampler=batch_sampler,
                      collate_fn=partial(collate_fn,
                                         task=dataset.task,
                                         src_process=dataset.padding['src'],
                                         trg_process=dataset.padding['trg'] if dataset.has_trg else None,
                                         pad_index=pad_index,
                                         device=device,
                                         normalization=normalization,
                                         specaugment=dataset.specaugment,
                                         cmvn=dataset.cmvn,
                                         is_train=dataset.is_train))
                      #num_workers=num_workers) # num_workers=0


class TsvDataset(Dataset):
    """
    TsvDataset which looks up input data.
    - used for tsv data with audio paths.
    - holds pointer to preprocessing functions.

    :param path: file name (w/o ext)
    :param exts: file ext (language code pair) # not used
    :param task: "MT" or "s2t"
    :param max_len: max length of src/trg instance
    :param min_len: min length of src/trg instance
    :param is_train: bool indicator for train set or not
    :param src_tokenizer: tokenizer for src
    :param trg_tokenizer: tokenizer for trg
    :param src_padding: padding function for src
    :param trg_padding: padding function for trg
    :param batch_load: whether load data in batch or at once
    """
    def __init__(self, path: Path,
                 exts: str = None, # not used
                 task: str = "s2t",
                 is_train: bool = False,
                 src_tokenizer: BasicTokenizer = None, # not used
                 trg_tokenizer: BasicTokenizer = None,
                 src_padding: Callable = None,
                 trg_padding: Callable = None,
                 online_load: bool = True,
                 **kwargs):
        self.task = task
        assert task == "s2t"

        self.online_load = online_load
        assert online_load is True

        # tokenizers
        self.tokenizer = {'src': src_tokenizer, 'trg': trg_tokenizer}

        # padding func: will be assigned after vocab is built
        self.padding = {'src': src_padding, 'trg': trg_padding}

        # read tsv data
        path = path.with_suffix(".tsv")
        assert path.is_file(), f"{path} not found. Abort."
        self.df = pd.read_csv(path.as_posix(), sep="\t", header=0,
                              encoding="utf-8", escapechar="\\",
                              quoting=3, na_filter=False)

        self.has_trg = "trg" in self.df.columns

        self.is_train = is_train
        if self.is_train:
            assert self.has_trg

        # filter by length
        self.max_len = kwargs.get('max_len', {'src': -1, 'trg': -1})
        self.min_len = kwargs.get('min_len', {'src': -1, 'trg': -1})

        # data augmentation
        self.specaugment = kwargs.get("specaugment", None) if self.is_train else None
        self.cmvn = kwargs.get("cmvn", None)

        # place holder
        self.cache = {}

        # random subset
        self.random_subset = kwargs.get("random_subset", -1)
        if len(self.df) >= self.random_subset > 0:
            self._initial_df = self.df.copy(deep=True)
            #self.sample_random_subset()  # subsampling just once

    def sample_random_subset(self, seed: int = 42) -> None:
        assert self._initial_df is not None
        self.df = self._initial_df.sample(
            n=self.random_subset, replace=False, random_state=seed).reset_index()

    def get_item(self, idx: int, side: str, sample: bool = False) \
            -> Union[SpeechInstance, List[str]]:
        line = self.df.loc[idx]
        if side == "src":
            item = SpeechInstance(fbank_path=line['src'],
                                  n_frames=line['n_frames'],
                                  idx=line['id'])
            if len(item) > self.max_len['src'] > 0 or len(item) < self.min_len['src']:
                item = None
        elif side == "trg":
            line = line['trg']
            item = self.tokenizer["trg"](line, sample=sample)
            if len(item) > self.max_len['trg'] > 0 or len(item) < self.min_len['trg']:
                item = None
        return item

    def cache_item_pair(self, idx: int) -> Tuple[int, int]:
        """cache item pair of given index. called by BatchSampler."""
        src, trg = None, None
        src = self.get_item(idx=idx, side='src', sample=self.is_train)
        if self.has_trg:
            trg = self.get_item(idx=idx, side='trg', sample=self.is_train)
            if trg is None:
                src = None
        self.cache[idx] = (src, trg)
        assert idx in self.cache, (idx, self.cache)

    def __getitem__(self, idx: int) -> Tuple[Union[List[str], SpeechInstance],
                                             Optional[List[str]]]:
        # when called outside iterator
        if idx not in self.cache:
            self.cache_item_pair(idx)

        # pass through
        src, trg = self.cache[idx]
        if self.online_load:
            del self.cache[idx]
        return src, trg

    def get_raw_texts(self) -> Tuple[None, List[str]]:
        trg_list = self.trg if self.has_trg else []
        if self.tokenizer['trg'] is not None:
            trg_list = [self.tokenizer['trg'].pre_process(x) for x in trg_list]
        return None, trg_list

    @property
    def src(self) -> List[str]:
        return self.df["src"].to_list()

    @property
    def trg(self) -> List[str]:
        return self.df["trg"].to_list() if self.has_trg else []

    def __len__(self) -> int:
        return len(self.df)

    def __repr__(self) -> str:
        return "%s(task=%s, len=%d, is_train=%r, " \
               "random_subset=%d, filter_src_length=(%d,%d), " \
               "filter_trg_length=(%d,%d), specaugment=%r, cmvn=%r)" % (
            self.__class__.__name__, self.task, len(self.df), self.is_train,
            self.random_subset, self.min_len['src'], self.max_len['src'],
            self.min_len['trg'], self.max_len['trg'],
            self.specaugment is not None, self.cmvn is not None)


class PlaintextDataset(TsvDataset):
    """
    PlaintextDataset which stores raw sentence pairs.
    - used for text file data in the format of one sentence per line.

    :param path: file name (w/o ext)
    :param exts: file ext (language code pair)
    :param task: "MT" or "s2t"
    :param is_train: bool indicator for train set or not
    :param src_tokenizer: tokenizer for src
    :param trg_tokenizer: tokenizer for trg
    :param src_padding: padding function for src
    :param trg_padding: padding function for trg
    :param min_len: min length of src/trg instance
    :param max_len: max length of src/trg instance
    """
    def __init__(self, path: Path,
                 exts: Tuple[str, Union[str, None]],
                 task: str = "MT",
                 is_train: bool = False,
                 src_tokenizer: BasicTokenizer = None,
                 trg_tokenizer: BasicTokenizer = None,
                 src_padding: Callable = None,
                 trg_padding: Callable = None,
                 online_load: bool = False, **kwargs):
        self.task = task
        assert task == "MT"

        self.online_load = online_load

        src_lang, trg_lang = exts
        self.has_trg = False if trg_lang is None else True

        self.is_train = is_train
        if self.is_train:
            assert self.has_trg

        # tokenizers
        self.tokenizer = {'src': src_tokenizer, 'trg': trg_tokenizer}

        # load data and store offsets of line breaks
        def _read_offsets(file_path: Path) -> List[int]:
            with file_path.open("r", encoding="utf-8") as f:
                num = 0
                offsets = [0]  # offsets list
                line = f.readline()
                while line and line != "\n": # stop at empty line
                    num += 1
                    offsets.append(f.tell())
                    line = f.readline()
            return offsets[:num]

        # filer by length
        self.max_len = kwargs.get("max_len", {'src': -1, 'trg': -1})
        self.min_len = kwargs.get("min_len", {'src': -1, 'trg': -1})

        # padding func: will be assigned after vocab is built
        self.padding = {'src': src_padding, 'trg': trg_padding}

        # place holder
        self.cache = {}

        self.file_path = {'src': path.with_suffix(f'{path.suffix}.{src_lang}')}
        if self.has_trg:
            self.file_path['trg'] = path.with_suffix(f'{path.suffix}.{trg_lang}')

        if self.online_load:
            # read offsets
            self.offsets = {'src': _read_offsets(self.file_path['src'])}
            # file IO objects
            self.file_objects = {'src': open(
                self.file_path['src'], "r", encoding="utf-8")}
            if self.has_trg:
                self.offsets['trg'] = _read_offsets(self.file_path['trg'])
                assert len(self.offsets['src']) == len(self.offsets['trg'])
                self.file_objects['trg'] = open(
                    self.file_path['trg'], "r", encoding="utf-8")
        else:
            src_list, trg_list = self.get_raw_texts()
            data = {'src': src_list}
            if self.has_trg:
                assert len(src_list) == len(trg_list)
                data['trg'] = trg_list
            self.df = pd.DataFrame.from_dict(data)

        # not used for task == "MT"
        self.specaugment = None
        self.cmvn = None

        # no random subset sampling for text
        self.random_subset = kwargs.get("random_subset", -1)
        if self.online_load:
            assert self.random_subset == -1, "random subset sampling not supported."
            # TODO: implement subsampling from self.offsets lists
        else:
            if len(self.df) >= self.random_subset > 0:
                self._initial_df = self.df.copy(deep=True)
                #self.sample_random_subset() # subsampling just once

    def sample_random_subset(self, seed: int = 42) -> None:
        # TODO: call this func i.e. in the beginning of each epoch.
        if self.online_load:
            raise NotImplementedError
        else:
            super().sample_random_subset(seed)

    def get_item(self, idx: int, side: str, sample: bool = False,
                 filter_by_length: bool = False) -> List[str]:
        if self.online_load:
            self.file_objects[side].seek(self.offsets[side][idx]) # seek line break
            line = self.file_objects[side].readline().rstrip('\n')
        else:
            line = self.df.loc[idx][side]

        item = self.tokenizer[side](line, sample=sample)
        if len(item) > self.max_len[side] > 0 or len(item) < self.min_len[side]:
            item = None
        return item

    def get_raw_texts(self) -> Tuple[List[str], List[str]]:
        """get non-tokenized string sentence pairs"""
        #if not self.online_load: #and len(self.df) > 1000000:
        #    logger.warning("This might raise a memory error.")

        src_list = read_list_from_file(self.file_path['src'])
        trg_list = read_list_from_file(self.file_path['trg']) if self.has_trg else []
        if self.tokenizer['src'] is not None:
            src_list = [self.tokenizer['src'].pre_process(x) for x in src_list]
        if self.tokenizer['trg'] is not None:
            trg_list = [self.tokenizer['trg'].pre_process(x) for x in trg_list]
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
        return len(self.cache['src']) if self.online_load else len(self.df)

    def __repr__(self) -> str:
        return "%s(task=%s, len=%d, has_trg=%r, is_train=%r, " \
               "filter_src_length=(%d,%d), filter_trg_length=(%d,%d)), " \
               "online_load=%r" % (
            self.__class__.__name__, self.task, self.__len__(), self.has_trg,
            self.is_train, self.min_len['src'], self.max_len['src'],
            self.min_len['trg'], self.max_len['trg'], self.online_load)


class StreamDataset(Dataset):
    """
    StreamDataset which loads interactive stream inputs.
    - called by `translate()` func in `prediction.py`.

    :param task:
    :param src_tokenizer: tokenizer for src
    :param src_padding: padding function for src
    """
    def __init__(self, task: str = "MT",
                 src_tokenizer: BasicTokenizer = None,
                 trg_tokenizer: BasicTokenizer = None,
                 src_padding: Callable = None,
                 **kwargs):
        self.task = task
        assert task in ["MT", "s2t"]
        self.is_train = False
        self.has_trg = False
        self.online_load = True

        # tokenizer
        self.tokenizer = {'src': src_tokenizer, 'trg': trg_tokenizer}

        # filter by length
        self.max_len = {'src': -1} # no cut-off
        self.min_len = {'src': -1}

        # padding func: will be assigned after vocab is built
        self.padding = {'src': src_padding}

        # data augmentation
        self.specaugment = None
        self.cmvn = kwargs.get("cmvn", None)

        # place holder
        self.cache = {}

    def set_item(self, line: str) -> None:
        """
        - for MT task: takes sentence string (i.e. `this is a test.`)
            no need to be pre-tokenized.
            tokenizer specified in config will be applied in this func.
        - for s2t task: takes audio file name (i.e. `commonvoice_en_10000.mp3`)
            this audio file must be located under `root_path` in config!

        :param line: (str)
        """
        idx = len(self.cache)
        if self.task == "MT":
            item = self.tokenizer['src'](line, sample=False)
        elif self.task == "s2t":
            path = self.padding["src"].keywords["root_path"] / line
            assert path.is_file(), f"file {path} not found."
            n_frames = None
            if path.suffix in [".mp3", ".wav"]:
                meta = torchaudio.info(path.as_posix())
                n_frames = get_n_frames(meta.num_frames, meta.sample_rate)
            elif path.suffix in [".npy"]:
                n_frames = int(np.load(path.as_posix()).shape[0])
            item = SpeechInstance(fbank_path=line, n_frames=n_frames, idx=idx)
        self.cache[idx] = (item, None)

    def cache_item_pair(self, idx: int) -> int:
        #assert idx in self.cache
        pass # pass through

    def __getitem__(self, idx: int) -> Tuple[List[Union[str, SpeechInstance]], None]:
        # pass through
        assert idx in self.cache, (idx, self.cache)
        src, trg = self.cache[idx]
        return src, None

    def __len__(self) -> int:
        return len(self.cache)

    def __repr__(self) -> str:
        return "%s(task=%s, len(src)=%d, is_train=%r)" % (
            self.__class__.__name__, self.task, len(self.cache), self.is_train)


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
            src, trg = d[idx] # call __getitem__()
            src_len = 0 if src is None else len(src)
            trg_len = 0 if trg is None else len(trg)
            n_tokens = 0 if src_len == 0 else max(src_len, trg_len)
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
    tokens (incl. padding). An instance longer than dataset.max_len or
    shorter than dataset.min_len will be filtered out.
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
            src, trg = d[idx]  # call __getitem__()
            src_len = 0 if src is None else len(src)
            trg_len = 0 if trg is None else len(trg)
            n_tokens = 0 if src_len == 0 else max(src_len, trg_len)
            if n_tokens > 0: # otherwise drop instance
                batch.append(idx)
                if n_tokens > max_tokens:
                    max_tokens = n_tokens
                if max_tokens * len(batch) >= self.batch_size:
                    yield batch
                    batch = []
                    max_tokens = 0
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        raise NotImplementedError

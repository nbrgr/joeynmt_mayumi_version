# coding: utf-8
"""
Collection of helper functions
"""
from __future__ import annotations

import copy
import functools
import logging
import operator
from pathlib import Path
import random
import shutil
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

import torch
from torch import nn, Tensor
from torch.nn.functional import pad as _pad
from torch.utils.tensorboard import SummaryWriter

import pkg_resources
import yaml

from joeynmt.plotting import plot_heatmap

if TYPE_CHECKING:
    from joeynmt.data import TsvDataset
    from joeynmt.vocabulary import Vocabulary  # to avoid circular import


class ConfigurationError(Exception):
    """ Custom exception for misspecifications of configuration """


def make_model_dir(model_dir: Path, overwrite: bool = False) -> Path:
    """
    Create a new directory for the model.

    :param model_dir: path to model directory
    :param overwrite: whether to overwrite an existing directory
    :return: path to model directory
    """
    model_dir = model_dir.absolute()
    if model_dir.is_dir():
        if not overwrite:
            raise FileExistsError(
                f"Model directory {model_dir} exists and overwriting is disabled.")
        # delete previous directory to start with empty dir again
        shutil.rmtree(model_dir)
    model_dir.mkdir()
    return model_dir


def make_logger(log_dir: Path = None, mode: str = "train") -> str:
    """
    Create a logger for logging the training/testing process.

    :param log_dir: path to file where log is stored as well
    :param mode: log file name. 'train', 'test' or 'translate'
    :return: joeynmt version number
    """
    logger = logging.getLogger("")  # root logger
    version = pkg_resources.require("joeynmt")[0].version

    # add handlers only once.
    if len(logger.handlers) == 0:
        logger.setLevel(level=logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(name)s - %(message)s')

        if log_dir is not None:
            if log_dir.is_dir():
                log_file = log_dir / f'{mode}.log'

                fh = logging.FileHandler(log_file.as_posix())
                fh.setLevel(level=logging.DEBUG)
                logger.addHandler(fh)
                fh.setFormatter(formatter)

        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(formatter)

        logger.addHandler(sh)
        logger.info("Hello! This is Joey-NMT (version %s).", version)

    return version


def log_cfg(cfg: Dict, prefix: str = "cfg") -> None:
    """
    Write configuration to log.

    :param cfg: configuration to log
    :param prefix: prefix for logging
    """
    logger = logging.getLogger(__name__)
    for k, v in cfg.items():
        if isinstance(v, dict):
            p = '.'.join([prefix, k])
            log_cfg(v, prefix=p)
        else:
            p = '.'.join([prefix, k])
            logger.info("%34s : %s", p, v)


def clones(module: nn.Module, n: int) -> nn.ModuleList:
    """
    Produce N identical layers. Transformer helper function.

    :param module: the module to clone
    :param n: clone this many times
    :return cloned modules
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def subsequent_mask(size: int) -> Tensor:
    """
    Mask out subsequent positions (to prevent attending to future positions)
    Transformer helper function.

    :param size: size of mask (2nd and 3rd dim)
    :return: Tensor with 0s and 1s of shape (1, size, size)
    """
    ones = torch.ones(size, size, dtype=torch.bool)
    return torch.tril(ones, out=ones).unsqueeze(0)


def set_seed(seed: int) -> None:
    """
    Set the random seed for modules torch, numpy and random.

    :param seed: random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed)


def log_data_info(src_vocab: Vocabulary, trg_vocab: Vocabulary,
                  train_data: Optional[TsvDataset],
                  valid_data: Optional[TsvDataset],
                  test_data: Optional[TsvDataset]) -> None:
    """
    Log statistics of data and vocabulary.

    :param src_vocab:
    :param trg_vocab:
    :param train_data:
    :param valid_data:
    :param test_data:
    """
    logger = logging.getLogger(__name__)
    logger.info("Train dataset: %s", train_data)
    logger.info("Valid dataset: %s", valid_data)
    logger.info(" Test dataset: %s", test_data)

    if train_data:
        src = ""
        if train_data.task == "MT":
            src = "\n\t[SRC] " +  " ".join(train_data.get_item(
                idx=0, side="src", sample=False, filter_by_length=False))
        logger.info("First training example:%s\n\t[TRG] %s",
                    src, " ".join(train_data.get_item(
                idx=0, side="trg", sample=False, filter_by_length=False)))

    if src_vocab is not None:
        logger.info("First 10 Src tokens: %s", src_vocab.log_vocab(10))
    logger.info("First 10 Trg tokens: %s", trg_vocab.log_vocab(10))

    if src_vocab is not None:
        logger.info("Number of unique Src tokens (vocab_size): %d", len(src_vocab))
    logger.info("Number of unique Trg tokens (vocab_size): %d", len(trg_vocab))


def load_config(path: Path = Path("configs/default.yaml")) -> Dict:
    """
    Loads and parses a YAML configuration file.

    :param path: path to YAML configuration file
    :return: configuration dictionary
    """
    with path.open('r', encoding="utf-8") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg


def write_list_to_file(output_path: Path, array: List[str]) -> None:
    """
    Write list of str to file in `output_path`.

    :param output_path: output file path
    :param array: list of strings
    """
    with output_path.open('w', encoding="utf-8") as opened_file:
        for entry in array:
            opened_file.write(f"{entry}\n")


def read_list_from_file(input_path: Path) -> List[str]:
    """
    Read list of str from file in `input_path`.

    :param input_path: input file path
    :return: list of strings
    """
    return [line.rstrip("\n") for line in
            input_path.read_text(encoding='utf-8').splitlines()]


def bpe_postprocess(string, bpe_type="subword-nmt") -> str:
    """
    Post-processor for BPE output. Recombines BPE-split tokens.

    :param string:
    :param bpe_type: one of {"sentencepiece", "subword-nmt"}
    :return: post-processed string
    """
    if bpe_type == "sentencepiece":
        ret = string.replace(" ", "").replace("▁", " ").strip()
    elif bpe_type == "subword-nmt":
        # Remove merge markers within the sentence.
        ret = string.replace("@@ ", "").strip()
        # Remove final merge marker.
        if ret.endswith("@@"):
            ret = ret[:-2]
    else:
        ret = string.strip()
    return ret


def store_attention_plots(attentions: np.ndarray,
                          targets: List[List[str]],
                          sources: List[List[str]],
                          output_prefix: str,
                          indices: List[int],
                          tb_writer: Optional[SummaryWriter] = None,
                          steps: int = 0) -> None:
    """
    Saves attention plots.

    :param attentions: attention scores
    :param targets: list of tokenized targets
    :param sources: list of tokenized sources
    :param output_prefix: prefix for attention plots
    :param indices: indices selected for plotting
    :param tb_writer: Tensorboard summary writer (optional)
    :param steps: current training steps, needed for tb_writer
    :param dpi: resolution for images
    """
    for i in indices:
        if i >= len(sources):
            continue
        plot_file = f"{output_prefix}.{i}.pdf"
        src = sources[i]
        trg = targets[i]
        attention_scores = attentions[i].T
        try:
            fig = plot_heatmap(scores=attention_scores,
                               column_labels=trg,
                               row_labels=src,
                               output_path=plot_file,
                               dpi=100)
            if tb_writer is not None:
                # lower resolution for tensorboard
                fig = plot_heatmap(scores=attention_scores,
                                   column_labels=trg,
                                   row_labels=src,
                                   output_path=None,
                                   dpi=50)
                tb_writer.add_figure(f"attention/{i}.",
                                     fig,
                                     global_step=steps)
        except Exception:   # pylint: disable=broad-except
            print("Couldn't plot example {}: src len {}, trg len {}, "
                  "attention scores shape {}".format(i, len(src), len(trg),
                                                     attention_scores.shape))
            continue


def get_latest_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    """
    Returns the latest checkpoint (by creation time, not the steps number!)
    from the given directory.
    If there is no checkpoint in this directory, returns None

    :param ckpt_dir:
    :return: latest checkpoint file
    """
    list_of_files = ckpt_dir.glob("*.ckpt")
    latest_checkpoint = None
    if list_of_files:
        latest_checkpoint = max(list_of_files, key=lambda f: f.stat().st_ctime)

    # check existence
    if latest_checkpoint is None:
        raise FileNotFoundError(
            f"No checkpoint found in directory {ckpt_dir}.")
    return latest_checkpoint


def load_checkpoint(path: Path, device: torch.device) -> Dict:
    """
    Load model from saved checkpoint.

    :param path: path to checkpoint
    :param device: cuda device name or cpu
    :return: checkpoint (dict)
    """
    logger = logging.getLogger(__name__)
    assert path.is_file(), "Checkpoint %s not found" % path
    checkpoint = torch.load(path.as_posix(), map_location=device)
    logger.info("Load model from %s.", path.resolve())
    return checkpoint


def resolve_ckpt_path(ckpt: str, load_model: str, model_dir: Path) -> Path:
    """
    resolve checkpoint path
    :param ckpt: str passed from stdin args (--ckpt)
    :param load_model: config entry (cfg['training']['load_model'])
    :param model_dir: Path(cfg['training']['model_dir'])
    :return:
    """
    if ckpt is None:
        if load_model is None:
            if (model_dir / "best.ckpt").is_file():
                ckpt = model_dir / "best.ckpt"
            else:
                ckpt = get_latest_checkpoint(model_dir)
        else:
            ckpt = Path(load_model)
    else:
        ckpt = Path(ckpt)
    return ckpt


# from onmt
def tile(x: Tensor, count: int, dim=0) -> Tensor:
    """
    Tiles x on dimension dim count times. From OpenNMT. Used for beam search.

    :param x: tensor to tile
    :param count: number of tiles
    :param dim: dimension along which the tensor is tiled
    :return: tiled tensor
    """
    if isinstance(x, tuple):
        h, c = x
        return tile(h, count, dim=dim), tile(c, count, dim=dim)

    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
        .transpose(0, 1) \
        .repeat(count, 1) \
        .transpose(0, 1) \
        .contiguous() \
        .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


def freeze_params(module: nn.Module) -> None:
    """
    Freeze the parameters of this module,
    i.e. do not update them during training

    :param module: freeze parameters of this module
    """
    for _, p in module.named_parameters():
        p.requires_grad = False


def delete_ckpt(to_delete: Path) -> None:
    """
    Delete checkpoint

    :param to_delete: checkpoint file to be deleted
    """
    logger = logging.getLogger(__name__)
    try:
        logger.info('delete %s', to_delete.as_posix())
        to_delete.unlink()

    except FileNotFoundError as e:
        logger.warning(
            "Wanted to delete old checkpoint %s but "
            "file does not exist. (%s)", to_delete, e)


def symlink_update(target: Path, link_name: Path) -> Optional[Path]:
    """
    This function finds the file that the symlink currently points to, sets it
    to the new target, and returns the previous target if it exists.

    :param target: A path to a file that we want the symlink to point to.
                    no parent dir, filename only, i.e. "10000.ckpt"
    :param link_name: This is the name of the symlink that we want to update.
                    link name with parent dir, i.e. "models/my_model/best.ckpt"

    :return:
        - current_last: This is the previous target of the symlink, before it is
            updated in this function. If the symlink did not exist before or did
            not have a target, None is returned instead.
    """
    if link_name.is_symlink():
        current_last = link_name.resolve()
        link_name.unlink()
        link_name.symlink_to(target)
        return current_last
    link_name.symlink_to(target)
    return None


def lengths_to_padding_mask(lens: torch.Tensor) -> torch.BoolTensor:
    bsz, max_lens = lens.size(0), torch.max(lens).item()
    mask = torch.arange(max_lens).to(lens.device).view(1, max_lens)
    mask = mask.expand(bsz, -1) >= lens.view(bsz, 1).expand(-1, max_lens)
    return ~mask


def pad(x, max_len, pad_index, dim=1):
    if dim == 1:
        batch_size, seq_len, _ = x.size()
        offset = max_len - seq_len
        new_x = _pad(x, (0, 0, 0, offset, 0, 0), "constant", pad_index) \
            if x.size(1) < max_len else x
    elif dim == -1:
        batch_size, _, seq_len = x.size()
        offset = max_len - seq_len
        new_x = _pad(x, (0, offset), "constant", pad_index) \
            if x.size(1) < max_len else x
    assert new_x.size(dim) == max_len, (x.size(), offset, new_x.size(), max_len)
    return new_x


def flatten(array: List[List[Any]]) -> List[Any]:
    """
    flatten a nested 2D list. faster even with a very long array than
    [item for subarray in array for item in subarray] or newarray.extend().
    :param array: a nested list
    :return: flattened list
    """
    return functools.reduce(operator.iconcat, array, [])


#from fairseq
def align_words_to_bpe(bpe_tokens: List[str], word_tokens: List[str],
                       bpe_type="sentencepiece", start=1) -> List[List[int]]:
    """
    align BPE to word tokenization formats.
    :params bpe_tokens: list of BPE tokens
    :params word_tokens: list of word tokens
    :return: mapping from *word_tokens* to corresponding *bpe_tokens*.
    """

    # remove whitespaces/delimiters
    postprocess = partial(bpe_postprocess, bpe_type=bpe_type)
    bpe_tokens = [postprocess(str(x)) for x in bpe_tokens]
    word_tokens = [postprocess(str(w)) for w in word_tokens]
    assert "".join(bpe_tokens) == "".join(word_tokens)

    # create alignment from every word to a list of BPE tokens
    words2bpe = []
    bpe_toks = filter(lambda item: item[1] != "", enumerate(bpe_tokens,
                                                            start=start))
    j, bpe_tok = next(bpe_toks)
    for word_tok in word_tokens:
        bpe_indices = []
        while True:
            if word_tok.startswith(bpe_tok):
                bpe_indices.append(j)
                word_tok = word_tok[len(bpe_tok) :]
                try:
                    j, bpe_tok = next(bpe_toks)
                except StopIteration:
                    j, bpe_tok = None, None
            elif bpe_tok.startswith(word_tok):
                # word_tok spans multiple BPE tokens
                bpe_indices.append(j)
                bpe_tok = bpe_tok[len(word_tok) :]
                word_tok = ""
            else:
                raise Exception(f'Cannot align "{word_tok}" and "{bpe_tok}"')
            if word_tok == "":
                break
        assert len(bpe_indices) > 0
        words2bpe.append(bpe_indices)
    assert len(words2bpe) == len(word_tokens)

    return words2bpe


def expand_reverse_index(reverse_index: List[int], n_best: int = 1) \
        -> List[int]:
    """
    expand resort_reverse_index for n_best prediction

    ex. 1) reverse_index = [1, 0, 2] and n_best = 2, then this will return
    [2, 3, 0, 1, 4, 5].

    ex. 2) reverse_index = [1, 0, 2] and n_best = 3, then this will return
    [3, 4, 5, 0, 1, 2, 6, 7, 8]

    :param reverse_index: reverse_index returned from batch.sort_by_src_length()
    :param n_best:
    :return: expanded sort_reverse_index
    """
    if n_best == 1:
        return reverse_index

    resort_reverse_index = []
    for ix in reverse_index:
        for n in range(0, n_best):
            resort_reverse_index.append(ix * n_best + n)
    assert len(resort_reverse_index) == len(reverse_index) * n_best
    return resort_reverse_index

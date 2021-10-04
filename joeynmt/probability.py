# coding: utf-8
"""
This modules holds methods for generating predictions from a model.
"""

from typing import List, Optional
import logging
import numpy as np
from collections import defaultdict

import torch
from torchtext.data import Dataset, Field

from joeynmt.helpers import bpe_postprocess, load_config, make_logger, \
    get_latest_checkpoint, load_checkpoint
from joeynmt.model import build_model, Model, _DataParallel
from joeynmt.decoders import TransformerDecoder
from joeynmt.batch import Batch
from joeynmt.data import load_data, make_data_iter
from joeynmt.constants import UNK_TOKEN, PAD_TOKEN, EOS_TOKEN
from joeynmt.prediction import parse_test_args

logger = logging.getLogger(__name__)

# pylint: disable=too-many-arguments,too-many-locals,no-member
def compute_prob(model: Model, data: Dataset,
                     batch_size: int,
                     use_cuda: bool,
                     n_gpu: int,
                     batch_class: Batch = Batch,
                     batch_type: str = "sentence") \
        -> List[np.array]:
    """
    Generate translations for the given data.
    If `compute_loss` is True and references are given,
    also compute the loss.

    :param model: model module
    :param data: dataset for validation
    :param batch_size: validation batch size
    :param batch_class: class type of batch
    :param use_cuda: if True, use CUDA
    :param n_gpu: number of GPUs
    :param batch_type: validation batch type (sentence or token)

    :return:
        - valid_probabilites: probabilities
    """
    assert batch_size >= n_gpu, "batch_size must be bigger than n_gpu."
    if batch_size > 1000 and batch_type == "sentence":
        logger.warning(
            "WARNING: Are you sure you meant to work on huge batches like "
            "this? 'batch_size' is > 1000 for sentence-batching. "
            "Consider decreasing it or switching to"
            " 'eval_batch_type: token'.")
    valid_iter = make_data_iter(
        dataset=data, batch_size=batch_size, batch_type=batch_type,
        shuffle=False, train=False)
    pad_index = model.src_vocab.stoi[PAD_TOKEN]

    assert isinstance(model.decoder, TransformerDecoder), \
        "only support TransformerDecoder."

    # disable dropout
    model.eval()

    # store token-wise probabilites
    valid_probs = []

    for valid_batch in iter(valid_iter):
        # run as during training to get validation loss (e.g. xent)

        batch = batch_class(valid_batch, pad_index, use_cuda=use_cuda)
        # sort batch now by src length and keep track of order
        sort_reverse_index = batch.sort_by_src_length()

        # need trg to compute probs
        assert batch.trg is not None

        # don't track gradients during validation
        with torch.no_grad():
            encoder_output, _, _, _ = model(
                return_type="encode", **vars(batch))

        src_mask = batch.src_mask
        pad_index = model.pad_index
        eos_index = model.eos_index
        bos_index = model.bos_index
        batch_size = batch.trg.size(0)
        output_length = batch.trg.size(1)

        # start with BOS-symbol for each sentence in the batch
        ys = encoder_output.new_full([batch_size, 1], bos_index, dtype=torch.long)

        # a subsequent mask is intersected with this in decoder forward pass
        trg_mask = src_mask.new_ones([1, 1, 1])
        if isinstance(model, torch.nn.DataParallel):
            trg_mask = torch.stack(
                [src_mask.new_ones([1, 1]) for _ in model.device_ids])

        # probability holder for the batch
        batch_prob = defaultdict(list)

        # compute probabilities by forced decoding from left to right
        for j in range(output_length):
            with torch.no_grad():
                logits, _, _, _ = model(
                    return_type="decode",
                    trg_input=ys, # model.trg_embed(ys) # embed the previous tokens
                    encoder_output=encoder_output,
                    encoder_hidden=None,
                    src_mask=src_mask,
                    unroll_steps=None,
                    decoder_hidden=None,
                    trg_mask=trg_mask
                )

            for i in range(batch_size):
                trg_id = batch.trg[i, j].item()
                if trg_id not in [pad_index, bos_index, eos_index]:
                    batch_prob[sort_reverse_index[i]].append((logits[:, -1])[i, trg_id].item())
                    #logger.debug("\t(%d %d %s %f)" % (i, j, model.trg_vocab.itos[trg_id],
                    #                                  batch_prob[sort_reverse_index[i]][-1]))
            next_word = batch.trg[:, j]
            ys = torch.cat([ys, next_word.unsqueeze(-1)], dim=1)

        for i in range(batch_size):
            valid_probs.append(batch_prob[i])
    return valid_probs


# pylint: disable-msg=logging-too-many-args
def prob(cfg_file, ckpt: str, output_path: str) -> None:
    """
    Main test function. Handles loading a model from checkpoint, generating
    translations and storing them and attention plots.

    :param cfg_file: path to configuration file
    :param ckpt: path to checkpoint to load
    :param output_path: path to output
    """
    assert output_path is not None

    cfg = load_config(cfg_file)
    model_dir = cfg["training"]["model_dir"]

    if len(logger.handlers) == 0:
        _ = make_logger(model_dir, mode="test")   # version string returned

    # when checkpoint is not specified, take latest (best) from model dir
    if ckpt is None:
        ckpt = get_latest_checkpoint(model_dir)

    # load the data
    _, dev_data, test_data, src_vocab, trg_vocab = load_data(
        data_cfg=cfg["data"], datasets=["dev", "test"])
    data_to_predict = {"dev": dev_data, "test": test_data}

    # parse test args
    batch_size, batch_type, use_cuda, device, n_gpu, level, eval_metrics, \
        max_output_length, beam_size, beam_alpha, postprocess, \
        bpe_type, sacrebleu, decoding_description, tokenizer_info \
        = parse_test_args(cfg, mode="test")

    # load model state from disk
    model_checkpoint = load_checkpoint(ckpt, use_cuda=use_cuda)

    # build model and load parameters into it
    model = build_model(cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab)
    model.load_state_dict(model_checkpoint["model_state"])
    logger.info(f"load model_state from {ckpt}.")

    if use_cuda:
        model.to(device)

    # multi-gpu eval
    if n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = _DataParallel(model)

    data_set_prob = {}

    for data_set_name, data_set in data_to_predict.items():
        if data_set is None:
            continue

        dataset_file = cfg["data"][data_set_name] + "." + cfg["data"]["trg"]
        assert "trg" in data_set.fields
        logger.info("Computing probabilities on %s set (%s)...", data_set_name, dataset_file)

        probabilites = compute_prob(
            model, data=data_set, batch_size=batch_size,
            batch_class=Batch, batch_type=batch_type,
            use_cuda=use_cuda, n_gpu=n_gpu)

        output_path_set = "{}.{}".format(output_path, data_set_name)
        with open(output_path_set, mode="w", encoding="utf-8") as out_file:
            for prob in probabilites:
                out_file.write(' '.join([str(p) for p in prob]) + "\n")
        logger.info("\tProbabilities saved to: %s", output_path_set)

        data_set_prob[data_set_name] = probabilites

    assert len(data_set_prob["dev"]) == len(data_set_prob["test"])
    total = []
    for i, (dev_probs, test_probs) in enumerate(zip(data_set_prob["dev"], data_set_prob["test"])):
        ratio = sum(dev_probs) / sum(test_probs)
        logger.debug("\t[instance %3d] Log-probability ratio: %.8f (%.8f)" % (i, ratio, np.abs(1 - ratio)))
        total.append(np.abs(1 - ratio))
    logger.info("Log-probability ratio: %.8f", sum(total)/len(total))


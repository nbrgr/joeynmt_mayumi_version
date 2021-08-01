# coding: utf-8
"""
This modules holds methods for generating predictions from a model.
"""
import logging
from pathlib import Path
import sys
from typing import List, Optional, Tuple

import numpy as np

import torch
from torch.utils.data import Dataset

from joeynmt.data import TranslationDataset, _tokenize, load_data, \
    make_data_iter
from joeynmt.helpers import bpe_postprocess, expand_reverse_index, \
    load_checkpoint, load_config, make_logger, resolve_ckpt_path, \
    store_attention_plots, write_list_to_file
from joeynmt.metrics import bleu, chrf, sequence_accuracy, token_accuracy
from joeynmt.model import Model, _DataParallel, build_model
from joeynmt.search import run_batch
from joeynmt.vocabulary import Vocabulary

logger = logging.getLogger(__name__)


def validate_on_data(model: Model,
                     data: Dataset,
                     batch_size: int,
                     device: torch.device,
                     n_gpu: int,
                     max_output_length: int,
                     level: str,
                     eval_metric: Optional[str],
                     compute_loss: bool = False,
                     beam_size: int = 1,
                     beam_alpha: int = -1,
                     batch_type: str = "sentence",
                     postprocess: bool = True,
                     bpe_type: str = "subword-nmt",
                     sacrebleu: dict = None,
                     n_best: int = 1) \
        -> Tuple[float, float, float, List[str], List[str], List[str],
                 List[List[str]], List[np.ndarray]]:
    """
    Generate translations for the given data.
    If `compute_loss` is True and references are given,
    also compute the loss.

    :param model: model module
    :param data: dataset for validation
    :param batch_size: validation batch size
    :param device: cpu or gpu
    :param n_gpu: number of GPUs
    :param max_output_length: maximum length for generated hypotheses
    :param level: segmentation level, one of "char", "bpe", "word"
    :param eval_metric: evaluation metric, e.g. "bleu"
    :param compute_loss: whether to computes a scalar loss
        for given inputs and targets
    :param beam_size: beam size for validation.
        If <2 then greedy decoding (default).
    :param beam_alpha: beam search alpha for length penalty,
        disabled if set to -1 (default).
    :param batch_type: validation batch type (sentence or token)
    :param postprocess: if True, remove BPE segmentation from translations
    :param bpe_type: bpe type, one of {"subword-nmt", "sentencepiece"}
    :param sacrebleu: sacrebleu options
    :param n_best: Amount of candidates to return

    :return:
        - current_valid_score: current validation score [eval_metric],
        - valid_loss: validation loss,
        - valid_ppl:, validation perplexity,
        - valid_sources: validation sources,
        - valid_references: validation references,
        - valid_hypotheses: validation_hypotheses,
        - decoded_valid: raw validation hypotheses (before post-processing),
        - valid_attention_scores: attention scores for validation hypotheses
    """
    # pylint: disable=too-many-arguments,too-many-locals,no-member
    assert batch_size >= n_gpu, "batch_size must be bigger than n_gpu."
    if sacrebleu is None:   # assign default value
        sacrebleu = {"remove_whitespace": True, "tokenize": "13a"}
    if batch_size > 1000 and batch_type == "sentence":
        logger.warning(
            "WARNING: Are you sure you meant to work on huge batches like "
            "this? 'batch_size' is > 1000 for sentence-batching. "
            "Consider decreasing it or switching to"
            " 'eval_batch_type: token'.")
    # caution: batch_size divided by beam_size, because a batch will be expanded
    # to batch_size*beam_size, and it could cause an out-of-memory error.
    valid_iter = make_data_iter(
        dataset=data, batch_size=batch_size, batch_type=batch_type,
        shuffle=False, pad_index=model.pad_index, device=device)
    # valid_sources_raw = data.src

    # disable dropout
    model.eval()

    all_outputs = []
    valid_attention_scores = []
    total_loss = 0
    total_ntokens = 0
    total_nseqs = 0
    for batch in valid_iter:
        if batch.nseqs < 1:
            continue

        # sort batch now by src length and keep track of order
        reverse_index = batch.sort_by_src_length()
        sort_reverse_index = expand_reverse_index(reverse_index, n_best)

        # run as during training to get validation loss (e.g. xent)
        if compute_loss and batch.trg is not None:
            # don't track gradients during validation
            with torch.no_grad():
                batch_loss, _, _, _ = model(return_type="loss", **vars(batch))
            if n_gpu > 1:
                batch_loss = batch_loss.mean() # average on multi-gpu
            total_loss += batch_loss
            total_ntokens += batch.ntokens
            total_nseqs += batch.nseqs

        # run as during inference to produce translations
        output, attention_scores = run_batch(
            model=model, batch=batch, beam_size=beam_size,
            beam_alpha=beam_alpha, max_output_length=max_output_length,
            n_best=n_best)

        # sort outputs back to original order
        all_outputs.extend(output[sort_reverse_index])
        valid_attention_scores.extend(attention_scores[sort_reverse_index]
                                      if attention_scores is not None else [])

    assert len(all_outputs) == len(data) * n_best

    if compute_loss and total_ntokens > 0:
        # total validation loss
        valid_loss = total_loss
        # exponent of token-level negative log prob
        valid_ppl = torch.exp(total_loss / total_ntokens)
    else:
        valid_loss = -1
        valid_ppl = -1

    # decode back to symbols
    decoded_valid = model.trg_vocab.arrays_to_sentences(arrays=all_outputs,
                                                        cut_at_eos=True)

    # evaluate with metric on full dataset
    join_char = " " if level in ["word", "bpe"] else ""
    valid_sources = [join_char.join(s) for s in data.src]
    valid_references = [join_char.join(t) for t in data.trg] \
        if data.trg is not None else []
    valid_hypotheses = [join_char.join(t) for t in decoded_valid]

    # post-process
    if level == "bpe" and postprocess:
        valid_sources = [bpe_postprocess(s, bpe_type=bpe_type)
                         for s in valid_sources]
        valid_references = [bpe_postprocess(v, bpe_type=bpe_type)
                            for v in valid_references]
        valid_hypotheses = [bpe_postprocess(v, bpe_type=bpe_type)
                            for v in valid_hypotheses]

    # if references are given, evaluate against them
    if valid_references:
        assert len(valid_hypotheses) == len(valid_references)

        current_valid_score = 0
        if eval_metric.lower() == 'bleu':
            # this version does not use any tokenization
            current_valid_score = bleu(
                valid_hypotheses, valid_references,
                tokenize=sacrebleu["tokenize"])
        elif eval_metric.lower() == 'chrf':
            current_valid_score = chrf(valid_hypotheses, valid_references,
                remove_whitespace=sacrebleu["remove_whitespace"])
        elif eval_metric.lower() == 'token_accuracy':
            current_valid_score = token_accuracy(   # supply List[List[str]]
                list(decoded_valid), list(data.trg))
        elif eval_metric.lower() == 'sequence_accuracy':
            current_valid_score = sequence_accuracy(
                valid_hypotheses, valid_references)
    else:
        current_valid_score = -1

    return current_valid_score, valid_loss, valid_ppl, valid_sources, \
        valid_references, valid_hypotheses, decoded_valid, \
        valid_attention_scores


def parse_test_args(cfg, mode="test"):
    """
    parse test args
    :param cfg: config object
    :param mode: 'test' or 'translate'
    :return:
    """
    if "test" not in cfg["data"].keys():
        raise ValueError("Test data must be specified in config.")

    batch_size = cfg["training"].get(
        "eval_batch_size", cfg["training"].get("batch_size", 1))
    batch_type = cfg["training"].get(
        "eval_batch_type", cfg["training"].get("batch_type", "sentence"))
    use_cuda = (cfg["training"].get("use_cuda", False)
                and torch.cuda.is_available())
    device = torch.device("cuda" if use_cuda else "cpu")
    if mode == 'test':
        n_gpu = torch.cuda.device_count() if use_cuda else 0
        #k = cfg["testing"].get("beam_size", 1)
        #batch_per_device = batch_size*k // n_gpu if n_gpu > 1 else batch_size*k
        batch_per_device = batch_size // n_gpu if n_gpu > 1 else batch_size
        logger.info("Process device: %s, n_gpu: %d, batch_size per device: %d",
                    device, n_gpu, batch_per_device)
        eval_metric = cfg["training"]["eval_metric"]

    elif mode == 'translate':
        # in multi-gpu, batch_size must be bigger than n_gpu!
        n_gpu = 1 if use_cuda else 0
        logger.debug("Process device: %s, n_gpu: %d", device, n_gpu)
        eval_metric = ""

    level = cfg["data"]["level"]
    max_output_length = cfg["training"].get("max_output_length", None)

    # whether to use beam search for decoding, 0: greedy decoding
    if "testing" in cfg.keys():
        beam_size = cfg["testing"].get("beam_size", 1) # positive integer only
        beam_alpha = cfg["testing"].get("alpha", -1)
        postprocess = cfg["testing"].get("postprocess", True)
        bpe_type = cfg["testing"].get("bpe_type", "subword-nmt")
        sacrebleu = {"remove_whitespace": True, "tokenize": "13a"}
        if "sacrebleu" in cfg["testing"].keys():
            sacrebleu["remove_whitespace"] = cfg["testing"]["sacrebleu"] \
                .get("remove_whitespace", True)
            sacrebleu["tokenize"] = cfg["testing"]["sacrebleu"] \
                .get("tokenize", "13a")

    else:
        beam_size = 1
        beam_alpha = -1
        postprocess = True
        bpe_type = "subword-nmt"
        sacrebleu = {"remove_whitespace": True, "tokenize": "13a"}

    decoding_description = "Greedy decoding" if beam_size < 2 else \
        f"Beam search decoding with beam size = {beam_size} " \
        f"and alpha = {beam_alpha}"
    tokenizer_info = sacrebleu['tokenize'] if eval_metric == "bleu" else ""

    # caution: batch_size divided by beam_size, because a batch will be expanded
    # to batch.nseqs * beam_size, and it could cause an out-of-memory error.
    if batch_size > beam_size:
        batch_size //= beam_size

    return batch_size, batch_type, device, n_gpu, level, \
        eval_metric, max_output_length, beam_size, beam_alpha, \
        postprocess, bpe_type, sacrebleu, decoding_description, tokenizer_info


def test(cfg_file,
         ckpt: str,
         output_path: str = None,
         save_attention: bool = False,
         datasets: dict = None) -> None:
    """
    Main test function. Handles loading a model from checkpoint, generating
    translations and storing them and attention plots.

    :param cfg_file: path to configuration file
    :param ckpt: path to checkpoint to load
    :param output_path: path to output
    :param datasets: datasets to predict
    :param save_attention: whether to save the computed attention weights
    """
    # pylint: disable-msg=logging-too-many-args,too-many-branches
    cfg = load_config(Path(cfg_file))
    model_dir = Path(cfg["training"]["model_dir"])

    if len(logger.handlers) == 0:
        _ = make_logger(model_dir, mode="test")   # version string returned

    # load the data
    if datasets is None:
        # set default vocab path
        src_vocab_file = model_dir / "src_vocab.txt"
        trg_vocab_file = model_dir / "trg_vocab.txt"
        if "src_vocab" not in cfg["data"]:
            assert src_vocab_file.isfile(), f"{src_vocab_file} not found."
            cfg["data"]["src_vocab"] = src_vocab_file
        if "trg_vocab" not in cfg["data"]:
            assert trg_vocab_file.isfile(), f"{trg_vocab_file} not found."
            cfg["data"]["trg_vocab"] = trg_vocab_file
        # load data
        src_vocab, trg_vocab , _, dev_data, test_data = load_data(
            data_cfg=cfg["data"], datasets=["dev", "test"])
        data_to_predict = {"dev": dev_data, "test": test_data}
    else:  # avoid to load data again
        data_to_predict = {"dev": datasets["dev"], "test": datasets["test"]}
        src_vocab = datasets["src_vocab"]
        trg_vocab = datasets["trg_vocab"]

    # parse test args
    batch_size, batch_type, device, n_gpu, level, eval_metric, \
        max_output_length, beam_size, beam_alpha, postprocess, \
        bpe_type, sacrebleu, decoding_description, tokenizer_info \
        = parse_test_args(cfg, mode="test")

    # build model and load parameters into it
    model = build_model(cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab)

    # when checkpoint is not specified, take latest (best) from model dir
    ckpt = resolve_ckpt_path(ckpt, cfg["training"].get("load_model", None),
                             model_dir)

    # load model state from disk
    model_checkpoint = load_checkpoint(ckpt, device=device)
    model.load_state_dict(model_checkpoint["model_state"])
    logger.info("Loading model_state from %s.", ckpt)

    if device.type == "cuda":
        model.to(device)

    # multi-gpu eval
    if n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = _DataParallel(model)

    for data_set_name, data_set in data_to_predict.items():
        if data_set is None:
            continue

        dataset_file = f'{cfg["data"][data_set_name]}.{cfg["data"]["src"]}'
        logger.info("Decoding on %s set (%s)...", data_set_name, dataset_file)

        # pylint: disable=unused-variable
        (score, loss, ppl, sources, references, hypotheses, hypotheses_raw,
         attention_scores) = validate_on_data(
            model, data=data_set, batch_size=batch_size, batch_type=batch_type,
            level=level, max_output_length=max_output_length,
            eval_metric=eval_metric, device=device, compute_loss=False,
            beam_size=beam_size, beam_alpha=beam_alpha, postprocess=postprocess,
            bpe_type=bpe_type, sacrebleu=sacrebleu, n_gpu=n_gpu)
        # pylint: enable=unused-variable

        if data_set.trg is not None:
            logger.info("%4s %s%s: %6.2f [%s]",
                        data_set_name, eval_metric, tokenizer_info,
                        score, decoding_description)
        else:
            logger.info("No references given for %s -> no evaluation.",
                        data_set_name)

        if save_attention:
            if attention_scores:
                attention_name = f"{data_set_name}.{ckpt.stem}.att"
                attention_path = (model_dir / attention_name).as_posix()
                logger.info("Saving attention plots. This might take a while..")
                store_attention_plots(attentions=attention_scores,
                                      targets=hypotheses_raw,
                                      sources=data_set.src,
                                      indices=range(len(hypotheses)),
                                      output_prefix=attention_path)
                logger.info("Attention plots saved to: %s", attention_path)
            else:
                logger.warning("Attention scores could not be saved. "
                               "Note that attention scores are not available "
                               "when using beam search. "
                               "Set beam_size to 1 for greedy decoding.")

        if output_path is not None:
            output_path_set = Path(f"{output_path}.{data_set_name}")
            write_list_to_file(output_path_set, hypotheses)
            logger.info("Translations saved to: %s.", output_path_set)


def translate(cfg_file: str,
              ckpt: str = None,
              output_path: str = None,
              n_best: int = 1) -> None:
    """
    Interactive translation function.
    Loads model from checkpoint and translates either the stdin input or
    asks for input to translate interactively.
    The input has to be pre-processed according to the data that the model
    was trained on, i.e. tokenized or split into subwords.
    Translations are printed to stdout.

    :param cfg_file: path to configuration file
    :param ckpt: path to checkpoint to load
    :param output_path: path to output file
    :param n_best: amount of candidates to display
    """

    def _translate_data(test_data):
        """ Translates given dataset, using parameters from outer scope. """
        # pylint: disable=unused-variable
        (score, loss, ppl, sources, references, hypotheses, hypotheses_raw,
         attention_scores) = validate_on_data(
            model, data=test_data, batch_size=batch_size, batch_type=batch_type,
            level=level, max_output_length=max_output_length, eval_metric="",
            device=device, compute_loss=False, beam_size=beam_size,
            beam_alpha=beam_alpha, postprocess=postprocess, bpe_type=bpe_type,
            sacrebleu=sacrebleu, n_gpu=n_gpu, n_best=n_best)
        # pylint: enable=unused-variable
        return hypotheses

    cfg = load_config(Path(cfg_file))
    model_dir = Path(cfg["training"]["model_dir"])

    _ = make_logger(model_dir, mode="translate")
    # version string returned

    # when checkpoint is not specified, take latest (best) from model dir
    ckpt = resolve_ckpt_path(ckpt, cfg["training"].get("load_model", None),
                             model_dir)

    # read vocabs
    src_vocab_file = cfg["data"].get("src_vocab", model_dir / "src_vocab.txt")
    trg_vocab_file = cfg["data"].get("trg_vocab", model_dir / "trg_vocab.txt")
    src_vocab = Vocabulary(file=Path(src_vocab_file))
    trg_vocab = Vocabulary(file=Path(trg_vocab_file))

    data_cfg = cfg["data"]
    lowercase = data_cfg["lowercase"]

    # parse test args
    batch_size, batch_type, device, n_gpu, level, _, \
        max_output_length, beam_size, beam_alpha, postprocess, \
        bpe_type, sacrebleu, _, _ = parse_test_args(cfg, mode="translate")

    # build model and load parameters into it
    model = build_model(cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab)

    # load model state from disk
    model_checkpoint = load_checkpoint(ckpt, device=device)
    model.load_state_dict(model_checkpoint["model_state"])

    if device.type == "cuda":
        model.to(device)

    if not sys.stdin.isatty():
        # input stream given
        test_src = [_tokenize(line.rstrip(), level, lowercase)
                    for line in sys.stdin.readlines()]
        test_data = TranslationDataset(test_src, None,
                                       src_padding=src_vocab.sentences_to_ids,
                                       trg_padding=trg_vocab.sentences_to_ids)
        all_hypotheses = _translate_data(test_data)
        assert len(all_hypotheses) == len(test_data) * n_best

        if output_path is not None:
            # write to outputfile if given
            output_path_set = Path(output_path).expanduser()

            if n_best > 1:
                for n in range(n_best):
                    file_basename = output_path_set.stem
                    file_extension = output_path_set.suffix
                    file_name = (output_path_set.parent
                                 / f"{file_basename}-{n}.{file_extension}")
                    write_list_to_file(
                        file_name,
                        [all_hypotheses[i]
                         for i in range(n, len(all_hypotheses), n_best)]
                    )
            else:
                write_list_to_file(output_path_set, all_hypotheses)

            logger.info("Translations saved to: %s.", output_path_set)

        else:
            # print to stdout
            for hyp in all_hypotheses:
                print(hyp)

    else:
        # enter interactive mode
        batch_size = 1
        batch_type = "sentence"
        while True:
            try:
                src_input = input("\nPlease enter a source sentence "
                                  "(pre-processed): \n")
                if not src_input.strip():
                    break

                # every line has to be made into dataset
                test_src = [_tokenize(src_input, level, lowercase)]
                test_data = TranslationDataset(
                    test_src, None,
                    src_padding=src_vocab.sentences_to_ids,
                    trg_padding=trg_vocab.sentences_to_ids)
                hypotheses = _translate_data(test_data)

                print("JoeyNMT: Hypotheses ranked by score")
                for i, hyp in enumerate(hypotheses):
                    print("JoeyNMT #{}: {}".format(i + 1, hyp))

            except (KeyboardInterrupt, EOFError):
                print("\nBye.")
                break

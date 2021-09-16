# coding: utf-8
"""
Training module
"""

import argparse
from collections import OrderedDict, defaultdict
import heapq
import logging
import math
from pathlib import Path
import shutil
import sys
import time
from typing import List, Tuple

import numpy as np

import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from joeynmt.batch import Batch
from joeynmt.builders import build_gradient_clipper, build_optimizer, \
    build_scheduler
from joeynmt.data import TsvDataset, TranslationDataset, load_data, make_data_iter
from joeynmt.helpers import delete_ckpt, load_checkpoint, load_config, \
    log_cfg, make_logger, make_model_dir, set_seed, store_attention_plots, \
    symlink_update, write_list_to_file
from joeynmt.model import Model, _DataParallel, build_model
from joeynmt.prediction import test, validate_on_data
from joeynmt.tokenizers import EvaluationTokenizer

logger = logging.getLogger(__name__)

# for fp16 training
try:
    from apex import amp
    amp.register_half_function(torch, "einsum")
except ImportError as no_apex:
    logger.debug(no_apex)
    # error handling in TrainManager object construction


class TrainManager:
    """ Manages training loop, validations, learning rate scheduling
    and early stopping."""
    # pylint: disable=too-many-instance-attributes

    def __init__(self, model: Model, config: dict) -> None:
        """
        Creates a new TrainManager for a model, specified as in configuration.
        Note: no need to pass batch_class here. see make_data_iter()

        :param model: torch module defining the model
        :param config: dictionary containing the training configurations
        """
        # pylint: disable=too-many-statements
        train_cfg = config["training"]
        data_cfg = config["data"]
        test_cfg = config["testing"]
        model_cfg = config["model"]

        self.task = data_cfg.get("task", "MT")
        if self.task not in {"MT", "s2t"}:
            raise ConfigurationError("Invalid setting for `task`. "
                                     "Valid options: 'MT', 's2t'.")

        # logging and storing
        self.model_dir = Path(train_cfg["model_dir"])
        assert self.model_dir.is_dir()
        self.logging_freq = train_cfg.get("logging_freq", 100)
        self.tb_writer = SummaryWriter(
            log_dir=(self.model_dir / "tensorboard").as_posix())

        # model
        self.model = model
        self.model.log_parameters_list()

        # objective
        self.model.loss_function = (train_cfg.get("loss", "crossentropy"),
                                    train_cfg.get("label_smoothing", 0.0),
                                    train_cfg.get("ctc_weight", 0.3))
        self.normalization = train_cfg.get("normalization", "batch")
        if self.normalization not in ["batch", "tokens", "none"]:
            raise ConfigurationError("Invalid normalization option."
                                     "Valid options: "
                                     "'batch', 'tokens', 'none'.")
        logger.info(self.model)

        # optimization
        self.learning_rate_min = train_cfg.get("learning_rate_min", 1.0e-8)

        self.clip_grad_fun = build_gradient_clipper(config=train_cfg)
        self.optimizer = build_optimizer(config=train_cfg,
                                         parameters=model.parameters())

        # save/delete checkpoints
        self.num_ckpts = train_cfg.get("keep_best_ckpts", 5)
        self.ckpt_queue: List[Tuple[float, Path]] = [] # heap queue
        # backward compatibility
        keep_last_ckpts = train_cfg.get("keep_last_ckpts", None)
        if keep_last_ckpts is not None:
            self.num_ckpts = keep_last_ckpts
            logger.warning("`keep_last_ckpts` option is outdated. "
                           "Please use `keep_best_ckpts`, instead.")

        # validation & early stopping
        self.validation_freq = train_cfg.get("validation_freq", 1000)
        self.log_valid_sents = train_cfg.get("print_valid_sents", [0, 1, 2])
        metrics = train_cfg.get("eval_metrics", "").split(",")
        self.eval_metrics = [s.strip().lower() for s in metrics if len(s) > 0]
        #backward compatibility
        if len(self.eval_metrics) == 0 and "eval_metric" in train_cfg.keys():
            self.eval_metrics = [train_cfg.get("eval_metric","bleu").lower()]
        assert len(self.eval_metrics) > 0
        for eval_metric in self.eval_metrics:
            if eval_metric not in [
                'bleu', 'chrf', 'token_accuracy', 'sequence_accuracy', 'wer']:
                raise ConfigurationError(
                    "Invalid setting for 'eval_metric'. Valid options: 'bleu': "
                    "'chrf', 'token_accuracy', 'sequence_accuracy', 'wer'.")
        self.early_stopping_metric = train_cfg.get("early_stopping_metric",
                                                      "eval_metric").lower()
        if self.early_stopping_metric not in ['loss', 'ppl', 'eval_metric']:
            raise ConfigurationError(
                "Invalid setting for 'early_stopping_metric'. "
                "Valid options: 'loss', 'ppl', 'eval_metric'.")
        self.early_stopping_metric_name = self.eval_metrics[0] \
            if self.early_stopping_metric == "eval_metric" \
            else self.early_stopping_metric
        # early_stopping_metric decides on how to find the early stopping point:
        # ckpts are written when there's a new high/low score for this metric.
        # If we schedule after BLEU/chrf/accuracy, we want to maximize the
        # score, else we want to minimize it.
        if self.early_stopping_metric_name in ["ppl", "loss", "wer"]:
            self.minimize_metric = True
        elif self.early_stopping_metric_name in [
            "acc", "bleu", "chrf", "token_accuracy", "sequence_accuracy"]:
          self.minimize_metric = False

        # eval options
        self.sacrebleu = {"remove_whitespace": True, "tokenize": "13a"}
        if "sacrebleu" in test_cfg.keys():
            self.sacrebleu["remove_whitespace"] = test_cfg["sacrebleu"] \
                .get("remove_whitespace", True)
            self.sacrebleu["tokenize"] = test_cfg["sacrebleu"] \
                .get("tokenize", "13a")
        if "wer" in self.eval_metrics:
            eval_tokenizer = EvaluationTokenizer(
                tokenize=self.sacrebleu["tokenize"],
                lowercase=data_cfg["trg"].get("lowercase", False),
                remove_punctuation=test_cfg["sacrebleu"] \
                    .get("remove_punctuation", False),
                level="char" if data_cfg["trg"]["level"] == "char" else "word")
            logger.info(eval_tokenizer)
            self.sacrebleu["tok_fun"] = eval_tokenizer

        # learning rate scheduling
        self.scheduler, self.scheduler_step_at = build_scheduler(
            config=train_cfg,
            scheduler_mode="min" if self.minimize_metric else "max",
            optimizer=self.optimizer,
            hidden_size=model_cfg["encoder"]["hidden_size"])

        # data & batch handling
        self.seed = train_cfg.get("random_seed", 42)
        self.shuffle = train_cfg.get("shuffle", True)
        self.epochs = train_cfg["epochs"]
        self.max_updates = train_cfg.get("updates", np.inf)
        self.batch_size = train_cfg["batch_size"]
        # Placeholder so that we can use the train_iter in other functions.
        self.train_iter, self.train_iter_state = None, None
        # per-device batch_size = self.batch_size // self.n_gpu
        self.batch_type = train_cfg.get("batch_type", "sentence")
        self.eval_batch_size = train_cfg.get("eval_batch_size", self.batch_size)
        # per-device eval_batch_size = self.eval_batch_size // self.n_gpu
        self.eval_batch_type = train_cfg.get("eval_batch_type", self.batch_type)

        self.batch_multiplier = train_cfg.get("batch_multiplier", 1)

        # generation
        self.max_output_length = train_cfg.get("max_output_length", None)

        # CPU / GPU
        use_cuda = train_cfg["use_cuda"] and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        if self.device.type == "cuda":
            self.model.to(self.device)
        self.n_gpu = torch.cuda.device_count() if use_cuda else 0
        #self.num_workers = train_cfg.get("num_workers", 0)
        #if self.num_workers > 0:
        #    self.num_workers = min(cpu_count(), self.num_workers)

        # fp16
        self.fp16 = train_cfg.get("fp16", False)
        if self.fp16:
            if 'apex' not in sys.modules:
                raise ImportError("Please install apex from "
                                  "https://www.github.com/nvidia/apex "
                                  "to use fp16 training.") from no_apex
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer,
                                                    opt_level='O1')
            # opt level: one of {"O0", "O1", "O2", "O3"}
            # see https://nvidia.github.io/apex/amp.html#opt-levels

        # initialize training statistics
        self.stats = self.TrainStatistics(
            steps=0,
            is_min_lr=False,
            is_max_update=False,
            total_tokens=0,
            best_ckpt_iter=0,
            best_ckpt_score=np.inf if self.minimize_metric else -np.inf,
            minimize_metric=self.minimize_metric)

        # model parameters
        if "load_model" in train_cfg.keys():
            self.init_from_checkpoint(
                Path(train_cfg["load_model"]),
                reset_best_ckpt=train_cfg.get("reset_best_ckpt", False),
                reset_scheduler=train_cfg.get("reset_scheduler", False),
                reset_optimizer=train_cfg.get("reset_optimizer", False),
                reset_iter_state=train_cfg.get("reset_iter_state", False))
        for enc_dec in ['encoder', 'decoder']:
            if f"load_{enc_dec}" in train_cfg.keys():
                self.init_layers(path=Path(train_cfg[f"load_{enc_dec}"]),
                                 layer=enc_dec)

        # multi-gpu training (should be after apex fp16 initialization)
        if self.n_gpu > 1:
            self.model = _DataParallel(self.model)

    def _save_checkpoint(self, new_best: bool, score: float) -> None:
        """
        Save the model's current parameters and the training state to a
        checkpoint.

        The training state contains the total number of training steps,
        the total number of training tokens,
        the best checkpoint score and iteration so far,
        and optimizer and scheduler states.

        :param new_best: This boolean signals which symlink we will use for the
                         new checkpoint. If it is true, we update best.ckpt.
        :param score: Validation score which is used as key of heap queue.
                        if score is float('nan'), the queue won't be updated.
        """
        model_path = Path(self.model_dir) / f"{self.stats.steps}.ckpt"
        model_state_dict = self.model.module.state_dict() \
            if isinstance(self.model, torch.nn.DataParallel) \
            else self.model.state_dict()
        state = {
            "steps": self.stats.steps,
            "total_tokens": self.stats.total_tokens,
            "best_ckpt_score": self.stats.best_ckpt_score,
            "best_ckpt_iteration": self.stats.best_ckpt_iter,
            "model_state": model_state_dict,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": (self.scheduler.state_dict()
                                if self.scheduler is not None else None),
            'amp_state': amp.state_dict() if self.fp16 else None,
            "train_iter_state":
                self.train_iter.batch_sampler.sampler.generator.get_state()
        }
        torch.save(state, model_path.as_posix())

        # update symlink
        symlink_target = Path(f"{self.stats.steps}.ckpt")
        # last symlink
        last_path = Path(self.model_dir) / "latest.ckpt"
        prev_path = symlink_update(symlink_target, last_path) # update always
        # best symlink
        best_path = Path(self.model_dir) / "best.ckpt"
        if new_best:
            prev_path = symlink_update(symlink_target, best_path)
            assert best_path.resolve().stem == str(self.stats.best_ckpt_iter)

        # push to and pop from the heap queue
        to_delete = None
        if not math.isnan(score) and self.num_ckpts > 0:
            if len(self.ckpt_queue) < self.num_ckpts:   # no pop, push only
                heapq.heappush(self.ckpt_queue, (score, model_path))
            else:   # push + pop the worst one in the queue
                if self.minimize_metric:
                    # pylint: disable=protected-access
                    heapq._heapify_max(self.ckpt_queue)
                    to_delete = heapq._heappop_max(self.ckpt_queue)
                    heapq.heappush(self.ckpt_queue, (score, model_path))
                    # pylint: enable=protected-access
                else:
                    to_delete = heapq.heappushpop(self.ckpt_queue,
                                          (score, model_path))

            if to_delete is not None:
                assert to_delete[1] != model_path   # don't delete the last ckpt
                if to_delete[1].stem != best_path.resolve().stem:
                    delete_ckpt(to_delete[1])   # don't delete the best ckpt

            assert len(self.ckpt_queue) <= self.num_ckpts

            # remove old symlink target if not in queue after push/pop
            if prev_path is not None and \
                    prev_path.stem not in [c[1].stem for c in self.ckpt_queue]:
                delete_ckpt(prev_path)

    def init_from_checkpoint(self,
                             path: Path,
                             reset_best_ckpt: bool = False,
                             reset_scheduler: bool = False,
                             reset_optimizer: bool = False,
                             reset_iter_state: bool = False) -> None:
        """
        Initialize the trainer from a given checkpoint file.

        This checkpoint file contains not only model parameters, but also
        scheduler and optimizer states, see `self._save_checkpoint`.

        :param path: path to checkpoint
        :param reset_best_ckpt: reset tracking of the best checkpoint,
                                use for domain adaptation with a new dev
                                set or when using a new metric for fine-tuning.
        :param reset_scheduler: reset the learning rate scheduler, and do not
                                use the one stored in the checkpoint.
        :param reset_optimizer: reset the optimizer, and do not use the one
                                stored in the checkpoint.
        :param reset_iter_state: reset the sampler's internal state and do not
                                use the one stored in the checkpoint.
        """
        logger.info("Loading model from %s", path)
        model_checkpoint = load_checkpoint(path=path, device=self.device)

        # restore model and optimizer parameters
        self.model.load_state_dict(model_checkpoint["model_state"])

        if not reset_optimizer:
            self.optimizer.load_state_dict(model_checkpoint["optimizer_state"])
        else:
            logger.info("Reset optimizer.")

        if not reset_scheduler:
            if model_checkpoint["scheduler_state"] is not None and \
                    self.scheduler is not None:
                self.scheduler.load_state_dict(
                    model_checkpoint["scheduler_state"])
        else:
            logger.info("Reset scheduler.")

        if not reset_best_ckpt:
            self.stats.best_ckpt_score = model_checkpoint["best_ckpt_score"]
            self.stats.best_ckpt_iter = model_checkpoint["best_ckpt_iteration"]
        else:
            logger.info("Reset tracking of the best checkpoint.")

        if not reset_iter_state:
            # restore counters
            self.stats.steps = model_checkpoint["steps"]
            self.stats.total_tokens = model_checkpoint["total_tokens"]
            self.train_iter_state = model_checkpoint["train_iter_state"]
        else:
            # reset counters if explicitly 'train_iter_state: True' in config
            logger.info("Reset data iterator (random seed: {%d}).", self.seed)

        # fp16
        if self.fp16 and model_checkpoint.get("amp_state", None) is not None:
            amp.load_state_dict(model_checkpoint['amp_state'])

    def init_layers(self, path: Path, layer: str) -> None:
        """
        Initialize encoder decoder layers from a given checkpoint file.

        :param path: path to checkpoint
        :param layer: layer name; 'encoder' or 'decoder' expected
        """
        assert path is not None
        layer_state_dict = OrderedDict()
        logger.info("Loading %s model from %s", layer, path)
        ckpt = load_checkpoint(path=path, device=self.device)
        for k, v in ckpt['model_state'].items():
            if k.startswith(layer):
                layer_state_dict[k] = v
        self.model.load_state_dict(layer_state_dict, strict=False)

    def train_and_validate(self, train_data: TsvDataset,
                           valid_data: TsvDataset) -> None:
        """
        Train the model and validate it from time to time on the validation set.

        :param train_data: training data
        :param valid_data: validation data
        """
        # pylint: disable=unnecessary-comprehension
        # pylint: disable=too-many-branches
        # pylint: disable=too-many-statements}
        self.train_iter = make_data_iter(dataset=train_data,
                                         batch_size=self.batch_size,
                                         batch_type=self.batch_type,
                                         seed=self.seed,
                                         shuffle=self.shuffle,
                                         #num_workers=self.num_workers,
                                         device=self.device,
                                         pad_index=self.model.pad_index,
                                         normalization=self.normalization)

        if self.train_iter_state is not None:
            self.train_iter.batch_sampler.sampler.generator.set_state(
                self.train_iter_state.cpu())

        #################################################################
        # simplify accumulation logic:
        #################################################################
        # for epoch in range(epochs):
        #     self.model.zero_grad()
        #     epoch_loss = 0.0
        #     batch_loss = 0.0
        #     for i, batch in enumerate(self.train_iter):
        #
        #         # gradient accumulation:
        #         # loss.backward() inside _train_step()
        #         batch_loss += self._train_step(inputs)
        #
        #         if (i + 1) % self.batch_multiplier == 0:
        #             self.optimizer.step()     # update!
        #             self.model.zero_grad()    # reset gradients
        #             self.steps += 1           # increment counter
        #
        #             epoch_loss += batch_loss  # accumulate batch loss
        #             batch_loss = 0            # reset batch loss
        #
        #     # leftovers are just ignored.
        #################################################################

        # pylint: disable=logging-too-many-args
        logger.info(
            "Train stats:\n"
            "\tdevice: %s\n"
            "\tn_gpu: %d\n"
            "\t16-bits training: %r\n"
            "\tgradient accumulation: %d\n"
            "\tbatch size per device: %d\n"
            "\ttotal batch size (w. parallel & accumulation): %d",
            self.device.type, self.n_gpu, self.fp16, self.batch_multiplier,
            self.batch_size // self.n_gpu if self.n_gpu > 1
            else self.batch_size, self.batch_size * self.batch_multiplier)
        # pylint: enable=logging-too-many-args

        try:
            for epoch_no in range(self.epochs):
                logger.info("EPOCH %d", epoch_no + 1)

                if self.scheduler_step_at == "epoch":
                    self.scheduler.step(epoch=epoch_no)

                self.model.train()

                # Reset statistics for each epoch.
                start = time.time()
                total_valid_duration = 0
                start_tokens = self.stats.total_tokens
                self.model.zero_grad()
                epoch_loss = 0
                batch_losses = defaultdict(float)

                batch: Batch    # yield a joeynmt Batch object
                for i, batch in enumerate(self.train_iter):
                    # sort batch now by src length and keep track of order
                    batch.sort_by_src_length()

                    # get batch loss
                    loss_dict = self._train_step(batch)
                    for k, v in loss_dict.items():
                        batch_losses[k] += v

                    # update!
                    if (i + 1) % self.batch_multiplier == 0:
                        # clip gradients (in-place)
                        if self.clip_grad_fun is not None:
                            if self.fp16:
                                self.clip_grad_fun(
                                    parameters=amp.master_params(self.optimizer))
                            else:
                                self.clip_grad_fun(
                                    parameters=self.model.parameters())

                        # make gradient step
                        self.optimizer.step()

                        # decay lr
                        if self.scheduler_step_at == "step":
                            self.scheduler.step(step=self.stats.steps)

                        # reset gradients
                        self.model.zero_grad()

                        # increment step counter
                        self.stats.steps += 1
                        if self.stats.steps >= self.max_updates:
                            self.stats.is_max_update = True

                        # log learning progress
                        if self.stats.steps % self.logging_freq == 0:
                            for k, v in batch_losses.items():
                                self.tb_writer.add_scalar(f"train/batch_{k}",
                                                          v, self.stats.steps)
                            elapsed = time.time() - start - total_valid_duration
                            elapsed_tokens = self.stats.total_tokens - start_tokens
                            logger.info(
                                "Epoch %3d, Step: %8d, Batch Loss: %12.6f, "
                                "Batch Acc: %.6f, Tokens per Sec: %8.0f, Lr: %.6f",
                                epoch_no + 1, self.stats.steps, batch_losses['loss'],
                                batch_losses['acc'], elapsed_tokens / elapsed,
                                self.optimizer.param_groups[0]["lr"])
                            start = time.time()
                            total_valid_duration = 0
                            start_tokens = self.stats.total_tokens

                        # update epoch_loss
                        epoch_loss += batch_losses['loss']  # accumulate loss
                        batch_losses = defaultdict(float)

                        # validate on the entire dev set
                        if self.stats.steps % self.validation_freq == 0:
                            if valid_data.random_subset > 0: # sample random subset
                                valid_data.sample_random_subset(seed=self.stats.steps)
                            valid_duration = self._validate(valid_data, epoch_no)
                            total_valid_duration += valid_duration

                        # check current_lr
                        current_lr = self.optimizer.param_groups[0]['lr']
                        if current_lr < self.learning_rate_min:
                            self.stats.is_min_lr = True

                        self.tb_writer.add_scalar("train/learning_rate",
                                                  current_lr, self.stats.steps)

                    if self.stats.is_min_lr or self.stats.is_max_update:
                        break

                if self.stats.is_min_lr or self.stats.is_max_update:
                    log_str = f"minimum lr {self.learning_rate_min}" \
                        if self.stats.is_min_lr \
                        else f"maximum num. of updates {self.max_updates}"
                    logger.info("Training ended since %s was reached.", log_str)
                    break

                logger.info('Epoch %3d: total training loss %.2f', epoch_no + 1,
                            epoch_loss)
            else:
                logger.info('Training ended after %3d epochs.', epoch_no + 1)
            logger.info('Best validation result (greedy) at step %8d: %6.2f %s.',
                        self.stats.best_ckpt_iter, self.stats.best_ckpt_score,
                        self.early_stopping_metric)
        except KeyboardInterrupt:
            self._save_checkpoint(False, float("nan"))

        self.tb_writer.close()  # close Tensorboard writer
        if isinstance(train_data, TranslationDataset):
            train_data.close_file()
        if isinstance(valid_data, TranslationDataset):
            valid_data.close_file()

    def _train_step(self, batch: Batch) -> Tensor:
        """
        Train the model on one batch: Compute the loss.

        :param batch: training batch
        :return: loss for batch (sum)
        """
        # pylint: disable=consider-iterating-dictionary
        # reactivate training
        self.model.train()

        # get loss (run as during training with teacher forcing)
        batch_loss, nll_loss, ctc_loss, correct_tokens = self.model(
            return_type="loss", **vars(batch))
        losses = {'loss': batch_loss, 'acc': correct_tokens}
        if torch.is_tensor(nll_loss): # nll_loss is not None
            losses['nll_loss'] = nll_loss
        if torch.is_tensor(ctc_loss): # ctc_loss is not None
            losses['ctc_loss'] = ctc_loss

        # sum multi-gpu losses
        if self.n_gpu > 1:
            for l in losses.keys():
                losses[l] = losses[l].sum()

        # normalize batch loss
        for l in losses.keys():
            denominator = batch.ntokens if l == 'acc' else batch.normalizer
            losses[l] = losses[l] / denominator

        if self.batch_multiplier > 1:
            for l in losses.keys():
                losses[l] = losses[l] / self.batch_multiplier

        # accumulate gradients
        if self.fp16:
            with amp.scale_loss(losses['loss'], self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            losses['loss'].backward()

        # increment token counter
        self.stats.total_tokens += batch.ntokens

        for l in losses.keys():
            losses[l] = losses[l].item()
        return losses

    def _validate(self, valid_data: TsvDataset, epoch_no: int):
        valid_start_time = time.time()

        (valid_scores, valid_sources, valid_references, valid_hypotheses,
         valid_hypotheses_raw, valid_attention_scores) = validate_on_data(
            data=valid_data,
            model=self.model,
            batch_size=self.eval_batch_size,
            batch_type=self.eval_batch_type,
            eval_metrics=self.eval_metrics,
            max_output_length=self.max_output_length,
            compute_loss=True,          # normalization == "batch"
            n_best=1, beam_size=1,      # 1best greedy decoding
            sacrebleu=self.sacrebleu,   # sacrebleu options
            device=self.device,
            n_gpu=self.n_gpu,
            normalization=self.normalization
        )

        #for eval_metric in ['loss', 'ppl', 'acc'] + self.eval_metrics:
        for eval_metric, score in valid_scores.items():
            if not math.isnan(score):
                self.tb_writer.add_scalar(
                    f"valid/{eval_metric}", score, self.stats.steps)

        ckpt_score = valid_scores[self.early_stopping_metric_name]

        if self.scheduler_step_at == "validation":
            self.scheduler.step(metrics=ckpt_score)

        # update new best
        new_best = self.stats.is_best(ckpt_score)
        if new_best:
            self.stats.best_ckpt_score = ckpt_score
            self.stats.best_ckpt_iter = self.stats.steps
            logger.info('Hooray! New best validation result [%s]!',
                        self.early_stopping_metric_name)

        # save checkpoints
        is_better = self.stats.is_better(ckpt_score, self.ckpt_queue) \
            if len(self.ckpt_queue) > 0 else True
        if self.num_ckpts < 0 or is_better:
            self._save_checkpoint(new_best, ckpt_score)

        # append to validation report
        self._add_report(valid_scores=valid_scores, new_best=new_best)

        self._log_examples(sources=valid_sources if self.task=="MT" else None,
                           references=valid_references,
                           hypotheses=valid_hypotheses,
                           hypotheses_raw=valid_hypotheses_raw,
                           data=valid_data)

        valid_duration = time.time() - valid_start_time
        score_str = ', '.join(["{:s}: {:6.2f}".format(eval_metric,
                                                      valid_scores[eval_metric])
            for eval_metric in self.eval_metrics + ['loss', 'ppl', 'acc']])
        logger.info('Validation result (greedy) at epoch %3d, '
                    'step %8d: %s, duration: %.4fs', epoch_no + 1,
                    self.stats.steps, score_str, valid_duration)

        # store validation set outputs
        write_list_to_file(self.model_dir / f"{self.stats.steps}.hyps",
                           valid_hypotheses)

        # store attention plots for selected valid sentences
        if valid_attention_scores:
            store_attention_plots(
                attentions=valid_attention_scores,
                targets=valid_hypotheses_raw,
                sources=valid_data.tokenizer['src'](valid_sources) \
                    if self.task== "MT" else None,
                indices=self.log_valid_sents,
                output_prefix=(self.model_dir / f"att.{self.stats.steps}"
                               ).as_posix(),
                tb_writer=self.tb_writer,
                steps=self.stats.steps)

        return valid_duration

    def _add_report(self, valid_scores: dict, new_best: bool = False) -> None:
        """
        Append a one-line report to validation logging file.

        :param valid_scores: validation evaluation score [eval_metric]
        :param new_best: whether this is a new best model
        """
        current_lr = self.optimizer.param_groups[0]['lr']

        valid_file = self.model_dir / "validations.txt"
        with valid_file.open('a', encoding="utf-8") as opened_file:
            score_str = "\t".join(
            ["Steps: {}".format(self.stats.steps)]
                + ["{}: {:.5f}".format(eval_metric,
                                       valid_scores[eval_metric.lower()])
                for eval_metric in ['Loss', 'PPL', 'Acc'] + self.eval_metrics]
                + ["LR: {:.8f}".format(current_lr), "*" if new_best else ""])
            opened_file.write(f"{score_str}\n")

    def _log_examples(self,
                      sources: List[str],
                      hypotheses: List[str],
                      references: List[str],
                      hypotheses_raw: List[List[str]],
                      data: TsvDataset) -> None:
        """
        Log a the first `self.log_valid_sents` sentences from given examples.

        :param sources: decoded sources (list of strings)
        :param hypotheses: decoded hypotheses (list of strings)
        :param references: decoded references (list of strings)
        :param hypotheses_raw: raw hypotheses (list of list of tokens)
        :param data: TsvDataset
        """
        for p in self.log_valid_sents:

            if p >= len(hypotheses):
                continue

            logger.info("Example #%d", p)
            # tokenized text
            if self.task == "MT":
                logger.debug("\tRaw source:     %s",
                             data.get_item(idx=p, side='src'))
            logger.debug("\tRaw reference:  %s",
                         data.get_item(idx=p, side='trg'))
            logger.debug("\tRaw hypothesis: %s", hypotheses_raw[p])

            # post-processed text
            if self.task == "MT":
                logger.info("\tSource:     %s", sources[p])
            logger.info("\tReference:  %s", references[p])
            logger.info("\tHypothesis: %s", hypotheses[p])


    class TrainStatistics:
        def __init__(self,
                     steps: int = 0,
                     is_min_lr: bool = False,
                     is_max_update: bool = False,
                     total_tokens: int = 0,
                     best_ckpt_iter: int = 0,
                     best_ckpt_score: float = np.inf,
                     minimize_metric: bool = True) -> None:
            # global update step counter
            self.steps = steps
            # stop training if this flag is True
            # by reaching learning rate minimum
            self.is_min_lr = is_min_lr
            # stop training if this flag is True
            # by reaching max num of updates
            self.is_max_update = is_max_update
            # number of total tokens seen so far
            self.total_tokens = total_tokens
            # store iteration point of best ckpt
            self.best_ckpt_iter = best_ckpt_iter
            # initial values for best scores
            self.best_ckpt_score = best_ckpt_score
            # minimize or maximize score
            self.minimize_metric = minimize_metric

        def is_best(self, score):
            if self.minimize_metric:
                is_best = score < self.best_ckpt_score
            else:
                is_best = score > self.best_ckpt_score
            return is_best

        def is_better(self, score: float, heap_queue: list):
            assert len(heap_queue) > 0
            if self.minimize_metric:
                is_better = score < heapq.nlargest(1, heap_queue)[0][0]
            else:
                is_better = score > heapq.nsmallest(1, heap_queue)[0][0]
            return is_better


def train(cfg_file: str, skip_test: bool = False) -> None:
    """
    Main training function. After training, also test on test data if given.

    :param cfg_file: path to configuration yaml file
    :param skip_test: whether a test should be run or not after training
    """
    # read config file
    cfg = load_config(Path(cfg_file))
    task = cfg["data"]["task"]  # "MT" or "s2t"

    # make logger
    model_dir = make_model_dir(Path(cfg["training"]["model_dir"]),
                               overwrite=cfg["training"].get(
                                   "overwrite", False))
    joeynmt_version = make_logger(model_dir, mode="train")
    if "joeynmt_version" in cfg:
        assert str(joeynmt_version) == str(cfg["joeynmt_version"]), \
            f'You are using JoeyNMT version {joeynmt_version}, ' \
            f'but {cfg["joeynmt_version"]} is expected in the given config.'
    # TODO: save version number in model checkpoints

    # write all entries of config to the log
    log_cfg(cfg)

    # store copy of original training config in model dir
    shutil.copy2(cfg_file, (model_dir / "config.yaml").as_posix())

    # set the random seed
    set_seed(seed=cfg["training"].get("random_seed", 42))

    # load the data
    src_vocab, trg_vocab, train_data, dev_data, test_data = load_data(
        data_cfg=cfg["data"])

    # store the vocabs
    if task == "MT":
        src_vocab.to_file(model_dir / "src_vocab.txt")
    trg_vocab.to_file(model_dir / "trg_vocab.txt")

    # build an encoder-decoder model
    model = build_model(cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab)

    # for training management, e.g. early stopping and model selection
    trainer = TrainManager(model=model, config=cfg)

    # train the model
    trainer.train_and_validate(train_data=train_data, valid_data=dev_data)

    if not skip_test:
        # predict with the best model on validation and test
        # (if test data is available)
        ckpt = model_dir / f"{trainer.stats.best_ckpt_iter}.ckpt"
        output_path = (model_dir
                       / "{:08d}.hyps".format(trainer.stats.best_ckpt_iter))
        datasets_to_test = {
            "dev": dev_data,
            "test": test_data,
            "src_vocab": src_vocab,
            "trg_vocab": trg_vocab
        }
        test(cfg_file,
             ckpt=ckpt.as_posix(),
             output_path=output_path.as_posix(),
             datasets=datasets_to_test)
    else:
        logger.info("Skipping test after training")


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Joey-NMT')
    parser.add_argument("config",
                        default="configs/default.yaml",
                        type=str,
                        help="Training configuration file (yaml).")
    args = parser.parse_args()
    train(cfg_file=args.config)

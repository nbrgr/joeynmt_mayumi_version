# coding: utf-8
"""
Implementation of a mini-batch.
"""
import logging
from typing import List, Optional
import numpy as np

import torch
from torch import Tensor

from joeynmt.constants import PAD_ID
logger = logging.getLogger(__name__)


class Batch:
    """
    Object for holding a batch of data with mask during training.
    Input is yielded from collate_fn in torch.data.utils.DataLoader.
    """
    def __init__(self, src: Tensor, src_length: Tensor, trg: Optional[Tensor],
                 trg_length: Optional[Tensor], device: torch.device,
                 pad_index: int = PAD_ID, normalization: str = "batch"):
        """
        Create a new joey batch. This batch supports attributes with
        src and trg length, masks, number of non-padded tokens in trg.
        Furthermore, it can be sorted by src length.

        :param src:
        :param src_length:
        :param trg:
        :param trg_length:
        :param device:
        :param pad_index: *must be the same for both src and trg
        :param normalization:
        """
        # pylint: disable=too-many-instance-attributes
        self.src = src
        self.src_length = src_length
        self.src_mask = (self.src != pad_index).unsqueeze(1)
        self.nseqs = self.src.size(0) # int
        self.trg_input: Optional[Tensor] = None
        self.trg: Optional[Tensor] = None
        self.trg_mask: Optional[Tensor] = None
        self.trg_length: Optional[Tensor] = None
        self.ntokens: Optional[int] = None
        self.has_trg = trg is not None and trg_length is not None

        if self.has_trg:
            # trg_input is used for teacher forcing, last one is cut off
            self.trg_input = trg[:, :-1]    # shape (batch_size, seq_length)
            self.trg_length = trg_length - 1
            # trg is used for loss computation, shifted by one since BOS
            self.trg = trg[:, 1:]           # shape (batch_size, seq_length)
            # we exclude the padded areas (and blank areas)
            # from the loss computation
            self.trg_mask = (self.trg != pad_index).unsqueeze(1)
            self.ntokens = (self.trg != pad_index).data.sum().item() # int

        self.normalizer = 1 if normalization == "none" \
                            else self.ntokens if normalization == "tokens" \
                            else self.nseqs #if normalization == "batch"

        if device.type == "cuda":
            self._make_cuda(device)

    def _make_cuda(self, device: torch.device) -> None:
        """
        Move the batch to GPU
        """
        self.src = self.src.to(device)
        self.src_length = self.src_length.to(device)
        if self.src_mask is not None:
            self.src_mask = self.src_mask.to(device)

        if self.has_trg:
            self.trg_input = self.trg_input.to(device)
            self.trg = self.trg.to(device)
            self.trg_mask = self.trg_mask.to(device)

    def sort_by_src_length(self) -> List[int]:
        """
        Sort by src length (descending) and return index to revert sort

        :return: list of indices
        """
        _, perm_index = self.src_length.sort(0, descending=True)
        rev_index = [0] * perm_index.size(0)
        for new_pos, old_pos in enumerate(perm_index.cpu().numpy()):
            rev_index[old_pos] = new_pos

        sorted_src_length = self.src_length[perm_index]
        sorted_src = self.src[perm_index]
        if self.src_mask is not None:
            sorted_src_mask = self.src_mask[perm_index]
        if self.has_trg:
            sorted_trg_input = self.trg_input[perm_index]
            sorted_trg_length = self.trg_length[perm_index]
            sorted_trg_mask = self.trg_mask[perm_index]
            sorted_trg = self.trg[perm_index]

        self.src = sorted_src
        self.src_length = sorted_src_length
        if self.src_mask is not None:
            self.src_mask = sorted_src_mask

        if self.has_trg:
            self.trg_input = sorted_trg_input
            self.trg_mask = sorted_trg_mask
            self.trg_length = sorted_trg_length
            self.trg = sorted_trg

        return rev_index

    def __repr__(self) -> str:
        return "%s(nseqs=%d, ntokens=%d, has_trg=%r, normalizer=%d)" % (
            self.__class__.__name__, self.nseqs, self.ntokens, self.has_trg,
            self.normalizer)


class SpeechBatch(Batch):
    """Batch object for speech data"""
    # pylint: disable=too-many-instance-attributes

    def __init__(self, src: np.ndarray, src_length: Tensor,
                 trg: Tensor, trg_length: Tensor, device: torch.device,
                 pad_index: int = PAD_ID, normalization: str = "batch",
                 **kwargs):

        self.is_train = kwargs["is_train"]
        #self.steps = kwargs.get("steps", -1)
        self.cmvn = kwargs.get("cmvn", None)
        self.specaugment = kwargs.get("specaugment", None)

        # data augmentation
        src, src_length, trg, trg_length = self._augment(
            src, src_length, trg, trg_length)
        src = torch.from_numpy(src).float().to(device)

        super().__init__(src, src_length, trg, trg_length,
                         device, pad_index, normalization)
        # note that src PAD_ID is BLANK_ID, which is different from trg PAD_ID!
        self.src_mask = None # will be constructed in encoder
        self.src_max_len = src.size(1)
        if self.is_train:
            assert self.has_trg

    def _augment(self, src_input: np.ndarray, src_length: Tensor,
                 trg_input: Tensor, trg_length: Tensor):
        """
        Augment Data

        :param src_input: np.ndarray, shape (batch_size, src_len, embed_size)
        :param src_length: torch.Tensor, shape (batch_size)
        :param trg_input: torch.Tensor, shape (batch_size)
        :param trg_length: torch.Tensor, shape (batch_size)
        :return: src_input_aug, src_length_aug, trg_input_aug, trg_length_aug
        """
        batch_size = len(src_input)
        src_input_aug = src_input.copy() # (batch_size, num_freq, num_frames)
        for i in range(batch_size):
            # you can implement here your own data augmentation methods

            #trg = trg_input[i]
            #t_l = trg_length[i]

            s_l = src_length[i]

            # apply both train and test data
            if self.cmvn and self.cmvn.before:
                src_input_aug[i, :s_l, :] = self.cmvn(
                    src_input_aug[i, :s_l, :])

            # train data only
            if self.is_train and self.specaugment:
                src_input_aug[i, :s_l, :] = self.specaugment(
                    src_input_aug[i, :s_l, :])
            
            if self.cmvn and not self.cmvn.before:
                src_input_aug[i, :s_l, :] = self.cmvn(
                    src_input_aug[i, :s_l, :])

        return src_input_aug, src_length, trg_input, trg_length

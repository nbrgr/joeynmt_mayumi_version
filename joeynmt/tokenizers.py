# coding: utf-8
"""
Tokenizer module
"""

import logging
import shutil
from pathlib import Path
from typing import Dict, List

import sentencepiece as sp
from subword_nmt import apply_bpe

from joeynmt.constants import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN
from joeynmt.helpers import ConfigurationError, remove_extra_spaces, unicode_normalize

logger = logging.getLogger(__name__)


class BasicTokenizer:
    SPACE = chr(32)  # ' ': half-width white space (ascii)
    SPACE_ESCAPE = chr(9601)  # '▁': sentencepiece default

    def __init__(
        self,
        level: str = "word",
        lowercase: bool = False,
        normalize: bool = False,
        max_length: int = -1,
        min_length: int = -1,
        **kwargs,
    ):
        self.level = level
        self.lowercase = lowercase
        self.normalize = normalize

        # filter by length
        self.max_length = max_length
        self.min_length = min_length

    def pre_process(self, raw_input: str) -> str:
        """
        Pre-process text
            - ex.) Lowercase, Normalize, Remove emojis,
                Pre-tokenize(add extra white space before punc) etc.
            - applied for all inputs both in training and inference.
        """
        if self.lowercase:
            raw_input = raw_input.lower()
        if self.normalize:
            raw_input = remove_extra_spaces(unicode_normalize(raw_input))
            # TODO: support other normalization(?)
        return raw_input

    def __call__(self, raw_input: str, is_train: bool = False) -> List[str]:
        """Tokenize single sentence"""
        sequence = self.pre_process(raw_input)
        if self.level == "word":
            sequence = sequence.split(self.SPACE)
        elif self.level == "char":
            sequence = list(sequence.replace(self.SPACE, self.SPACE_ESCAPE))

        if is_train and self._filter_by_length(len(sequence)):
            return None
        return sequence

    def _filter_by_length(self, length: int) -> bool:
        """
        Check if the given seq length is out of the valid range.

        :param length: (int) number of tokens
        :return: True if the length is invalid(= to be filtered out), False if valid.
        """
        if length > self.max_length > 0 or self.min_length > length > 0:
            return True
        else:
            return False

    def _remove_special(self, sequence: List[str], remove_unk: bool = False):
        specials = [BOS_TOKEN, EOS_TOKEN, PAD_TOKEN]
        if remove_unk:
            specials.append(UNK_TOKEN)
        return [token for token in sequence if token not in specials]

    def post_process(self,
                     sequence: List[str],
                     remove_unk: bool = False) -> str:
        """Detokenize"""
        sequence = self._remove_special(sequence, remove_unk=remove_unk)
        if self.level == "word":
            detokenized = self.SPACE.join(sequence)
        elif self.level == "char":
            detokenized = "".join(sequence).replace(self.SPACE_ESCAPE,
                                                    self.SPACE)

        # Remove extra spaces
        if self.normalize:
            detokenized = remove_extra_spaces(detokenized)
        return detokenized

    def __repr__(self):
        return (f"{self.__class__.__name__}(level={self.level}, "
                f"lowercase={self.lowercase}, normalize={self.normalize}, "
                f"filter_by_length=({self.min_length}, {self.max_length}))")


class SentencePieceTokenizer(BasicTokenizer):

    def __init__(
        self,
        level: str = "bpe",
        lowercase: bool = False,
        normalize: bool = False,
        max_length: int = -1,
        min_length: int = -1,
        **kwargs,
    ):
        super().__init__(level, lowercase, normalize, max_length, min_length)
        assert self.level == "bpe"

        self.model_file: Path = Path(kwargs["model_file"])
        assert self.model_file.is_file(
        ), f"model file {self.model_file} not found."

        self.spm = sp.SentencePieceProcessor()
        self.spm.load(kwargs["model_file"])

        self.nbest_size: bool = kwargs.get("nbest_size", 5)
        self.alpha: float = kwargs.get("alpha", 0.0)

    def __call__(self, raw_input: str, is_train: bool = False) -> List[str]:
        if is_train and self.alpha > 0:
            tokenized = self.spm.sample_encode_as_pieces(
                raw_input,
                nbest_size=self.nbest_size,
                alpha=self.alpha,
            )
        else:
            tokenized = self.spm.encode(raw_input, out_type=str)

        if is_train and self._filter_by_length(len(tokenized)):
            return None
        return tokenized

    def post_process(self,
                     sequence: List[str],
                     remove_unk: bool = False) -> str:
        sequence = self._remove_special(sequence, remove_unk=remove_unk)

        # Decode back to str
        detokenized = self.spm.decode(sequence)

        # Remove extra spaces
        if self.normalize:
            detokenized = remove_extra_spaces(detokenized)
        return detokenized

    def set_vocab(self, itos: List[str]) -> None:
        """
        Set vocab
        :param itos: (list) indices-to-symbols mapping
        """
        self.spm.SetVocabulary(itos)

    def copy_cfg_file(self, model_dir: Path) -> None:
        """Copy confg file to model_dir"""
        if (model_dir / self.model_file.name).is_file():
            logger.warning(
                "%s already exists. Stop copying.",
                (model_dir / self.model_file.name).as_posix(),
            )
        shutil.copy2(self.model_file,
                     (model_dir / self.model_file.name).as_posix())

    def __repr__(self):
        return (f"{self.__class__.__name__}(level={self.level}, "
                f"lowercase={self.lowercase}, normalize={self.normalize}, "
                f"filter_by_length=({self.min_length}, {self.max_length}), "
                f"tokenizer={self.spm.__class__.__name__}, "
                f"nbest_size={self.nbest_size}, alpha={self.alpha})")


class SubwordNMTTokenizer(BasicTokenizer):

    def __init__(
        self,
        level: str = "bpe",
        lowercase: bool = False,
        normalize: bool = False,
        max_length: int = -1,
        min_length: int = -1,
        **kwargs,
    ):
        super().__init__(level, lowercase, normalize, max_length, min_length)
        assert self.level == "bpe"

        self.codes: Path = Path(kwargs["codes"])
        assert self.codes.is_file(), f"codes file {self.codes} not found."

        bpe_parser = apply_bpe.create_parser()
        bpe_args = bpe_parser.parse_args([
            "--codes", kwargs["codes"], "--separator",
            kwargs.get("separator", "@@")
        ])
        self.bpe = apply_bpe.BPE(
            bpe_args.codes,
            bpe_args.merges,
            bpe_args.separator,
            None,
            bpe_args.glossaries,
        )
        self.separator: str = kwargs.get("separator", "@@")
        self.dropout: float = kwargs.get("dropout", 0.0)

    def __call__(self, raw_input: str, is_train: bool = False) -> List[str]:
        dropout = self.dropout if is_train else 0.0
        tokenized = self.bpe.process_line(raw_input, dropout).strip().split()
        if is_train and self._filter_by_length(len(tokenized)):
            return None
        return tokenized

    def post_process(self,
                     sequence: List[str],
                     remove_unk: bool = False) -> str:
        sequence = self._remove_special(sequence, remove_unk=remove_unk)

        # Remove separators, join with spaces
        detokenized = self.SPACE.join(sequence).replace(
            self.separator + self.SPACE, "")
        # Remove final merge marker.
        if detokenized.endswith(self.separator):
            detokenized = detokenized[:-2]

        # Remove extra spaces
        if self.normalize:
            detokenized = remove_extra_spaces(detokenized)
        return detokenized

    def copy_cfg_file(self, model_dir: Path) -> None:
        shutil.copy2(self.codes, (model_dir / self.codes.name).as_posix())

    def __repr__(self):
        return (f"{self.__class__.__name__}(level={self.level}, "
                f"lowercase={self.lowercase}, normalize={self.normalize}, "
                f"filter_by_length=({self.min_length}, {self.max_length}), "
                f"tokenizer={self.bpe.__class__.__name__}, "
                f"separator={self.separator}, dropout={self.dropout})")


def _build_tokenizer(cfg: Dict) -> BasicTokenizer:
    """Builds tokenizer."""
    tokenizer = None
    if cfg["level"] in ["word", "char"]:
        tokenizer = BasicTokenizer(
            level=cfg["level"],
            lowercase=cfg.get("lowercase", False),
            normalize=cfg.get("normalize", False),
            max_length=cfg.get("max_length", -1),
            min_length=cfg.get("min_length", -1),
        )
    elif cfg["level"] == "bpe":
        tokenizer_type = cfg.get("tokenizer_type",
                                 cfg.get("bpe_type", "sentencepiece"))
        if tokenizer_type == "sentencepiece":
            assert "tokenizer_cfg" in cfg and "model_file" in cfg[
                "tokenizer_cfg"]
            tokenizer = SentencePieceTokenizer(
                level=cfg["level"],
                lowercase=cfg.get("lowercase", False),
                normalize=cfg.get("normalize", False),
                max_length=cfg.get("max_length", -1),
                min_length=cfg.get("min_length", -1),
                **cfg["tokenizer_cfg"],
            )
        elif tokenizer_type == "subword-nmt":
            assert "tokenizer_cfg" in cfg and "codes" in cfg["tokenizer_cfg"]
            tokenizer = SubwordNMTTokenizer(
                level=cfg["level"],
                lowercase=cfg.get("lowercase", False),
                normalize=cfg.get("normalize", False),
                max_length=cfg.get("max_length", -1),
                min_length=cfg.get("min_length", -1),
                **cfg["tokenizer_cfg"],
            )
        else:
            raise ConfigurationError(
                f"{tokenizer_type}: Unknown tokenizer type.")
    else:
        raise ConfigurationError(
            f"{cfg['level']}: Unknown tokenization level.")
    return tokenizer


def build_tokenizer(data_cfg: Dict) -> Dict[str, BasicTokenizer]:
    src_lang = data_cfg["src"]["lang"]
    trg_lang = data_cfg["trg"]["lang"]
    tokenizer = {
        src_lang: _build_tokenizer(data_cfg["src"]),
        trg_lang: _build_tokenizer(data_cfg["trg"]),
    }
    logger.info("%s tokenizer: %s", src_lang, tokenizer[src_lang])
    logger.info("%s tokenizer: %s", trg_lang, tokenizer[trg_lang])
    return tokenizer

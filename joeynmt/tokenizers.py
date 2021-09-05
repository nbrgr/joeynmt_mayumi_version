# coding: utf-8
"""
Data module
"""

import logging
import sentencepiece as sp
from typing import List

from joeynmt.helpers_for_audio import remove_punc

logger = logging.getLogger(__name__)


class BasicTokenizer:
    SPACE = chr(32) # ' ': half-width white space (ascii)
    SPACE_ESCAPE = chr(9601) # '▁': sentencepiece default

    def __init__(self, level: str = "word", lowercase: bool = False,
                 remove_punctuation: bool = False):
        self.level = level
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation

    def pre_process(self, raw_input: str):
        if self.lowercase:
            raw_input = raw_input.lower()
        if self.remove_punctuation:
            raw_input = remove_punc(raw_input)
        return raw_input

    def __call__(self, raw_input: str, sample: bool = False) -> List[str]:
        raw_input = self.pre_process(raw_input)
        if self.level == "word":
            tokenized = raw_input.split(self.SPACE)
        elif self.level == "char":
            tokenized = list(raw_input.replace(self.SPACE, self.SPACE_ESCAPE))
        else:
            raise ConfigurationError(f"Unknown tokenization level {self.level}")
        return tokenized

    def post_process(self, output: List[str]) -> str:
        if self.level == "word":
            detokenized = self.SPACE.join(output)
        elif self.level == "char":
            detokenized = "".join(output).replace(self.SPACE_ESCAPE, self.SPACE)
        else:
            raise ConfigurationError(f"Unknown tokenization level {self.level}")
        return detokenized

    def __repr__(self):
        return "%s(level=%r, lowercase=%r, remove_punctuation=%r)" % (
            self.__class__.__name__, self.level, self.lowercase,
            self.remove_punctuation)


class SentencePieceTokenizer(BasicTokenizer):
    def __init__(self, level: str = "bpe", lowercase: bool = False,
                 remove_punctuation: bool = False, model_file: str = "",
                 enable_sampling: bool = True, alpha: float = 0.1,
                 nbest_size: int = -1):
        # pylint: unexpected-keyword-arg
        super().__init__(level, lowercase, remove_punctuation)
        assert self.level == "bpe"

        self.spm = sp.SentencePieceProcessor(model_file=model_file)

        self.enable_sampling = enable_sampling
        self.alpha = alpha
        self.nbest_size = nbest_size

    def __call__(self, raw_input: str, sample: bool = False) -> List[str]:
        raw_input = self.pre_process(raw_input)
        if sample:
            tokenized = self.spm.encode(raw_input, out_type=str,
                                       enable_sampling=self.enable_sampling,
                                       alpha=self.alpha,
                                       nbest_size=self.nbest_size)
        else:
            tokenized = self.spm.encode(raw_input, out_type=str)
        return tokenized

    def post_process(self, output: List[str]) -> str:
        # return "".join(output).replace(self.SPACE, "").replace("▁", self.SPACE).strip()
        return self.spm.decode(output)

    def __repr__(self):
        return "%s(level=%r, lowercase=%r, remove_punctuation=%r, tokenizer=%r)" % (
            self.__class__.__name__, self.level, self.lowercase,
            self.remove_punctuation, self.spm.__class__.__name__)

# from fairseq
class EvaluationTokenizer(BasicTokenizer):
    """A generic evaluation-time tokenizer, which leverages built-in tokenizers
    in sacreBLEU (https://github.com/mjpost/sacrebleu). It additionally provides
    lowercasing, punctuation removal and character tokenization, which are
    applied after sacreBLEU tokenization.

    :param tokenize: (str) the type of sacreBLEU tokenizer to apply.
    :param lowercase: (bool) lowercase the text.
    :param remove_punctuation: (bool) remove punctuation (based on unicode
        category) from text.
    :param level: (str) tokenization level. {"word", "bpe", "char"}
    """
    ALL_TOKENIZER_TYPES = ["none", "13a", "intl", "zh", "ja-mecab"]

    def __init__(self, level: str = "word", lowercase: bool = False,
                 remove_punctuation: bool = False, tokenize: str = "13a"):
        # pylint: disable=import-outside-toplevel
        super().__init__(level, lowercase, remove_punctuation)

        from sacrebleu.metrics.bleu import _get_tokenizer

        assert tokenize in self.ALL_TOKENIZER_TYPES, \
            f"{tokenize}, {self.ALL_TOKENIZER_TYPES}"
        self.tokenizer = _get_tokenizer(tokenize)()

    def __call__(self, raw_input: str):
        if self.level == "char":
            tokenized = self.SPACE.join(list(
                raw_input.replace(self.SPACE, self.SPACE_ESCAPE)))
        else:
            tokenized = self.tokenizer(raw_input)

        tokenized = self.pre_process(tokenized) # lowercase etc.
        return tokenized.split()

    def __repr__(self):
        return "%s(level=%r, lowercase=%r, remove_punctuation=%r, " \
               "tokenizer=%r)" % (self.__class__.__name__, self.level,
                                  self.lowercase, self.remove_punctuation,
                                  self.tokenizer.__class__.__name__)


def build_tokenizer(task, data_cfg):
    tokenizer = {}
    if data_cfg["level"] == "bpe":
        if data_cfg.get("bpe_type", "sentencepiece") == "sentencepiece":
            if task == "MT":
                tokenizer['src'] = SentencePieceTokenizer(
                    level=data_cfg["level"], lowercase=data_cfg["lowercase"],
                    remove_punctuation=data_cfg.get("remove_punctuation", False),
                    **data_cfg["src_spm"])
            tokenizer['trg'] = SentencePieceTokenizer(
                level=data_cfg["level"], lowercase=data_cfg["lowercase"],
                remove_punctuation=data_cfg.get("remove_punctuation", False),
                **data_cfg["trg_spm"])
        else:
            raise ConfigurationError("We currently support sentencepiece bpe"
                                     " only.")
            # TODO: support subword-nmt

    elif data_cfg["level"] in ["word", "char"]:
        basic_tokenizer = BasicTokenizer(
            level=data_cfg["level"], lowercase=data_cfg["lowercase"],
            remove_punctuation=data_cfg.get("remove_punctuation", False))
        tokenizer = {'src': basic_tokenizer, 'trg': basic_tokenizer}
    else:
        raise ConfigurationError("Invalid tokenization level. Valid options:"
                                 " 'bpe', 'word', 'char'.")
        # TODO: support different src tokenization from trg
    if "src" in tokenizer:
        logger.info(f'Src tokenizer: {tokenizer["src"]}')
    logger.info(f'Trg tokenizer: {tokenizer["trg"]}')
    return tokenizer

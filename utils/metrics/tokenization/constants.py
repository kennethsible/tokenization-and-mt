from enum import StrEnum
from pathlib import Path
from typing import Any, Callable, TypeAlias, Union

from transformers import PreTrainedTokenizer
from tensor2tensor.data_generators.text_encoder import SubwordTextEncoder

from utils.data.corpora import BaseCorpusDataset

DerivationMap: TypeAlias = dict[str, list[tuple[str, str]]]
InflectionMap: TypeAlias = dict[str, list[tuple[str, list[str]]]]
Paradigm: TypeAlias = dict[str, int]
ParadigmConstructor: TypeAlias = Callable[[DerivationMap, InflectionMap], list[Paradigm]]
SubwordTokenizer: TypeAlias = Union[PreTrainedTokenizer, SubwordTextEncoder]

CorpusMetric: TypeAlias = Callable[
    [SubwordTokenizer, BaseCorpusDataset, dict[str, Any], ...], float
]
ParadigmMetric: TypeAlias = Callable[[SubwordTokenizer, list[Paradigm], dict[str, Any]], float]


class MorphologyDataSource(StrEnum):
    UNIMORPH: str = "unimorph"
    WORD_FORMATION_LEXICON: str = "wfl"


class NamedLanguageModel(StrEnum):
    CANINE_C: str = "canine-c"
    CANINE_S: str = "canine-s"
    LATIN_BERT: str = "latin-bert"
    LABERTA: str = "laberta"
    MULTILINGUAL_BERT: str = "mbert"
    PHILBERTA: str = "philberta"
    SPHILBERTA: str = "sphilberta"


class NamedCorpusTokenizationMetric(StrEnum):
    AVERAGE_TOKENS_PER_SENTENCE: str = "average-tps"
    FERTILITY: str = "fertility"


class NamedMorphologyTokenizationMetric(StrEnum):
    PARADIGM_COHERENCE: str = "paradigm-coherence"


class TokenizationLanguage(StrEnum):
    LATIN: str = "latin"


MODELS_BY_LANGUAGE: dict[str, set[str]] = {
    TokenizationLanguage.LATIN: {
        NamedLanguageModel.CANINE_C,
        NamedLanguageModel.CANINE_S,
        NamedLanguageModel.LABERTA,
        NamedLanguageModel.LATIN_BERT,
        NamedLanguageModel.MULTILINGUAL_BERT,
        NamedLanguageModel.PHILBERTA,
        NamedLanguageModel.SPHILBERTA,
    }
}

DEFAULT_TOKENIZER_FILEPATHS: dict[NamedLanguageModel, Path] = {
    NamedLanguageModel.CANINE_C: Path("resources/multi/canine-c"),
    NamedLanguageModel.CANINE_S: Path("resources/multi/canine-s"),
    NamedLanguageModel.LATIN_BERT: Path(
        "resources/lat/latin-bert/subword_tokenizer_latin/latin.subword.encoder"
    ),
    NamedLanguageModel.MULTILINGUAL_BERT: Path("resources/multi/mbert"),
    NamedLanguageModel.LABERTA: Path("resources/lat/laberta"),
    NamedLanguageModel.PHILBERTA: Path("resources/multi/philberta"),
    NamedLanguageModel.SPHILBERTA: Path("resources/multi/sphilberta"),
}

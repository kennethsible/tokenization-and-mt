from enum import StrEnum
from pathlib import Path
from typing import Any, Callable, TypeAlias, Union

from transformers import PreTrainedTokenizer
from tensor2tensor.data_generators.text_encoder import SubwordTextEncoder

DerivationMap: TypeAlias = dict[str, list[tuple[str, str]]]
InflectionMap: TypeAlias = dict[str, list[tuple[str, list[str]]]]
Paradigm: TypeAlias = dict[str, int]
ParadigmConstructor: TypeAlias = Callable[[DerivationMap, InflectionMap], list[Paradigm]]
SubwordTokenizer: TypeAlias = Union[PreTrainedTokenizer, SubwordTextEncoder]
ParadigmMetric: TypeAlias = Callable[[SubwordTokenizer, list[Paradigm], dict[str, Any]], float]


class NamedLanguageModel(StrEnum):
    CANINE_C: str = "canine-c"
    CANINE_S: str = "canine-s"
    LATIN_BERT: str = "latin-bert"
    LABERTA: str = "laberta"
    MULTILINGUAL_BERT: str = "mbert"
    PHILBERTA: str = "philberta"
    SPHILBERTA: str = "sphilberta"


class NamedTokenizationMetric(StrEnum):
    PARADIGM_COHERENCE: str = "paradigm-coherence"


class TokenizationDataSource(StrEnum):
    UNIMORPH: str = "unimorph"
    WORD_FORMATION_LEXICON: str = "wfl"


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
    NamedLanguageModel.CANINE_C: Path("resources/canine-c"),
    NamedLanguageModel.CANINE_S: Path("resources/canine-s"),
    NamedLanguageModel.LATIN_BERT: Path(
        "resources/latin-bert/subword_tokenizer_latin/latin.subword.encoder"
    ),
    NamedLanguageModel.MULTILINGUAL_BERT: Path("resources/mbert"),
    NamedLanguageModel.LABERTA: Path("resources/laberta"),
    NamedLanguageModel.PHILBERTA: Path("resources/philberta"),
    NamedLanguageModel.SPHILBERTA: Path("resources/sphilberta"),
}

from enum import StrEnum
from pathlib import Path
from typing import Any, Callable, NamedTuple, Optional, TypeAlias, Union

from transformers import PreTrainedTokenizer
from tensor2tensor.data_generators.text_encoder import SubwordTextEncoder

from utils.data.corpora import BaseCorpusDataset

DerivationMap: TypeAlias = dict[str, list[tuple[str, str]]]
InflectionMap: TypeAlias = dict[str, list[tuple[str, list[str]]]]


class MorphemeTable(NamedTuple):
    derivations: Optional[int] = None
    inflections: Optional[int] = None
    stem: Optional[int] = None

    def count_morphemes(self):
        morpheme_sources: list[int] = [self.derivations, self.inflections, self.stem]
        if any([source is None for source in morpheme_sources]):
            raise ValueError("Counting is not defined for an incomplete morpheme table.")

        return sum(morpheme_sources)


Paradigm: TypeAlias = dict[str, MorphemeTable]
ParadigmConstructor: TypeAlias = Callable[
    [InflectionMap, ..., Optional[DerivationMap]], list[Paradigm]
]
SubwordTokenizer: TypeAlias = Union[PreTrainedTokenizer, SubwordTextEncoder]

CorpusMetric: TypeAlias = Callable[
    [SubwordTokenizer, BaseCorpusDataset, dict[str, Any], ...], float
]
ParadigmMetric: TypeAlias = Callable[[SubwordTokenizer, list[Paradigm], dict[str, Any]], float]


class MorphologyDataSource(StrEnum):
    UNIMORPH: str = "unimorph"
    UNIMORPH_CORRECTED: str = "unimorph-corrected"
    WORD_FORMATION_LEXICON: str = "wfl"


class NamedLanguageModel(StrEnum):
    CANINE_C: str = "canine-c"
    CANINE_S: str = "canine-s"
    ICEBERT: str = "icebert"
    IS_ROBERTA: str = "is-roberta"
    LATIN_BERT: str = "latin-bert"
    LABERTA: str = "laberta"
    MULTILINGUAL_BERT: str = "mbert"
    PHILBERTA: str = "philberta"
    SPHILBERTA: str = "sphilberta"
    XLM_ROBERTA: str = "xlm-roberta"


class NamedCorpusTokenizationMetric(StrEnum):
    AVERAGE_TOKENS_PER_SENTENCE: str = "average-tps"
    FERTILITY: str = "fertility"


class NamedMorphologyTokenizationMetric(StrEnum):
    DERIVATIONALLY_AWARE_PARADIGM_COHERENCE: str = "da-paradigm-coherence"
    PARADIGM_ADHERENCE: str = "paradigm-adherence"
    PARADIGM_COHERENCE: str = "paradigm-coherence"


class TokenizationLanguage(StrEnum):
    ICELANDIC: str = "icelandic"
    LATIN: str = "latin"


MODELS_BY_LANGUAGE: dict[str, set[str]] = {
    TokenizationLanguage.ICELANDIC: {
        NamedLanguageModel.ICEBERT,
        NamedLanguageModel.IS_ROBERTA,
        NamedLanguageModel.MULTILINGUAL_BERT,
        NamedLanguageModel.XLM_ROBERTA,
    },
    TokenizationLanguage.LATIN: {
        NamedLanguageModel.CANINE_C,
        NamedLanguageModel.CANINE_S,
        NamedLanguageModel.LABERTA,
        NamedLanguageModel.LATIN_BERT,
        NamedLanguageModel.MULTILINGUAL_BERT,
        NamedLanguageModel.PHILBERTA,
        NamedLanguageModel.SPHILBERTA,
        NamedLanguageModel.XLM_ROBERTA,
    },
}

DEFAULT_TOKENIZER_FILEPATHS: dict[NamedLanguageModel, Path] = {
    NamedLanguageModel.CANINE_C: Path("resources/multi/canine-c"),
    NamedLanguageModel.CANINE_S: Path("resources/multi/canine-s"),
    NamedLanguageModel.ICEBERT: Path("resources/isl/icebert"),
    NamedLanguageModel.IS_ROBERTA: Path("resources/isl/is-roberta"),
    NamedLanguageModel.LATIN_BERT: Path(
        "resources/lat/latin-bert/subword_tokenizer_latin/latin.subword.encoder"
    ),
    NamedLanguageModel.MULTILINGUAL_BERT: Path("resources/multi/mbert"),
    NamedLanguageModel.LABERTA: Path("resources/lat/laberta"),
    NamedLanguageModel.PHILBERTA: Path("resources/multi/philberta"),
    NamedLanguageModel.SPHILBERTA: Path("resources/multi/sphilberta"),
    NamedLanguageModel.XLM_ROBERTA: Path("resources/multi/xlm-roberta-base"),
}

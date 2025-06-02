from pathlib import Path
from typing import Any, Callable, NamedTuple, Optional, TypeAlias, Union

from transformers import PreTrainedTokenizer
from tensor2tensor.data_generators.text_encoder import SubwordTextEncoder

from utils.data.corpora import BaseCorpusDataset
from utils.data.unimorph import CategoryMap


class MorphemeTable(NamedTuple):
    derivations: Optional[int] = None
    inflections: Optional[int] = None
    stem: Optional[int] = None

    def count_morphemes(self):
        morpheme_sources: list[int] = [self.derivations, self.inflections, self.stem]
        if any([source is None for source in morpheme_sources]):
            raise ValueError("Counting is not defined for an incomplete morpheme table.")

        return sum(morpheme_sources)


DerivationMap: TypeAlias = dict[str, list[tuple[str, str]]]
InflectionMap: TypeAlias = dict[str, list[tuple[str, Optional[list[str]]]]]
Paradigm: TypeAlias = dict[tuple[str, Any], MorphemeTable]
ParadigmConstructor: TypeAlias = Callable[
    [InflectionMap, ..., Optional[DerivationMap]], list[Paradigm]
]
FeaturedWordlist: TypeAlias = list[tuple[str, dict[str, str]]]

SubwordTokenizer: TypeAlias = Union[PreTrainedTokenizer, SubwordTextEncoder]
TokenizerKwargs: TypeAlias = dict[str, bool]

AggregateParadigmMetric: TypeAlias = Callable[
    [SubwordTokenizer, list[Paradigm], TokenizerKwargs, ...], float
]
CorpusMetric: TypeAlias = Callable[
    [SubwordTokenizer, BaseCorpusDataset, TokenizerKwargs, ...], float
]
IndividualParadigmMetric: TypeAlias = Callable[
    [SubwordTokenizer, Paradigm, TokenizerKwargs], tuple[..., float, dict[str, Any]]
]
IndividualParadigmWriter: TypeAlias = Callable[
    [Path, SubwordTokenizer, list[Paradigm], list[TokenizerKwargs]], None
]

CoherenceFunction: TypeAlias = Callable[
    [SubwordTokenizer, Paradigm, TokenizerKwargs], tuple[int, float, dict[str, Any]]
]

AggregateFeatureMetric: TypeAlias = Callable[
    [SubwordTokenizer, CategoryMap, FeaturedWordlist, TokenizerKwargs], tuple[float, dict[str, Any]]
]

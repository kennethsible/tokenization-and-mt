from pathlib import Path
from typing import Any, Callable, NamedTuple, Optional, TypeAlias, Union

from transformers import PreTrainedTokenizer
from tensor2tensor.data_generators.text_encoder import SubwordTextEncoder

from utils.data.corpora import BaseCorpusDataset


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
SubwordTokenizer: TypeAlias = Union[PreTrainedTokenizer, SubwordTextEncoder]

AggregateParadigmMetric: TypeAlias = Callable[
    [SubwordTokenizer, list[Paradigm], dict[str, Any]], float
]
CorpusMetric: TypeAlias = Callable[
    [SubwordTokenizer, BaseCorpusDataset, dict[str, Any], ...], float
]
IndividualParadigmMetric: TypeAlias = Callable[
    [SubwordTokenizer, Paradigm, dict[str, Any]], tuple[..., float, dict[str, Any]]
]
IndividualParadigmWriter: TypeAlias = Callable[
    [Path, SubwordTokenizer, list[Paradigm], list[dict[str, Any]]], None
]

CoherenceFunction: TypeAlias = Callable[
    [SubwordTokenizer, Paradigm, dict[str, Any]], tuple[int, float, dict[str, Any]]
]

from pathlib import Path
from typing import Callable, Optional

from .constants import (
    MorphologyDataSource,
    NamedCorpusTokenizationMetric,
    NamedFeatureTokenizationMetric,
    NamedLanguageModel,
    NamedParadigmTokenizationMetric,
    TokenizationLanguage,
)
from .constructors import (
    construct_paradigms,
    load_unimorph_latin_derivations,
    load_unimorph_latin_inflections,
    load_wfl_derivations,
)
from .helpers import collect_tokenizer_filepaths, get_tokenizers, retrieve_default_filepath
from .metrics import (
    compute_morphological_rajski_distance,
    compute_paradigm_adherence,
    compute_paradigm_coherence,
    compute_average_fertility,
    compute_average_tps,
)
from .tables import (
    AGGREGATE_FEATURE_METRIC_MAPPING,
    AGGREGATE_PARADIGM_METRIC_MAPPING,
    CORPUS_METRIC_MAPPING,
    DEFAULT_DERIVATION_FILEPATHS,
    DEFAULT_DERIVATION_FUNCTIONS,
    DEFAULT_INFLECTION_FILEPATHS,
    DEFAULT_INFLECTION_FUNCTIONS,
    DEFAULT_TOKENIZER_FILEPATHS,
    INDIVIDUAL_PARADIGM_METRIC_MAPPING,
    INDIVIDUAL_PARADIGM_WRITER_MAPPING,
    MODELS_BY_LANGUAGE,
)
from .types import (
    AggregateFeatureMetric,
    AggregateParadigmMetric,
    CorpusMetric,
    DerivationMap,
    FeaturedWordlist,
    IndividualParadigmMetric,
    IndividualParadigmWriter,
    InflectionMap,
    Paradigm,
    ParadigmConstructor,
    SubwordTokenizer,
)


def derive_paradigms(
    language: TokenizationLanguage,
    inflection_source: MorphologyDataSource,
    derivation_source: Optional[MorphologyDataSource] = None,
) -> list[Paradigm]:
    paradigm_builder: ParadigmConstructor = construct_paradigms

    # Derivations are not required for all morphological metrics.
    if any(language in key for key in DEFAULT_DERIVATION_FUNCTIONS.keys()):
        try:
            derivation_function: Callable = DEFAULT_DERIVATION_FUNCTIONS[
                (language, derivation_source)
            ]
            derivation_location: Path = DEFAULT_DERIVATION_FILEPATHS[(language, derivation_source)]
        except KeyError:
            raise ValueError(
                f"The derivation source <{derivation_source}> is not known for <{language}>."
            )

        derivations: Optional[DerivationMap] = derivation_function(derivation_location)
    else:
        derivations = None

    # Inflections (or, more specifically, lemma-inflection mappings) are required for all morphological metrics.
    try:
        inflection_function: Callable = DEFAULT_INFLECTION_FUNCTIONS[(language, inflection_source)]
        inflection_location: Path = DEFAULT_INFLECTION_FILEPATHS[(language, inflection_source)]
    except KeyError:
        raise ValueError(
            f"The inflection source <{derivation_source}> is not known for <{language}>."
        )

    inflections: InflectionMap = inflection_function(inflection_location)

    paradigms: list[Paradigm] = paradigm_builder(inflections, language, derivations)
    return paradigms


def get_tokenization_corpus_metric(metric: str):
    try:
        metric_function: CorpusMetric = CORPUS_METRIC_MAPPING[metric]
    except KeyError:
        raise ValueError(f"The metric <{metric}> is not currently supported.")

    return metric_function


def get_tokenization_aggregate_paradigm_metric(metric: str):
    try:
        metric_function: AggregateParadigmMetric = AGGREGATE_PARADIGM_METRIC_MAPPING[metric]
    except KeyError:
        raise ValueError(f"The metric <{metric}> is not currently supported.")

    return metric_function


def get_tokenization_aggregate_feature_metric(metric: str):
    try:
        metric_function: AggregateFeatureMetric = AGGREGATE_FEATURE_METRIC_MAPPING[metric]
    except KeyError:
        raise ValueError(f"The metric <{metric}> is not currently supported.")

    return metric_function


def get_tokenization_individual_paradigm_metric(metric: str):
    try:
        metric_function: IndividualParadigmMetric = INDIVIDUAL_PARADIGM_METRIC_MAPPING[metric]
    except KeyError:
        raise ValueError(f"The metric <{metric}> is not currently supported.")

    return metric_function


def get_tokenization_paradigm_writer(metric: str) -> IndividualParadigmWriter:
    try:
        output_function: IndividualParadigmWriter = INDIVIDUAL_PARADIGM_WRITER_MAPPING[metric]
    except KeyError:
        raise ValueError(f"The metric <{metric}> is not currently supported.")

    return output_function


def get_tokenization_feature_metric(metric: str):
    try:
        metric_function: AggregateFeatureMetric = AGGREGATE_FEATURE_METRIC_MAPPING[metric]
    except KeyError:
        raise ValueError(f"The metric <{metric}> is not currently supported.")

    return metric_function

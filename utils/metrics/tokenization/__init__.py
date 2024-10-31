from .constants import (
    CorpusMetric,
    DerivationMap,
    InflectionMap,
    MorphologyDataSource,
    NamedCorpusTokenizationMetric,
    NamedLanguageModel,
    NamedMorphologyTokenizationMetric,
    Paradigm,
    ParadigmConstructor,
    ParadigmMetric,
    SubwordTokenizer,
    TokenizationLanguage,
    DEFAULT_TOKENIZER_FILEPATHS,
    MODELS_BY_LANGUAGE,
)
from .constructors import (
    construct_latin_paradigms,
    construct_paradigms,
    load_unimorph_inflections,
    load_unimorph_derivations,
    load_wfl_derivations,
    DEFAULT_DERIVATION_FILEPATHS,
    DEFAULT_INFLECTION_FILEPATHS,
)
from .helpers import collect_tokenizer_filepaths, get_tokenizers, retrieve_default_filepath
from .metrics import (
    compute_paradigm_adherence,
    compute_paradigm_coherence,
    compute_average_fertility,
    compute_average_tps,
    CORPUS_METRIC_MAPPING,
    MORPHOLOGY_METRIC_MAPPING,
)


def get_tokenization_corpus_metric(metric: str):
    try:
        metric_function: CorpusMetric = CORPUS_METRIC_MAPPING[metric]
    except KeyError:
        raise ValueError(f"The metric <{metric}> is not currently supported.")

    return metric_function


def get_tokenization_morphology_metric(metric: str):
    try:
        metric_function: ParadigmMetric = MORPHOLOGY_METRIC_MAPPING[metric]
    except KeyError:
        raise ValueError(f"The metric <{metric}> is not currently supported.")

    return metric_function

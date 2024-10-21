from .constants import (
    DerivationMap,
    InflectionMap,
    NamedLanguageModel,
    NamedTokenizationMetric,
    Paradigm,
    ParadigmConstructor,
    ParadigmMetric,
    SubwordTokenizer,
    TokenizationDataSource,
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
from .helpers import get_tokenizer, resolve_filepaths, retrieve_default_filepath
from .metrics import compute_paradigm_coherence, METRIC_MAPPING


def get_tokenization_metric(metric: str):
    try:
        metric_function: ParadigmMetric = METRIC_MAPPING[metric]
    except KeyError:
        raise ValueError(f"The metric <{metric}> is not currently supported.")

    return metric_function

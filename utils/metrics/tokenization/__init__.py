from .constants import NamedLanguageModel, NamedTokenizationMetric, Paradigm, ParadigmConstructor, \
    ParadigmMetric, SubwordTokenizer, TokenizationLanguage, DEFAULT_PARADIGM_FILEPATHS, DEFAULT_TOKENIZER_FILEPATHS, \
    MODELS_BY_LANGUAGE
from .constructors import construct_paradigms, construct_latin_unimorph_paradigms
from .helpers import get_tokenizer, resolve_filepaths, retrieve_default_filepath
from .metrics import compute_paradigm_coherence, METRIC_MAPPING


def get_tokenization_metric(metric: str):
    try:
        metric_function: ParadigmMetric = METRIC_MAPPING[metric]
    except KeyError:
        raise ValueError(f"The metric <{metric}> is not currently supported.")

    return metric_function

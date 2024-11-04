from pathlib import Path
from typing import Callable, Optional

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
    construct_paradigms,
    load_unimorph_latin_derivations,
    load_unimorph_latin_inflections,
    load_wfl_derivations,
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


DEFAULT_DERIVATION_FILEPATHS: dict[tuple[TokenizationLanguage, MorphologyDataSource], Path] = {
    (TokenizationLanguage.LATIN, MorphologyDataSource.UNIMORPH): Path(
        "data/unimorph/lat/lat.derivations"
    ),
    (TokenizationLanguage.LATIN, MorphologyDataSource.UNIMORPH_CORRECTED): Path(
        "data/unimorph/lat-corrected/lat.derivations"
    ),
    (TokenizationLanguage.LATIN, MorphologyDataSource.WORD_FORMATION_LEXICON): Path(
        "data/word-formation-lexicon/wfl_derivations.tsv"
    ),
}

DEFAULT_INFLECTION_FILEPATHS: dict[tuple[TokenizationLanguage, MorphologyDataSource], Path] = {
    (TokenizationLanguage.LATIN, MorphologyDataSource.UNIMORPH): Path(
        "data/unimorph/lat/lat.segmentations"
    ),
    (TokenizationLanguage.LATIN, MorphologyDataSource.UNIMORPH_CORRECTED): Path(
        "data/unimorph/lat-corrected/lat.segmentations"
    ),
}


DEFAULT_DERIVATION_FUNCTIONS: dict[tuple[TokenizationLanguage, MorphologyDataSource], Callable] = {
    (TokenizationLanguage.LATIN, MorphologyDataSource.UNIMORPH): load_unimorph_latin_derivations,
    (
        TokenizationLanguage.LATIN,
        MorphologyDataSource.UNIMORPH_CORRECTED,
    ): load_unimorph_latin_derivations,
    (TokenizationLanguage.LATIN, MorphologyDataSource.WORD_FORMATION_LEXICON): load_wfl_derivations,
}


DEFAULT_INFLECTION_FUNCTIONS: dict[tuple[TokenizationLanguage, MorphologyDataSource], Callable] = {
    (TokenizationLanguage.LATIN, MorphologyDataSource.UNIMORPH): load_unimorph_latin_inflections
}


def derive_paradigms(
    language: TokenizationLanguage,
    inflection_source: MorphologyDataSource,
    derivation_source: Optional[MorphologyDataSource] = None,
) -> list[Paradigm]:
    match language:
        case TokenizationLanguage.LATIN:
            paradigm_builder: ParadigmConstructor = construct_paradigms

            try:
                derivation_function: Callable = DEFAULT_DERIVATION_FUNCTIONS[
                    (language, derivation_source)
                ]
                derivation_location: Path = DEFAULT_DERIVATION_FILEPATHS[
                    (language, derivation_source)
                ]
            except KeyError:
                raise ValueError(
                    f"The derivation source <{derivation_source}> is not known for <{language}>."
                )

            derivations: DerivationMap = derivation_function(derivation_location)

            try:
                inflection_function: Callable = DEFAULT_INFLECTION_FUNCTIONS[
                    (language, inflection_source)
                ]
                inflection_location: Path = DEFAULT_INFLECTION_FILEPATHS[
                    (language, inflection_source)
                ]
            except KeyError:
                raise ValueError(
                    f"The inflection source <{derivation_source}> is not known for <{language}>."
                )

            inflections: InflectionMap = inflection_function(inflection_location)
        case _:
            raise ValueError(f"Language {language} not currently supported.")

    paradigms: list[Paradigm] = paradigm_builder(inflections, language, derivations)
    return paradigms


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

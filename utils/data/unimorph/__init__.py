from pathlib import Path

from cltk.alphabet.lat import remove_accents, remove_macrons

from .constants import GRC_ARTICLES, GRC_ARTICLE_REGEX, GRC_NORMALIZATION_MAP, UnimorphLanguage
from .corpus import UnimorphCorpus
from .dimension import UnimorphDimension
from .feature import *
from .mapping import compose_regex, DIMENSION_MAP
from .preprocessors import *
from .types import CategoryMap, StringPreprocessor


LANGUAGE_PATH_MAPPING: dict[UnimorphLanguage, Path] = {
    UnimorphLanguage.ANCIENT_GREEK: Path("data/unimorph/grc/grc"),
    UnimorphLanguage.ICELANDIC: Path("data/unimorph/isl/isl"),
    UnimorphLanguage.LATIN: Path("data/unimorph/lat/lat-filtered"),
    UnimorphLanguage.TURKMEN: Path("data/unimorph/tuk/tuk"),
}


LANGUAGE_PREPROCESSOR_MAPPING: dict[UnimorphLanguage, list[StringPreprocessor]] = {
    UnimorphLanguage.ANCIENT_GREEK: [clean_alternatives, clean_articles_grc, normalize_grc],
    UnimorphLanguage.ICELANDIC: [],  # TODO: are there any required preprocessors for Icelandic?
    UnimorphLanguage.LATIN: [remove_macrons, remove_accents],
    UnimorphLanguage.TURKMEN: [],  # TODO: are there any required preprocessors for Turkmen?
}


def get_language_path(language: UnimorphLanguage) -> Path:
    try:
        language_path: Path = LANGUAGE_PATH_MAPPING[language]
    except KeyError:
        raise ValueError(f"The language <{language}> was not found in the path mapping.")

    return language_path


def get_language_preprocessors(language: UnimorphLanguage) -> list[StringPreprocessor]:
    try:
        language_preprocessors: list[StringPreprocessor] = LANGUAGE_PREPROCESSOR_MAPPING[language]
    except KeyError:
        raise ValueError(f"The language <{language}> was not found in the preprocessor mapping.")

    return language_preprocessors

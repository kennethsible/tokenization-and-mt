from enum import StrEnum
from re import compile, Pattern
from typing import Sequence


class UnimorphOperation(StrEnum):
    CONJUNCTION: str = "conjunction"
    DISJUNCTION: str = "disjunction"
    NEGATION: str = "negation"


class UnimorphLanguage(StrEnum):
    ANCIENT_GREEK: str = "grc"
    ICELANDIC: str = "isl"
    LATIN: str = "lat"
    TURKMEN: str = "tur"


GRC_NORMALIZATION_MAP: dict[str, str] = {
    "ᾱ": "α",
    "ᾰ": "α",
    "Ᾱ": "Α",
    "Ᾰ": "Α",
    "ῑ": "ι",
    "ῐ": "ι",
    "Ῑ": "Ι",
    "Ῐ": "Ι",
    "ῡ": "υ",
    "ῠ": "υ",
    "Ῡ": "Υ",
    "Ῠ": "Υ",
    "(": "",
    ")": "",
    "̄": "",
    "̆": "",
}

GRC_ARTICLES: Sequence[str] = (
    "ὁ",
    "ἡ",
    "τὸ",
    "τοῦ",
    "τῆς",
    "τῷ",
    "τῇ",
    "τὸν",
    "τὴν",
    "οἱ",
    "αἱ",
    "τὰ",
    "τῶν",
    "τοῖς",
    "ταῖς",
    "τοὺς",
    "τὰς",
    "τὰ",
    "τοῖν",
    "τὼ",
)
GRC_ARTICLE_REGEX: Pattern = compile(f"((?:(?:{'|'.join(GRC_ARTICLES)})(?:,)? +)+)")

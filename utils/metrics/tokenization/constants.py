from enum import StrEnum


# Writing Format Constants:
DA_PARADIGM_COHERENCE_HEADER: str = "Derivationally-Aware Paradigm Coherence: Samples ({0})\n\n"
PARADIGM_ADHERENCE_HEADER: str = "Paradigm Adherence: Samples ({0})\n\n"
PARADIGM_COHERENCE_HEADER: str = "Paradigm Coherence: Samples ({0})\n\n"

PARADIGM_SUBHEADER: str = "Paradigm {0} (Score: {1}):"
PARADIGM_BULLET: str = "\n\t* {0}: ({1})"

PARADIGM_ADHERENCE_SUBBULLET: str = "\n\t\t- {0} (Expected) vs. {1} (Actual); Deviation: {2}"


GREEK_NORMALIZATION_MAP: dict[str, str] = {
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


class MorphologyDataSource(StrEnum):
    UNIMORPH: str = "unimorph"
    UNIMORPH_CORRECTED: str = "unimorph-corrected"
    WORD_FORMATION_LEXICON: str = "wfl"


class NamedLanguageModel(StrEnum):
    ANCIENT_GREEK_BERT: str = "ancient-greek-bert"
    CANINE_C: str = "canine-c"
    CANINE_S: str = "canine-s"
    GREBERTA: str = "greberta"
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
    ANCIENT_GREEK: str = "ancient-greek"
    ICELANDIC: str = "icelandic"
    LATIN: str = "latin"

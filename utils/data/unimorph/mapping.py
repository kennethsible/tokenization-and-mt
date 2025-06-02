from re import compile, Pattern

from .dimension import UnimorphDimension
from .feature import (
    Aktionsart,
    Animacy,
    Aspect,
    Case,
    Comparison,
    Definiteness,
    Deixis,
    Evidentiality,
    Finiteness,
    Gender,
    InformationStructure,
    Interrogativity,
    LanguageSpecificFeature,
    Mood,
    Number,
    PartOfSpeech,
    Person,
    Polarity,
    Politeness,
    Possession,
    SwitchReference,
    Tense,
    UnimorphFeature,
    Valency,
    Voice,
)

DIMENSION_MAP: dict[UnimorphDimension, UnimorphFeature] = {
    UnimorphDimension.AKTIONSART: Aktionsart,
    UnimorphDimension.ANIMACY: Animacy,
    UnimorphDimension.ASPECT: Aspect,
    UnimorphDimension.CASE: Case,
    UnimorphDimension.COMPARISON: Comparison,
    UnimorphDimension.DEFINITENESS: Definiteness,
    UnimorphDimension.DEIXIS: Deixis,
    UnimorphDimension.EVIDENTIALITY: Evidentiality,
    UnimorphDimension.FINITENESS: Finiteness,
    UnimorphDimension.GENDER: Gender,
    UnimorphDimension.INFORMATION_STRUCTURE: InformationStructure,
    UnimorphDimension.INTERROGATIVITY: Interrogativity,
    UnimorphDimension.LANGUAGE_SPECIFIC_FEATURE: LanguageSpecificFeature,
    UnimorphDimension.MOOD: Mood,
    UnimorphDimension.NUMBER: Number,
    UnimorphDimension.PART_OF_SPEECH: PartOfSpeech,
    UnimorphDimension.PERSON: Person,
    UnimorphDimension.POLARITY: Polarity,
    UnimorphDimension.POLITENESS: Politeness,
    UnimorphDimension.POSSESSION: Possession,
    UnimorphDimension.SWITCH_REFERENCE: SwitchReference,
    UnimorphDimension.TENSE: Tense,
    UnimorphDimension.VALENCY: Valency,
    UnimorphDimension.VOICE: Voice,
}


def compose_regex(dimensions: list[UnimorphDimension] = tuple(DIMENSION_MAP.keys())) -> Pattern:
    components: list[str] = [DIMENSION_MAP[dimension].to_regex() for dimension in dimensions]
    pattern: Pattern = compile("|".join(components))
    return pattern

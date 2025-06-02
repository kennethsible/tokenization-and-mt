from abc import abstractmethod
from enum import Enum
from re import findall, fullmatch, Match, Pattern
from typing import Iterator, Optional


# TODO: We currently do not yet model argument marking.


class UnimorphFeatureMixin:
    @classmethod
    @abstractmethod
    def names(cls) -> Iterator[str]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def features(cls) -> Iterator[str]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def labels(cls) -> Iterator[str]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_feature(cls, label: str) -> str:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_label(cls, feature: str) -> str:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def has_label(cls, label: str) -> bool:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def has_feature(cls, feature: str) -> bool:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def has_name(cls, name: str) -> bool:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def to_regex(cls) -> str:
        raise NotImplementedError

    @classmethod
    def derive_class_referent(cls):
        capital_matches: list[str] = findall("[A-Z]", cls.__name__)
        if len(capital_matches) > 1:
            class_referent: str = "".join([letter.lower() for letter in capital_matches])
        else:
            class_referent: str = cls.__name__.lower()

        return class_referent


class UnimorphFeature(UnimorphFeatureMixin, Enum):
    def __init__(self, feature: str, label: str):
        self.feature = feature
        self.label = label

    @classmethod
    def names(cls) -> Iterator[str]:
        return iter([name for name in cls.__members__.keys()])

    @classmethod
    def features(cls) -> Iterator[str]:
        return iter([pair.feature for pair in cls.__members__.values()])

    @classmethod
    def labels(cls) -> Iterator[str]:
        return iter([pair.label for pair in cls.__members__.values()])

    @classmethod
    def get_feature(cls, label: str) -> str:
        for pair in cls.__members__.values():
            if pair.label == label:
                feature: str = pair.feature
                break
        else:
            raise ValueError(f"Label <{label}> not known to {cls.__name__}")

        return feature

    @classmethod
    def get_label(cls, feature: str) -> str:
        for pair in cls.__members__.values():
            if pair.feature == feature:
                label: str = pair.label
                break
        else:
            raise ValueError(f"UnimorphFeature <{feature}> not known to {cls.__name__}")

        return label

    @classmethod
    def has_label(cls, label: str) -> bool:
        return label in cls.labels()

    @classmethod
    def has_feature(cls, feature: str) -> bool:
        return feature in cls.features()

    @classmethod
    def has_name(cls, name: str) -> bool:
        return name in cls.names()

    @classmethod
    def to_regex(cls) -> str:
        regex_group: str = f"(?P<{cls.derive_class_referent()}>{'|'.join(list(cls.labels()))})"
        return regex_group


class LanguageSpecificMixin:
    FEATURE_REGEX: Pattern = r"language-specific-feature-(?P<instance>\d+)"
    LABEL_REGEX: Pattern = r"LGSPEC(?P<instance>\d+)"

    FEATURE_TEMPLATE: str = "language-specific-feature-{0}"
    LABEL_TEMPLATE: str = "LGSPEC{0}"


class Aktionsart(UnimorphFeature):
    ACCOMPLISHMENT: tuple[str, str] = ("accomplishment", "ACCMP")
    ACHIEVEMENT: tuple[str, str] = ("achievement", "ACH")
    ACTIVITY: tuple[str, str] = ("activity", "ACTY")
    ATELIC: tuple[str, str] = ("atelic", "ATEL")
    DURATIVE: tuple[str, str] = ("durative", "DUR")
    DYNAMIC: tuple[str, str] = ("dynamic", "DYN")
    PUNCTUAL: tuple[str, str] = ("punctual", "PCT")
    SEMELFACTIVE: tuple[str, str] = ("semelfactive", "SEMEL")
    STATIVE: tuple[str, str] = ("stative", "STAT")
    TELIC: tuple[str, str] = ("telic", "TEL")


class Animacy(UnimorphFeature):
    ANIMATE: tuple[str, str] = ("animate", "ANIM")
    HUMAN: tuple[str, str] = ("human", "HUM")
    INANIMATE: tuple[str, str] = ("inanimate", "INAN")
    NONHUMAN: tuple[str, str] = ("nonhuman", "NHUM")


class Aspect(UnimorphFeature):
    HABITUAL: tuple[str, str] = ("habitual", "HAB")
    IMPERFECTIVE: tuple[str, str] = ("imperfective", "IPFV")
    ITERATIVE: tuple[str, str] = ("iterative", "ITER")
    PERFECTIVE: tuple[str, str] = ("perfective", "PFV")
    PERFECT: tuple[str, str] = ("perfect", "PRF")
    PROGRESSIVE: tuple[str, str] = ("progressive", "PROG")
    PROSPECTIVE: tuple[str, str] = ("prospective", "PROSP")


class Case(UnimorphFeature):
    ABLATIVE: tuple[str, str] = ("ablative", "ABL")
    ABSOLUTIVE: tuple[str, str] = ("absolutive", "ABS")
    ACCUSATIVE: tuple[str, str] = ("accusative", "ACC")
    ALLATIVE: tuple[str, str] = ("allative", "ALL")
    NEAR_IN_FRONT: tuple[str, str] = ("near", "ANTE")
    APPROXIMATIVE: tuple[str, str] = ("approximative", "APPRX")
    NEXT_TO: tuple[str, str] = ("next-to", "APUD")
    AT: tuple[str, str] = ("at", "AT")
    AVERSIVE: tuple[str, str] = ("aversive", "AVR")
    BENEFACTIVE: tuple[str, str] = ("benefactive", "BEN")
    ESSIVE_MODAL: tuple[str, str] = ("essive-modal", "BYWAY")
    NEAR: tuple[str, str] = ("near", "CIRC")
    COMITATIVE: tuple[str, str] = ("comitative", "COM")
    COMPARATIVE: tuple[str, str] = ("comparative", "COMPV")
    DATIVE: tuple[str, str] = ("dative", "DAT")
    EQUATIVE: tuple[str, str] = ("equative", "EQTV")
    ERGATIVE: tuple[str, str] = ("ergative", "ERG")
    ESSIVE: tuple[str, str] = ("essive", "ESS")
    FORMAL: tuple[str, str] = ("formal", "FRML")
    GENITIVE: tuple[str, str] = ("genitive", "GEN")
    IN: tuple[str, str] = ("in", "IN")
    INSTRUMENTAL: tuple[str, str] = ("instrumental", "INS")
    AMONG: tuple[str, str] = ("among", "INTER")
    NOMINATIVE: tuple[str, str] = ("nominative", "NOM")
    NOMINATIVE_S_ONLY: tuple[str, str] = ("nominative-s-only", "NOMS")
    ON: tuple[str, str] = ("on", "ON")
    ON_HORIZONTAL: tuple[str, str] = ("on-horizontal", "ONHR")
    ON_VERTICAL: tuple[str, str] = ("on-vertical", "ONVR")
    BEHIND: tuple[str, str] = ("behind", "POST")
    PRIVATIVE: tuple[str, str] = ("privative", "PRIV")
    PROLATIVE_TRANSLATIVE: tuple[str, str] = ("prolative-translative", "PROL")
    PROPRIETIVE: tuple[str, str] = ("proprietive", "PROPR")
    PROXIMATE: tuple[str, str] = ("proximate", "PROX")
    PURPOSIVE: tuple[str, str] = ("purposive", "PRP")
    PARTITIVE: tuple[str, str] = ("partitive", "PRT")
    RELATIVE: tuple[str, str] = ("relative", "REL")
    DISTAL: tuple[str, str] = ("distal", "REM")
    UNDER: tuple[str, str] = ("under", "SUB")
    TERMINATIVE: tuple[str, str] = ("terminative", "TERM")
    TRANSLATIVE: tuple[str, str] = ("translative", "TRANS")
    VERSATIVE: tuple[str, str] = ("versative", "VERS")
    VOCATIVE: tuple[str, str] = ("vocative", "VOC")

    # NOT ACCORDING TO SYLAK-GLASSMAN 2016 STANDARD:
    LOCATIVE: tuple[str, str] = ("locative", "LOC")


class Comparison(UnimorphFeature):
    ABSOLUTE: tuple[str, str] = ("absolute", "AB")
    COMPARATIVE: tuple[str, str] = ("comparative", "CMPR")
    EQUATIVE: tuple[str, str] = ("equative", "EQT")
    RELATIVE: tuple[str, str] = ("relative", "RL")
    SUPERLATIVE: tuple[str, str] = ("superlative", "SPRL")


class Definiteness(UnimorphFeature):
    DEFINITE: tuple[str, str] = ("definite", "DEF")
    INDEFINITE: tuple[str, str] = ("indefinite", "INDF")
    NON_SPECIFIC: tuple[str, str] = ("nonspecific", "NSPEC")
    SPECIFIC: tuple[str, str] = ("specific", "SPEC")


class Deixis(UnimorphFeature):
    ABOVE: tuple[str, str] = ("above", "ABV")
    BELOW: tuple[str, str] = ("below", "BEL")
    EVEN: tuple[str, str] = ("even", "EVEN")
    MEDIAL: tuple[str, str] = ("medial", "MED")
    NO_REFERENCE_POINT: tuple[str, str] = ("no-reference-point", "NOREF")
    INVISIBLE: tuple[str, str] = ("invisible", "NVIS")
    PHORIC: tuple[str, str] = ("phoric", "PHOR")
    PROXIMATE: tuple[str, str] = ("proximate", "PROX")
    FIRST_PERSON_REFERENCE_POINT: tuple[str, str] = ("first-person-reference-point", "REF1")
    SECOND_PERSON_REFERENCE_POINT: tuple[str, str] = ("second-person-reference-point", "REF2")
    REMOTE: tuple[str, str] = ("remote", "REMT")
    VISIBLE: tuple[str, str] = ("visible", "VIS")


class Evidentiality(UnimorphFeature):
    ASSUMED: tuple[str, str] = ("assumed", "ASSUM")
    AUDITORY: tuple[str, str] = ("auditory", "AUD")
    DIRECT: tuple[str, str] = ("direct", "DRCT")
    FIRSTHAND: tuple[str, str] = ("firsthand", "FH")
    HEARSAY: tuple[str, str] = ("hearsay", "HRSY")
    INFERRED: tuple[str, str] = ("inferred", "INFER")
    NON_FIRSTHAND: tuple[str, str] = ("non-firsthand", "NFH")
    NON_VISUAL_SENSORY: tuple[str, str] = ("non-visual-sensory", "NVSEN")
    QUOTATIVE: tuple[str, str] = ("quotative", "QUOT")
    REPORTED: tuple[str, str] = ("reported", "RPRT")
    SENSORY: tuple[str, str] = ("sensory", "SEN")


class Finiteness(UnimorphFeature):
    FINITE: tuple[str, str] = ("finite", "FIN")
    NONFINITE: tuple[str, str] = ("nonfinite", "NFIN")


class Gender(UnimorphFeature):
    BANTU_NOUN_CLASSES: tuple[str, str] = ("bantu-noun-classes", "BANTU1-23")
    FEMININE: tuple[str, str] = ("feminine", "FEM")
    MASCULINE: tuple[str, str] = ("masculine", "MASC")
    NAKH_DAGHESTANIAN_NOUN_CLASSES: tuple[str, str] = ("nakh-daghestanian-noun-classes", "NAKH1-8")
    NEUTER: tuple[str, str] = ("neuter", "NEUT")


class InformationStructure(UnimorphFeature):
    FOCUS: tuple[str, str] = ("focus", "FOC")
    TOPIC: tuple[str, str] = ("topic", "TOP")


class Interrogativity(UnimorphFeature):
    DECLARATIVE: tuple[str, str] = ("declarative", "DECL")
    INTERROGATIVE: tuple[str, str] = ("interrogative", "INT")


class LanguageSpecificFeature(LanguageSpecificMixin, UnimorphFeature):
    @classmethod
    def names(cls):
        raise TypeError(f"Names not defined for {cls.__name__}")

    @classmethod
    def features(cls):
        raise TypeError(f"Fixed iterables not defined for {cls.__name__}")

    @classmethod
    def labels(cls):
        raise TypeError(f"Fixed iterables not defined for {cls.__name__}")

    @classmethod
    def has_label(cls, label: str):
        match: Optional[Match] = fullmatch(cls.LABEL_REGEX, label)
        return match is not None

    @classmethod
    def has_feature(cls, feature: str):
        match: Optional[Match] = fullmatch(cls.FEATURE_REGEX, feature)
        return match is not None

    @classmethod
    def has_name(cls, name: str):
        raise TypeError(f"Names not defined for {cls.__name__}")

    @classmethod
    def get_feature(cls, label: str):
        match: Optional[Match] = fullmatch(cls.LABEL_REGEX, label)
        if match is None:
            raise ValueError(f"Label <{label}> not known to {cls.__name__}")
        else:
            feature: str = cls.FEATURE_TEMPLATE.format(match.group("instance"))

        return feature

    @classmethod
    def get_label(cls, feature: str):
        match: Optional[Match] = fullmatch(cls.FEATURE_REGEX, feature)
        if match is None:
            raise ValueError(f"UnimorphFeature <{feature}> not known to {cls.__name__}")
        else:
            label: str = cls.LABEL_TEMPLATE.format(match.group("instance"))

        return label

    @classmethod
    def get_names(cls, name: str):
        raise TypeError(f"Names not defined for {cls.__name__}")

    @classmethod
    def to_regex(cls):
        regex_group: str = f"(?P<{cls.derive_class_referent()}>{cls.LABEL_REGEX})"
        return regex_group


class Mood(UnimorphFeature):
    ADMIRATIVE: tuple[str, str] = ("admirative", "ADM")
    AUSTRALIAN_NON_PURPOSIVE: tuple[str, str] = ("australian-non-purposive", "AUNPRP")
    AUSTRALIAN_PURPOSIVE: tuple[str, str] = ("australian-purposive", "AUPRP")
    CONDITIONAL: tuple[str, str] = ("conditional", "COND")
    DEBITIVE: tuple[str, str] = ("debitive", "DEB")
    DEDUCTIVE: tuple[str, str] = ("deductive", "DED")
    IMPERATIVE_JUSSIVE: tuple[str, str] = ("imperative-jussive", "IMP")
    INDICATIVE: tuple[str, str] = ("indicative", "IND")
    INTENTIVE: tuple[str, str] = ("intentive", "INTEN")
    IRREALIS: tuple[str, str] = ("irrealis", "IRR")
    LIKELY: tuple[str, str] = ("likely", "LKLY")
    OBLIGATIVE: tuple[str, str] = ("obligative", "OBLIG")
    OPTATIVE_DESIDERATIVE: tuple[str, str] = ("optative-desiderative", "OPT")
    PERMISSIVE: tuple[str, str] = ("permissive", "PERM")
    POTENTIAL: tuple[str, str] = ("potential", "POT")
    GENERAL_PURPOSIVE: tuple[str, str] = ("general-purposive", "PURP")
    REALIS: tuple[str, str] = ("realis", "REAL")
    SUBJUNCTIVE: tuple[str, str] = ("subjunctive", "SBJV")
    SIMULATIVE: tuple[str, str] = ("simulative", "SIM")


class Number(UnimorphFeature):
    DUAL: tuple[str, str] = ("dual", "DU")
    GREATER_PAUCAL: tuple[str, str] = ("greater-paucal", "GPAUC")
    GREATER_PLURAL: tuple[str, str] = ("greater-plural", "GRPL")
    INVERSE: tuple[str, str] = ("inverse", "INVN")
    PAUCAL: tuple[str, str] = ("paucal", "PAUC")
    PLURAL: tuple[str, str] = ("plural", "PL")
    SINGULAR: tuple[str, str] = ("singular", "SG")
    TRIAL: tuple[str, str] = ("trial", "TRI")


class PartOfSpeech(UnimorphFeature):
    ADJECTIVE: tuple[str, str] = ("adjective", "ADJ")
    ADPOSITION: tuple[str, str] = ("adposition", "ADP")
    ADVERB: tuple[str, str] = ("adverb", "ADV")
    ARTICLE: tuple[str, str] = ("article", "ART")
    AUXILIARY: tuple[str, str] = ("auxiliary", "AUX")
    CLASSIFIER: tuple[str, str] = ("classifier", "CLF")
    COMPLEMENTIZER: tuple[str, str] = ("complementizer", "COMP")
    CONJUNCTION: tuple[str, str] = ("conjunction", "CONJ")
    DETERMINER: tuple[str, str] = ("determiner", "DET")
    INTERJECTION: tuple[str, str] = ("interjection", "INTJ")
    NOUN: tuple[str, str] = ("noun", "N")
    NUMERAL: tuple[str, str] = ("numeral", "NUM")
    PARTICLE: tuple[str, str] = ("particle", "PART")
    PRONOUN: tuple[str, str] = ("pronoun", "PRO")
    PROPER_NAME: tuple[str, str] = ("proper-name", "PROPN")
    VERB: tuple[str, str] = ("verb", "V")
    CONVERB: tuple[str, str] = ("converb", "V.CVB")
    MASDAR: tuple[str, str] = ("masdar", "V.MSDR")
    PARTICIPLE: tuple[str, str] = ("participle", "V.PTCP")


class Person(UnimorphFeature):
    ZERO_PERSON: tuple[str, str] = ("zero-person", "0")
    FIRST_PERSON: tuple[str, str] = ("first-person", "1")
    SECOND_PERSON: tuple[str, str] = ("second-person", "2")
    THIRD_PERSON: tuple[str, str] = ("third-person", "3")
    FOURTH_PERSON: tuple[str, str] = ("fourth-person", "4")
    EXCLUSIVE: tuple[str, str] = ("exclusive", "EXCL")
    INCLUSIVE: tuple[str, str] = ("inclusive", "INCL")
    OBVIATIVE: tuple[str, str] = ("obviative", "OBV")
    PROXIMATE: tuple[str, str] = ("proximate", "PRX")


class Polarity(UnimorphFeature):
    POSITIVE: tuple[str, str] = ("positive", "POS")
    NEGATIVE: tuple[str, str] = ("negative", "NEG")


class Politeness(UnimorphFeature):
    AVOIDANCE_STYLE: tuple[str, str] = ("avoidance-style", "AVOID")
    COLLOQUIAL: tuple[str, str] = ("colloquial", "COL")
    FORMAL_REFERENT_ELEVATING: tuple[str, str] = ("formal-register-elevating", "ELEV")
    FORMAL_REGISTER: tuple[str, str] = ("formal-register", "FOREG")
    FORMAL: tuple[str, str] = ("formal", "FORM")
    HIGH_STATUS: tuple[str, str] = ("high-status", "HIGH")
    FORMAL_SPEAKER_HUMBLING: tuple[str, str] = ("formal-speaker-humbling", "HUMB")
    INFORMAL: tuple[str, str] = ("informal", "INFM")
    LITERARY: tuple[str, str] = ("literary", "LIT")
    LOW_STATUS: tuple[str, str] = ("low-status", "LOW")
    POLITE: tuple[str, str] = ("polite", "POL")
    HIGH_STATUS_ELEVATED: tuple[str, str] = ("high-status-elevated", "STELEV")
    HIGH_STATUS_SUPREME: tuple[str, str] = ("high-status-supreme", "STSUPR")


class Possession(UnimorphFeature):
    ALIENABLE: tuple[str, str] = ("alienable-possession", "ALN")
    INALIENABLE: tuple[str, str] = ("inalienable-possession", "NALN")
    POSSESSION_BY_FIRST_DUAL: tuple[str, str] = ("possession-by-1.DU", "PSS1D")
    POSSESSION_BY_FIRST_DUAL_EXCLUSIVE: tuple[str, str] = ("possession-by-1.DU.EXCL", "PSS1DE")
    POSSESSION_BY_FIRST_DUAL_INCLUSIVE: tuple[str, str] = ("possession-by-1.DU.INCL", "PSS1DI")
    POSSESSION_BY_FIRST_PLURAL: tuple[str, str] = ("possession-by-1.PL", "PSS1P")
    POSSESSION_BY_FIRST_PLURAL_EXCLUSIVE: tuple[str, str] = ("possession-by-1.PL.EXCL", "PSS1PE")
    POSSESSION_BY_FIRST_PLURAL_INCLUSIVE: tuple[str, str] = ("possession-by-1.PL.INCL", "PSS1PI")
    POSSESSION_BY_FIRST_SINGULAR: tuple[str, str] = ("possession-by-1.SG", "PSS1S")
    POSSESSION_BY_SECOND_DUAL: tuple[str, str] = ("possession-by-2.DU", "PSS2D")
    POSSESSION_BY_SECOND_DUAL_FEMININE: tuple[str, str] = ("possession-by-2.DU.FEM", "PSS2DF")
    POSSESSION_BY_SECOND_DUAL_MASCULINE: tuple[str, str] = ("possession-by-2.DU.MASC", "PSS2DM")
    POSSESSION_BY_SECOND_PLURAL: tuple[str, str] = ("possession-by-2.PL", "PSS2P")
    POSSESSION_BY_SECOND_PLURAL_FEMININE: tuple[str, str] = ("possession-by-2.PL.FEM", "PSS2PF")
    POSSESSION_BY_SECOND_PLURAL_MASCULINE: tuple[str, str] = ("possession-by-2.PL.MASC", "PSS2PM")
    POSSESSION_BY_SECOND_SINGULAR: tuple[str, str] = ("possession-by-2.SG", "PSS2S")
    POSSESSION_BY_SECOND_SINGULAR_FEMININE: tuple[str, str] = ("possession-by-2.SG.FEM", "PSS2SF")
    POSSESSION_BY_SECOND_SINGULAR_FORMAL: tuple[str, str] = ("possession-by-2.SG.FORM", "PSS2SFORM")
    POSSESSION_BY_SECOND_SINGULAR_INFORMAL: tuple[str, str] = (
        "possession-by-2.SG.INFM",
        "PSS2SINFM",
    )
    POSSESSION_BY_SECOND_SINGULAR_MASCULINE: tuple[str, str] = ("possession-by-2.SG.MASC", "PSS2SM")
    POSSESSION_BY_THIRD_DUAL: tuple[str, str] = ("possession-by-3.DU", "PSS3D")
    POSSESSION_BY_THIRD_DUAL_FEMININE: tuple[str, str] = ("possession-by-3.DU.FEM", "PSS3DF")
    POSSESSION_BY_THIRD_DUAL_MASCULINE: tuple[str, str] = ("possession-by-3.DU.MASC", "PSS3DM")
    POSSESSION_BY_THIRD_PLURAL: tuple[str, str] = ("possession-by-3.PL", "PSS3P")
    POSSESSION_BY_THIRD_PLURAL_FEMININE: tuple[str, str] = ("possession-by-3.PL.FEM", "PSS3PF")
    POSSESSION_BY_THIRD_PLURAL_MASCULINE: tuple[str, str] = ("possession-by-3.PL.MASC", "PSS3PM")
    POSSESSION_BY_THIRD_SINGULAR: tuple[str, str] = ("possession-by-3.SG", "PSS3S")
    POSSESSION_BY_THIRD_SINGULAR_FEMININE: tuple[str, str] = ("possession-by-3.SG.FEM", "PSS3SF")
    POSSESSION_BY_THIRD_SINGULAR_MASCULINE: tuple[str, str] = ("possession-by-3.SG.MASC", "PSS3SM")
    POSSESSED: tuple[str, str] = ("possessed", "PSSD")


class SwitchReference(UnimorphFeature):
    SWITCH_REFERENCE_AMONG_NP_ARGUMENTS: tuple[str, str] = (
        "switch-reference-among-np-arguments",
        "CN_R_MN",
    )
    DIFFERENT_SUBJECT: tuple[str, str] = ("different-subject", "DS")
    DIFFERENT_SUBJECT_ADVERBIAL: tuple[str, str] = ("different-subject-adverbial", "DSADV")
    LOGOPHORIC: tuple[str, str] = ("logophoric", "LOG")
    OPEN_REFERENCE: tuple[str, str] = ("open-reference", "OR")
    SEQUENTIAL_MULTICLAUSAL_ASPECT: tuple[str, str] = ("sequential-multiclausal-aspect", "SEQMA")
    SIMULTANEOUS_MULTICLAUSAL_ASPECT: tuple[str, str] = (
        "simultaneous-multiclausal-aspect",
        "SIMMA",
    )
    SAME_SUBJECT: tuple[str, str] = ("same-subject", "SS")
    SAME_SUBJECT_ADVERBIAL: tuple[str, str] = ("same-subject-adverbial", "SSADV")


class Tense(UnimorphFeature):
    WITHIN_ONE_DAY: tuple[str, str] = ("within-one-day", "1DAY")
    FUTURE: tuple[str, str] = ("future", "FUT")
    HODIERNAL: tuple[str, str] = ("hodiernal", "HOD")
    IMMEDIATE: tuple[str, str] = ("immediate", "IMMED")
    PRESENT: tuple[str, str] = ("present", "PRS")
    PAST: tuple[str, str] = ("past", "PST")
    RECENT: tuple[str, str] = ("recent", "RCT")
    REMOTE: tuple[str, str] = ("remote", "RMT")


class Valency(UnimorphFeature):
    APPLICATIVE: tuple[str, str] = ("applicative", "APPL")
    CAUSATIVE: tuple[str, str] = ("causative", "CAUS")
    DITRANSITIVE: tuple[str, str] = ("ditransitive", "DITR")
    IMPERSONAL: tuple[str, str] = ("impersonal", "IMPRS")
    INTRANSITIVE: tuple[str, str] = ("intransitive", "INTR")
    RECIPROCAL: tuple[str, str] = ("reciprocal", "RECP")
    REFLEXIVE: tuple[str, str] = ("reflexive", "REFL")
    TRANSITIVE: tuple[str, str] = ("transitive", "TR")


class Voice(UnimorphFeature):
    ACCOMPANIER_FOCUS: tuple[str, str] = ("accompanier-focus", "ACFOC")
    ACTIVE: tuple[str, str] = ("active", "ACT")
    AGENT_FOCUS: tuple[str, str] = ("agent-focus", "AGFOC")
    ANTIPASSIVE: tuple[str, str] = ("antipassive", "ANTIP")
    BENEFICIARY_FOCUS: tuple[str, str] = ("beneficiary-focus", "BFOC")
    CONVEYED_FOCUS: tuple[str, str] = ("conveyed-focus", "CFOC")
    DIRECT: tuple[str, str] = ("direct", "DIR")
    INSTRUMENT_FOCUS: tuple[str, str] = ("instrument-focus", "IFOC")
    INVERSE: tuple[str, str] = ("inverse", "INV")
    LOCATION_FOCUS: tuple[str, str] = ("location-focus", "LFOC")
    MIDDLE: tuple[str, str] = ("middle", "MID")
    PASSIVE: tuple[str, str] = ("passive", "PASS")
    PATIENT_FOCUS: tuple[str, str] = ("patient-focus", "PFOC")

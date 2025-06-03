from itertools import product
from pathlib import Path
from typing import Optional

from cltk.alphabet.lat import remove_accents, remove_macrons
from tqdm import tqdm

from .types import DerivationMap, InflectionMap, MorphemeTable, Paradigm

from utils.data.unimorph import GRC_NORMALIZATION_MAP


def load_unimorph_latin_derivations(derivation_filepath: Path) -> DerivationMap:
    derivations: DerivationMap = {}
    with derivation_filepath.open(encoding="utf-8", mode="r") as derivations_file:
        for line in tqdm(derivations_file, desc="Loading Derivations (Latin, Unimorph)"):
            base, derivation, _, affix = line.strip().split("\t")
            base, derivation, affix = (
                remove_accents(remove_macrons(base)),
                remove_accents(remove_macrons(derivation)),
                remove_accents(remove_macrons(affix)),
            )

            if derivation not in derivations:
                derivations[derivation] = []

            affixed_base: tuple[str, str] = (base, affix)
            if affixed_base not in derivations[derivation]:
                derivations[derivation].append(affixed_base)

    return derivations


def load_wfl_derivations(derivation_filepath: Path) -> DerivationMap:
    derivations: DerivationMap = {}
    with derivation_filepath.open(encoding="utf-8", mode="r") as derivations_file:
        for line in tqdm(derivations_file, desc="Loading Derivations (Latin, WFL)"):
            base, derivation, _, affix, *_ = line.strip().split("\t")
            base, derivation, affix = (
                remove_accents(remove_macrons(base)),
                remove_accents(remove_macrons(derivation)),
                remove_accents(remove_macrons(affix)),
            )

            base_alternates: list[str] = []
            for form in base.split("/"):
                if form.startswith("-"):
                    # In this case, an alternate inflection is given.
                    # In the future, I could attempt to integrate these in a systematic way,
                    #   as the notation does not lend itself to determining forms readily.
                    continue
                else:
                    base_alternates.append(form)

            derivation_alternates: list[str] = []
            for form in derivation.split("/"):
                if form.startswith("-"):
                    # In this case, an alternate inflection is given.
                    # In the future, I could attempt to integrate these in a systematic way,
                    #   as the notation does not lend itself to determining forms readily.
                    continue
                else:
                    derivation_alternates.append(form)

            for base, derivation in product(base_alternates, derivation_alternates):
                if derivation not in derivations:
                    derivations[derivation] = []

                affixed_base: tuple[str, str] = (base, affix)
                if affixed_base not in derivations[derivation]:
                    derivations[derivation].append(affixed_base)

    return derivations


def load_unimorph_latin_inflections(inflection_filepath: Path) -> InflectionMap:
    inflections: InflectionMap = {}
    with inflection_filepath.open(encoding="utf-8", mode="r") as inflections_file:
        for line in tqdm(inflections_file, desc="Loading Inflections (Latin, Unimorph)"):
            base, inflection, tags, segmentation = line.strip().split("\t")
            base, inflection, segmentation = (
                remove_accents(remove_macrons(base)),
                remove_accents(remove_macrons(inflection)),
                remove_accents(remove_macrons(segmentation)),
            )

            if base.startswith("*") or base.startswith("-"):
                continue

            tagged_segmentation: tuple[str, list[str]] = (tags, segmentation.split("|"))
            if base not in inflections:
                inflections[base] = []

            if tagged_segmentation not in inflections[base] and segmentation != "-":
                inflections[base].append(tagged_segmentation)

    # Some headwords have no available segmentations.
    # The two loops below filter out those segmentations.
    empty_headwords: list[str] = []
    for headword, tagged_segmentations in inflections.items():
        if len(tagged_segmentations) == 0:
            empty_headwords.append(headword)

    for headword in empty_headwords:
        del inflections[headword]

    return inflections


def load_unimorph_ancient_greek_inflections(inflection_filepath: Path) -> InflectionMap:
    inflections: InflectionMap = {}
    with inflection_filepath.open(encoding="utf-8", mode="r") as inflections_file:
        for line in tqdm(inflections_file, desc="Loading Inflections (Ancient Greek, Unimorph)"):
            if line.strip() == "":
                continue
            else:
                base, inflection, tags = line.strip().split("\t")
                forms: list[str] = [base, inflection]
                for i in range(0, len(forms)):
                    # If there are two spelling variants, we take the first.
                    if "/" in forms[i]:
                        forms[i] = forms[i].split("/")[0].strip()

                    # We remove parentheses (which indicate possible spelling variants
                    #   depending on the next word) as well as macrons and breves.
                    for key, value in GRC_NORMALIZATION_MAP.items():
                        forms[i] = forms[i].replace(key, value)

                base, inflection = forms
                # We filter out articles for nominal forms.
                if inflection.count(" ") > 0:
                    inflection = inflection[inflection.index(" ") + 1 :]

                # We don't have a segmentation, so we just supply the full inflected form here.
                tagged_inflection: tuple[str, list[str]] = (tags, [inflection])
                if base not in inflections:
                    inflections[base] = []

                if tagged_inflection not in inflections[base]:
                    inflections[base].append(tagged_inflection)

    # Some headwords have no available inflections.
    # The two loops below filter out those inflections.
    empty_headwords: list[str] = []
    for headword, tagged_segmentations in inflections.items():
        if len(tagged_segmentations) == 0:
            empty_headwords.append(headword)

    for headword in empty_headwords:
        del inflections[headword]

    return inflections


def construct_paradigms(
    inflections: InflectionMap, language: str, derivations: Optional[DerivationMap] = None
) -> list[Paradigm]:
    paradigms: list[Paradigm] = []

    # We combine derivation and inflection information into paradigms.
    for headword, tagged_segmentations in tqdm(
        inflections.items(), desc=f"Loading Paradigms ({language.title()})"
    ):
        # First (if applicable), across the paradigm, we take derivations into account.
        if derivations is not None:
            derivational_affix_count: Optional[int] = 0
            current_derivations: list[str] = [headword]
            prior_derivations: list[str] = []
            while len(current_derivations) > 0:
                current_derivation = current_derivations.pop(0)
                if current_derivation in prior_derivations:
                    continue  # It's possible to get in an infinite loop due to circular derivations.
                elif derivations.get(current_derivation, None) is not None:
                    derivational_affix_count += len(derivations[current_derivation])
                    prior_derivations.append(current_derivation)
                    current_derivations.extend(
                        {base for (base, affix) in derivations[current_derivation]}
                    )
        else:
            derivational_affix_count = None

        # Second, we take into account the number of morphemes in each inflection.
        paradigm: Paradigm = {}
        for tag, segmentation in tagged_segmentations:
            if segmentation is not None:
                inflectional_affix_count: Optional[int] = len(segmentation) - 1
                stem_count: Optional[int] = 1
            else:
                inflectional_affix_count, stem_count = None, None

            paradigm[("".join(segmentation), tag)] = MorphemeTable(
                derivational_affix_count, inflectional_affix_count, stem_count
            )
        else:
            paradigms.append(paradigm)

    return paradigms

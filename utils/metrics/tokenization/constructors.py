from itertools import product
from pathlib import Path
from typing import Callable, Optional

from cltk.alphabet.lat import remove_accents, remove_macrons
from tqdm import tqdm

from .constants import (
    DerivationMap,
    InflectionMap,
    MorphemeTable,
    MorphologyDataSource,
    Paradigm,
    TokenizationLanguage,
)


def load_unimorph_derivations(derivation_filepath: Path) -> DerivationMap:
    derivations: DerivationMap = {}
    with derivation_filepath.open(encoding="utf-8", mode="r") as derivations_file:
        for line in tqdm(derivations_file, desc="Loading Derivations (Latin, Unimorph)"):
            base, derivation, _, affix = line.strip().split("\t")
            base, derivation, affix = (
                remove_macrons(base),
                remove_macrons(derivation),
                remove_macrons(affix),
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
                remove_macrons(remove_accents(base)),
                remove_macrons(remove_accents(derivation)),
                remove_macrons(remove_accents(affix)),
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


def load_unimorph_inflections(inflection_filepath: Path) -> InflectionMap:
    inflections: InflectionMap = {}
    with inflection_filepath.open(encoding="utf-8", mode="r") as inflections_file:
        for line in tqdm(inflections_file, desc="Loading Inflections (Unimorph)"):
            base, inflection, tags, segmentation = line.strip().split("\t")
            base, inflection, segmentation = (
                remove_macrons(base),
                remove_macrons(inflection),
                remove_macrons(segmentation),
            )

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
            paradigm["".join(segmentation)] = MorphemeTable(
                derivational_affix_count, len(segmentation) - 1, 1
            )
        else:
            paradigms.append(paradigm)

    return paradigms

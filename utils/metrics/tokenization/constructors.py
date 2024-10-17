from pathlib import Path

from cltk.alphabet.lat import remove_macrons
from tqdm import tqdm

from .constants import Paradigm, ParadigmConstructor, TokenizationLanguage


def construct_paradigms(filepath: Path, language: TokenizationLanguage) -> list[Paradigm]:
    match language:
        case TokenizationLanguage.LATIN:
            paradigm_builder: ParadigmConstructor = construct_latin_unimorph_paradigms
        case _:
            raise ValueError(f"Language {language} not currently supported.")

    paradigms: list[Paradigm] = paradigm_builder(filepath)
    return paradigms


def construct_latin_unimorph_paradigms(filepath: Path) -> list[Paradigm]:
    paradigms: list[Paradigm] = []

    # First, we gather derivations.
    derivation_filepath: Path = filepath.joinpath("lat.derivations")
    derivations: dict[str, list[tuple[str, str]]] = {}
    with derivation_filepath.open(encoding="utf-8", mode="r") as derivations_file:
        for line in tqdm(derivations_file, desc="Loading Derivations (Latin, Unimorph)"):
            base, derivation, _, affix = line.strip().split("\t")
            base, derivation, affix = remove_macrons(base), remove_macrons(derivation), remove_macrons(affix)

            if derivation not in derivations:
                derivations[derivation] = []

            affixed_base: tuple[str, str] = (base, affix)
            if affixed_base not in derivations[derivation]:
                derivations[derivation].append(affixed_base)

    # Next, we gather inflections.
    inflection_filepath: Path = filepath.joinpath("lat.segmentations")
    inflections: dict[str, list[tuple[str, list[str]]]] = {}
    with inflection_filepath.open(encoding="utf-8", mode="r") as inflections_file:
        for line in tqdm(inflections_file, desc="Loading Inflections (Latin, Unimorph)"):
            base, inflection, tags, segmentation = line.strip().split("\t")
            base, inflection, segmentation = \
                remove_macrons(base), remove_macrons(inflection), remove_macrons(segmentation)

            tagged_segmentation: tuple[str, list[str]] = (tags, segmentation.split("|"))
            if base not in inflections:
                inflections[base] = []

            if tagged_segmentation not in inflections[base] and segmentation != "-":
                inflections[base].append(tagged_segmentation)

    # Finally, we combine this information into paradigms.
    for headword, tagged_segmentations in tqdm(inflections.items(), desc="Loading Paradigms (Latin, Unimorph)"):
        # First, across the paradigm, we take inflections into account.
        derivational_affix_count: int = 0
        current_derivations: list[str] = [headword]
        prior_derivations: list[str] = []
        while len(current_derivations) > 0:
            current_derivation = current_derivations.pop(0)
            if current_derivation in prior_derivations:
                continue   # It's possible to get in an infinite loop due to circular derivations.
            elif derivations.get(current_derivation, None) is not None:
                derivational_affix_count += len(derivations[current_derivation])
                prior_derivations.append(current_derivation)
                current_derivations.extend({base for (base, affix) in derivations[current_derivation]})

        # Second, we take into account the number of morphemes in each inflection.
        paradigm: Paradigm = {}
        for (tag, segmentation) in tagged_segmentations:
            paradigm["".join(segmentation)] = derivational_affix_count + len(segmentation)
        else:
            paradigms.append(paradigm)

    return paradigms

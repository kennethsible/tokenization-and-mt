from pathlib import Path
from typing import Any

from tqdm import tqdm

from .constants import (
    IndividualParadigmWriter,
    NamedMorphologyTokenizationMetric,
    Paradigm,
    SubwordTokenizer,
)

DA_PARADIGM_COHERENCE_HEADER: str = "Derivationally-Aware Paradigm Coherence: Samples ({0})\n\n"
PARADIGM_ADHERENCE_HEADER: str = "Paradigm Adherence: Samples ({0})\n\n"
PARADIGM_COHERENCE_HEADER: str = "Paradigm Coherence: Samples ({0})\n\n"

PARADIGM_SUBHEADER: str = "Paradigm {0} (Score: {1}):"
PARADIGM_BULLET: str = "\n\t* {0}: ({1})"

PARADIGM_ADHERENCE_SUBBULLET: str = "\n\t\t- {0} (Expected) vs. {1} (Actual); Deviation: {2}"


def write_paradigm_adherence_results(
    output_filepath: Path,
    tokenizer: SubwordTokenizer,
    paradigms: list[Paradigm],
    samples: list[dict[str, Any]],
):
    with output_filepath.open(encoding="utf-8", mode="w+") as output_file:
        output_file.write(PARADIGM_ADHERENCE_HEADER.format(len(paradigms)))

        output_text: str = ""
        for i in tqdm(range(0, len(paradigms)), desc="Writing Adherence Samples"):
            paradigm: Paradigm = paradigms[i]
            sample: dict[str, Any] = samples[i]

            paradigm_subheading: str = PARADIGM_SUBHEADER.format(
                i + 1, f"{sample['adherence']:.4f}"
            )
            output_text += paradigm_subheading
            for j in range(0, len(paradigm)):
                tokens: list[str] = [
                    tokenizer.decode([token]) for token in sample["tokenizations"][j]
                ]
                main_bullet: str = PARADIGM_BULLET.format(sample["forms"][j], ", ".join(tokens))
                subbullet: str = PARADIGM_ADHERENCE_SUBBULLET.format(
                    sample["expected_lengths"][j],
                    sample["actual_lengths"][j],
                    sample["deviations"][j],
                )
                output_text += main_bullet + subbullet

            else:
                output_text += "\n\n"
        else:
            output_file.write(output_text)


def write_paradigm_coherence_results(
    output_filepath: Path,
    tokenizer: SubwordTokenizer,
    paradigms: list[Paradigm],
    samples: list[dict[str, Any]],
):
    with output_filepath.open(encoding="utf-8", mode="w+") as output_file:
        output_file.write(PARADIGM_COHERENCE_HEADER.format(len(paradigms)))

        output_text: str = ""
        for i in tqdm(range(0, len(paradigms)), desc="Writing Coherence Results"):
            paradigm: Paradigm = paradigms[i]
            sample: dict[str, Any] = samples[i]

            cohering_token: str = tokenizer.decode([sample['cohering_token']])
            maximal_coherence_output: str = (
                f"{sample['cohering_value']} / {len(paradigm)}; <{cohering_token}>"
            )
            paradigm_subheading: str = PARADIGM_SUBHEADER.format(i + 1, maximal_coherence_output)
            output_text += paradigm_subheading
            for j in range(0, len(paradigm)):
                tokens: list[str] = [
                    tokenizer.decode([token]) for token in sample["tokenizations"][j]
                ]
                main_bullet: str = PARADIGM_BULLET.format(sample["forms"][j], ", ".join(tokens))
                output_text += main_bullet
            else:
                output_text += "\n\n"
        else:
            output_file.write(output_text)


def write_da_paradigm_coherence_results(
    output_filepath: Path,
    tokenizer: SubwordTokenizer,
    paradigms: list[Paradigm],
    samples: list[dict[str, Any]],
):
    with output_filepath.open(encoding="utf-8", mode="w+") as output_file:
        output_file.write(DA_PARADIGM_COHERENCE_HEADER.format(len(paradigms)))

        output_text: str = ""
        for i in tqdm(range(0, len(paradigms)), desc="Writing DA Coherence Results"):
            paradigm: Paradigm = paradigms[i]
            sample: dict[str, Any] = samples[i]

            cohering_tokens: list[str] = [
                tokenizer.decode([token]) for token in sample['cohering_token_set']
            ]
            maximal_coherence_output: str = (
                f"{sample['cohering_value']} / {len(paradigm)}; "
                f"{sample['derivational_affix_count']}; "
                f"<{', '.join(cohering_tokens)}>"
            )

            paradigm_subheading: str = PARADIGM_SUBHEADER.format(i + 1, maximal_coherence_output)
            output_text += paradigm_subheading
            for j in range(0, len(paradigm)):
                tokens: list[str] = [
                    tokenizer.decode([token]) for token in sample["tokenizations"][j]
                ]
                main_bullet: str = PARADIGM_BULLET.format(sample["forms"][j], ", ".join(tokens))
                output_text += main_bullet
            else:
                output_text += "\n\n"
        else:
            output_file.write(output_text)


INDIVIDUAL_MORPHOLOGY_WRITER_MAPPING: dict[str, IndividualParadigmWriter] = {
    NamedMorphologyTokenizationMetric.DERIVATIONALLY_AWARE_PARADIGM_COHERENCE: write_da_paradigm_coherence_results,
    NamedMorphologyTokenizationMetric.PARADIGM_ADHERENCE: write_paradigm_adherence_results,
    NamedMorphologyTokenizationMetric.PARADIGM_COHERENCE: write_paradigm_coherence_results,
}

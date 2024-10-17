from typing import Any

from tqdm import tqdm

from .constants import NamedTokenizationMetric, Paradigm, ParadigmMetric, SubwordTokenizer


def compute_paradigm_coherence(
    tokenizer: SubwordTokenizer, paradigms: list[Paradigm], tokenizer_kwargs: dict[str, Any]
) -> float:
    total_coherence: int = 0
    total_forms: int = 0
    for paradigm in tqdm(paradigms, desc="Examining Paradigms"):
        forms: list[str] = list(paradigm.keys())
        tokenizations: list[list[int]] = [
            tokenizer.encode(form, **tokenizer_kwargs) for form in forms
        ]
        tokens: set[int] = set()
        for tokenization in tokenizations:
            tokens.update(set(tokenization))

        maximally_cohering_token: int = -1
        maximally_cohering_value: int = -1
        for token in tokens:
            token_coherence: int = sum([token in tokenization for tokenization in tokenizations])
            if maximally_cohering_token == -1 or token_coherence > maximally_cohering_value:
                # For now, the first item found takes precedence. Ties could be broken in another way.
                maximally_cohering_token = token
                maximally_cohering_value = token_coherence
            else:
                continue
        else:
            total_coherence += maximally_cohering_value
            total_forms += len(forms)

    paradigm_coherence: float = total_coherence / total_forms
    return paradigm_coherence


METRIC_MAPPING: dict[str, ParadigmMetric] = {
    NamedTokenizationMetric.PARADIGM_COHERENCE: compute_paradigm_coherence
}

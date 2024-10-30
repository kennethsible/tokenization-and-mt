from functools import partial
from multiprocessing import Pool
from typing import Any

from tqdm import tqdm

from .constants import (
    NamedCorpusTokenizationMetric,
    NamedMorphologyTokenizationMetric,
    Paradigm,
    ParadigmMetric,
    SubwordTokenizer,
    CorpusMetric,
)
from utils.data.corpora import BaseCorpusDataset


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


def compute_average_tps(
    tokenizer: SubwordTokenizer,
    corpus: BaseCorpusDataset,
    tokenizer_kwargs: dict[str, Any],
    processes: int = 1,
    chunk_size: int = 1,
) -> float:
    sentence_length_map: dict[int, int] = {}
    if processes > 1:
        static_kwargs: dict[str, Any] = {
            "tokenizer": tokenizer,
            "tokenizer_kwargs": tokenizer_kwargs,
        }
        with Pool(processes=processes) as pool:
            with tqdm(total=len(corpus), desc="Checking Corpus TPS") as completion_tracker:
                tps_partial: partial = partial(get_mapped_sentence_count, **static_kwargs)
                for result in pool.imap_unordered(
                    tps_partial, enumerate(corpus), chunksize=chunk_size
                ):
                    key, value = result
                    sentence_length_map[key] = value
                    completion_tracker.update()
    else:
        for sentence_index, sentence in tqdm(enumerate(corpus), desc="Checking Corpus Average TPS"):
            # noinspection PyTypeChecker
            tokenized_sentence_length: int = get_sentence_token_count(
                tokenizer, tokenizer_kwargs, sentence
            )
            sentence_length_map[sentence_index] = tokenized_sentence_length

    sentence_lengths: list[int] = list(sentence_length_map.values())
    assert len(sentence_lengths) == len(corpus)

    average_tokens_per_sentence: float = sum(sentence_lengths) / len(sentence_lengths)
    return average_tokens_per_sentence


def get_sentence_token_count(
    tokenizer: SubwordTokenizer, tokenizer_kwargs: dict[str, Any], sentence: str
) -> int:
    tokenized_sentence: list[int] = tokenizer.encode(sentence, **tokenizer_kwargs)
    tokenized_sentence_length: int = len(tokenized_sentence)
    return tokenized_sentence_length


def get_mapped_sentence_count(
    sentence: tuple[int, str], tokenizer: SubwordTokenizer, tokenizer_kwargs: dict[str, Any]
) -> tuple[int, int]:
    sentence_id, sentence_text = sentence
    sentence_token_count: int = get_sentence_token_count(tokenizer, tokenizer_kwargs, sentence_text)
    return sentence_id, sentence_token_count


def compute_average_fertility(
    tokenizer: SubwordTokenizer,
    corpus: BaseCorpusDataset,
    tokenizer_kwargs: dict[str, Any],
    processes: int = 1,
    chunk_size: int = 1,
) -> float:
    if processes > 1:
        fertility_map: dict[int, list[int]] = {}
        static_kwargs: dict[str, Any] = {
            "tokenizer": tokenizer,
            "tokenizer_kwargs": tokenizer_kwargs,
        }
        with Pool(processes=processes) as pool:
            with tqdm(total=len(corpus), desc="Checking Corpus Fertility") as completion_tracker:
                fertility_partial: partial = partial(
                    get_mapped_sentence_fertilities, **static_kwargs
                )
                for result in pool.imap_unordered(
                    fertility_partial, enumerate(corpus), chunksize=chunk_size
                ):
                    key, value = result
                    fertility_map[key] = value
                    completion_tracker.update()

        assert len(corpus) == len(fertility_map)

        word_fertilities: list[int] = []
        for sentence_fertilities in fertility_map.values():
            word_fertilities.extend(sentence_fertilities)
    else:
        word_fertilities: list[int] = []
        for sentence in tqdm(corpus, desc="Checking Corpus Fertility"):
            sentence_fertilities: list[int] = get_sentence_fertilities(
                tokenizer, tokenizer_kwargs, sentence
            )
            word_fertilities.extend(sentence_fertilities)

    average_fertility: float = sum(word_fertilities) / len(word_fertilities)
    return average_fertility


def get_sentence_fertilities(
    tokenizer: SubwordTokenizer, tokenizer_kwargs: dict[str, Any], sentence: str
):
    sentence_fertilities: list[int] = []
    words: list[str] = sentence.split()
    for word in words:
        tokenized_word: list[int] = tokenizer.encode(word, **tokenizer_kwargs)
        tokenized_word_length: int = len(tokenized_word)
        sentence_fertilities.append(tokenized_word_length)

    return sentence_fertilities


def get_mapped_sentence_fertilities(
    sentence: tuple[int, str], tokenizer: SubwordTokenizer, tokenizer_kwargs: dict[str, Any]
) -> tuple[int, list[int]]:
    sentence_id, sentence_text = sentence
    sentence_fertilities: list[int] = get_sentence_fertilities(
        tokenizer, tokenizer_kwargs, sentence_text
    )
    return sentence_id, sentence_fertilities


CORPUS_METRIC_MAPPING: dict[str, CorpusMetric] = {
    NamedCorpusTokenizationMetric.AVERAGE_TOKENS_PER_SENTENCE: compute_average_tps,
    NamedCorpusTokenizationMetric.FERTILITY: compute_average_fertility,
}

MORPHOLOGY_METRIC_MAPPING: dict[str, ParadigmMetric] = {
    NamedMorphologyTokenizationMetric.PARADIGM_COHERENCE: compute_paradigm_coherence
}

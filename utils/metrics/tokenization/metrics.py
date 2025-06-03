from csv import DictWriter, QUOTE_NONNUMERIC
from functools import partial
from itertools import combinations, product
from multiprocessing import Pool
from statistics import mean
from typing import Any, Iterator, Optional, Sequence, Union

from more_itertools import chunked
from multiset import FrozenMultiset
from numpy import array, float32, log2, uint8, uint16, uint32, uint64
from numpy.typing import DTypeLike, NDArray
from scipy.sparse import csr_array, dok_array
from tqdm import tqdm

from .types import (
    CategoryMap,
    CoherenceFunction,
    FeaturedWordlist,
    Paradigm,
    SubwordTokenizer,
    TokenizerKwargs,
)

from utils.data.corpora import BaseCorpusDataset


def compute_aggregate_paradigm_adherence(
    tokenizer: SubwordTokenizer, paradigms: list[Paradigm], tokenizer_kwargs: TokenizerKwargs
) -> float:
    total_adherence: float = 0.0
    total_paradigms: int = 0
    for paradigm in tqdm(paradigms, desc="Examining Paradigms for Adherence"):
        individual_adherence, _ = compute_paradigm_adherence(tokenizer, paradigm, tokenizer_kwargs)
        total_adherence += individual_adherence
        total_paradigms += 1

    paradigm_adherence: float = total_adherence / total_paradigms
    return paradigm_adherence


def compute_paradigm_adherence(
    tokenizer: SubwordTokenizer, paradigm: Paradigm, tokenizer_kwargs: TokenizerKwargs
) -> tuple[float, dict[str, Any]]:
    tagged_forms: list[tuple[str, str]] = list(paradigm.keys())
    forms: list[str] = [form for form, _ in paradigm.keys()]
    tokenizations: list[list[int]] = [tokenizer.encode(form, **tokenizer_kwargs) for form in forms]
    expected_tokenization_lengths: list[int] = [
        paradigm[tagged_form].count_morphemes() for tagged_form in tagged_forms
    ]
    actual_tokenization_lengths: list[int] = [len(tokenization) for tokenization in tokenizations]

    assert len(expected_tokenization_lengths) == len(actual_tokenization_lengths)
    form_deviations: list[int] = [
        abs(expected_tokenization_lengths[i] - actual_tokenization_lengths[i])
        for i in range(0, len(expected_tokenization_lengths))
    ]

    individual_deviation: int = sum(form_deviations)
    individual_total: int = sum(expected_tokenization_lengths)
    individual_adherence: float = 1 - (individual_deviation / individual_total)

    computation_stages: dict[str, Any] = {
        "forms": forms,
        "tokenizations": tokenizations,
        "expected_lengths": expected_tokenization_lengths,
        "actual_lengths": actual_tokenization_lengths,
        "deviations": form_deviations,
        "adherence": individual_adherence,
    }

    return individual_adherence, computation_stages


def compute_aggregate_paradigm_coherence(
    tokenizer: SubwordTokenizer,
    paradigms: list[Paradigm],
    tokenizer_kwargs: dict[str, Any],
    coherence_function: CoherenceFunction,
) -> float:
    total_coherence: int = 0
    total_forms: int = 0
    for paradigm in tqdm(paradigms, desc="Examining Paradigms for Coherence"):
        paradigm_size, maximally_cohering_value, _ = coherence_function(
            tokenizer, paradigm, tokenizer_kwargs
        )
        total_coherence += maximally_cohering_value
        total_forms += paradigm_size

    paradigm_coherence: float = total_coherence / total_forms
    return paradigm_coherence


def compute_paradigm_coherence(
    tokenizer: SubwordTokenizer, paradigm: Paradigm, tokenizer_kwargs: TokenizerKwargs
) -> tuple[int, float, dict[str, Any]]:
    forms: list[str] = [form for form, _ in paradigm.keys()]
    tokenizations: list[list[int]] = [tokenizer.encode(form, **tokenizer_kwargs) for form in forms]
    tokens: set[int] = set()
    for tokenization in tokenizations:
        tokens.update(set(tokenization))

    maximally_cohering_token: int = -1
    maximally_cohering_value: int = 0
    for token in tokens:
        token_coherence: int = sum([token in tokenization for tokenization in tokenizations])
        if maximally_cohering_token == -1 or token_coherence > maximally_cohering_value:
            # For now, the first item found takes precedence. Ties could be broken in another way.
            maximally_cohering_token = token
            maximally_cohering_value = token_coherence
        else:
            continue

    assert maximally_cohering_value <= len(forms)

    computation_stages: dict[str, Any] = {
        "forms": forms,
        "tokenizations": tokenizations,
        "cohering_token": maximally_cohering_token,
        "cohering_value": maximally_cohering_value,
    }

    return len(forms), maximally_cohering_value, computation_stages


def compute_aggregate_da_paradigm_coherence(
    tokenizer: SubwordTokenizer, paradigms: list[Paradigm], tokenizer_kwargs: TokenizerKwargs
) -> float:
    total_coherence: int = 0
    total_forms: int = 0
    for paradigm in tqdm(paradigms, desc="Examining Paradigms for DA Coherence"):
        paradigm_size, maximally_cohering_value, _ = compute_da_paradigm_coherence(
            tokenizer, paradigm, tokenizer_kwargs
        )
        total_coherence += maximally_cohering_value
        total_forms += paradigm_size

    paradigm_coherence: float = total_coherence / total_forms
    return paradigm_coherence


def compute_da_paradigm_coherence(
    tokenizer: SubwordTokenizer, paradigm: Paradigm, tokenizer_kwargs: TokenizerKwargs
) -> tuple[int, float, dict[str, Any]]:
    tagged_forms: list[tuple[str, str]] = list(paradigm.keys())
    forms: list[str] = [form for form, _ in paradigm.keys()]
    tokenizations: list[list[int]] = [tokenizer.encode(form, **tokenizer_kwargs) for form in forms]

    derivational_affix_count: int = paradigm[tagged_forms[0]].derivations
    for tagged_form in tagged_forms:
        assert derivational_affix_count == paradigm[tagged_form].derivations

    token_sets: set[FrozenMultiset[int]] = set()
    for tokenization in tokenizations:
        token_combinations: list[Sequence[int]] = combinations(
            tokenization, derivational_affix_count + 1
        )
        token_multisets: list[FrozenMultiset[int]] = [
            FrozenMultiset(combination) for combination in token_combinations
        ]
        token_sets.update(token_multisets)

    multiset_tokenizations: list[FrozenMultiset[int]] = [
        FrozenMultiset(tokenization) for tokenization in tokenizations
    ]
    maximally_cohering_token_set: Optional[FrozenMultiset[int]] = None
    maximally_cohering_value: int = 0
    for token_set in token_sets:
        token_coherence: int = sum(
            [
                token_set.intersection(tokenization) == token_set
                for tokenization in multiset_tokenizations
            ]
        )
        if maximally_cohering_token_set is None or token_coherence > maximally_cohering_value:
            # For now, the first item found takes precedence. Ties could be broken in another way.
            maximally_cohering_token_set = token_set
            maximally_cohering_value = token_coherence
        else:
            continue

    computation_stages: dict[str, Any] = {
        "forms": forms,
        "tokenizations": tokenizations,
        "derivational_affix_count": derivational_affix_count,
        "cohering_token_set": maximally_cohering_token_set,
        "cohering_value": maximally_cohering_value,
    }

    assert maximally_cohering_value <= len(forms)
    return len(forms), maximally_cohering_value, computation_stages


def compute_morphological_rajski_distance(
    tokenizer: SubwordTokenizer,
    categories: CategoryMap,
    featured_forms: FeaturedWordlist,
    tokenizer_kwargs: TokenizerKwargs,
    add_null_feature: bool = False,
    processes: int = 1,
    batch_size: int = 1,
) -> tuple[float, dict[str, float]]:
    mutual_information_map: dict[str, float] = {}

    joint_frequency_matrices: list[csr_array] = build_joint_frequency_matrices(
        tokenizer,
        categories,
        featured_forms,
        tokenizer_kwargs,
        add_null_feature,
        processes,
        batch_size,
    )

    for index, category in tqdm(
        enumerate(categories.keys()), desc="Handling Categories", total=len(categories)
    ):
        csr_frequency_matrix: csr_array = joint_frequency_matrices.pop(0).tocsr()
        joint_distribution_matrix: csr_array = (
            csr_frequency_matrix / csr_frequency_matrix.sum()
        ).astype(dtype=float32)
        mutual_information_map[category] = compute_categorical_rajski_distance(
            joint_distribution_matrix, processes, batch_size
        )

    print(mutual_information_map)

    return mean(list(mutual_information_map.values())), mutual_information_map


def build_feature_matrices(
    categories: CategoryMap, add_null_feature: bool, matrix_dtype: DTypeLike, vocabulary_size: int
) -> list[dok_array]:
    tag_modifier: int = 1 if add_null_feature is True else 0
    joint_frequency_matrices: list[dok_array] = [
        dok_array((vocabulary_size, len(category_tags) + tag_modifier), dtype=matrix_dtype)
        for category_tags in categories.values()
    ]
    return joint_frequency_matrices


def build_joint_frequency_matrices(
    tokenizer: SubwordTokenizer,
    categories: CategoryMap,
    featured_forms: FeaturedWordlist,
    tokenizer_kwargs: TokenizerKwargs,
    add_null_feature: bool,
    processes: int,
    batch_size: int,
):
    tokenized_featured_forms: list[tuple[list[int], dict[str, str]]] = [
        (tokenizer.encode(form, **tokenizer_kwargs), features)
        for (form, features) in tqdm(featured_forms, desc="Tokenizing Forms")
    ]
    batches: list[list[tuple[list[int], dict[str, str]]]] = list(
        chunked(tokenized_featured_forms, batch_size)
    )

    maximum_matrix_value: int = len(featured_forms)
    if maximum_matrix_value < 256:
        matrix_dtype: DTypeLike = uint8
    elif maximum_matrix_value < 65536:
        matrix_dtype = uint16
    elif maximum_matrix_value < 4294967296:
        matrix_dtype = uint32
    else:
        matrix_dtype = uint64

    static_kwargs: dict[str, Any] = {
        "add_null_feature": add_null_feature,
        "categories": categories,
        "matrix_dtype": matrix_dtype,
        "vocabulary_size": tokenizer.vocab_size,
    }

    batch_matrices: list[list[dok_array]] = []
    with Pool(processes=processes) as pool:
        with tqdm(
            total=len(batches), desc="Populating Frequency Matrices by Batch"
        ) as completion_tracker:
            frequency_partial: partial = partial(accumulate_feature_frequencies, **static_kwargs)
            for result in pool.imap_unordered(frequency_partial, batches):
                batch_matrices.append(result)
                completion_tracker.update()
                completion_tracker.refresh()

    # noinspection PyTypeChecker
    joint_frequency_matrices: list[dok_array] = [
        sum([matrices[i] for matrices in batch_matrices]) for i in range(0, len(categories))
    ]

    return joint_frequency_matrices


def accumulate_feature_frequencies(
    batch: list[tuple[list[int], dict[str, str]]],
    categories: CategoryMap,
    add_null_feature: bool,
    vocabulary_size: int,
    matrix_dtype: DTypeLike,
) -> list[dok_array]:
    matrices: list[dok_array] = build_feature_matrices(
        categories, add_null_feature, matrix_dtype, vocabulary_size
    )
    for item in batch:
        tokenized_form, features = item
        for category, tag in features.items():
            category_index: int = list(categories).index(category)
            tag_index: int = categories[category].index(tag) if tag in categories[category] else -1
            if tag_index != -1 or add_null_feature is True:
                for token_id in tokenized_form:
                    matrices[category_index][token_id, tag_index] += 1
            else:
                continue

    return matrices


def write_frequencies(
    tokenizer: SubwordTokenizer, categories: CategoryMap, joint_frequency_matrices: list[dok_array]
):
    for category_index, (category, tags) in tqdm(
        enumerate(categories.items()), desc="Writing Categorical Frequencies"
    ):
        with open(
            f"results/test/{category}_frequencies.tsv", encoding="utf-8", mode="w+", newline=""
        ) as output_file:
            writer: DictWriter = DictWriter(
                output_file,
                fieldnames=["Token", *tags, "NULL"],
                delimiter="\t",
                quoting=QUOTE_NONNUMERIC,
            )
            writer.writeheader()

            matrix: dok_array = joint_frequency_matrices[category_index]
            lines: list[dict[Union[int, str], str]] = []
            for row_index, row in enumerate(matrix):
                line: dict[Union[int, str], str] = {"Token": tokenizer.decode([row_index])}
                for item_index, item in enumerate(row):
                    tag: str = tags[item_index] if item_index < len(tags) else "NULL"
                    line[tag] = item.item()
                else:
                    lines.append(line)
            else:
                writer.writerows(lines)


def compute_categorical_rajski_distance(
    joint_distribution_matrix: csr_array, processes: int, batch_size: int
) -> float:
    mutual_information: csr_array = compute_mutual_information(
        joint_distribution_matrix, processes, batch_size
    )
    joint_entropy: csr_array = compute_entropy(joint_distribution_matrix)
    rajski_distance: float = (1.0 - (mutual_information / joint_entropy)).item()
    return rajski_distance


def compute_mutual_information(
    joint_matrix: csr_array, processes: int, batch_size: int
) -> csr_array:
    x_matrix, y_matrix = joint_matrix.sum(axis=1), joint_matrix.sum(axis=0)

    x_range: list[int] = [i for i in range(0, len(x_matrix)) if x_matrix[i] != 0.0]
    y_range: list[int] = [j for j in range(0, len(y_matrix)) if y_matrix[j] != 0.0]
    index_pairs: product = product(x_range, y_range)
    batched_index_pairs: Iterator = chunked(index_pairs, n=batch_size)

    batch_probabilities: dict[Sequence[tuple[int, int]], NDArray] = {}
    marginal_probability_matrix: dok_array = dok_array(joint_matrix.shape, dtype=joint_matrix.dtype)
    static_kwargs: dict[str, Any] = {"x_matrix": x_matrix, "y_matrix": y_matrix}
    with Pool(processes=processes) as pool:
        with tqdm(
            total=len(x_range) * len(y_range), desc="Computing Marginal Probabilities"
        ) as completion_tracker:
            probability_partial: partial = partial(get_batched_probabilities, **static_kwargs)
            for result in pool.imap_unordered(probability_partial, batched_index_pairs):
                batch, probabilities = result
                batch_probabilities[batch] = probabilities
                completion_tracker.update(len(batch))
                completion_tracker.refresh()

    for batch, probabilities in tqdm(
        batch_probabilities.items(), desc="Storing Marginal Probabilities"
    ):
        for k in range(0, len(batch)):
            marginal_probability_matrix[batch[k]] = probabilities[k].item()
    else:
        marginal_probability_matrix: csr_array = marginal_probability_matrix.tocsr()

    # It is possible that a token is not predicted at all by the model for a dataset of a limited size.
    #   In this case, the joint probability is 0.0; so regardless of the resolution of the other term,
    #   the mutual information remains 0.0, and nothing is added.

    dependence_matrix: csr_array = joint_matrix.copy()
    rows, columns = dependence_matrix.nonzero()
    for row, column in zip(rows, columns):
        if marginal_probability_matrix[row, column].item() > 0.0:
            dependence_matrix[row, column] = (
                joint_matrix[row, column] / marginal_probability_matrix[row, column]
            )

    dependence_matrix.data[:] = log2(dependence_matrix.data)

    mutual_information_values: csr_array = joint_matrix * dependence_matrix
    mutual_information: csr_array = mutual_information_values.sum()

    return mutual_information


def compute_entropy(distribution: csr_array) -> csr_array:
    log_distribution: csr_array = distribution.copy()
    log_distribution.data[:] = log2(log_distribution.data)

    joint_log_distribution: csr_array = distribution * log_distribution
    joint_entropy: csr_array = -1 * joint_log_distribution.sum()

    return joint_entropy


def get_batched_probabilities(
    indices: list[tuple[int, int]], x_matrix: csr_array, y_matrix: csr_array
) -> tuple[Sequence[tuple[int, int]], NDArray]:
    x_values, y_values = zip(*indices)
    x_array: NDArray = array([x_matrix[i] for i in x_values], dtype=x_matrix.dtype)
    y_array: NDArray = array([y_matrix[j] for j in y_values], dtype=y_matrix.dtype)
    probabilities: NDArray = x_array * y_array

    return tuple(indices), probabilities


def compute_average_tps(
    tokenizer: SubwordTokenizer,
    corpus: BaseCorpusDataset,
    tokenizer_kwargs: TokenizerKwargs,
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
    tokenizer: SubwordTokenizer, tokenizer_kwargs: TokenizerKwargs, sentence: str
) -> int:
    tokenized_sentence: list[int] = tokenizer.encode(sentence, **tokenizer_kwargs)
    tokenized_sentence_length: int = len(tokenized_sentence)
    return tokenized_sentence_length


def get_mapped_sentence_count(
    sentence: tuple[int, str], tokenizer: SubwordTokenizer, tokenizer_kwargs: TokenizerKwargs
) -> tuple[int, int]:
    sentence_id, sentence_text = sentence
    sentence_token_count: int = get_sentence_token_count(tokenizer, tokenizer_kwargs, sentence_text)
    return sentence_id, sentence_token_count


def compute_average_fertility(
    tokenizer: SubwordTokenizer,
    corpus: BaseCorpusDataset,
    tokenizer_kwargs: TokenizerKwargs,
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
    tokenizer: SubwordTokenizer, tokenizer_kwargs: TokenizerKwargs, sentence: str
) -> list[int]:
    sentence_fertilities: list[int] = []
    words: list[str] = sentence.split()
    for word in words:
        tokenized_word: list[int] = tokenizer.encode(word, **tokenizer_kwargs)
        tokenized_word_length: int = len(tokenized_word)
        sentence_fertilities.append(tokenized_word_length)

    return sentence_fertilities


def get_mapped_sentence_fertilities(
    sentence: tuple[int, str], tokenizer: SubwordTokenizer, tokenizer_kwargs: TokenizerKwargs
) -> tuple[int, list[int]]:
    sentence_id, sentence_text = sentence
    sentence_fertilities: list[int] = get_sentence_fertilities(
        tokenizer, tokenizer_kwargs, sentence_text
    )
    return sentence_id, sentence_fertilities

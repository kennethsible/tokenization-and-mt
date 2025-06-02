from functools import partial
from pathlib import Path
from typing import Callable

from .constants import (
    MorphologyDataSource,
    NamedLanguageModel,
    NamedCorpusTokenizationMetric,
    NamedParadigmTokenizationMetric,
    TokenizationLanguage,
    NamedFeatureTokenizationMetric,
)
from .constructors import (
    load_unimorph_ancient_greek_inflections,
    load_unimorph_latin_derivations,
    load_unimorph_latin_inflections,
    load_wfl_derivations,
)
from .metrics import (
    compute_average_fertility,
    compute_average_tps,
    compute_aggregate_paradigm_adherence,
    compute_aggregate_paradigm_coherence,
    compute_paradigm_adherence,
    compute_paradigm_coherence,
    compute_da_paradigm_coherence,
    compute_morphological_rajski_distance,
)
from .types import (
    AggregateParadigmMetric,
    CorpusMetric,
    IndividualParadigmMetric,
    IndividualParadigmWriter,
    AggregateFeatureMetric,
)
from .writers import (
    write_paradigm_adherence_results,
    write_paradigm_coherence_results,
    write_da_paradigm_coherence_results,
)

MODELS_BY_LANGUAGE: dict[str, set[str]] = {
    TokenizationLanguage.ANCIENT_GREEK: {
        NamedLanguageModel.ANCIENT_GREEK_BERT,
        NamedLanguageModel.GREBERTA,
        NamedLanguageModel.PHILBERTA,
    },
    TokenizationLanguage.ICELANDIC: {
        NamedLanguageModel.ICEBERT,
        NamedLanguageModel.IS_ROBERTA,
        NamedLanguageModel.MULTILINGUAL_BERT,
        NamedLanguageModel.XLM_ROBERTA,
    },
    TokenizationLanguage.LATIN: {
        NamedLanguageModel.CANINE_C,
        NamedLanguageModel.CANINE_S,
        NamedLanguageModel.LABERTA,
        NamedLanguageModel.LATIN_BERT,
        NamedLanguageModel.MULTILINGUAL_BERT,
        NamedLanguageModel.PHILBERTA,
        NamedLanguageModel.SPHILBERTA,
        NamedLanguageModel.XLM_ROBERTA,
    },
}

DEFAULT_TOKENIZER_FILEPATHS: dict[NamedLanguageModel, Path] = {
    NamedLanguageModel.ANCIENT_GREEK_BERT: Path("resources/grc/ancient-greek-bert"),
    NamedLanguageModel.CANINE_C: Path("resources/multi/canine-c"),
    NamedLanguageModel.CANINE_S: Path("resources/multi/canine-s"),
    NamedLanguageModel.GREBERTA: Path("resources/grc/greberta"),
    NamedLanguageModel.ICEBERT: Path("resources/isl/icebert"),
    NamedLanguageModel.IS_ROBERTA: Path("resources/isl/IsRoBERTa"),
    NamedLanguageModel.LATIN_BERT: Path(
        "resources/lat/latin-bert/subword_tokenizer_latin/latin.subword.encoder"
    ),
    NamedLanguageModel.MULTILINGUAL_BERT: Path("resources/multi/mbert"),
    NamedLanguageModel.LABERTA: Path("resources/lat/laberta"),
    NamedLanguageModel.PHILBERTA: Path("resources/multi/philberta"),
    NamedLanguageModel.SPHILBERTA: Path("resources/multi/sphilberta"),
    NamedLanguageModel.XLM_ROBERTA: Path("resources/multi/xlm-roberta-base"),
}


DEFAULT_DERIVATION_FUNCTIONS: dict[tuple[TokenizationLanguage, MorphologyDataSource], Callable] = {
    (TokenizationLanguage.LATIN, MorphologyDataSource.UNIMORPH): load_unimorph_latin_derivations,
    (
        TokenizationLanguage.LATIN,
        MorphologyDataSource.UNIMORPH_CORRECTED,
    ): load_unimorph_latin_derivations,
    (TokenizationLanguage.LATIN, MorphologyDataSource.WORD_FORMATION_LEXICON): load_wfl_derivations,
}


DEFAULT_INFLECTION_FUNCTIONS: dict[tuple[TokenizationLanguage, MorphologyDataSource], Callable] = {
    (
        TokenizationLanguage.ANCIENT_GREEK,
        MorphologyDataSource.UNIMORPH,
    ): load_unimorph_ancient_greek_inflections,
    (TokenizationLanguage.LATIN, MorphologyDataSource.UNIMORPH): load_unimorph_latin_inflections,
}


DEFAULT_DERIVATION_FILEPATHS: dict[tuple[TokenizationLanguage, MorphologyDataSource], Path] = {
    (TokenizationLanguage.LATIN, MorphologyDataSource.UNIMORPH): Path(
        "data/unimorph/lat/lat.derivations"
    ),
    (TokenizationLanguage.LATIN, MorphologyDataSource.WORD_FORMATION_LEXICON): Path(
        "data/word-formation-lexicon/wfl_derivations.tsv"
    ),
}

DEFAULT_INFLECTION_FILEPATHS: dict[tuple[TokenizationLanguage, MorphologyDataSource], Path] = {
    (TokenizationLanguage.ANCIENT_GREEK, MorphologyDataSource.UNIMORPH): Path(
        "data/unimorph/grc/grc"
    ),
    (TokenizationLanguage.LATIN, MorphologyDataSource.UNIMORPH): Path(
        "data/unimorph/lat/lat.segmentations"
    ),
}

CORPUS_METRIC_MAPPING: dict[str, CorpusMetric] = {
    NamedCorpusTokenizationMetric.AVERAGE_TOKENS_PER_SENTENCE: compute_average_tps,
    NamedCorpusTokenizationMetric.FERTILITY: compute_average_fertility,
}

AGGREGATE_FEATURE_METRIC_MAPPING: dict[NamedFeatureTokenizationMetric, AggregateFeatureMetric] = {
    NamedFeatureTokenizationMetric.MORPHOLOGICAL_RAJSKI_DISTANCE: compute_morphological_rajski_distance
}

AGGREGATE_PARADIGM_METRIC_MAPPING: dict[str, AggregateParadigmMetric] = {
    NamedParadigmTokenizationMetric.DERIVATIONALLY_AWARE_PARADIGM_COHERENCE: partial(
        compute_aggregate_paradigm_coherence, coherence_function=compute_da_paradigm_coherence
    ),
    NamedParadigmTokenizationMetric.PARADIGM_ADHERENCE: compute_aggregate_paradigm_adherence,
    NamedParadigmTokenizationMetric.PARADIGM_COHERENCE: partial(
        compute_aggregate_paradigm_coherence, coherence_function=compute_paradigm_coherence
    ),
}

INDIVIDUAL_PARADIGM_METRIC_MAPPING: dict[str, IndividualParadigmMetric] = {
    NamedParadigmTokenizationMetric.DERIVATIONALLY_AWARE_PARADIGM_COHERENCE: compute_da_paradigm_coherence,
    NamedParadigmTokenizationMetric.PARADIGM_ADHERENCE: compute_paradigm_adherence,
    NamedParadigmTokenizationMetric.PARADIGM_COHERENCE: compute_paradigm_coherence,
}

INDIVIDUAL_PARADIGM_WRITER_MAPPING: dict[str, IndividualParadigmWriter] = {
    NamedParadigmTokenizationMetric.DERIVATIONALLY_AWARE_PARADIGM_COHERENCE: write_da_paradigm_coherence_results,
    NamedParadigmTokenizationMetric.PARADIGM_ADHERENCE: write_paradigm_adherence_results,
    NamedParadigmTokenizationMetric.PARADIGM_COHERENCE: write_paradigm_coherence_results,
}

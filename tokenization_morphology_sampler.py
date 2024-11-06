from argparse import ArgumentParser, Namespace
from pathlib import Path
from random import sample, seed
from typing import Any

from utils.metrics.tokenization import (
    collect_tokenizer_filepaths,
    derive_paradigms,
    get_tokenization_individual_morphology_metric,
    get_tokenization_morphology_writer,
    get_tokenizers,
    IndividualParadigmMetric,
    IndividualParadigmWriter,
    MorphologyDataSource,
    NamedLanguageModel,
    NamedMorphologyTokenizationMetric,
    Paradigm,
    TokenizationLanguage,
)

if __name__ == "__main__":
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("--language", type=str, choices=list(TokenizationLanguage), required=True)
    parser.add_argument(
        "--language-model", type=str, choices=list(NamedLanguageModel), required=True
    )
    parser.add_argument(
        "--metric", type=str, choices=list(NamedMorphologyTokenizationMetric), required=True
    )
    parser.add_argument(
        "--derivation-source", type=str, choices=list(MorphologyDataSource), default=None
    )
    parser.add_argument(
        "--inflection-source", type=str, choices=list(MorphologyDataSource), required=True
    )
    parser.add_argument("--tokenizer-filepath", type=str, nargs="?", default="auto")
    parser.add_argument("--output-filepath", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--random-seed", type=int, default=42)
    args: Namespace = parser.parse_args()

    print(f"Starting sample generation with random seed <{args.random_seed}>...")
    seed(args.random_seed)

    # Sets up appropriate paths for loading pre-trained model and tokenizer.
    tokenizer_paths: list[Path] = collect_tokenizer_filepaths(
        [args.language_model], [args.tokenizer_filepath]
    )
    tokenizer, tokenizer_kwargs = list(get_tokenizers([args.language_model], tokenizer_paths))[-1]
    paradigms: list[Paradigm] = derive_paradigms(
        args.language, args.inflection_source, args.derivation_source
    )

    metric_function: IndividualParadigmMetric = get_tokenization_individual_morphology_metric(
        args.metric
    )
    sample_paradigms: list[Paradigm] = sample(paradigms, k=args.sample_size)

    sample_results: list[dict[str, Any]] = []
    for paradigm in sample_paradigms:
        *_, results = metric_function(tokenizer, paradigm, tokenizer_kwargs)
        sample_results.append(results)

    output_function: IndividualParadigmWriter = get_tokenization_morphology_writer(args.metric)
    output_function(args.output_filepath, tokenizer, sample_paradigms, sample_results)

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Optional

from utils.metrics.tokenization import (
    collect_tokenizer_filepaths,
    derive_paradigms,
    get_tokenization_morphology_metric,
    get_tokenizers,
    MorphologyDataSource,
    NamedLanguageModel,
    NamedMorphologyTokenizationMetric,
    Paradigm,
    ParadigmMetric,
    SubwordTokenizer,
    TokenizationLanguage,
)

if __name__ == "__main__":
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("--language", type=str, choices=list(TokenizationLanguage), required=True)
    parser.add_argument("--language-models", nargs="+", type=str, choices=list(NamedLanguageModel))
    parser.add_argument(
        "--metric", type=str, nargs="+", choices=list(NamedMorphologyTokenizationMetric)
    )
    parser.add_argument(
        "--derivation-source", type=str, choices=list(MorphologyDataSource), default=None
    )
    parser.add_argument(
        "--inflection-source", type=str, choices=list(MorphologyDataSource), required=True
    )
    parser.add_argument("--tokenizer-filepaths", type=str, nargs="*", default=["auto"])
    args: Namespace = parser.parse_args()

    # Sets up appropriate paths for loading pre-trained model and tokenizer.
    tokenizer_paths: list[Path] = collect_tokenizer_filepaths(
        args.language_models, args.tokenizer_filepaths
    )
    full_tokenizers: list[tuple[SubwordTokenizer, dict[str, Any]]] = get_tokenizers(
        args.language_models, tokenizer_paths
    )
    paradigms: list[Paradigm] = derive_paradigms(
        args.language, args.inflection_source, args.derivation_source
    )

    for i, (tokenizer, tokenizer_kwargs) in enumerate(full_tokenizers):
        for metric in args.metric:
            metric_function: ParadigmMetric = get_tokenization_morphology_metric(metric)
            result: float = metric_function(tokenizer, paradigms, tokenizer_kwargs)
            print(
                f"For the language <{args.language}> and the model <{args.language_models[i]}>, "
                f"the metric <{metric}> attains a result of <{result}>."
            )

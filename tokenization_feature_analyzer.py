from argparse import ArgumentParser, BooleanOptionalAction, Namespace
from copy import deepcopy
from pathlib import Path
from math import ceil, sqrt
from os import cpu_count
from typing import Any
from tqdm import tqdm

from utils.data.unimorph import (
    CategoryMap,
    get_language_path,
    get_language_preprocessors,
    StringPreprocessor,
    UnimorphCorpus,
    UnimorphLanguage,
)
from utils.metrics.tokenization import (
    collect_tokenizer_filepaths,
    get_tokenizers,
    get_tokenization_aggregate_feature_metric,
    AggregateFeatureMetric,
    FeaturedWordlist,
    NamedLanguageModel,
    NamedFeatureTokenizationMetric,
    SubwordTokenizer,
)

# from utils.visualizations import compare_tokenizer_scores

if __name__ == "__main__":
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("--add-semantic-category", action=BooleanOptionalAction, default=False)
    parser.add_argument("--add-null-feature", action=BooleanOptionalAction, default=False)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--language", type=str, choices=list(UnimorphLanguage), required=True)
    parser.add_argument("--language-models", nargs="+", type=str, choices=list(NamedLanguageModel))
    parser.add_argument(
        "--metrics", type=str, nargs="+", choices=list(NamedFeatureTokenizationMetric)
    )
    parser.add_argument("--processes", type=int, default=ceil(sqrt(cpu_count())))
    parser.add_argument("--tokenizer-filepaths", type=str, nargs="*", default=["auto"])
    # parser.add_argument("--visualize", action=BooleanOptionalAction, default=False)
    args: Namespace = parser.parse_args()

    # Sets up appropriate paths for loading pre-trained model and tokenizer.
    tokenizer_paths: list[Path] = collect_tokenizer_filepaths(
        args.language_models, args.tokenizer_filepaths
    )
    full_tokenizers: list[tuple[SubwordTokenizer, dict[str, Any]]] = get_tokenizers(
        args.language_models, tokenizer_paths
    )

    language_filepath: Path = get_language_path(args.language)
    language_preprocessors: list[StringPreprocessor] = get_language_preprocessors(args.language)
    unimorph_corpus: UnimorphCorpus = UnimorphCorpus(language_filepath, language_preprocessors)

    categories: CategoryMap = deepcopy(unimorph_corpus.categories)
    if args.add_semantic_category is True:
        categories["semantics"] = list(
            set(
                [
                    lemma
                    for (_, lemma, _) in tqdm(
                        unimorph_corpus, desc="Collecting Semantic Categories"
                    )
                ]
            )
        )
        for featured_form in tqdm(unimorph_corpus, desc="Adding Semantic Features"):
            inflection, lemma, features = featured_form
            features["semantics"] = lemma

    featured_wordlist: FeaturedWordlist = [
        (inflection, features) for (inflection, _, features) in unimorph_corpus
    ]

    metric_kwargs: dict[str, Any] = {
        "add_null_feature": args.add_null_feature,
        "batch_size": args.batch_size,
        "processes": args.processes,
    }
    metric_scores: dict[str, list[float]] = {metric: [] for metric in args.metrics}
    for i, (tokenizer, tokenizer_kwargs) in tqdm(
        enumerate(full_tokenizers), desc="Examining Tokenizers", total=len(tokenizer_paths)
    ):
        for metric in args.metrics:
            metric_function: AggregateFeatureMetric = get_tokenization_aggregate_feature_metric(
                metric
            )
            aggregate_result, *_ = metric_function(
                tokenizer, categories, featured_wordlist, tokenizer_kwargs, **metric_kwargs
            )
            metric_scores[metric].append(aggregate_result)
            print(
                f"For the language <{args.language}> and the model <{args.language_models[i]}>, "
                f"the metric <{metric}> attains a result of <{aggregate_result}>."
            )
    # else:
    #     if args.visualize is True:
    #         for metric in args.metrics:
    #             compare_tokenizer_scores(metric, args.language_models, metric_scores[metric])

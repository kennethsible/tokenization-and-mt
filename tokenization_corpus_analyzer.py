from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

from utils.data.corpora import BaseCorpusDataset, NamedCorpus, load_corpus
from utils.metrics.tokenization import (
    CorpusMetric,
    NamedCorpusTokenizationMetric,
    NamedLanguageModel,
    collect_tokenizer_filepaths,
    get_tokenization_corpus_metric,
    get_tokenizers,
    SubwordTokenizer,
)


# TODO: should datasets be customized for each tokenizer?
#  (e.g., Latin BERT's has segmentation based on Latin BERT already.)
#  The main concern I have about this involves -ne, -que, and other affixes...


if __name__ == "__main__":
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=1)
    parser.add_argument("--corpus", type=str, choices=list(NamedCorpus), required=True)
    parser.add_argument("--language-models", nargs="+", type=str, choices=list(NamedLanguageModel))
    parser.add_argument(
        "--metric", type=str, nargs="+", choices=list(NamedCorpusTokenizationMetric)
    )
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--tokenizer-filepaths", type=str, nargs="*", default=["auto"])
    args: Namespace = parser.parse_args()

    multiprocessing_kwargs: dict[str, int] = {
        "processes": args.processes,
        "chunk_size": args.chunk_size,
    }

    tokenizer_filepaths: list[Path] = collect_tokenizer_filepaths(
        args.language_models, args.tokenizer_filepaths
    )
    full_tokenizers: list[tuple[SubwordTokenizer, dict[str, Any]]] = get_tokenizers(
        args.language_models, tokenizer_filepaths
    )
    corpus: BaseCorpusDataset = load_corpus(args.corpus)

    for i, (tokenizer, tokenizer_kwargs) in enumerate(full_tokenizers):
        for metric in args.metric:
            metric_function: CorpusMetric = get_tokenization_corpus_metric(metric)
            result: float = metric_function(
                tokenizer, corpus, tokenizer_kwargs, **multiprocessing_kwargs
            )
            print(
                f"For the model <{args.language_models[i]}>, "
                f"the metric <{metric}> attains a result of <{result}>."
            )

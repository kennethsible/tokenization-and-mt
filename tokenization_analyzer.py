from argparse import ArgumentParser, Namespace
from typing import Any

from utils.metrics.tokenization import NamedLanguageModel, NamedTokenizationMetric, Paradigm, ParadigmMetric, \
    TokenizationLanguage, construct_paradigms, get_tokenizer, get_tokenization_metric, resolve_filepaths

if __name__ == "__main__":
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("--language", type=str, choices=list(TokenizationLanguage), required=True)
    parser.add_argument("--language-model", type=str, choices=list(NamedLanguageModel))
    parser.add_argument("--metric", type=str, choices=list(NamedTokenizationMetric), required=True)
    parser.add_argument("--paradigm-filepath", type=str, default="auto")
    parser.add_argument("--tokenizer-filepath", type=str, default="auto")
    args: Namespace = parser.parse_args()
    kwargs: dict[str, Any] = vars(args)

    resolve_filepaths(kwargs)   # Sets up appropriate paths for loading pre-trained model and tokenizer.
    tokenizer, tokenizer_kwargs = get_tokenizer(args.language_model, kwargs["tokenizer_filepath"])
    paradigms: list[Paradigm] = construct_paradigms(kwargs["paradigm_filepath"], args.language)
    metric: ParadigmMetric = get_tokenization_metric(args.metric)
    result: float = metric(tokenizer, paradigms, tokenizer_kwargs)
    print(f"For the language <{args.language}> and the model <{args.language_model}>, "
          f"the metric <{args.metric}> attains a result of <{result}>.")

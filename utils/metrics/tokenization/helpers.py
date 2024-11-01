from pathlib import Path
from typing import Any

from tensor2tensor.data_generators.text_encoder import SubwordTextEncoder
from transformers import AutoTokenizer, CanineTokenizer, PreTrainedTokenizer

from .constants import (
    DEFAULT_TOKENIZER_FILEPATHS,
    NamedLanguageModel,
    SubwordTokenizer,
)


def get_tokenizers(
    language_model_names: list[NamedLanguageModel], tokenizer_filepaths: list[Path]
) -> list[tuple[SubwordTokenizer, dict[str, Any]]]:
    assert len(language_model_names) == len(tokenizer_filepaths)

    tokenizers: list[SubwordTokenizer] = []
    tokenizer_arguments: list[dict[str, Any]] = []
    for i in range(0, len(language_model_names)):
        language_model: str = language_model_names[i]
        tokenizer_filepath: Path = tokenizer_filepaths[i]
        match language_model:
            case NamedLanguageModel.LATIN_BERT:
                subword_tokenizer: SubwordTextEncoder = SubwordTextEncoder(tokenizer_filepath)
                tokenizer_kwargs: dict[str, Any] = {}
            case (
                NamedLanguageModel.LABERTA
                | NamedLanguageModel.PHILBERTA
                | NamedLanguageModel.SPHILBERTA
                | NamedLanguageModel.MULTILINGUAL_BERT
                | NamedLanguageModel.XLM_ROBERTA
                | NamedLanguageModel.ICEBERT
                | NamedLanguageModel.IS_ROBERTA
            ):
                subword_tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_filepath
                )
                tokenizer_kwargs: dict[str, Any] = {"add_special_tokens": False}
            case NamedLanguageModel.CANINE_C | NamedLanguageModel.CANINE_S:
                # We do not need to make use of the pretrained filepaths,
                # since CANINE's tokenizer is the same regardless.
                subword_tokenizer: CanineTokenizer = CanineTokenizer()
                tokenizer_kwargs: dict[str, Any] = {"add_special_tokens": False}
            case _:
                raise ValueError(
                    f"The language model <{language_model}> is not currently recognized."
                )

        tokenizers.append(subword_tokenizer)
        tokenizer_arguments.append(tokenizer_kwargs)

    full_tokenizers: list[tuple[SubwordTokenizer, dict[str, Any]]] = zip(
        tokenizers, tokenizer_arguments
    )
    return full_tokenizers


def collect_tokenizer_filepaths(
    language_models: list[NamedLanguageModel], tokenizer_path_selections: list[str]
):
    assert len(language_models) == len(tokenizer_path_selections) or (
        len(tokenizer_path_selections) == 1 and tokenizer_path_selections[-1] == "auto"
    )

    if len(language_models) == len(tokenizer_path_selections):
        model_path_pairs: tuple[str, str] = zip(language_models, tokenizer_path_selections)
    else:
        model_path_pairs = zip(language_models, tokenizer_path_selections * len(language_models))

    tokenizer_filepaths: list[Path] = []
    for language_model, tokenizer_filepath in model_path_pairs:
        if tokenizer_filepath == "auto":
            tokenizer_filepath: Path = retrieve_default_filepath(language_model)
        else:
            tokenizer_filepath: Path = Path(tokenizer_filepath)

        tokenizer_filepaths.append(tokenizer_filepath)

    return tokenizer_filepaths


def retrieve_default_filepath(language_model: NamedLanguageModel) -> Path:
    try:
        default_filepath: Path = DEFAULT_TOKENIZER_FILEPATHS[language_model]
    except KeyError:
        raise ValueError(
            f"The language model <{language_model}> does not currently have a default filepath."
        )

    return default_filepath

from pathlib import Path
from typing import Any, Type

from tensor2tensor.data_generators.text_encoder import SubwordTextEncoder
from transformers import BertTokenizer, CanineTokenizer, PreTrainedTokenizer, RobertaTokenizer

from .constants import DEFAULT_TOKENIZER_FILEPATHS, NamedLanguageModel, SubwordTokenizer, DEFAULT_PARADIGM_FILEPATHS


def get_tokenizer(lm_name: NamedLanguageModel, filepath: str) -> tuple[SubwordTokenizer, dict[str, Any]]:
    match lm_name:
        case NamedLanguageModel.LATIN_BERT:
            subword_tokenizer: SubwordTextEncoder = SubwordTextEncoder(filepath)
            tokenizer_kwargs: dict[str, Any] = {}
        case NamedLanguageModel.LABERTA | NamedLanguageModel.PHILBERTA | NamedLanguageModel.SPHILBERTA | \
             NamedLanguageModel.MULTILINGUAL_BERT:
            tokenizer_class: Type[PreTrainedTokenizer] = BertTokenizer \
                if lm_name == NamedLanguageModel.MULTILINGUAL_BERT else RobertaTokenizer
            subword_tokenizer: PreTrainedTokenizer = tokenizer_class.from_pretrained(filepath)
            tokenizer_kwargs: dict[str, Any] = {"add_special_tokens": False}
        case NamedLanguageModel.CANINE_C | NamedLanguageModel.CANINE_S:
            # We do not need to make use of the pretrained filepaths, since CANINE's tokenizer is the same regardless.
            subword_tokenizer: CanineTokenizer = CanineTokenizer()
            tokenizer_kwargs: dict[str, Any] = {"add_special_tokens": False}
        case _:
            raise ValueError(f"The language model <{lm_name}> is not currently recognized.")

    return subword_tokenizer, tokenizer_kwargs


def resolve_filepaths(kwargs: dict[str, Any]):
    for key in ("paradigm_filepath", "tokenizer_filepath"):
        if kwargs.get(key, None) is None:
            raise ValueError(f"<{key}> was not defined.")
        elif kwargs[key] == "auto":
            if key == "paradigm_filepath":
                table = DEFAULT_PARADIGM_FILEPATHS
                table_key = kwargs["language"]
            elif key == "tokenizer_filepath":
                table = DEFAULT_TOKENIZER_FILEPATHS
                table_key = kwargs["language_model"]
            else:
                raise ValueError(f"<{key}> for embedding-related filepaths not recognized.")

            kwargs[key] = retrieve_default_filepath(table_key, key, table)
        else:
            continue


def retrieve_default_filepath(table_key: str, key: str, table: dict[str, Path]) -> Path:
    try:
        default_filepath: Path = table[table_key]
    except KeyError:
        raise ValueError(f"The table key <{table_key}> is not present in the table for key <{key}> ")

    return default_filepath

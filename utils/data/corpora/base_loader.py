from abc import abstractmethod
from pathlib import Path
from re import sub
from typing import Any

MULTISPACE_REGEX: str = r"[\s]{2,}"


class BaseCorpusDataset:
    def __init__(self, **kwargs):
        self.data: list[str] = kwargs["sequences"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> str:
        return self.data[index]

    def __iter__(self):
        return iter(self.data)

    @abstractmethod
    def _load_dataset(self, dataset_filepath: Path, **kwargs) -> list[str]:
        raise NotImplementedError

    @staticmethod
    def _clean_sequence(sequence: str) -> str:
        # We perform some basic whitespace cleaning...
        revised_sequence: str = sequence.strip().replace("\n", " ")
        revised_sequence: str = sub(MULTISPACE_REGEX, " ", revised_sequence)
        return revised_sequence

    @abstractmethod
    def _tokenize_sentences(self, sequence: str, **tokenizer_kwargs: dict[str, Any]) -> list[str]:
        raise NotImplementedError

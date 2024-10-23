from abc import abstractmethod
from pathlib import Path
from typing import Any

from cltk.tokenizers.lat.lat import LatinPunktSentenceTokenizer

from .base_loader import BaseCorpusDataset


class LatinCorpusDataset(BaseCorpusDataset):
    sentence_tokenizer: LatinPunktSentenceTokenizer = LatinPunktSentenceTokenizer(strict=True)

    def __init__(self, sequences: list[str]):
        super().__init__(sequences=sequences)

    @abstractmethod
    def _load_dataset(self, dataset_filepath: Path, **kwargs) -> list[str]:
        raise NotImplementedError

    def _tokenize_sentences(self, sequence: str, **tokenizer_kwargs: dict[str, Any]) -> list[str]:
        return self.sentence_tokenizer.tokenize(sequence)

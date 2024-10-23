from pathlib import Path

from tqdm import tqdm

from .latin_loader import LatinCorpusDataset


class LatinBERTDataset(LatinCorpusDataset):
    def __init__(self, dataset_filepath: Path, **loading_kwargs):
        sequences: list[str] = self._load_dataset(dataset_filepath, **loading_kwargs)
        super().__init__(sequences)

    def _load_dataset(self, dataset_filepath: Path, **kwargs) -> list[str]:
        data: list[str] = []
        with dataset_filepath.open(encoding="utf-8", mode="r") as dataset_file:
            # We use this list to track groups of text in the document.
            current_sequences: list[str] = []
            for line in tqdm(dataset_file, "Dataset Loading (Lines)"):
                if line.isspace() and len(current_sequences) > 0:
                    joined_sequences: str = " ".join(current_sequences)
                    cleaned_sequence: str = self._clean_sequence(joined_sequences)
                    tokenized_sentences: list[str] = self._tokenize_sentences(cleaned_sequence)
                    data.extend(tokenized_sentences)

                    # We clear the data structure to start on the next group of text.
                    current_sequences.clear()
                else:
                    current_sequences.append(line.strip())

        return data

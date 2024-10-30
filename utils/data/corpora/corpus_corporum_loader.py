from pathlib import Path

# noinspection PyPep8Naming
from xml.etree import ElementTree as ET
from xml.etree.ElementTree import Element, ElementTree

from tqdm import tqdm

from .latin_loader import LatinCorpusDataset


class CorpusCorporumDataset(LatinCorpusDataset):
    def __init__(self, dataset_filepath: Path, **loading_kwargs):
        sequences: list[str] = self._load_dataset(dataset_filepath, **loading_kwargs)
        super().__init__(sequences)

    def _load_dataset(self, dataset_filepath: Path, **kwargs) -> list[str]:
        sequences: list[str] = []

        filepaths: list[str] = self._gather_files(dataset_filepath)
        for document_path in tqdm(filepaths, desc="Dataset Loading (Files)"):
            # First, we gather the text.
            xml_tree: ElementTree = ET.parse(document_path)
            xml_root: Element = xml_tree.getroot()
            child: Element = xml_root.find("text", {"": "http://www.tei-c.org/ns/1.0"})
            segments: list[str] = [
                sequence for sequence in child.itertext() if not sequence.isspace()
            ]

            # Then, we unite and clean that text.
            document_sequence: str = self._clean_sequence(" ".join(segments))
            tokenized_sentences: list[str] = self._tokenize_sentences(document_sequence)
            sequences.extend(tokenized_sentences)

        return sequences

    @staticmethod
    def _gather_files(head_directory: Path) -> list[str]:
        filepaths: list[str] = []
        for filepath in head_directory.glob("**/*.xml"):
            filepaths.append(str(filepath).replace("\\", "/"))

        return filepaths

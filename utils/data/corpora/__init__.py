from pathlib import Path
from typing import Type

from .base_loader import BaseCorpusDataset
from .constants import NamedCorpus, CORPUS_PATHS
from .corpus_corporum_loader import CorpusCorporumDataset
from .latin_bert_loader import LatinBERTDataset
from .latin_loader import LatinCorpusDataset

CORPUS_CLASSES: dict[NamedCorpus, Type[BaseCorpusDataset]] = {
    NamedCorpus.CORPUS_CORPORUM: CorpusCorporumDataset,
    NamedCorpus.LATIN_BERT: LatinBERTDataset,
}


def load_corpus(corpus_name: NamedCorpus) -> BaseCorpusDataset:
    try:
        corpus_path: Path = CORPUS_PATHS[corpus_name]
        corpus_loader: Type[BaseCorpusDataset] = CORPUS_CLASSES[corpus_name]
    except KeyError:
        raise ValueError(f"The corpus <{corpus_name}> is not currently supported.")

    corpus: BaseCorpusDataset = corpus_loader(dataset_filepath=corpus_path)
    return corpus

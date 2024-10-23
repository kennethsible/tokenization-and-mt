from enum import StrEnum
from pathlib import Path


class NamedCorpus(StrEnum):
    CORPUS_CORPORUM: str = "corpus-corporum"
    LATIN_BERT: str = "latin-bert"


CORPUS_PATHS: dict[NamedCorpus, Path] = {
    NamedCorpus.CORPUS_CORPORUM: Path("data/corpora/corpus-corporum"),
    NamedCorpus.LATIN_BERT: Path("data/corpora/latin-bert/training_base.txt"),
}

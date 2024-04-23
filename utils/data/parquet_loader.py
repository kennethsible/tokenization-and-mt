from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

from pandas import DataFrame, read_parquet
from tqdm import tqdm


class RosenthalItem(NamedTuple):
    item_id: int
    latin: str
    english: str
    origin: str


class RosenthalDataset:
    def __init__(self, input_path: Path):
        self.data: list[RosenthalItem] = self._load_dataset(input_path)

    @staticmethod
    def _load_dataset(input_path: Path) -> list[RosenthalItem]:
        frame: DataFrame = read_parquet(input_path)
        listed_frame: list[tuple[int, str, str, str]] = zip(*[column.tolist() for (_, column) in frame.items()])
        data: list[RosenthalItem] = []
        for listed_item in tqdm(listed_frame, desc="Loading Rosenthal Items"):
            item_id, latin_sentence, english_sentence, source = listed_item
            new_item: RosenthalItem = RosenthalItem(item_id, latin_sentence, english_sentence, source)
            data.append(new_item)

        return data

    def __getitem__(self, index: int):
        return self.data[index]

    def __len__(self):
        return len(self.data)

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import TextIO

from tqdm import tqdm

from utils.data.preprocessing.parquet_loader import RosenthalDataset


DEFAULT_INPUT_PATH: Path = Path("data", "rosenthal", "original")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--input-type", type=str, choices=["parquet"], default="parquet")
    parser.add_argument("--language-codes", type=str, nargs=2, default=["la", "en"])
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "validation", "test"])
    args: Namespace = parser.parse_args()

    if not args.input_path.is_dir():
        raise ValueError("The input path is not a valid directory.")

    input_filepaths: list[Path] = []
    for split in args.splits:
        split_path: Path = args.input_path / f"{split}.{args.input_type}"
        if split_path.exists():
            input_filepaths.append(split_path)
        else:
            raise ValueError(
                f"The path for the split <{split}> of type <{args.input_type}> does not exist."
            )

    split_datasets: list[RosenthalDataset] = []
    for input_filepath in tqdm(input_filepaths, desc="Loading"):
        split_dataset: RosenthalDataset = RosenthalDataset(input_path=input_filepath)
        split_datasets.append(split_dataset)

    first_code, second_code = args.language_codes
    for split_number, split_name in enumerate(args.splits):
        first_file: TextIO = (args.output_path / f"{split_name}.{first_code}").open(
            encoding="utf-8", mode="w+"
        )
        second_file: TextIO = (args.output_path / f"{split_name}.{second_code}").open(
            encoding="utf-8", mode="w+"
        )

        split: RosenthalDataset = split_datasets[split_number]
        for sentence in split:
            first_file.write(f"{sentence.latin}\n")
            second_file.write(f"{sentence.english}\n")

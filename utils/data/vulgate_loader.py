from pathlib import Path
from re import fullmatch, Match, sub
from typing import Optional


class VulgateDataset:
    IUXTA_HEBRAICA_PATH: Path = Path("data", "vulgate", "jerome.vulgate.part.21a.old_latin_psalms.tess")
    IUXTA_SEPTUAGINTA_PATH: Path = Path("data", "vulgate", "jerome.vulgate.part.21.psalms.tess")
    MULTISPACE_PATTERN: str = r"( ){2,}"
    TESSERAE_LINE_PATTERN: str = r"(?P<annotation><.+>)[\s](?P<text>.+)[;]?[\n]?"

    def __init__(self):
        self.hebrew_verses: list[str] = self._load_verses(self.IUXTA_HEBRAICA_PATH)
        self.septuagint_verses: list[str] = self._load_verses(self.IUXTA_SEPTUAGINTA_PATH)

    def __iter__(self):
        return iter(zip(self.hebrew_verses, self.septuagint_verses))

    def __getitem__(self, index: int) -> tuple[str, str]:
        item: tuple[str, str] = (self.hebrew_verses[index], self.septuagint_verses[index])
        return item

    def __len__(self):
        assert len(self.hebrew_verses) == len(self.septuagint_verses)
        return len(self.hebrew_verses)

    def _load_verses(self, input_path: Path) -> list[str]:
        verses: list[str] = []
        with input_path.open(encoding="utf-8", mode="r") as input_file:
            for line in input_file:
                matched_line: Optional[Match] = fullmatch(self.TESSERAE_LINE_PATTERN, line)
                if matched_line is not None:
                    line_text: str = matched_line.group("text")
                    line_text: str = line_text.replace("%", "").replace("*", "")
                    line_text: str = line_text[:-1] if line_text[-1] == ";" else line_text
                    line_text = sub(self.MULTISPACE_PATTERN, " ", line_text)
                    verses.append(line_text)
                else:
                    raise ValueError(f"The line <{line}> does not match the expected pattern.")

        return verses

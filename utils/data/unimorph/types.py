from typing import Callable, TypeAlias

UnimorphTuple: TypeAlias = tuple[str, str, dict[str, str]]
CategoryMap: TypeAlias = dict[str, list[str]]
StringPreprocessor: TypeAlias = Callable[[str], str]

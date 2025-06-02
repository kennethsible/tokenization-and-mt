from csv import DictReader
from itertools import product
from pathlib import Path
from re import compile, fullmatch, search, Match, Pattern
from typing import Iterator, Optional, Sequence, Union

from natsort import natsorted
from tqdm import tqdm

from .constants import UnimorphOperation
from .mapping import compose_regex
from .types import CategoryMap, UnimorphTuple, StringPreprocessor


# TODO: permit LGSPEC to be used more than once for the same entry.
#   Should LGSPEC be omitted from the metric? It's hard to tell how many axes LGSPEC spans.


class UnimorphCorpus:
    # Operation Patterns:
    OPERATION_REGEX: Pattern = compile(
        r"(?P<conjunction>\+)|(?P<disjunction>/)|(?P<negation>non{[^}]+})"
    )
    # Conjunction realizes all conjuncts in one form, combining multiple simple features.
    # Disjunction realizes each disjunct in a separate instance of the form.
    # Negation is syntactic sugar for a more extensive disjunction.

    # General Pattern:
    TAG_REGEX: Pattern = compose_regex()

    def __init__(self, corpus_path: Path, preprocessors: list[StringPreprocessor] = tuple()):
        self.categories: CategoryMap = self._load_corpus_categories(corpus_path)
        self.data: list[UnimorphTuple] = self._load_corpus(corpus_path, preprocessors)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> UnimorphTuple:
        return self.data[index]

    def __iter__(self) -> Iterator[UnimorphTuple]:
        return iter(self.data)

    def _load_corpus_categories(self, corpus_path: Path) -> CategoryMap:
        category_map: dict[str, set[str]] = {}
        with corpus_path.open(encoding="utf-8", mode="r", newline='') as corpus_file:
            reader: DictReader = DictReader(
                corpus_file, fieldnames=["lemma", "inflection", "tags"], delimiter="\t"
            )
            for line_index, line in tqdm(
                enumerate(reader), desc=f"Gathering Corpus Categories <{corpus_path}>"
            ):
                raw_features: list[str] = line["tags"].split(";")
                for raw_feature in raw_features:
                    operation_match: Optional[Match[str]] = search(
                        self.OPERATION_REGEX, raw_feature
                    )
                    if operation_match is not None:
                        if operation_match.group(UnimorphOperation.CONJUNCTION) is not None:
                            internal_features: list[str] = list(natsorted(raw_feature.split("+")))
                            operation: Optional[str] = UnimorphOperation.CONJUNCTION
                        elif operation_match.group(UnimorphOperation.DISJUNCTION) is not None:
                            internal_features: list[str] = raw_feature.strip("{}").split("/")
                            operation: Optional[str] = UnimorphOperation.DISJUNCTION
                        elif operation_match.group(UnimorphOperation.NEGATION) is not None:
                            internal_features: list[str] = [raw_feature.strip("non{}")]
                            operation: Optional[str] = UnimorphOperation.NEGATION
                        else:
                            raise ValueError(
                                f"The operation present in <{raw_feature}> is not supported."
                            )
                    else:
                        internal_features: list[str] = [raw_feature]
                        operation: Optional[str] = None

                    if all(
                        [
                            self._is_standard_feature(internal_feature)
                            for internal_feature in internal_features
                        ]
                    ):
                        if operation == UnimorphOperation.CONJUNCTION:
                            feature_matches: list[Match] = [
                                fullmatch(self.TAG_REGEX, conjunct)
                                for conjunct in internal_features
                            ]
                            matched_dimensions: list[str] = [
                                dimension
                                for feature_match in feature_matches
                                for (dimension, feature) in feature_match.groupdict().items()
                                if feature is not None
                            ]

                            if not len(set(matched_dimensions)) == 1:
                                raise ValueError(
                                    f"Features in tag <{raw_feature}> not of same (or any) category."
                                )
                            else:
                                label: str = "+".join(internal_features)
                                if matched_dimensions[-1] not in category_map:
                                    category_map[matched_dimensions[-1]] = set()
                                category_map[matched_dimensions[-1]].add(label)
                        else:  # operation in (UnimorphOperation.DISJUNCTION, UnimorphOperation.NEGATION, None):
                            for internal_feature in internal_features:
                                feature_match: Match = fullmatch(self.TAG_REGEX, internal_feature)
                                for feature, label in feature_match.groupdict().items():
                                    if label is not None:
                                        if feature not in category_map:
                                            category_map[feature] = set()
                                        category_map[feature].add(label)
                                        break

        category_map: CategoryMap = {key: list(value) for key, value in category_map.items()}
        return category_map

    def _is_standard_feature(self, feature: str) -> bool:
        standard_bool: bool = fullmatch(self.TAG_REGEX, feature) is not None
        return standard_bool

    def _load_corpus(
        self, corpus_path: Path, preprocessors: list[StringPreprocessor]
    ) -> list[UnimorphTuple]:
        data: list[UnimorphTuple] = []
        with corpus_path.open(encoding="utf-8", mode="r", newline='') as corpus_file:
            reader: DictReader = DictReader(
                corpus_file, fieldnames=["lemma", "inflection", "tags"], delimiter="\t"
            )
            for line_index, line in tqdm(
                enumerate(reader), desc=f"Processing Lines <{corpus_path}>"
            ):
                lemma, inflection = line["lemma"], line["inflection"]
                for preprocessor in preprocessors:
                    lemma, inflection = preprocessor(lemma), preprocessor(inflection)

                realization_sets: list[Sequence[str]] = self._get_realization_sets(line["tags"])
                for realization_set in realization_sets:
                    features: dict[str, str] = self._parse_realization_set(realization_set)
                    featured_form: UnimorphTuple = (inflection, lemma, features)
                    data.append(featured_form)

        return data

    def _get_realization_sets(self, tag: str) -> list[Sequence[str]]:
        features: list[str] = tag.split(";")

        # TODO: the below is limited by the combination of operations.
        #   For instance, if conjunction or disjunction are used with negation, this won't work correctly.

        options: list[list[Union[str, Sequence[str]]]] = []
        for feature in features:
            operation_match: Optional[Match] = search(self.OPERATION_REGEX, feature)
            if operation_match is not None:
                if operation_match.group("conjunction") is not None:
                    conjoined_feature: str = "+".join(natsorted(feature.split("+")))
                    options.append([conjoined_feature])
                elif operation_match.group("disjunction") is not None:
                    disjuncts: list[str] = feature.strip("{}").split("/")
                    options.append(disjuncts)
                elif operation_match.group("negation") is not None:
                    # Because we're previously looked at all categories, we can assume that all categories are valid.
                    excluded_feature: str = feature.strip("non{}")
                    category: str = self._get_feature_category(feature)
                    possible_disjuncts: set[str] = set(self.categories[category])
                    possible_disjuncts.remove(excluded_feature)
                    options.append(list(possible_disjuncts))
                else:
                    raise ValueError(f"Match for <{feature}> does not fall into any defined group.")
            else:
                options.append([feature])

        realizations: list[Sequence[str]] = list(product(*options))
        return realizations

    def _parse_realization_set(
        self, realization_set: Sequence[str]
    ) -> dict[str, Union[None, str, Sequence[str]]]:
        feature_mapping: dict[str, Union[None, str, Sequence[str]]] = {
            category: None for category in self.categories
        }
        for feature in realization_set:
            category: str = self._get_feature_category(feature)
            if feature_mapping[category] is not None:
                raise ValueError(
                    f"Both <{feature}> and <{feature_mapping[category]}> were seen for <{category}>."
                )
            else:
                feature_mapping[category] = feature

        return feature_mapping

    def _get_feature_category(self, feature: str) -> str:
        for category, labels in self.categories.items():
            if feature in labels:
                relevant_category: str = category
                break
        else:
            raise ValueError(f"The feature <{feature}> was not recognized.")

        return relevant_category

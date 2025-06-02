from re import Match, match
from typing import Optional

from .constants import GRC_ARTICLE_REGEX, GRC_NORMALIZATION_MAP


def clean_alternatives(form_string: str) -> str:
    form: str = form_string.split("/")[0].strip() if "/" in form_string else form_string
    return form


def normalize_grc(form: str) -> str:
    for source, target in GRC_NORMALIZATION_MAP.items():
        form = form.replace(source, target)

    return form


def clean_articles_grc(form: str) -> str:
    article_match: Optional[Match] = match(GRC_ARTICLE_REGEX, form)
    form: str = form[len(article_match.group(0)) :] if article_match is not None else form
    return form

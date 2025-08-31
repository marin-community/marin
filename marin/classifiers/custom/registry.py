"""
Registry of custom attribute functions.

Each function should take a document, a list of attributes, and an optional additional kwargs.
It should return a dictionary with the attribute name as the key and the attribute value as the value.
We use this registry so that we can version the attribute functions within ExecutorSteps.
"""

from collections.abc import Callable
from typing import Any

from marin.classifiers.types import Attribute, Document

REGISTRY = {}


def register(func: Callable[[Document, list[Attribute], Any | None], dict]):
    REGISTRY[func.__name__] = func
    return func


@register
def max_quality_score(
    doc: list[Document],
    input_attrs: list[Attribute],
    input_attr_names: list[str],
    score_name: str,
    output_attr_name: str,
):
    """
    Take the maximum of the input attributes.
    """
    return {
        output_attr_name: {
            "score": max(
                attr["attributes"][input_attr_names[classifier_id]][score_name]
                for classifier_id, attr in enumerate(input_attrs)
            )
        }
    }

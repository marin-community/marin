import re
from dataclasses import asdict, is_dataclass
from string import punctuation
from typing import Any


def asdict_without_nones(obj: Any) -> dict[str, Any]:
    """
    Convert a dataclass instance to a dictionary, excluding None values.

    This function transforms a dataclass object into a dictionary representation,
    but unlike the standard dataclasses.asdict() function, it filters out any
    attributes with None values from the resulting dictionary.

    Args:
        obj: A dataclass instance to be converted to a dictionary.

    Returns:
        A dictionary containing all non-None attributes from the dataclass.

    Raises:
        ValueError: If the provided object is not a dataclass.
    """
    if not is_dataclass(obj):
        raise ValueError(f"Expected dataclass, got '{obj}'")
    return asdict(obj, dict_factory=lambda x: {k: v for (k, v) in x if v is not None})


class DefaultTokenizer:
    """
    Normalize and tokenize texts by converting all characters to the lower case and
    splitting on whitespaces and punctuations.
    """

    def __init__(self):
        super().__init__()
        self.r = re.compile(rf"[\s{re.escape(punctuation)}]+")

    def tokenize(self, text: str) -> list[str]:
        return self.r.split(text.lower())

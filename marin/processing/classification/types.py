from dataclasses import dataclass
from enum import Enum


class DatasetFormat(Enum):
    """Dataset formats when curating a dataset for training

    Attributes:
        DOLMA_FORMATTED_JSONL: The given input dataset contains Dolma formatted jsonl files.
        This means that the dataset must contain the key "text" in the jsonl file.

        FASTTEXT: The given input dataset contains fasttext formatted jsonl files.
                  This is in the format of __label__<label> <text>
    """

    DOLMA_FORMATTED_JSONL = "dolma_formatted_jsonl"
    FASTTEXT = "fasttext"


@dataclass
class Example:
    """Data structure representing a single input with its quality label"""

    label: str
    text: str

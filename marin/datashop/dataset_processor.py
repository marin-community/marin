import re
from typing import ClassVar

from marin.generation.dataset import DatasetOutputProcessor


class MeduDatasetOutputProcessor(DatasetOutputProcessor):
    SCORE_OPTIONS_DICT: ClassVar[dict[str, int]] = {
        "Great": 5,
        "Good": 4,
        "Okay": 3,
        "Poor": 2,
        "Useless": 1,
    }

    def __init__(self, input_path: str, output_path: str):
        super().__init__(input_path, output_path, list(MeduDatasetOutputProcessor.SCORE_OPTIONS_DICT.values()))

    @staticmethod
    def extract_score(text: str) -> int:

        # Match "Final Score: " followed by one of the score options
        match = re.search(r"Final Score:\s*(Great|Good|Okay|Poor|Useless)", text)
        if match:
            score_text = match.group(1)
            return MeduDatasetOutputProcessor.SCORE_OPTIONS_DICT[score_text]

        return -1

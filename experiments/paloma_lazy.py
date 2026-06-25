# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Paloma perplexity-eval subsets as lazy validation ``Dataset`` handles.

The Paloma HF download is already pinned (``raw/paloma-fc6827``), so each subset
tokenizes its ``val`` split straight from that location into a fresh explicit cache
— no re-download. Handle version of ``paloma_tokenized`` in ``paloma.py``.
"""

from marin.execution.lazy import Dataset
from marin.experiment.data import tokenized

from experiments.llama import llama3_tokenizer
from experiments.paloma import PALOMA_DATASETS_TO_DIR

# Pinned Paloma download (paloma.py: with_output_path("raw/paloma-fc6827").cd("65cd6fc")).
_PALOMA_RAW = "raw/paloma-fc6827/65cd6fc"


def paloma_validation(*, tokenizer: str = llama3_tokenizer) -> list[Dataset]:
    """One validation ``Dataset`` handle per Paloma subset, keyed by ``paloma/<subset>``."""
    return [
        tokenized(
            f"paloma/{subset}",
            tokenizer=tokenizer,
            version="llama3",
            paths=[f"{_PALOMA_RAW}/{path_part}/val/val*.jsonl.gz"],
            validation=True,
        )
        for subset, path_part in PALOMA_DATASETS_TO_DIR.items()
    ]

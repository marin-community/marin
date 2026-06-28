# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Paloma perplexity-eval subsets as lazy validation ``Dataset`` handles.

The Paloma HF download is already pinned (``raw/paloma-fc6827``), so each subset
tokenizes its ``val`` split straight from that location into a fresh explicit cache
— no re-download.
"""

from marin.execution.lazy import Dataset
from marin.experiment.data import tokenized

from experiments.llama import llama3_tokenizer

# Pinned Paloma download.
_PALOMA_RAW = "raw/paloma-fc6827/65cd6fc"

# The Paloma eval subsets and their directories within the HF dataset
# (https://huggingface.co/datasets/allenai/paloma). The subset name keys the handle;
# the directory locates its shards.
_PALOMA_SUBSETS = {
    "4chan": "4chan_meta_sep",
    "c4_100_domains": "c4_100_domains",
    "c4_en": "c4_en",
    "dolma-v1_5": "dolma-v1_5",
    "dolma_100_programing_languages": "dolma_100_programing_languages",
    "dolma_100_subreddits": "dolma_100_subreddits",
    "falcon-refinedweb": "falcon-refinedweb",
    "gab": "gab",
    "m2d2_s2orc_unsplit": "m2d2_s2orc_unsplit",
    "m2d2_wikipedia_unsplit": "m2d2_wikipedia_unsplit",
    "manosphere_meta_sep": "manosphere_meta_sep",
    "mc4": "mc4",
    "ptb": "ptb",
    "redpajama": "redpajama",
    "twitterAAE_HELM_fixed": "twitterAAE_HELM_fixed",
    "wikitext_103": "wikitext_103",
}


def paloma_validation(*, tokenizer: str = llama3_tokenizer) -> list[Dataset]:
    """One validation ``Dataset`` handle per Paloma subset, keyed by ``paloma/<subset>``."""
    return [
        tokenized(
            f"paloma/{subset}",
            tokenizer=tokenizer,
            version="llama3",
            paths=[f"{_PALOMA_RAW}/{directory}/val/val*.jsonl.gz"],
            validation=True,
        )
        for subset, directory in _PALOMA_SUBSETS.items()
    ]

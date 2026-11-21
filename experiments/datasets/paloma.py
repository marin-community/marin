# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Paloma perplexity-eval subsets as lazy validation ``Dataset`` handles.

The Paloma HF download is already pinned (``raw/paloma-fc6827``), so each subset
tokenizes its ``val`` split straight from that location into a fresh explicit cache
— no re-download.
"""

from marin.execution.lazy import ArtifactStep
from marin.experiment.data import dataset_main, tokenized
from marin.processing.tokenize.tokenize import TokenizedCache

from experiments.llama import llama3_tokenizer

# Pinned Paloma download (out-of-region: gated HF, no in-region download step).
_PALOMA_RAW = "raw/paloma-fc6827/65cd6fc"
# In-region byte-lossless text reconstruction of the eval sets (see experiments/grug/fast_track/paloma_detok.py).
# Use this raw_prefix to rebuild the caches under any tokenizer without the out-of-region raw.
_PALOMA_DETOK_RAW = "raw/paloma-detok"
# Default cache version; bump when repointing raw or rebuilding under a new tokenizer.
_PALOMA_VERSION = "2026.06.28"

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


def paloma_dataset(
    subset: str,
    *,
    tokenizer: str = llama3_tokenizer,
    tag: str = "llama3",
    raw_prefix: str = _PALOMA_RAW,
    version: str = _PALOMA_VERSION,
) -> ArtifactStep[TokenizedCache]:
    """One Paloma subset as a validation handle.

    ``tag`` names the tokenizer in the cache path. The cache is content-addressed by name+version
    only (not by the tokenizer), so a non-default ``tokenizer`` MUST pass a distinct ``tag`` or it
    silently resolves to the existing ``llama3`` cache -- feeding the wrong vocab to the model.

    ``raw_prefix`` selects the raw source: the out-of-region pinned download (default) or the in-region
    detokenized reconstruction (``_PALOMA_DETOK_RAW``). Pair the detok prefix with a fresh ``version``
    so a new-tokenizer build writes a clean cache instead of colliding with a failed prior stub.
    """
    return tokenized(
        f"paloma/{subset}-{tag}",
        tokenizer=tokenizer,
        version=version,
        paths=[f"{raw_prefix}/{_PALOMA_SUBSETS[subset]}/val/val*.jsonl.gz"],
        validation=True,
    )


def paloma_datasets(
    *,
    tokenizer: str = llama3_tokenizer,
    tag: str = "llama3",
    raw_prefix: str = _PALOMA_RAW,
    version: str = _PALOMA_VERSION,
) -> dict[str, ArtifactStep[TokenizedCache]]:
    """All Paloma subsets, keyed by subset name."""
    return {
        subset: paloma_dataset(subset, tokenizer=tokenizer, tag=tag, raw_prefix=raw_prefix, version=version)
        for subset in _PALOMA_SUBSETS
    }


if __name__ == "__main__":
    dataset_main(paloma_datasets())

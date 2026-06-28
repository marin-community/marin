# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Nemotron CC splits as lazy ``Dataset`` handles (the flat dataset catalog).

One handle per quality split, tokenizing the split's glob straight from the pinned raw
download. The llama3-tokenized splits are pinned to their existing caches so referencing
them never re-tokenizes the multi-TiB corpus. This is the catalog only — handles, no
weights; the mixture weights are policy and live in the experiment that chooses them.
"""

from fray.types import ResourceConfig
from marin.execution.lazy import ArtifactStep
from marin.experiment.data import tokenized
from marin.processing.tokenize.tokenize import TokenizedCache

from experiments.llama import llama3_tokenizer

# The pinned raw Nemotron-CC download. (The bucket path carries an upstream "nemotro-cc"
# typo; it is the real location, so it is reproduced verbatim.)
_NEMOTRON_RAW = "raw/nemotro-cc-eeb783/contrib/Nemotron/Nemotron-CC/data-jsonl"

# The glob selecting each quality split's shards within the raw download.
NEMOTRON_SPLIT_GLOBS = {
    "hq_actual": "quality=high/kind=actual/**/*.jsonl.*",
    "hq_synth": "quality=high/kind=synthetic/**/*.jsonl.*",
    "medium_high": "quality=medium-high/**/*.jsonl.*",
    "medium": "quality=medium/**/*.jsonl.*",
    "medium_low": "quality=medium-low/**/*.jsonl.*",
    "low_actual": "quality=low/kind=actual/**/*.jsonl.*",
    "low_synth": "quality=low/kind=synthetic/**/*.jsonl.*",
}

# Pinned llama3-tokenized caches (hashes predate a hashing change; never recompute).
_NEMOTRON_LLAMA3_PINS = {
    "hq_actual": "tokenized/nemotron_cc/hq_actual-5af4cc",
    "hq_synth": "tokenized/nemotron_cc/hq_synth-3525e2",
    "medium_high": "tokenized/nemotron_cc/medium_high-d21701",
    "medium": "tokenized/nemotron_cc/medium-d86506",
    "medium_low": "tokenized/nemotron_cc/medium_low-0fdb07",
    "low_actual": "tokenized/nemotron_cc/low_actual-cb3f2c",
    "low_synth": "tokenized/nemotron_cc/low_synth-3c57b3",
}

# Levanter store consolidation needs the extra RAM for these multi-TiB splits.
_TOKENIZE_RESOURCES = ResourceConfig(ram="32g", cpu=2)


def nemotron_datasets(*, tokenizer: str = llama3_tokenizer) -> dict[str, ArtifactStep[TokenizedCache]]:
    """One :class:`Dataset` handle per Nemotron CC split, keyed by split name."""
    return {
        split: tokenized(
            f"nemotron_cc/{split}-llama3",
            tokenizer=tokenizer,
            version="2026.06.28",
            paths=[f"{_NEMOTRON_RAW}/{glob}"],
            pin=_NEMOTRON_LLAMA3_PINS.get(split) if tokenizer == llama3_tokenizer else None,
            resources=_TOKENIZE_RESOURCES,
        )
        for split, glob in NEMOTRON_SPLIT_GLOBS.items()
    }

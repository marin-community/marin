# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""FineWeb (GPT-2 tokenized) data for the arXiv 2609.19107 replication.

The paper pretrains on FineWeb with the GPT-2 tokenizer (vocab 50,257 padded
to 50,304). No GPT-2-tokenized FineWeb cache exists in the project's storage,
so this module declares the download + tokenize steps for the FineWeb
``sample/10BT`` slice (~10B tokens) plus one held-out validation file from
the full corpus. These materialize once; training launchers then depend on the
resulting caches.
"""

from marin.execution.lazy import ArtifactStep
from marin.experiment.data import hf_download, tokenized
from marin.processing.tokenize.tokenize import TokenizedCache

# Pinned FineWeb revision (HuggingFaceFW/fineweb, resolved 2026-09-19).
_FINEWEB_REVISION = "9bb295ddab0e05d785b879661af7260fed5140fc"
_FINEWEB_VERSION = "2026.09.19"

# Validation: one full-corpus crawl file (~6.9B tokens, ~5.7M docs), capped at
# 400k docs (~500M tokens); evals consume at most ~100M tokens per pass.
# (tokenize's sample_count is per shard.)
_VAL_SAMPLE_COUNT = 400_000
# A single crawl file — the full corpus is sharded per crawl as
# data/CC-MAIN-<crawl>/*.parquet (unlike the fineweb-edu flat layout); one
# file (~2.2GB, ~5.7M docs) is plenty for validation.
_VAL_FILE = "data/CC-MAIN-2024-10/000_00000.parquet"


def fineweb_10bt_train() -> ArtifactStep[TokenizedCache]:
    """FineWeb sample/10BT, GPT-2 tokenized — the replication training corpus.

    Deviation from the paper (documented in the variant README): the paper
    trains on full FineWeb; we use the 10BT uniformly-sampled slice.
    """
    raw = hf_download(
        "raw/fineweb-sample-10bt",
        hf_id="HuggingFaceFW/fineweb",
        revision=_FINEWEB_REVISION,
        urls_glob=["sample/10BT/*"],
        version=_FINEWEB_VERSION,
    )
    return tokenized(
        "fineweb-10bt-gpt2",
        tokenizer="gpt2",
        raw=raw,
        glob="sample/10BT/*.parquet",
        version=_FINEWEB_VERSION,
    )


def fineweb_validation() -> ArtifactStep[TokenizedCache]:
    """Held-out FineWeb documents (one full-corpus crawl file), GPT-2 tokenized."""
    raw = hf_download(
        "raw/fineweb-val",
        hf_id="HuggingFaceFW/fineweb",
        revision=_FINEWEB_REVISION,
        urls_glob=[_VAL_FILE],
        version=_FINEWEB_VERSION,
    )
    return tokenized(
        "fineweb-val-gpt2",
        tokenizer="gpt2",
        raw=raw,
        glob=_VAL_FILE,
        validation=True,
        sample_count=_VAL_SAMPLE_COUNT,
        version=_FINEWEB_VERSION,
    )


__all__ = ["fineweb_10bt_train", "fineweb_validation"]

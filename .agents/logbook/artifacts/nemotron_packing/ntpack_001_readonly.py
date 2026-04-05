# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read-only packing stats for exp3490b full Nemotron Terminal-Corpus caches.

This script intentionally avoids the experiment module because that import path
is broken in this checkout (`experiments.qwen3_chat_template`), and because we
want a hard no-rebuild guarantee. Every dataset component points directly at a
verified finished token cache and uses `auto_build_caches=False`, so any miss
raises instead of tokenizing.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Final

from experiments.chat_templates.qwen3_chat_template import QWEN_3_CHAT_TEMPLATE
from levanter.data.text import ChatLmDatasetFormat
from levanter.data.text.datasets import (
    DatasetComponent,
    LmDataConfig,
    _get_token_key_for_component,
    dataset_for_component,
)
from levanter.utils import fsspec_utils

logger = logging.getLogger("ntpack_001")

SEQ_LEN: Final[int] = 32768

CACHE_ROOTS: Final[dict[str, str]] = {
    "nvidia/Nemotron-Terminal-Corpus/dataset_adapters": (
        "gs://marin-us-east5/tokenized/dataset_adapters_qwen3_8b_tokenizer-c319d9"
    ),
    "nvidia/Nemotron-Terminal-Corpus/skill_based_easy": (
        "gs://marin-us-east5/tokenized/skill_based_easy_qwen3_8b_tokenizer-a5dd6f"
    ),
    "nvidia/Nemotron-Terminal-Corpus/skill_based_medium": (
        "gs://marin-us-east5/tokenized/skill_based_medium_qwen3_8b_tokenizer-045f3b"
    ),
    "nvidia/Nemotron-Terminal-Corpus/skill_based_mixed": (
        "gs://marin-us-east5/tokenized/skill_based_mixed_qwen3_8b_tokenizer-3d0352"
    ),
}

WEIGHTS: Final[dict[str, float]] = {
    "nvidia/Nemotron-Terminal-Corpus/dataset_adapters": 226313.0,
    "nvidia/Nemotron-Terminal-Corpus/skill_based_easy": 44800.0,
    "nvidia/Nemotron-Terminal-Corpus/skill_based_medium": 89300.0,
    "nvidia/Nemotron-Terminal-Corpus/skill_based_mixed": 5690.0,
}


def build_config() -> LmDataConfig:
    chat_format = ChatLmDatasetFormat(chat_template=QWEN_3_CHAT_TEMPLATE)
    components = {
        name: DatasetComponent(
            source=None,
            cache_dir=cache_root,
            format=chat_format,
        )
        for name, cache_root in CACHE_ROOTS.items()
    }
    return LmDataConfig(
        tokenizer="Qwen/Qwen3-8B",
        cache_dir=None,
        enforce_eos=True,
        auto_build_caches=False,
        shuffle=False,
        block_cross_document_attention=True,
        components=components,
        train_weights=WEIGHTS,
    )


def verify_cache_roots() -> dict[str, str]:
    train_paths: dict[str, str] = {}
    for name, cache_root in CACHE_ROOTS.items():
        train_path = os.path.join(cache_root, "train")
        if not fsspec_utils.exists(train_path):
            raise FileNotFoundError(
                f"Missing cache for {name}: expected existing train cache at {train_path}. "
                "Read-only mode forbids rebuilding tokenization."
            )
        train_paths[name] = train_path
        logger.info("Verified existing cache for %s at %s", name, train_path)
    return train_paths


def count_train_only(config: LmDataConfig, seq_len: int) -> dict:
    """Train-only variant of count_corpus_sizes that skips validation caches."""
    from haliax import Axis

    stats: dict[str, object] = {}
    train_caches = config.build_caches("train")
    Pos = Axis("position", seq_len)

    train_weights = config.train_weights or {name: 1.0 for name in train_caches}
    if isinstance(train_weights, list):
        train_weights = train_weights[0][1]
    total_weight = sum(train_weights.values()) if train_weights else 1.0
    weights = {name: weight / total_weight for name, weight in train_weights.items()}

    for name, cache in train_caches.items():
        prefix = f"train/{name}/"
        component = config.components[name]
        token_key = _get_token_key_for_component(component)
        total_tokens = cache.store.tree[token_key].data_size
        total_docs = cache.store.tree[token_key].num_rows
        train_set = dataset_for_component(
            component,
            Pos,
            cache,
            eos_id=None,
            block_cross_document_attention=config.block_cross_document_attention,
        )
        train_seqs = len(train_set.as_sync_dataset())
        padding_fraction = 1 - (total_tokens / (train_seqs * seq_len))

        stats[f"{prefix}total_tokens"] = total_tokens
        stats[f"{prefix}total_docs"] = total_docs
        stats[f"{prefix}total_seqs"] = train_seqs
        if padding_fraction < 0:
            stats[f"{prefix}truncation_fraction"] = -padding_fraction
        else:
            stats[f"{prefix}padding_fraction"] = padding_fraction
        if name in weights:
            stats[f"{prefix}weight"] = weights[name]

    return stats


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    train_paths = verify_cache_roots()
    config = build_config()
    logger.info("auto_build_caches=%s", config.auto_build_caches)
    logger.info("Counting corpus sizes at seq_len=%s (train only)", SEQ_LEN)
    stats = count_train_only(config, seq_len=SEQ_LEN)

    payload = {
        "experiment_id": "NTPACK-001",
        "seq_len": SEQ_LEN,
        "read_only_cache_inspection": True,
        "auto_build_caches": config.auto_build_caches,
        "train_cache_paths": train_paths,
        "weights": WEIGHTS,
        "stats": stats,
    }

    print("NTPACK_RESULT_BEGIN")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("NTPACK_RESULT_END")


if __name__ == "__main__":
    main()

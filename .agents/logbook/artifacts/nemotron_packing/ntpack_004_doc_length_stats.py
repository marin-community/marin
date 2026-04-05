# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""NTPACK-004: Document token-length distribution per subset.

Reports percentiles, mean, median, max, and fraction of documents exceeding
various length thresholds for each subset of the Nemotron Terminal-Corpus.
"""

from __future__ import annotations

import json
import logging
from typing import Final

import numpy as np

from experiments.chat_templates.qwen3_chat_template import QWEN_3_CHAT_TEMPLATE
from levanter.data.text import ChatLmDatasetFormat
from levanter.data.text.datasets import (
    DatasetComponent,
    LmDataConfig,
    _get_token_key_for_component,
)

logger = logging.getLogger("ntpack_004")

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
    )


def get_doc_lengths(cache, token_key: str) -> np.ndarray:
    store = cache.store.tree[token_key]
    offsets = store.offsets[0 : store.num_rows + 1].read().result().copy()
    offsets[0] = 0
    return offsets[1:] - offsets[:-1]


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = build_config()
    train_caches = config.build_caches("train")
    out: dict[str, dict] = {}

    for name, cache in train_caches.items():
        logger.info("Computing length stats for %s", name)
        component = config.components[name]
        token_key = _get_token_key_for_component(component)
        lengths = get_doc_lengths(cache, token_key)

        out[name] = {
            "docs": len(lengths),
            "total_tokens": int(lengths.sum()),
            "mean": float(lengths.mean()),
            "median": float(np.median(lengths)),
            "min": int(lengths.min()),
            "max": int(lengths.max()),
            "std": float(lengths.std()),
            "pct_gt_1024": float((lengths > 1024).mean()),
            "pct_gt_4096": float((lengths > 4096).mean()),
            "pct_gt_8192": float((lengths > 8192).mean()),
            "pct_gt_16384": float((lengths > 16384).mean()),
            "pct_gt_32768": float((lengths > 32768).mean()),
            "percentiles": {
                "p10": float(np.percentile(lengths, 10)),
                "p25": float(np.percentile(lengths, 25)),
                "p50": float(np.percentile(lengths, 50)),
                "p75": float(np.percentile(lengths, 75)),
                "p90": float(np.percentile(lengths, 90)),
                "p95": float(np.percentile(lengths, 95)),
                "p99": float(np.percentile(lengths, 99)),
                "p99_5": float(np.percentile(lengths, 99.5)),
                "p99_9": float(np.percentile(lengths, 99.9)),
            },
        }

    payload = {
        "experiment_id": "NTPACK-004",
        "results": out,
    }

    print("NTPACK_RESULT_BEGIN")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("NTPACK_RESULT_END")


if __name__ == "__main__":
    main()

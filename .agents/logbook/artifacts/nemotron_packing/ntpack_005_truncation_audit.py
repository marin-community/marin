# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""NTPACK-005: Assistant-token loss under current left truncation at seq_len=32768.

For every document longer than SEQ_LEN, reads the assistant_masks from the token
cache to measure how many supervised tokens are kept (first 32k) vs lost (beyond 32k).
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

logger = logging.getLogger("ntpack_005")

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


def audit_subset(cache, token_key: str, seq_len: int) -> dict:
    ids_store = cache.store.tree[token_key]
    offsets = ids_store.offsets[0 : ids_store.num_rows + 1].read().result().copy()
    offsets[0] = 0
    lengths = offsets[1:] - offsets[:-1]
    long_indices = np.nonzero(lengths > seq_len)[0]

    total_docs = len(lengths)
    long_docs = len(long_indices)

    if long_docs == 0:
        return {
            "total_docs": total_docs,
            "docs_gt_seq_len": 0,
            "assistant_tokens_kept": 0,
            "assistant_tokens_lost": 0,
            "assistant_loss_fraction": 0.0,
            "tail_heavy_docs": 0,
            "tail_heavy_fraction": 0.0,
            "total_tokens_lost": 0,
            "mean_tokens_lost_per_long_doc": 0.0,
        }

    # Check if assistant_masks exists in the cache
    if "assistant_masks" not in cache.store.tree:
        logger.warning("No assistant_masks in cache, falling back to raw token counts only")
        total_tokens_lost = int(sum(max(0, lengths[i] - seq_len) for i in long_indices))
        return {
            "total_docs": total_docs,
            "docs_gt_seq_len": long_docs,
            "assistant_masks_available": False,
            "total_tokens_lost": total_tokens_lost,
            "mean_tokens_lost_per_long_doc": total_tokens_lost / long_docs,
        }

    mask_store = cache.store.tree["assistant_masks"]

    total_kept_assistant = 0
    total_lost_assistant = 0
    tail_heavy_docs = 0

    for idx in long_indices.tolist():
        start = int(offsets[idx])
        end = int(offsets[idx + 1])
        mask = np.asarray(mask_store.data[start:end].read().result())

        kept = int(mask[:seq_len].sum())
        lost = int(mask[seq_len:].sum())
        total_kept_assistant += kept
        total_lost_assistant += lost

        if mask.sum() > 0 and lost > kept:
            tail_heavy_docs += 1

    total_assistant = total_kept_assistant + total_lost_assistant
    total_tokens_lost = int(sum(max(0, lengths[i] - seq_len) for i in long_indices))

    return {
        "total_docs": total_docs,
        "docs_gt_seq_len": long_docs,
        "assistant_tokens_kept": total_kept_assistant,
        "assistant_tokens_lost": total_lost_assistant,
        "assistant_loss_fraction": float(total_lost_assistant / total_assistant) if total_assistant else 0.0,
        "tail_heavy_docs": tail_heavy_docs,
        "tail_heavy_fraction": float(tail_heavy_docs / long_docs) if long_docs else 0.0,
        "total_tokens_lost": total_tokens_lost,
        "mean_tokens_lost_per_long_doc": float(total_tokens_lost / long_docs),
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = build_config()
    train_caches = config.build_caches("train")
    out: dict[str, dict] = {}

    for name, cache in train_caches.items():
        logger.info("Auditing truncation for %s", name)
        component = config.components[name]
        token_key = _get_token_key_for_component(component)
        out[name] = audit_subset(cache, token_key, SEQ_LEN)
        logger.info(
            "  %s: %d long docs, loss_fraction=%.4f",
            name,
            out[name]["docs_gt_seq_len"],
            out[name].get("assistant_loss_fraction", 0.0),
        )

    payload = {
        "experiment_id": "NTPACK-005",
        "seq_len": SEQ_LEN,
        "slice_strategy": "left",
        "results": out,
    }

    print("NTPACK_RESULT_BEGIN")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("NTPACK_RESULT_END")


if __name__ == "__main__":
    main()

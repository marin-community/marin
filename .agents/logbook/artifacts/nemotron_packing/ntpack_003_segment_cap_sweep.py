# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""NTPACK-003: Segment-cap sweep at seq_len=32768 with the current greedy packer.

Sweeps max_segments_per_example over [1, 4, 8, 16, 32, 64, 128] to determine
whether the default cap of 64 is the main source of residual padding.
"""

from __future__ import annotations

import json
import logging
from typing import Final

from haliax import Axis

from experiments.chat_templates.qwen3_chat_template import QWEN_3_CHAT_TEMPLATE
from levanter.data.text import ChatLmDatasetFormat
from levanter.data.text.datasets import (
    DatasetComponent,
    LmDataConfig,
    _get_token_key_for_component,
    dataset_for_component,
)

logger = logging.getLogger("ntpack_003")

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

SEGMENT_CAPS: Final[list[int]] = [1, 4, 8, 16, 32, 64, 128]


def build_config(pack_cap: int | None = None) -> LmDataConfig:
    chat_format = ChatLmDatasetFormat(chat_template=QWEN_3_CHAT_TEMPLATE, pack=pack_cap)
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


def measure_padding(config: LmDataConfig, seq_len: int) -> dict[str, dict[str, object]]:
    train_caches = config.build_caches("train")
    Pos = Axis("position", seq_len)
    result: dict[str, dict[str, object]] = {}

    for name, cache in train_caches.items():
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
        total_seqs = len(train_set.as_sync_dataset())
        padding_fraction = 1 - (total_tokens / (total_seqs * seq_len))
        result[name] = {
            "total_docs": total_docs,
            "total_seqs": total_seqs,
            "total_tokens": total_tokens,
            "padding_fraction": max(padding_fraction, 0.0),
            "truncation_fraction": max(-padding_fraction, 0.0),
        }

    return result


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    out: dict[str, dict] = {}
    for cap in SEGMENT_CAPS:
        logger.info("Measuring padding with max_segments_per_example=%d", cap)
        config = build_config(pack_cap=cap)
        out[str(cap)] = measure_padding(config, SEQ_LEN)

    payload = {
        "experiment_id": "NTPACK-003",
        "seq_len": SEQ_LEN,
        "segment_caps": SEGMENT_CAPS,
        "results": out,
    }

    print("NTPACK_RESULT_BEGIN")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("NTPACK_RESULT_END")


if __name__ == "__main__":
    main()

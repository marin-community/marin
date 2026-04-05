# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""NTPACK-008: Compare packing strategies (greedy vs sorted vs BFD) at seq_len=32768.

Measures padding fraction for each strategy on the full Nemotron Terminal-Corpus.
Uses only existing token caches — no re-tokenization.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Final, Literal

from haliax import Axis

from experiments.chat_templates.qwen3_chat_template import QWEN_3_CHAT_TEMPLATE
from levanter.data.text import ChatLmDatasetFormat
from levanter.data.text.datasets import (
    DatasetComponent,
    LmDataConfig,
    _get_token_key_for_component,
    dataset_for_component,
)

logger = logging.getLogger("ntpack_008")

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


def measure_strategy(
    config: LmDataConfig,
    seq_len: int,
    strategy: Literal["greedy", "sorted", "bfd"],
) -> dict[str, dict]:
    train_caches = config.build_caches("train")
    Pos = Axis("position", seq_len)
    result: dict[str, dict] = {}

    for name, cache in train_caches.items():
        component = config.components[name]
        token_key = _get_token_key_for_component(component)
        total_tokens = cache.store.tree[token_key].data_size
        total_docs = cache.store.tree[token_key].num_rows

        t0 = time.time()
        train_set = dataset_for_component(
            component,
            Pos,
            cache,
            eos_id=None,
            block_cross_document_attention=config.block_cross_document_attention,
            packing_strategy=strategy,
        )
        total_seqs = len(train_set.as_sync_dataset())
        pack_time = time.time() - t0

        padding_fraction = 1 - (total_tokens / (total_seqs * seq_len))
        result[name] = {
            "total_docs": total_docs,
            "total_seqs": total_seqs,
            "total_tokens": total_tokens,
            "padding_fraction": max(padding_fraction, 0.0),
            "truncation_fraction": max(-padding_fraction, 0.0),
            "pack_time_seconds": round(pack_time, 2),
        }

    return result


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = build_config()
    strategies: list[Literal["greedy", "sorted", "bfd"]] = ["greedy", "sorted", "bfd"]
    out: dict[str, dict] = {}

    for strategy in strategies:
        logger.info("Measuring strategy: %s", strategy)
        out[strategy] = measure_strategy(config, SEQ_LEN, strategy)
        logger.info("  Done with %s", strategy)

    payload = {
        "experiment_id": "NTPACK-008",
        "seq_len": SEQ_LEN,
        "strategies": strategies,
        "results": out,
    }

    print("NTPACK_RESULT_BEGIN")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("NTPACK_RESULT_END")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build a deterministic byte-window token store for the JPEG tokenizer baseline."""

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from datasets import load_dataset

from experiments.jpeg_tokenizer.base.data import (
    TokenSequenceRecord,
    TokenStoreMetadata,
    TokenStoreSplitInfo,
    write_token_store,
)
from experiments.jpeg_tokenizer.base.jpeg_codecs import (
    V0_BYTE_WINDOW_CONFIG,
    V0_CANONICAL_JPEG_CONFIG,
    byte_window_vocab_size,
    canonicalize_image,
    encode_jpeg_bytes,
    window_byte_tokens,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="frgfm/imagenette")
    parser.add_argument("--dataset-config", default="320px")
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--max-train-examples", type=int)
    parser.add_argument("--max-validation-examples", type=int)
    parser.add_argument("--window-size", type=int, default=8192)
    parser.add_argument("--stride", type=int, default=8192)
    parser.add_argument("--output-dir", default="artifacts/jpeg_tokenizer/token_store/imagenette_bytes_w8192_v0")
    parser.add_argument("--log-every", type=int, default=500)
    return parser.parse_args()


def _build_split(
    *,
    dataset,
    split_name: str,
    image_column: str,
    window_config,
    log_every: int,
) -> tuple[np.ndarray, list[TokenSequenceRecord], TokenStoreSplitInfo]:
    windows: list[np.ndarray] = []
    records: list[TokenSequenceRecord] = []

    for source_index, example in enumerate(dataset):
        canonical = canonicalize_image(example[image_column])
        byte_tokens = encode_jpeg_bytes(canonical)
        byte_windows = window_byte_tokens(byte_tokens, config=window_config)
        for window_index, window_tokens in enumerate(byte_windows):
            windows.append(window_tokens)
            records.append(
                TokenSequenceRecord(
                    example_id=f"{split_name}:{source_index}:{window_index}",
                    split=split_name,
                    num_tokens=len(window_tokens),
                    checksum=canonical.checksum,
                    source_index=source_index,
                )
            )

        if (source_index + 1) % log_every == 0:
            logger.info(
                "Processed %s %s/%s source images into %s windows",
                split_name,
                source_index + 1,
                len(dataset),
                len(windows),
            )

    token_matrix = np.stack(windows, axis=0).astype(np.int32)
    split_info = TokenStoreSplitInfo(
        num_examples=len(windows),
        seq_len=window_config.window_size,
        tokens_path=f"{split_name}_tokens.npy",
        manifest_path=f"{split_name}_manifest.jsonl",
    )
    return token_matrix, records, split_info


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    output_dir = Path(args.output_dir)

    window_config = replace(V0_BYTE_WINDOW_CONFIG, window_size=args.window_size, stride=args.stride)
    vocab_size = byte_window_vocab_size(window_config)

    split_specs = {
        "train": (args.train_split, args.max_train_examples),
        "validation": (args.validation_split, args.max_validation_examples),
    }
    split_tokens: dict[str, np.ndarray] = {}
    split_records: dict[str, list[TokenSequenceRecord]] = {}
    split_infos: dict[str, TokenStoreSplitInfo] = {}

    for split_name, (hf_split, max_examples) in split_specs.items():
        logger.info("Loading dataset %s config=%s split=%s", args.dataset, args.dataset_config, hf_split)
        dataset = load_dataset(args.dataset, args.dataset_config, split=hf_split)
        if max_examples is not None:
            if max_examples > len(dataset):
                raise ValueError(f"Requested {max_examples} examples from split {hf_split} of size {len(dataset)}")
            dataset = dataset.select(range(max_examples))

        token_matrix, records, split_info = _build_split(
            dataset=dataset,
            split_name=split_name,
            image_column=args.image_column,
            window_config=window_config,
            log_every=args.log_every,
        )
        split_tokens[split_name] = token_matrix
        split_records[split_name] = records
        split_infos[split_name] = split_info
        logger.info(
            "Built %s split with %s source images -> %s windows",
            split_name,
            len(dataset),
            split_info.num_examples,
        )

    metadata = TokenStoreMetadata(
        dataset=args.dataset,
        dataset_config=args.dataset_config,
        image_column=args.image_column,
        vocab_size=vocab_size,
        seq_len=window_config.window_size,
        canonical_config=asdict(V0_CANONICAL_JPEG_CONFIG),
        tokenizer_config=asdict(window_config),
        splits=split_infos,
    )
    write_token_store(output_dir, metadata=metadata, split_tokens=split_tokens, split_records=split_records)
    logger.info("Wrote byte-window token store to %s", output_dir)


if __name__ == "__main__":
    main()

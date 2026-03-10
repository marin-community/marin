#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build a deterministic whole-image scan-payload byte token store for the JPEG tokenizer baseline."""

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict
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
    V0_CANONICAL_JPEG_CONFIG,
    V0_WHOLE_IMAGE_BYTE_CONFIG,
    canonicalize_image,
    encode_jpeg_scan_bytes,
    pad_whole_image_byte_tokens,
    whole_image_byte_length,
    whole_image_byte_vocab_size,
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
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--pad-to-multiple", type=int)
    parser.add_argument("--output-dir", default="artifacts/jpeg_tokenizer/token_store/imagenette_scan_bytes_whole_v0")
    parser.add_argument("--log-every", type=int, default=500)
    return parser.parse_args()


def _load_split(dataset_name: str, dataset_config: str, split_name: str, max_examples: int | None):
    dataset = load_dataset(dataset_name, dataset_config, split=split_name)
    if max_examples is not None:
        if max_examples > len(dataset):
            raise ValueError(f"Requested {max_examples} examples from split {split_name} of size {len(dataset)}")
        dataset = dataset.select(range(max_examples))
    return dataset


def _resolve_seq_len(
    *,
    datasets_by_split: dict[str, object],
    image_column: str,
    explicit_seq_len: int | None,
    pad_to_multiple: int | None,
    log_every: int,
) -> int:
    if explicit_seq_len is not None:
        return explicit_seq_len

    max_length = 0
    for split_name, dataset in datasets_by_split.items():
        for source_index, example in enumerate(dataset):
            canonical = canonicalize_image(example[image_column])
            byte_tokens = encode_jpeg_scan_bytes(canonical)
            max_length = max(max_length, whole_image_byte_length(byte_tokens))
            if (source_index + 1) % log_every == 0:
                logger.info(
                    "Scanned %s %s/%s examples for whole-image scan-byte length (current max=%s)",
                    split_name,
                    source_index + 1,
                    len(dataset),
                    max_length,
                )

    if pad_to_multiple is not None:
        if pad_to_multiple <= 0:
            raise ValueError(f"pad_to_multiple must be positive, got {pad_to_multiple}")
        remainder = max_length % pad_to_multiple
        if remainder:
            max_length += pad_to_multiple - remainder
    return max_length


def _build_split(
    *,
    dataset,
    split_name: str,
    image_column: str,
    seq_len: int,
    log_every: int,
) -> tuple[np.ndarray, list[TokenSequenceRecord], TokenStoreSplitInfo]:
    tokens = np.empty((len(dataset), seq_len), dtype=np.uint16)
    records: list[TokenSequenceRecord] = []

    for source_index, example in enumerate(dataset):
        canonical = canonicalize_image(example[image_column])
        byte_tokens = encode_jpeg_scan_bytes(canonical)
        padded = pad_whole_image_byte_tokens(byte_tokens, seq_len=seq_len)
        tokens[source_index] = padded.astype(np.uint16, copy=False)
        records.append(
            TokenSequenceRecord(
                example_id=f"{split_name}:{source_index}",
                split=split_name,
                num_tokens=whole_image_byte_length(byte_tokens),
                checksum=canonical.checksum,
                source_index=source_index,
            )
        )
        if (source_index + 1) % log_every == 0:
            logger.info("Processed %s %s/%s whole-image scan-byte examples", split_name, source_index + 1, len(dataset))

    split_info = TokenStoreSplitInfo(
        num_examples=len(dataset),
        seq_len=seq_len,
        tokens_path=f"{split_name}_tokens.npy",
        manifest_path=f"{split_name}_manifest.jsonl",
    )
    return tokens, records, split_info


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    output_dir = Path(args.output_dir)

    datasets_by_split = {
        "train": _load_split(args.dataset, args.dataset_config, args.train_split, args.max_train_examples),
        "validation": _load_split(
            args.dataset,
            args.dataset_config,
            args.validation_split,
            args.max_validation_examples,
        ),
    }
    seq_len = _resolve_seq_len(
        datasets_by_split=datasets_by_split,
        image_column=args.image_column,
        explicit_seq_len=args.seq_len,
        pad_to_multiple=args.pad_to_multiple,
        log_every=args.log_every,
    )
    logger.info("Resolved whole-image scan-byte seq_len=%s", seq_len)

    split_tokens: dict[str, np.ndarray] = {}
    split_records: dict[str, list[TokenSequenceRecord]] = {}
    split_infos: dict[str, TokenStoreSplitInfo] = {}
    for split_name, dataset in datasets_by_split.items():
        token_matrix, records, split_info = _build_split(
            dataset=dataset,
            split_name=split_name,
            image_column=args.image_column,
            seq_len=seq_len,
            log_every=args.log_every,
        )
        split_tokens[split_name] = token_matrix
        split_records[split_name] = records
        split_infos[split_name] = split_info

    tokenizer_config = asdict(V0_WHOLE_IMAGE_BYTE_CONFIG)
    tokenizer_config["byte_source"] = "scan_payload"
    tokenizer_config["loss_mask_ignore_id"] = V0_WHOLE_IMAGE_BYTE_CONFIG.pad_token_id
    metadata = TokenStoreMetadata(
        dataset=args.dataset,
        dataset_config=args.dataset_config,
        image_column=args.image_column,
        vocab_size=whole_image_byte_vocab_size(V0_WHOLE_IMAGE_BYTE_CONFIG),
        seq_len=seq_len,
        canonical_config=asdict(V0_CANONICAL_JPEG_CONFIG),
        tokenizer_config=tokenizer_config,
        splits=split_infos,
    )
    write_token_store(output_dir, metadata=metadata, split_tokens=split_tokens, split_records=split_records)
    logger.info("Wrote whole-image scan-byte token store to %s", output_dir)


if __name__ == "__main__":
    main()

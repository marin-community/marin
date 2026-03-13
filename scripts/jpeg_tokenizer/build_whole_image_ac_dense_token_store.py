#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build a fixed-length DC-delta + dense-AC token store for whole-image JPEG modeling."""

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
    V0_AC_DENSE_CONFIG,
    V0_CANONICAL_JPEG_CONFIG,
    CoefficientTokenSource,
    ac_dense_vocab_size,
    canonicalize_image,
    encode_jpeg_ac_dense_tokens,
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
    parser.add_argument(
        "--source",
        choices=[source.value for source in CoefficientTokenSource],
        default=CoefficientTokenSource.LIBJPEG.value,
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/jpeg_tokenizer/token_store/imagenette_ac_dense_whole_libjpeg_v0",
    )
    parser.add_argument("--log-every", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    output_dir = Path(args.output_dir)

    tokenizer_config = replace(V0_AC_DENSE_CONFIG, source=CoefficientTokenSource(args.source))
    blocks_per_axis = V0_CANONICAL_JPEG_CONFIG.resolution // 8
    seq_len = blocks_per_axis * blocks_per_axis * 64
    vocab_size = ac_dense_vocab_size(tokenizer_config)

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

        num_examples = len(dataset)
        tokens = np.empty((num_examples, seq_len), dtype=np.int32)
        records: list[TokenSequenceRecord] = []

        for index, example in enumerate(dataset):
            canonical = canonicalize_image(example[args.image_column])
            token_ids = encode_jpeg_ac_dense_tokens(canonical, config=tokenizer_config)
            if len(token_ids) != seq_len:
                raise ValueError(f"Expected seq_len={seq_len} for {split_name} example {index}, got {len(token_ids)}")
            tokens[index] = token_ids
            records.append(
                TokenSequenceRecord(
                    example_id=f"{split_name}:{index}",
                    split=split_name,
                    num_tokens=seq_len,
                    checksum=canonical.checksum,
                    source_index=index,
                )
            )
            if (index + 1) % args.log_every == 0:
                logger.info("Processed %s %s/%s AC-dense examples", split_name, index + 1, num_examples)

        split_tokens[split_name] = tokens
        split_records[split_name] = records
        split_infos[split_name] = TokenStoreSplitInfo(
            num_examples=num_examples,
            seq_len=seq_len,
            tokens_path=f"{split_name}_tokens.npy",
            manifest_path=f"{split_name}_manifest.jsonl",
        )

    metadata = TokenStoreMetadata(
        dataset=args.dataset,
        dataset_config=args.dataset_config,
        image_column=args.image_column,
        vocab_size=vocab_size,
        seq_len=seq_len,
        canonical_config=asdict(V0_CANONICAL_JPEG_CONFIG),
        tokenizer_config=asdict(tokenizer_config),
        splits=split_infos,
    )
    write_token_store(output_dir, metadata=metadata, split_tokens=split_tokens, split_records=split_records)
    logger.info("Wrote whole-image AC-dense token store to %s", output_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inspect JPEG-tokenizer representations on a small clean image corpus.

This Phase 0 script canonicalizes a sample of images, emits the three initial
token families, and writes summary stats to disk so the V0 decisions can be
revisited with concrete numbers.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from datasets import load_dataset

from experiments.jpeg_tokenizer.base.eval import compute_token_sequence_stats
from experiments.jpeg_tokenizer.base.jpeg_codecs import (
    V0_CANONICAL_JPEG_CONFIG,
    V0_COEFFICIENT_CONFIG,
    V0_SYMBOL_CONFIG,
    canonicalize_image,
    coefficient_vocab_size,
    encode_dct_coeffs,
    encode_jpeg_bytes,
    encode_jpeg_symbols,
    symbol_vocab_size,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="frgfm/imagenette")
    parser.add_argument("--dataset-config", default="320px")
    parser.add_argument("--split", default="train")
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--max-examples", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="artifacts/jpeg_tokenizer/phase0/imagenette")
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    output_dir = Path(args.output_dir)
    stats_dir = output_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset %s config=%s split=%s", args.dataset, args.dataset_config, args.split)
    dataset = load_dataset(args.dataset, args.dataset_config, split=args.split)
    if args.max_examples > len(dataset):
        raise ValueError(f"Requested {args.max_examples} examples from split of size {len(dataset)}")
    dataset = dataset.shuffle(seed=args.seed).select(range(args.max_examples))

    bytes_sequences = []
    symbols_sequences = []
    coeff_sequences = []
    checksum_verified_examples = 0

    for index, example in enumerate(dataset):
        image = example[args.image_column]
        canonical = canonicalize_image(image)
        repeated = canonicalize_image(image)
        if canonical.jpeg_bytes != repeated.jpeg_bytes:
            raise ValueError(f"Canonical JPEG mismatch for example index {index}")
        checksum_verified_examples += 1

        bytes_sequences.append(encode_jpeg_bytes(canonical))
        symbols_sequences.append(encode_jpeg_symbols(canonical))
        coeff_sequences.append(encode_dct_coeffs(canonical))

        if (index + 1) % args.log_every == 0:
            logger.info("Processed %s/%s examples", index + 1, args.max_examples)

    family_records = {
        "bytes": {
            **compute_token_sequence_stats(bytes_sequences).to_dict(),
            "configured_vocab_size": 256,
        },
        "symbols": {
            **compute_token_sequence_stats(symbols_sequences).to_dict(),
            "configured_vocab_size": symbol_vocab_size(),
        },
        "coeffs": {
            **compute_token_sequence_stats(coeff_sequences).to_dict(),
            "configured_vocab_size": coefficient_vocab_size(),
        },
    }

    for family_name, record in family_records.items():
        with (stats_dir / f"{family_name}.json").open("w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, sort_keys=True)
            handle.write("\n")

    summary_lines = [
        "# JPEG Tokenizer Phase 0 Summary",
        "",
        f"- Dataset: `{args.dataset}`",
        f"- Dataset config: `{args.dataset_config}`",
        f"- Split: `{args.split}`",
        f"- Examples: `{args.max_examples}`",
        f"- Canonical image config: `{V0_CANONICAL_JPEG_CONFIG}`",
        f"- Coefficient config: `{V0_COEFFICIENT_CONFIG}`",
        f"- Symbol config: `{V0_SYMBOL_CONFIG}`",
        f"- Deterministic canonicalization checks: `{checksum_verified_examples}` examples passed",
        "",
        "## Family Stats",
        "",
    ]
    for family_name, record in family_records.items():
        summary_lines.append(
            f"- `{family_name}`: mean_length={record['mean_length']:.2f}, "
            f"median_length={record['median_length']:.2f}, "
            f"p90_length={record['p90_length']:.2f}, "
            f"p95_length={record['p95_length']:.2f}, "
            f"p99_length={record['p99_length']:.2f}, "
            f"min_length={record['min_length']}, max_length={record['max_length']}, "
            f"observed_unique_tokens={record['unique_tokens']}, configured_vocab_size={record['configured_vocab_size']}"
        )

    with (output_dir / "summary.md").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(summary_lines) + "\n")

    logger.info("Wrote stats to %s", output_dir)


if __name__ == "__main__":
    main()

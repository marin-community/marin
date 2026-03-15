#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure local-token context dependence across JPEG-derived tokenizers."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from datasets import load_dataset
from PIL import Image

from experiments.jpeg_tokenizer.base.eval import summarize_metric
from experiments.jpeg_tokenizer.base.jpeg_codecs import (
    V0_AC_DENSE_CONFIG,
    V0_CANONICAL_JPEG_CONFIG,
    V0_COEFFICIENT_CONFIG,
    V0_HUFFMAN_EVENT_CONFIG,
    V0_SYMBOL_CONFIG,
    CoefficientTokenSource,
    CoefficientTokenizerConfig,
    HuffmanEventTokenizerConfig,
    SymbolTokenizerConfig,
    canonicalize_image,
    encode_dct_coeffs,
    encode_jpeg_ac_dense_tokens,
    quantized_luma_blocks,
)

logger = logging.getLogger(__name__)

_ZIGZAG_ORDER = np.asarray(
    [
        0,
        1,
        8,
        16,
        9,
        2,
        3,
        10,
        17,
        24,
        32,
        25,
        18,
        11,
        4,
        5,
        12,
        19,
        26,
        33,
        40,
        48,
        41,
        34,
        27,
        20,
        13,
        6,
        7,
        14,
        21,
        28,
        35,
        42,
        49,
        56,
        57,
        50,
        43,
        36,
        29,
        22,
        15,
        23,
        30,
        37,
        44,
        51,
        58,
        59,
        52,
        45,
        38,
        31,
        39,
        46,
        53,
        60,
        61,
        54,
        47,
        55,
        62,
        63,
    ],
    dtype=np.int32,
)


@dataclass(frozen=True)
class BlockContextMetrics:
    """Context-dependence metrics for one representation."""

    flip_rates: list[float]
    base_block_lengths: list[int]
    mixed_block_lengths: list[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="frgfm/imagenette")
    parser.add_argument("--dataset-config", default="320px")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--max-examples", type=int, default=1024)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pair-offset", type=int, default=1)
    parser.add_argument("--target-block-row", type=int)
    parser.add_argument("--target-block-col", type=int)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--output-dir",
        default="artifacts/jpeg_tokenizer/analysis/context_dependence",
    )
    return parser.parse_args()


def _symbol_block_tokens(
    *,
    canonical,
    config: SymbolTokenizerConfig,
) -> list[np.ndarray]:
    """Encode one JPEG symbol stream split per block."""

    dc_offset = 2
    max_category = config.ac_bound.bit_length()
    ac_offsets = _ac_symbol_offsets(max_category)
    eob_token = 0
    zrl_token = 1

    blocks = quantized_luma_blocks(
        canonical,
        config=CoefficientTokenizerConfig(quality=config.quality, source=config.source),
    )
    zigzag = blocks.reshape(-1, 64)[:, _ZIGZAG_ORDER]
    previous_dc = 0
    per_block: list[np.ndarray] = []

    for block in zigzag:
        tokens: list[int] = []
        dc_value = int(block[0])
        dc_delta = dc_value - previous_dc
        previous_dc = dc_value
        tokens.append(dc_offset + _encode_bounded_value(dc_delta, config.dc_bound, "dc_delta"))

        zero_run = 0
        for coeff in block[1:]:
            coeff_value = int(coeff)
            if coeff_value == 0:
                zero_run += 1
                if zero_run == 16:
                    tokens.append(zrl_token)
                    zero_run = 0
                continue
            magnitude_category = abs(coeff_value).bit_length()
            if magnitude_category == 0 or magnitude_category > max_category:
                raise ValueError(f"AC magnitude category {magnitude_category} exceeds configured bounds")
            token = (
                2
                + (2 * config.dc_bound + 1)
                + ac_offsets[(zero_run, magnitude_category)]
                + _encode_category_value(coeff_value, magnitude_category, "ac_coefficient")
            )
            tokens.append(token)
            zero_run = 0

        if zero_run > 0:
            tokens.append(eob_token)

        per_block.append(np.asarray(tokens, dtype=np.int32))

    return per_block


def _huffman_block_tokens(
    *,
    canonical,
    config: HuffmanEventTokenizerConfig,
) -> list[np.ndarray]:
    """Encode one huffman-event stream split per block."""

    eob_token = 0
    zrl_token = 1
    max_dc_category = config.dc_bound.bit_length()
    max_ac_category = config.ac_bound.bit_length()
    dc_event_offset = 2
    ac_event_offset = dc_event_offset + (max_dc_category + 1)
    amplitude_offset = ac_event_offset + (16 * max_ac_category)

    blocks = quantized_luma_blocks(
        canonical,
        config=CoefficientTokenizerConfig(quality=config.quality, source=config.source),
    )
    zigzag = blocks.reshape(-1, 64)[:, _ZIGZAG_ORDER]
    previous_dc = 0
    per_block: list[np.ndarray] = []

    for block in zigzag:
        tokens: list[int] = []
        dc_value = int(block[0])
        dc_delta = dc_value - previous_dc
        previous_dc = dc_value

        dc_category = abs(dc_delta).bit_length()
        if dc_category > max_dc_category:
            raise ValueError(f"DC category {dc_category} exceeds configured maximum {max_dc_category}")
        tokens.append(dc_event_offset + dc_category)
        if dc_category > 0:
            tokens.append(amplitude_offset + _encode_category_value(dc_delta, dc_category, "dc_delta"))

        zero_run = 0
        for coeff in block[1:]:
            coeff_value = int(coeff)
            if coeff_value == 0:
                zero_run += 1
                if zero_run == 16:
                    tokens.append(zrl_token)
                    zero_run = 0
                continue

            magnitude_category = abs(coeff_value).bit_length()
            if magnitude_category == 0 or magnitude_category > max_ac_category:
                raise ValueError(f"AC magnitude category {magnitude_category} exceeds configured bounds")

            tokens.append(ac_event_offset + zero_run * max_ac_category + (magnitude_category - 1))
            tokens.append(amplitude_offset + _encode_category_value(coeff_value, magnitude_category, "ac_coefficient"))
            zero_run = 0

        if zero_run > 0:
            tokens.append(eob_token)

        per_block.append(np.asarray(tokens, dtype=np.int32))

    return per_block


def _ac_symbol_offsets(max_category: int) -> dict[tuple[int, int], int]:
    offsets: dict[tuple[int, int], int] = {}
    cursor = 0
    for run in range(16):
        for category in range(1, max_category + 1):
            offsets[(run, category)] = cursor
            cursor += 1 << category
    return offsets


def _encode_bounded_value(value: int, bound: int, name: str) -> int:
    if abs(value) > bound:
        raise ValueError(f"{name}={value} exceeds configured bound +/-{bound}")
    return value + bound


def _encode_category_value(value: int, category: int, name: str) -> int:
    if category <= 0:
        raise ValueError(f"{name} category must be positive, got {category}")
    if value == 0 or abs(value).bit_length() != category:
        raise ValueError(f"{name}={value} is not valid for category={category}")
    if value < 0:
        return value + (1 << category) - 1
    return (1 << (category - 1)) + value - (1 << (category - 1))


def _normalized_edit_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Return Levenshtein distance normalized by max sequence length."""

    n, m = len(a), len(b)
    if n == 0 and m == 0:
        return 0.0
    if n == 0 or m == 0:
        return 1.0

    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        curr[0] = i
        ai = int(a[i - 1])
        for j in range(1, m + 1):
            substitution_cost = 0 if ai == int(b[j - 1]) else 1
            curr[j] = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + substitution_cost,
            )
        prev, curr = curr, prev

    return float(prev[m] / max(n, m))


def _hamming_rate(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) != len(b):
        raise ValueError(f"Expected equal-length arrays, got {len(a)} and {len(b)}")
    return float(np.mean(np.asarray(a, dtype=np.int32) != np.asarray(b, dtype=np.int32)))


def _block_index(row: int, col: int, *, blocks_per_row: int) -> int:
    return row * blocks_per_row + col


def _mixed_canonical(base_luma: np.ndarray, context_luma: np.ndarray, *, row: int, col: int):
    mixed = np.asarray(context_luma, dtype=np.uint8).copy()
    r0 = row * 8
    c0 = col * 8
    mixed[r0 : r0 + 8, c0 : c0 + 8] = np.asarray(base_luma, dtype=np.uint8)[r0 : r0 + 8, c0 : c0 + 8]
    return canonicalize_image(Image.fromarray(mixed))


def _metrics_payload(metrics: BlockContextMetrics) -> dict[str, object]:
    flip = np.asarray(metrics.flip_rates, dtype=np.float64)
    base_lengths = np.asarray(metrics.base_block_lengths, dtype=np.float64)
    mixed_lengths = np.asarray(metrics.mixed_block_lengths, dtype=np.float64)
    return {
        "num_examples": len(flip),
        "flip_rate": summarize_metric(flip.tolist()).to_dict(),
        "exact_match_rate": float(np.mean(flip == 0.0)),
        "base_block_tokens": summarize_metric(base_lengths.tolist()).to_dict(),
        "mixed_block_tokens": summarize_metric(mixed_lengths.tolist()).to_dict(),
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_summary(path: Path, payload: dict[str, object]) -> None:
    lines = [
        "# JPEG Context Dependence",
        "",
        "Local block kept fixed, surrounding context replaced from a paired image.",
        "",
        f"- Dataset: `{payload['dataset']}` ({payload['dataset_config']})",
        f"- Split: `{payload['split']}`",
        f"- Num pairs: `{payload['num_examples']}`",
        f"- Target block: row `{payload['target_block_row']}`, col `{payload['target_block_col']}`",
        "",
        "## Representation Scores",
        "",
        "| Representation | Mean flip | P95 flip | Exact match rate |",
        "| --- | ---: | ---: | ---: |",
    ]
    results = payload["results"]
    for key in ("coeff_k64", "ac_dense", "symbols", "huffman_events"):
        metric = results[key]
        mean_flip = metric["flip_rate"]["mean"]
        p95_flip = metric["flip_rate"]["p95"]
        exact_match = metric["exact_match_rate"]
        lines.append(f"| `{key}` | {mean_flip:.4f} | {p95_flip:.4f} | {exact_match:.4f} |")
    control = payload["controls"]["coeff_absolute_control"]
    lines.extend(
        [
            "",
            "## Control",
            "",
            f"- `coeff_absolute_control` mean flip: {control['flip_rate']['mean']:.6f}",
            f"- `coeff_absolute_control` exact match rate: {control['exact_match_rate']:.6f}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.max_examples <= 1:
        raise ValueError(f"max-examples must be >1, got {args.max_examples}")
    if args.pair_offset <= 0:
        raise ValueError(f"pair-offset must be positive, got {args.pair_offset}")

    dataset = load_dataset(args.dataset, args.dataset_config, split=args.split)
    if args.shuffle:
        dataset = dataset.shuffle(seed=args.seed)
    if args.max_examples > len(dataset):
        raise ValueError(f"Requested {args.max_examples} examples from split of size {len(dataset)}")
    dataset = dataset.select(range(args.max_examples))

    coeff_config = dataclasses.replace(
        V0_COEFFICIENT_CONFIG,
        zigzag_coefficients=64,
        source=CoefficientTokenSource.LIBJPEG,
    )
    ac_dense_config = dataclasses.replace(V0_AC_DENSE_CONFIG, source=CoefficientTokenSource.LIBJPEG)
    symbol_config = dataclasses.replace(V0_SYMBOL_CONFIG, source=CoefficientTokenSource.LIBJPEG)
    huffman_config = dataclasses.replace(V0_HUFFMAN_EVENT_CONFIG, source=CoefficientTokenSource.LIBJPEG)

    blocks_per_axis = V0_CANONICAL_JPEG_CONFIG.resolution // coeff_config.block_size
    target_row = args.target_block_row if args.target_block_row is not None else blocks_per_axis // 2
    target_col = args.target_block_col if args.target_block_col is not None else blocks_per_axis // 2
    if not (0 <= target_row < blocks_per_axis and 0 <= target_col < blocks_per_axis):
        raise ValueError(
            f"Target block ({target_row}, {target_col}) is out of range for {blocks_per_axis}x{blocks_per_axis}"
        )
    target_block_index = _block_index(target_row, target_col, blocks_per_row=blocks_per_axis)

    coeff_metrics = BlockContextMetrics([], [], [])
    ac_dense_metrics = BlockContextMetrics([], [], [])
    symbol_metrics = BlockContextMetrics([], [], [])
    huffman_metrics = BlockContextMetrics([], [], [])
    coeff_control_metrics = BlockContextMetrics([], [], [])

    num_examples = len(dataset)
    for i in range(num_examples):
        base_example = dataset[i]
        context_example = dataset[(i + args.pair_offset) % num_examples]
        base_canonical = canonicalize_image(base_example[args.image_column])
        context_canonical = canonicalize_image(context_example[args.image_column])
        mixed_canonical = _mixed_canonical(
            base_canonical.luma_plane,
            context_canonical.luma_plane,
            row=target_row,
            col=target_col,
        )

        base_coeff = encode_dct_coeffs(base_canonical, config=coeff_config)
        mixed_coeff = encode_dct_coeffs(mixed_canonical, config=coeff_config)
        coeff_start = target_block_index * 64
        coeff_end = coeff_start + 64
        base_coeff_block = base_coeff[coeff_start:coeff_end]
        mixed_coeff_block = mixed_coeff[coeff_start:coeff_end]
        coeff_metrics.flip_rates.append(_hamming_rate(base_coeff_block, mixed_coeff_block))
        coeff_metrics.base_block_lengths.append(len(base_coeff_block))
        coeff_metrics.mixed_block_lengths.append(len(mixed_coeff_block))

        coeff_control_metrics.flip_rates.append(_hamming_rate(base_coeff_block, mixed_coeff_block))
        coeff_control_metrics.base_block_lengths.append(len(base_coeff_block))
        coeff_control_metrics.mixed_block_lengths.append(len(mixed_coeff_block))

        base_ac_dense = encode_jpeg_ac_dense_tokens(base_canonical, config=ac_dense_config)
        mixed_ac_dense = encode_jpeg_ac_dense_tokens(mixed_canonical, config=ac_dense_config)
        ac_start = target_block_index * 64
        ac_end = ac_start + 64
        base_ac_dense_block = base_ac_dense[ac_start:ac_end]
        mixed_ac_dense_block = mixed_ac_dense[ac_start:ac_end]
        ac_dense_metrics.flip_rates.append(_hamming_rate(base_ac_dense_block, mixed_ac_dense_block))
        ac_dense_metrics.base_block_lengths.append(len(base_ac_dense_block))
        ac_dense_metrics.mixed_block_lengths.append(len(mixed_ac_dense_block))

        base_symbol_blocks = _symbol_block_tokens(canonical=base_canonical, config=symbol_config)
        mixed_symbol_blocks = _symbol_block_tokens(canonical=mixed_canonical, config=symbol_config)
        base_symbol_block = base_symbol_blocks[target_block_index]
        mixed_symbol_block = mixed_symbol_blocks[target_block_index]
        symbol_metrics.flip_rates.append(_normalized_edit_distance(base_symbol_block, mixed_symbol_block))
        symbol_metrics.base_block_lengths.append(len(base_symbol_block))
        symbol_metrics.mixed_block_lengths.append(len(mixed_symbol_block))

        base_huffman_blocks = _huffman_block_tokens(canonical=base_canonical, config=huffman_config)
        mixed_huffman_blocks = _huffman_block_tokens(canonical=mixed_canonical, config=huffman_config)
        base_huffman_block = base_huffman_blocks[target_block_index]
        mixed_huffman_block = mixed_huffman_blocks[target_block_index]
        huffman_metrics.flip_rates.append(_normalized_edit_distance(base_huffman_block, mixed_huffman_block))
        huffman_metrics.base_block_lengths.append(len(base_huffman_block))
        huffman_metrics.mixed_block_lengths.append(len(mixed_huffman_block))

        if (i + 1) % args.log_every == 0:
            logger.info("Processed %s/%s context pairs", i + 1, num_examples)

    output_dir = Path(args.output_dir)
    payload = {
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "num_examples": num_examples,
        "pair_offset": args.pair_offset,
        "target_block_row": target_row,
        "target_block_col": target_col,
        "results": {
            "coeff_k64": _metrics_payload(coeff_metrics),
            "ac_dense": _metrics_payload(ac_dense_metrics),
            "symbols": _metrics_payload(symbol_metrics),
            "huffman_events": _metrics_payload(huffman_metrics),
        },
        "controls": {
            "coeff_absolute_control": _metrics_payload(coeff_control_metrics),
        },
    }
    _write_json(output_dir / "context_dependence.json", payload)
    _write_summary(output_dir / "summary.md", payload)
    logger.info("Wrote context dependence outputs to %s", output_dir)


if __name__ == "__main__":
    main()

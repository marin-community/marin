#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure how lossy the coefficient tokenizer is using SSIM as a perceptual proxy."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path

from datasets import load_dataset

from experiments.jpeg_tokenizer.base.eval import compute_reconstruction_metrics, summarize_metric
from experiments.jpeg_tokenizer.base.tokenizers import (
    V0_CANONICAL_JPEG_CONFIG,
    V0_COEFFICIENT_CONFIG,
    canonicalize_image,
    encode_dct_coeffs,
    reconstruct_luma_from_coeff_tokens,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="frgfm/imagenette")
    parser.add_argument("--dataset-config", default="320px")
    parser.add_argument("--split", default="train")
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--max-examples", type=int, default=9469)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k-values", default="4,8,16,64")
    parser.add_argument("--output-dir", default="artifacts/jpeg_tokenizer/phase0/coeff_reconstruction")
    parser.add_argument("--log-every", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    k_values = [int(value) for value in args.k_values.split(",") if value]
    logger.info("Loading dataset %s config=%s split=%s", args.dataset, args.dataset_config, args.split)
    dataset = load_dataset(args.dataset, args.dataset_config, split=args.split)
    if args.max_examples > len(dataset):
        raise ValueError(f"Requested {args.max_examples} examples from split of size {len(dataset)}")
    dataset = dataset.shuffle(seed=args.seed).select(range(args.max_examples))

    metric_lists: dict[int, dict[str, list[float]]] = {
        k: {"mse": [], "psnr": [], "ssim": [], "dssim": []} for k in k_values
    }

    for index, example in enumerate(dataset):
        canonical = canonicalize_image(example[args.image_column])
        for k in k_values:
            coeff_config = replace(V0_COEFFICIENT_CONFIG, zigzag_coefficients=k)
            coeff_tokens = encode_dct_coeffs(canonical, config=coeff_config)
            reconstructed = reconstruct_luma_from_coeff_tokens(
                coeff_tokens,
                canonical.luma_plane.shape,
                config=coeff_config,
            )
            metrics = compute_reconstruction_metrics(canonical.luma_plane, reconstructed)
            metric_lists[k]["mse"].append(metrics.mse)
            metric_lists[k]["psnr"].append(metrics.psnr)
            metric_lists[k]["ssim"].append(metrics.ssim)
            metric_lists[k]["dssim"].append(metrics.dssim)

        if (index + 1) % args.log_every == 0:
            logger.info("Processed %s/%s examples", index + 1, args.max_examples)

    summary: dict[str, object] = {
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "examples": args.max_examples,
        "canonical_config": repr(V0_CANONICAL_JPEG_CONFIG),
        "base_coefficient_config": repr(V0_COEFFICIENT_CONFIG),
        "k_values": k_values,
        "results": {},
    }

    markdown_lines = [
        "# Coefficient Reconstruction Evaluation",
        "",
        f"- Dataset: `{args.dataset}`",
        f"- Dataset config: `{args.dataset_config}`",
        f"- Split: `{args.split}`",
        f"- Examples: `{args.max_examples}`",
        f"- Canonical config: `{V0_CANONICAL_JPEG_CONFIG}`",
        f"- Base coefficient config: `{V0_COEFFICIENT_CONFIG}`",
        "",
        "## Results",
        "",
    ]

    for k in k_values:
        aggregated = {name: summarize_metric(values).to_dict() for name, values in metric_lists[k].items()}
        summary["results"][str(k)] = aggregated
        markdown_lines.append(
            f"- `K={k}`: SSIM mean={aggregated['ssim']['mean']:.4f}, "
            f"p95 DSSIM={aggregated['dssim']['p95']:.4f}, "
            f"PSNR mean={aggregated['psnr']['mean']:.2f} dB, "
            f"MSE mean={aggregated['mse']['mean']:.2f}"
        )

    with (output_dir / "coeff_reconstruction.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    with (output_dir / "coeff_reconstruction.md").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(markdown_lines) + "\n")

    logger.info("Wrote reconstruction metrics to %s", output_dir)


if __name__ == "__main__":
    main()

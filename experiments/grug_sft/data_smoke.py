# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check the realized SFT mixture and sample padding from its remote stores."""

import argparse
import json
import math
import time
from collections import Counter, defaultdict

import jax
import numpy as np
from haliax import Axis
from levanter.data.mixture import MixtureDataset
from rigging.filesystem.storage_path import StoragePath

from experiments.grug_sft.special_token_lr import BATCH, CONTEXT, DEFAULT_STEPS, data_config

DEFAULT_SAMPLE_SIZE = 64
MAX_PADDING_FRACTION = 0.07
POLL_INTERVAL = 60


def _wait_for_manifest(path: str, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while not StoragePath(path).exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for {path}")
        time.sleep(POLL_INTERVAL)


def _padding_tokens(example) -> tuple[int, int]:
    segment_ids = example.attn_mask.segment_ids
    if segment_ids is None:
        return 0, int(example.tokens.size)
    query_segment_ids = np.asarray(segment_ids[0])
    return int(np.count_nonzero(query_segment_ids < 0)), int(query_segment_ids.size)


def smoke(stores_manifest: str, output_path: str, sample_size: int) -> dict:
    if sample_size <= 0:
        raise ValueError("Sample size must be positive")

    config, _ = data_config(DEFAULT_STEPS, stores_manifest)
    mix_key, shuffle_key = jax.random.split(jax.random.PRNGKey(0))
    datasets = config.train_sets(Axis("position", CONTEXT), key=shuffle_key, initial_batch_size=BATCH)
    mixture = MixtureDataset(
        datasets=datasets,
        weights=config.train_weights,
        stop_strategy=config.stop_strategy,
        key=mix_key,
        block_size=config.mixture_block_size,
    )
    if not isinstance(config.train_weights, dict):
        raise ValueError("The SFT smoke test requires a fixed mixture")

    block = np.asarray(mixture._get_block(0))
    component_counts = np.bincount(block >> 16, minlength=len(mixture.dataset_index))
    realized_rates = {
        name: int(component_counts[index]) / mixture.block_size for index, name in enumerate(mixture.dataset_index)
    }
    expected_rates = {name: float(config.train_weights[name]) for name in mixture.dataset_index}
    deviations = {name: realized_rates[name] - expected_rates[name] for name in mixture.dataset_index}
    max_rate_error = max(abs(value) for value in deviations.values())
    rounding_tolerance = (len(mixture.dataset_index) + 1) / mixture.block_size
    if max_rate_error > rounding_tolerance:
        raise ValueError(f"Mixture sampling differs from requested weights by {max_rate_error:.6f}")
    zero_components = [name for name, count in zip(mixture.dataset_index, component_counts, strict=True) if count == 0]
    if zero_components:
        raise ValueError(f"Mixture components receive no samples: {zero_components}")

    rng = np.random.default_rng(0)
    indices = sorted(int(index) for index in rng.choice(mixture.block_size, size=sample_size, replace=False))
    examples = mixture.as_sync_dataset().get_batch(indices)
    sampled_components: Counter[str] = Counter()
    padding_by_component: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for index, example in zip(indices, examples, strict=True):
        component = mixture.dataset_index[int(block[index]) >> 16]
        sampled_components[component] += 1
        padding, tokens = _padding_tokens(example)
        padding_by_component[component][0] += padding
        padding_by_component[component][1] += tokens

    padding_tokens = sum(counts[0] for counts in padding_by_component.values())
    sampled_tokens = sum(counts[1] for counts in padding_by_component.values())
    padding_fraction = padding_tokens / sampled_tokens
    if padding_fraction > MAX_PADDING_FRACTION:
        raise ValueError(f"Sampled padding fraction {padding_fraction:.4%} exceeds {MAX_PADDING_FRACTION:.2%}")

    report = {
        "stores_manifest": stores_manifest,
        "steps": DEFAULT_STEPS,
        "batch_size": BATCH,
        "context_length": CONTEXT,
        "mixture_block_size": mixture.block_size,
        "component_count": len(mixture.dataset_index),
        "max_rate_error": max_rate_error,
        "rounding_tolerance": rounding_tolerance,
        "expected_rates": expected_rates,
        "realized_rates": realized_rates,
        "sample_size": sample_size,
        "sampled_components": dict(sorted(sampled_components.items())),
        "padding_fraction": padding_fraction,
        "padding_by_component": {
            name: {
                "examples": sampled_components[name],
                "padding_fraction": counts[0] / counts[1],
            }
            for name, counts in sorted(padding_by_component.items())
        },
    }
    if not math.isclose(sum(expected_rates.values()), 1.0):
        raise ValueError("Mixture weights do not sum to one")
    StoragePath(output_path).write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stores-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--wait-timeout", type=int, default=0)
    args = parser.parse_args()
    if args.wait_timeout:
        _wait_for_manifest(args.stores_manifest, args.wait_timeout)
    report = smoke(args.stores_manifest, args.output, args.sample_size)
    print(json.dumps({key: report[key] for key in ("component_count", "max_rate_error", "padding_fraction")}))


if __name__ == "__main__":
    main()

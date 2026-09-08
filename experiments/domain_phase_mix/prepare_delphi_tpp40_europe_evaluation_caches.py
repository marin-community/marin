# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize the Paloma and Uncheatable caches used by Europe TPP40 runs."""

from __future__ import annotations

import argparse
import os

from fray.cluster import ResourceConfig
from marin.execution.lazy import ArtifactStep, run
from rigging.filesystem import marin_prefix_for_region

from experiments.domain_phase_mix import launch_delphi_augmented_swarm_3e18 as base
from experiments.llama import llama3_tokenizer

SUPPORTED_REGIONS = ("us-east5", "europe-west4")


def evaluation_steps(*, region: str = "europe-west4") -> dict[str, ArtifactStep]:
    """Return the fixed region-local validation cache graph."""
    if region not in SUPPORTED_REGIONS:
        raise ValueError(f"Unsupported evaluation-cache region {region!r}")
    expected_prefix = marin_prefix_for_region(region)
    current_prefix = os.environ.get("MARIN_PREFIX")
    if current_prefix != expected_prefix:
        raise ValueError(f"MARIN_PREFIX must be {expected_prefix!r}, got {current_prefix!r}")
    return base._default_validation_sets(
        tokenizer=llama3_tokenizer,
        resources=ResourceConfig(regions=[region]),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", choices=SUPPORTED_REGIONS, default="europe-west4")
    parser.add_argument("--max-concurrent", type=int, default=12)
    parser.add_argument("--force-run-failed", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    steps = evaluation_steps(region=args.region)
    run(*steps.values(), max_concurrent=args.max_concurrent, force_run_failed=args.force_run_failed)


if __name__ == "__main__":
    main()

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fail closed unless every Delphi validation cache is complete in one region."""

from __future__ import annotations

import argparse
import json

import fsspec
from marin.processing.tokenize import step_to_lm_mixture_component
from rigging.filesystem import marin_prefix_for_region

from experiments.domain_phase_mix.prepare_delphi_tpp40_europe_evaluation_caches import (
    SUPPORTED_REGIONS,
    evaluation_steps,
)
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import executor_status_succeeded

EXPECTED_EVALUATION_CACHES = 23


def _read_text(path: str) -> str:
    with fsspec.open(path, "rt") as handle:
        return handle.read()


def _path_exists(path: str) -> bool:
    fs, _, paths = fsspec.get_fs_token_paths(path)
    if len(paths) != 1:
        raise ValueError(f"Expected one cache path for {path!r}, got {paths}")
    return fs.exists(paths[0])


def evaluation_cache_dirs(*, region: str) -> tuple[str, ...]:
    """Return the unique validation cache directories bound by the Delphi launcher."""
    required_prefix = marin_prefix_for_region(region).rstrip("/") + "/"
    cache_dirs = tuple(
        sorted(
            step_to_lm_mixture_component(step, include_raw_paths=False).cache_dir
            for step in evaluation_steps(region=region).values()
        )
    )
    if len(cache_dirs) != EXPECTED_EVALUATION_CACHES:
        raise ValueError(f"Expected {EXPECTED_EVALUATION_CACHES} validation caches, got {len(cache_dirs)}")
    if len(set(cache_dirs)) != len(cache_dirs):
        raise ValueError("Validation cache bindings contain duplicate paths")
    cross_region = tuple(path for path in cache_dirs if not path.startswith(required_prefix))
    if cross_region:
        raise ValueError(f"Validation cache bindings are not region-local: {cross_region}")
    return cache_dirs


def validate_evaluation_caches(*, region: str) -> tuple[str, ...]:
    """Require successful executor status and validation statistics for every cache."""
    incomplete: list[str] = []
    for cache_dir in evaluation_cache_dirs(region=region):
        status_path = f"{cache_dir.rstrip('/')}/.executor_status"
        try:
            status = _read_text(status_path)
        except FileNotFoundError:
            incomplete.append(f"missing executor status: {cache_dir}")
            continue
        if not executor_status_succeeded(status):
            incomplete.append(f"executor status is not successful: {cache_dir}")
            continue
        stats_path = f"{cache_dir.rstrip('/')}/validation/.stats.json"
        if not _path_exists(stats_path):
            incomplete.append(f"missing validation statistics: {cache_dir}")

    if incomplete:
        details = "\n".join(f"- {item}" for item in incomplete)
        raise ValueError(f"{len(incomplete)} validation caches are not ready in {region}:\n{details}")
    return evaluation_cache_dirs(region=region)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", choices=SUPPORTED_REGIONS, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cache_dirs = validate_evaluation_caches(region=args.region)
    print(json.dumps({"status": "ready", "region": args.region, "evaluation_caches": len(cache_dirs)}, sort_keys=True))


if __name__ == "__main__":
    main()

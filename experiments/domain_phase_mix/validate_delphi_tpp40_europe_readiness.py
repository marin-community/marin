# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["fsspec[gcs]"]
# ///

"""Fail closed unless every Europe-local input for the Delphi TPP40 swarm is ready."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import fsspec
from marin.execution.artifact import read_record
from marin.processing.tokenize import step_to_lm_mixture_component

from experiments.domain_phase_mix.audit_delphi_tpp40_evaluation_caches import evaluation_paths_sha256
from experiments.domain_phase_mix.delphi_tpp40_europe_runtime_caches import (
    EUROPE_HISTORICAL_STACK_INPUT_PREFIX,
    EUROPE_HISTORICAL_STACK_MERGED_PREFIX,
    EXPECTED_STACK_ELEMENTS,
    EXPECTED_STACK_TOKENS,
)
from experiments.domain_phase_mix.launch_delphi_augmented_swarm_tpp40 import (
    _runtime_cache_paths,
    _runtime_paths_sha256,
)
from experiments.domain_phase_mix.prepare_delphi_tpp40_europe_evaluation_caches import evaluation_steps
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import (
    PREBUILT_MERGED_RUNTIME_CACHE_PATHS_BY_REGION,
    executor_status_succeeded,
)

EUROPE_PREFIX = "gs://marin-eu-west4"
STACK_ROOT = PREBUILT_MERGED_RUNTIME_CACHE_PATHS_BY_REGION["europe-west4"]["dolma3_stack_edu"]
STACK_RAW_ROOT = f"{EUROPE_PREFIX}/raw/stack_edu-ad3cf0"
EXPECTED_STACK_INPUTS = frozenset(
    {
        "stack_edu/C",
        "stack_edu/CSharp",
        "stack_edu/Cpp",
        "stack_edu/Go",
        "stack_edu/Java",
        "stack_edu/JavaScript",
        "stack_edu/Markdown",
        "stack_edu/PHP",
        "stack_edu/Python",
        "stack_edu/Ruby",
        "stack_edu/Rust",
        "stack_edu/SQL",
        "stack_edu/Shell",
        "stack_edu/Swift",
        "stack_edu/TypeScript",
    }
)
EXPECTED_RUNTIME_CACHE_PAIRS = 140
EXPECTED_LOGICAL_DIGEST_PAIRS = 9
EXPECTED_STACK_LOGICAL_PAIRS = 1
EXPECTED_DIRECT_CRC_PAIRS = EXPECTED_RUNTIME_CACHE_PAIRS - EXPECTED_LOGICAL_DIGEST_PAIRS - EXPECTED_STACK_LOGICAL_PAIRS


@dataclass(frozen=True)
class RegionalAsset:
    path: str
    size: int
    sha256: str | None = None
    crc32c: str | None = None


REGIONAL_ASSETS = (
    RegionalAsset(
        path=(
            f"{EUROPE_PREFIX}/pinlin_calvin_xu/data_mixture/delphi_augmented_swarm_3e18_20260714/"
            "source/fit_panel_table9_macro-4f283bacb4ef269c.csv"
        ),
        size=710_359,
        sha256="4f283bacb4ef269c396277cbd518ef74212a51741c909a1e1e9ace040751d507",
    ),
    RegionalAsset(
        path=(
            f"{EUROPE_PREFIX}/pinlin_calvin_xu/data_mixture/"
            "delphi_baseline_mixtures_issue6607_20260623/analysis-af9355/isoflop_analysis_result.json"
        ),
        size=2_198,
        sha256="097328aada40b0beb8b38c765ae0b30bf1767623a2b2eacd6c5c02a77af49f2b",
    ),
    RegionalAsset(
        path=f"{EUROPE_PREFIX}/raw/eval-datasets/olmo_base_eval_table9/v2/manifest.json",
        size=3_486,
        sha256="207b67e47259eb861426fd81d6a25170149c6feecb6bba61a1384b19edc87e9f",
    ),
    RegionalAsset(
        path=f"{EUROPE_PREFIX}/raw/eval-datasets/olmo_base_eval_table9/v2/requests.jsonl",
        size=152_344_523,
        crc32c="bP7vxw==",
    ),
)


def _read_bytes(path: str) -> bytes:
    with fsspec.open(path, "rb") as handle:
        return handle.read()


def _read_json(path: str) -> dict[str, object]:
    return json.loads(_read_bytes(path))


def _executor_succeeded(path: str) -> bool:
    try:
        status = _read_bytes(f"{path.rstrip('/')}/.executor_status").decode().strip()
    except FileNotFoundError:
        return False
    return executor_status_succeeded(status)


def validate_stack_artifact(*, artifact: dict[str, object], stats: dict[str, object]) -> tuple[str, ...]:
    input_configs = artifact.get("input_configs")
    if not isinstance(input_configs, dict):
        raise ValueError("Stack-Edu artifact lacks input_configs")
    if frozenset(input_configs) != EXPECTED_STACK_INPUTS:
        raise ValueError("Stack-Edu artifact does not contain the frozen 15-language input set")

    cache_dirs: list[str] = []
    for name, config in input_configs.items():
        if not isinstance(config, dict) or not isinstance(config.get("cache_dir"), str):
            raise ValueError(f"Stack-Edu input {name!r} lacks a cache_dir")
        cache_dir = config["cache_dir"]
        if not cache_dir.startswith(EUROPE_HISTORICAL_STACK_INPUT_PREFIX):
            raise ValueError(f"Stack-Edu input {name!r} is not in the historical Europe namespace: {cache_dir}")
        cache_dirs.append(cache_dir)

    if stats.get("total_tokens") != EXPECTED_STACK_TOKENS:
        raise ValueError(f"Unexpected Stack-Edu token count: {stats.get('total_tokens')}")
    if stats.get("total_elements") != EXPECTED_STACK_ELEMENTS:
        raise ValueError(f"Unexpected Stack-Edu element count: {stats.get('total_elements')}")
    return tuple(sorted(cache_dirs))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-audit", type=Path, required=True)
    parser.add_argument("--evaluation-cache-audit", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    fs = fsspec.filesystem("gcs")
    asset_results: list[dict[str, object]] = []
    for asset in REGIONAL_ASSETS:
        info = fs.info(asset.path)
        if info["size"] != asset.size:
            raise ValueError(f"Unexpected size for {asset.path}: {info['size']} != {asset.size}")
        if asset.sha256 is not None:
            digest = hashlib.sha256(_read_bytes(asset.path)).hexdigest()
            if digest != asset.sha256:
                raise ValueError(f"Unexpected SHA-256 for {asset.path}: {digest}")
        if asset.crc32c is not None and info.get("crc32c") != asset.crc32c:
            raise ValueError(f"Unexpected CRC32C for {asset.path}: {info.get('crc32c')} != {asset.crc32c}")
        asset_results.append({"path": asset.path, "size": asset.size, "sha256": asset.sha256, "crc32c": asset.crc32c})

    cache_audit = json.loads(args.cache_audit.read_text())
    if cache_audit.get("status") != "training_payload_equivalent":
        raise ValueError(f"Runtime-cache audit did not pass: {cache_audit.get('status')}")
    if cache_audit.get("cache_pairs") != EXPECTED_RUNTIME_CACHE_PAIRS or cache_audit.get("top_level_domains") != 39:
        raise ValueError("Runtime-cache audit does not cover all 140 cache pairs and 39 domains")
    if cache_audit.get("direct_crc_exact_cache_pairs") != EXPECTED_DIRECT_CRC_PAIRS:
        raise ValueError(f"Runtime-cache audit does not certify exactly {EXPECTED_DIRECT_CRC_PAIRS} direct-CRC pairs")
    if cache_audit.get("logical_digest_exact_cache_pairs") != EXPECTED_LOGICAL_DIGEST_PAIRS:
        raise ValueError(
            f"Runtime-cache audit does not certify exactly {EXPECTED_LOGICAL_DIGEST_PAIRS} logical-digest pairs"
        )
    if cache_audit.get("stack_input_shard_crc_exact_cache_pairs") != EXPECTED_STACK_LOGICAL_PAIRS:
        raise ValueError("Runtime-cache audit does not certify exactly one Stack input-shard-CRC cache pair")
    audited_runtime_hashes = cache_audit.get("runtime_paths_sha256")
    if not isinstance(audited_runtime_hashes, dict):
        raise ValueError("Runtime-cache audit lacks bound launcher path hashes")
    current_runtime_hashes = {
        "east5": _runtime_paths_sha256(_runtime_cache_paths("us-east5")),
        "europe": _runtime_paths_sha256(_runtime_cache_paths("europe-west4")),
    }
    if audited_runtime_hashes != current_runtime_hashes:
        raise ValueError("Runtime-cache audit is stale relative to current launcher bindings")
    evaluation_cache_audit = json.loads(args.evaluation_cache_audit.read_text())
    if evaluation_cache_audit.get("status") != "evaluation_payload_equivalent":
        raise ValueError("Evaluation-cache audit did not pass")
    if evaluation_cache_audit.get("evaluation_caches") != 23:
        raise ValueError("Evaluation-cache audit does not cover all 23 validation caches")

    if not _executor_succeeded(STACK_RAW_ROOT):
        raise ValueError(f"Raw Stack-Edu source is incomplete: {STACK_RAW_ROOT}")
    if not STACK_ROOT.startswith(EUROPE_HISTORICAL_STACK_MERGED_PREFIX):
        raise ValueError(f"Merged Stack-Edu cache is outside the historical Europe namespace: {STACK_ROOT}")
    if not _executor_succeeded(STACK_ROOT):
        raise ValueError(f"Merged Stack-Edu cache is incomplete: {STACK_ROOT}")
    artifact_record = read_record(STACK_ROOT)
    if artifact_record is None or artifact_record.result is None:
        raise ValueError(f"Stack-Edu artifact record lacks a result payload: {STACK_ROOT}")
    artifact = artifact_record.result
    stats = _read_json(f"{STACK_ROOT}/train/.stats.json")
    cache_dirs = validate_stack_artifact(artifact=artifact, stats=stats)
    for cache_dir in cache_dirs:
        if not _executor_succeeded(cache_dir):
            raise ValueError(f"Stack-Edu language cache is incomplete: {cache_dir}")
        _read_json(f"{cache_dir}/train/.stats.json")

    if os.environ.get("MARIN_PREFIX") != EUROPE_PREFIX:
        raise ValueError(f"MARIN_PREFIX must be {EUROPE_PREFIX!r} for Europe readiness validation")
    eval_steps = evaluation_steps(region="europe-west4")
    if len(eval_steps) != 23:
        raise ValueError(f"Expected 23 evaluation caches, got {len(eval_steps)}")
    eval_cache_paths = {name: step.path() for name, step in eval_steps.items()}
    eval_cache_dirs = tuple(
        sorted(step_to_lm_mixture_component(step, include_raw_paths=False).cache_dir for step in eval_steps.values())
    )
    audited_evaluation_hashes = evaluation_cache_audit.get("evaluation_paths_sha256")
    if not isinstance(audited_evaluation_hashes, dict):
        raise ValueError("Evaluation-cache audit lacks bound cache-path hashes")
    current_europe_evaluation_hash = evaluation_paths_sha256(eval_cache_paths)
    if audited_evaluation_hashes.get("europe") != current_europe_evaluation_hash:
        raise ValueError("Evaluation-cache audit is stale relative to current Europe cache bindings")
    for cache_dir in eval_cache_dirs:
        if not _executor_succeeded(cache_dir):
            raise ValueError(f"Evaluation cache is incomplete: {cache_dir}")

    print(
        json.dumps(
            {
                "status": "ready",
                "region": "europe-west4",
                "assets": asset_results,
                "stack_root": STACK_ROOT,
                "stack_inputs": len(cache_dirs),
                "stack_total_tokens": stats["total_tokens"],
                "stack_total_elements": stats["total_elements"],
                "runtime_cache_audit": str(args.cache_audit),
                "runtime_cache_pairs": cache_audit["cache_pairs"],
                "runtime_cache_east5_training_objects": cache_audit["east5_training_objects"],
                "runtime_cache_europe_training_objects": cache_audit["europe_training_objects"],
                "logical_runtime_contract_sha256": cache_audit["logical_runtime_contract_sha256"],
                "evaluation_caches": len(eval_cache_dirs),
                "evaluation_paths_sha256": current_europe_evaluation_hash,
                "evaluation_cache_audit": str(args.evaluation_cache_audit),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate two frozen uncapped MARINER and two cap-8 HPR/MARINER Llama policies.

Run as a module with the Marin uv workspace. Training uses the original Llama
swarm recipes and full Uncheatable evaluation, with paired data/subset seeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fsspec
import numpy as np
from fray.cluster import ResourceConfig
from iris.client.client import get_iris_ctx
from levanter.data.text.datasets import LMMixtureDatasetConfig
from levanter.main.train_lm import TrainLmConfig
from marin.execution.context import executor_context
from marin.execution.executor import Executor, ExecutorMainConfig, executor_main
from marin.execution.types import ExecutorStep, this_output_path
from marin.training.training import TrainLmOnPodConfig

from experiments.datasets.uncheatable import UNCHEATABLE_SUBSETS
from experiments.domain_phase_mix.config import WeightConfig
from experiments.domain_phase_mix.dolma3_dolmino_top_level_domains import (
    TOP_LEVEL_DOMAIN_TOKEN_COUNTS,
    TOP_LEVEL_TOTAL_AVAILABLE_TOKENS,
)
from experiments.domain_phase_mix.experiment import DEFAULT_MUON_CONFIG
from experiments.domain_phase_mix.launch_fixed_aggregate_phase_order_panel_60m_1p2b import (
    _configure_training_step,
    _validate_uncheatable_caches,
)
from experiments.domain_phase_mix.proxy_sweep import regmix_60m_proxy, regmix_300m_muonh_base, regmix_300m_proxy
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import (
    BATCH_SIZE,
    DOMAIN_NAMES,
    SEQ_LEN,
    TARGET_BUDGET,
    create_two_phase_dolma3_dolmino_top_level_experiment,
    resolve_two_phase_wsd_boundary_schedule,
)

logger = logging.getLogger(__name__)

NAME = "calvin/dm/llama_mariner_hpr_val_20260910"
REFERENCE = Path(__file__).resolve().parent / "exploratory/two_phase_many/reference_outputs"
OUTPUT = REFERENCE / "llama_uncheatable_validation_20260910"
CANDIDATES = OUTPUT / "candidates.json"
CANDIDATES_SHA256 = "5f7066323ac3929efdc0b3b63688e898c9a6b3392cfef4f008447c6bb7b45f96"
REGION = "us-east5"
ZONE = "us-east5-a"
TPU_TYPE = "v5p-8"
PREFIX = "gs://marin-us-east5"
RUN_COUNT = 4
# The archived surrogate uses the six-decimal proportional anchor from swarm39_harness_20260725.
SURROGATE_PROPORTIONAL_EPOCHS = 0.905353


@dataclass(frozen=True)
class SaveValidationManifestConfig:
    output_path: str
    manifest_json: str


def save_validation_manifest(config: SaveValidationManifestConfig) -> None:
    """Preserve exact policies and predictions next to the remote run outputs."""
    with fsspec.open(f"{config.output_path}/run_manifest.json", "w") as handle:
        handle.write(config.manifest_json)


def load_candidates() -> dict[str, Any]:
    """Load only the four approved, hash-pinned policies."""
    source = CANDIDATES.read_bytes()
    if hashlib.sha256(source).hexdigest() != CANDIDATES_SHA256:
        raise ValueError("The approved candidate file has changed")
    manifest = json.loads(source)
    runs = manifest["runs"]
    assert len(runs) == RUN_COUNT
    assert len({run["run_name"] for run in runs}) == RUN_COUNT
    for run in runs:
        schedule = resolve_two_phase_wsd_boundary_schedule(experiment_budget=run["experiment_budget"])
        assert schedule.total_steps == run["num_train_steps"]
        assert schedule.boundary_step == run["boundary_step"]
        inventory = run["epochs_per_unit_weight"]
        assert set(inventory) == set(DOMAIN_NAMES)
        for bucket in DOMAIN_NAMES:
            expected = (
                SURROGATE_PROPORTIONAL_EPOCHS * TOP_LEVEL_TOTAL_AVAILABLE_TOKENS / TOP_LEVEL_DOMAIN_TOKEN_COUNTS[bucket]
            )
            assert np.isclose(inventory[bucket], expected, rtol=1e-10, atol=1e-10), bucket
        phases = run["phase_weights"]
        for weights in phases.values():
            assert set(weights) == set(DOMAIN_NAMES)
            assert sum(weights.values()) == 1.0
            assert all(w >= 0 and w * 2048 == round(w * 2048) for w in weights.values())
        if run["run_name"].endswith("1p_uncapped"):
            assert phases["phase_0"] == phases["phase_1"]
            continue
        for alpha in (0.8, schedule.boundary_step / schedule.total_steps):
            epochs = [
                (alpha * phases["phase_0"][b] + (1 - alpha) * phases["phase_1"][b]) * inventory[b] for b in DOMAIN_NAMES
            ]
            assert max(epochs) <= 8.0 + 1e-10
    return manifest


def validate_training_step(step: ExecutorStep, run: dict[str, Any]) -> dict[str, Any]:
    """Check the lowered training schedule, mixture, seeds, and evaluation isolation."""
    config = step.config
    assert isinstance(config, TrainLmOnPodConfig)
    train = config.train_config
    assert isinstance(train, TrainLmConfig)
    data = train.data
    assert isinstance(data, LMMixtureDatasetConfig)
    assert train.trainer.num_train_steps == run["num_train_steps"]
    assert train.trainer.train_batch_size == BATCH_SIZE
    assert train.train_seq_len == SEQ_LEN
    assert train.trainer.seed == run["trainer_seed"]
    assert train.data_seed == run["data_seed"]
    assert data.simulated_epoch_subset_seed == run["simulated_epoch_subset_seed"]
    assert data.experiment_budget == run["num_train_steps"] * BATCH_SIZE * SEQ_LEN
    assert data.target_budget == TARGET_BUDGET
    assert data.mixture_block_size == 2048
    assert data.num_validation_sequences is None
    assert train.trainer.max_eval_batches is None
    assert train.eval_harness is None
    expected_eval = {f"uncheatable_eval/{subset}" for subset in UNCHEATABLE_SUBSETS}
    assert set(data.components) - set(DOMAIN_NAMES) == expected_eval
    assert isinstance(data.train_weights, list)
    assert [start for start, _ in data.train_weights] == [0, run["boundary_step"]]
    for index, (_, weights) in enumerate(data.train_weights):
        assert {b: weights[b] for b in DOMAIN_NAMES} == run["phase_weights"][f"phase_{index}"]
        assert all(weights.get(b, 0) == 0 for b in expected_eval)
    assert config.env_vars is not None and config.env_vars["MARIN_PREFIX"] == PREFIX
    return {
        "run_name": run["run_name"],
        "training_step": step.name,
        "num_train_steps": train.trainer.num_train_steps,
        "boundary_step": run["boundary_step"],
        "realized_training_tokens": data.experiment_budget,
        "target_budget": data.target_budget,
        "uncheatable_components": sorted(expected_eval),
        "optimizer": str(train.optimizer),
        "model": str(train.model),
        "resources": str(config.resources),
        "passed": True,
    }


def build_launch_artifacts(manifest: dict[str, Any]) -> tuple[list[ExecutorStep], list[dict[str, Any]]]:
    """Build the original Llama recipes with exact frozen policy weights."""
    steps = []
    checks = []
    for run in manifest["runs"]:
        small = run["artifact_scale"] == "60m"
        experiment = create_two_phase_dolma3_dolmino_top_level_experiment(
            name=NAME,
            experiment_budget=run["experiment_budget"],
            target_budget=TARGET_BUDGET,
            batch_size=BATCH_SIZE,
            seq_len=SEQ_LEN,
            model_config=regmix_60m_proxy if small else regmix_300m_proxy,
            optimizer_config=DEFAULT_MUON_CONFIG if small else regmix_300m_muonh_base,
            resources=ResourceConfig.with_tpu(TPU_TYPE, regions=[REGION], zone=ZONE),
            eval_harness_tasks=(),
            runtime_cache_region=REGION,
        )
        step = experiment.create_training_step(
            weight_config=WeightConfig(run_id=run["run_id"], phase_weights=run["phase_weights"]),
            name_prefix=NAME,
            run_name=run["run_name"],
            data_seed=run["data_seed"],
            trainer_seed=run["trainer_seed"],
            simulated_epoch_subset_seed=run["simulated_epoch_subset_seed"],
        )
        step = _configure_training_step(step, tpu_region=REGION)
        checks.append(validate_training_step(step, run))
        steps.append(step)
    remote_manifest = ExecutorStep(
        name=f"{NAME}/manifest",
        fn=save_validation_manifest,
        config=SaveValidationManifestConfig(
            output_path=this_output_path(),
            manifest_json=json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        ),
    )
    return [remote_manifest, *steps], checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--tpu-type", choices=[TPU_TYPE], default=TPU_TYPE)
    parser.add_argument("--tpu-region", choices=[REGION], default=REGION)
    parser.add_argument("--tpu-zone", choices=[ZONE], default=ZONE)
    parser.add_argument("--max-concurrent", type=int, choices=[RUN_COUNT], default=RUN_COUNT)
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    logging.basicConfig(level=logging.INFO)
    if os.environ.get("MARIN_PREFIX", PREFIX) != PREFIX:
        raise ValueError("This validation is pinned to gs://marin-us-east5")
    os.environ["MARIN_PREFIX"] = PREFIX
    if not args.dry_run and get_iris_ctx() is None:
        raise ValueError("Submit through Iris; use --dry-run for local graph validation")
    manifest = load_candidates()
    _validate_uncheatable_caches(REGION)
    with executor_context():
        steps, checks = build_launch_artifacts(manifest)
    executor = Executor(prefix=PREFIX, executor_info_base_path=f"{PREFIX}/experiments")
    for step in steps:
        executor.compute_version(step, is_pseudo_dep=False)
    for row, step in zip(checks, steps[1:], strict=True):
        row["output_path"] = executor.output_paths[step]
        logger.info("Validated %s: %s", row["run_name"], row["output_path"])
    OUTPUT.mkdir(exist_ok=True, parents=True)
    (OUTPUT / "launch_checks.json").write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n")
    if os.environ.get("IRIS_OUTPUT_DIR"):
        (Path(os.environ["IRIS_OUTPUT_DIR"]) / "launch_checks.json").write_text(
            json.dumps(checks, indent=2, sort_keys=True) + "\n"
        )
    logger.info("Validated all four policies; launch checks at %s", OUTPUT / "launch_checks.json")
    if args.dry_run:
        return
    executor_main(
        ExecutorMainConfig(prefix=PREFIX, max_concurrent=RUN_COUNT),
        steps=steps,
        description="Four matched-seed Llama validations: uncapped MARINER 1p and cap-8 HPR/MARINER 2p.",
    )


if __name__ == "__main__":
    main()

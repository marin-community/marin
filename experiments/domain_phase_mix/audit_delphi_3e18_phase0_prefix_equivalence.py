# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# dependencies = ["wandb"]
# ///

"""Compare the phase-0 replay config with the original July W&B config."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
from pathlib import Path
from typing import Any

import wandb
from fray.cluster import ResourceConfig
from marin.processing.tokenize import step_to_lm_mixture_component
from marin.training.training import TrainLmOnPodConfig, _prepare_training_run
from wandb.util import json_friendly_val

from experiments.domain_phase_mix import launch_delphi_3e18_phase0_prefix_replay as replay
from experiments.llama import llama3_tokenizer

ORIGINAL_WANDB_RUN = "marin-community/marin/fit_000_baseline_proportional-90c474"
ORIGINAL_OUTPUT_PATH = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_augmented_swarm_3e18_20260714/"
    "fit_000_baseline_proportional-90c474"
)
DEFAULT_OUTPUT = replay.LOCAL_ARTIFACT_DIR / "config_equivalence_audit.json"


def _drop_empty_schema_additions(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _drop_empty_schema_additions(item)
            for key, item in value.items()
            if not (key in {"advanced_configuration", "shared_mapping"} and item == {})
        }
    if isinstance(value, list):
        return [_drop_empty_schema_additions(item) for item in value]
    return value


def _diff(expected: Any, actual: Any, path: str = "") -> list[dict[str, Any]]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        differences: list[dict[str, Any]] = []
        for key in sorted(expected.keys() | actual.keys()):
            child = f"{path}.{key}" if path else key
            if key not in expected:
                differences.append({"path": child, "expected": "<missing>", "actual": actual[key]})
            elif key not in actual:
                differences.append({"path": child, "expected": expected[key], "actual": "<missing>"})
            else:
                differences.extend(_diff(expected[key], actual[key], child))
        return differences
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [{"path": path, "expected": expected, "actual": actual}]
        differences = []
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual, strict=True)):
            differences.extend(_diff(expected_item, actual_item, f"{path}[{index}]"))
        return differences
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if math.isclose(float(expected), float(actual), rel_tol=1e-12, abs_tol=1e-15):
            return []
    if expected == actual:
        return []
    return [{"path": path, "expected": expected, "actual": actual}]


def _replay_config(replay_code_commit: str) -> dict[str, Any]:
    os.environ["MARIN_PREFIX"] = "gs://marin-us-east5"
    specs, _ = replay.load_replay_specs(
        source_panel=replay.base.DEFAULT_SOURCE_PANEL,
        analysis_output_path=replay.base.DEFAULT_ANALYSIS_OUTPUT_PATH,
        tpu_region=replay.base.DEFAULT_TPU_REGION,
        tpu_zone=replay.base.DEFAULT_TPU_ZONE,
    )
    run_spec = specs[0]
    if run_spec.source_run_name != "baseline_proportional":
        raise ValueError(f"First canonical row changed: {run_spec.source_run_name}")

    validation_steps = replay.base._default_validation_sets(tokenizer=llama3_tokenizer)
    validation_configs = {
        name: step_to_lm_mixture_component(step, include_raw_paths=False) for name, step in validation_steps.items()
    }
    scaling_fits = replay.base._read_scaling_fits(replay.base.DEFAULT_ANALYSIS_OUTPUT_PATH)
    candidate = replay.base._candidate_for_budget(scaling_fits=scaling_fits)
    config = replay._prefix_train_config(
        run_spec=run_spec,
        candidate=candidate,
        validation_configs=validation_configs,
        replay_code_commit=replay_code_commit,
    )
    resources = ResourceConfig.with_tpu(
        run_spec.tpu_type,
        regions=[run_spec.tpu_region],
        zone=run_spec.tpu_zone,
    )
    pod_config, config, _ = _prepare_training_run(
        TrainLmOnPodConfig(
            train_config=config,
            resources=resources,
            output_path=ORIGINAL_OUTPUT_PATH,
            env_vars={"RUN_ID": "fit_000_baseline_proportional-90c474"},
        )
    )
    del pod_config
    return _drop_empty_schema_additions(json_friendly_val(dataclasses.asdict(config)))


def audit() -> dict[str, Any]:
    """Return an exact normalized config comparison against the original run."""
    original = _drop_empty_schema_additions(wandb.Api().run(ORIGINAL_WANDB_RUN).config)
    replay_code_commit = replay.get_git_commit()
    if replay_code_commit is None or len(replay_code_commit) != 40:
        raise ValueError(f"Could not resolve replay code commit: {replay_code_commit!r}")
    replay_config = _replay_config(replay_code_commit)

    if replay_config["trainer"]["profiler"]["enabled"]:
        raise ValueError("Profiler must remain disabled for bitwise phase-0 replay")

    optimizer_horizon = replay_config.pop("optimizer_schedule_num_train_steps")
    if optimizer_horizon != original["trainer"]["num_train_steps"]:
        raise ValueError(f"Optimizer horizon changed: {optimizer_horizon}")

    execution_horizons = {
        "original": original["trainer"]["num_train_steps"],
        "replay": replay_config["trainer"]["num_train_steps"],
    }
    if execution_horizons != {"original": 3007, "replay": 2400}:
        raise ValueError(f"Unexpected execution horizons: {execution_horizons}")
    replay_config["trainer"]["num_train_steps"] = original["trainer"]["num_train_steps"]

    original_tags = original["trainer"]["tracker"].pop("tags")
    replay_tags = replay_config["trainer"]["tracker"].pop("tags")
    added_tags = {"phase0-prefix-replay", f"replay_code_commit={replay_code_commit}"}
    if set(replay_tags) != set(original_tags) | added_tags:
        raise ValueError("Replay W&B tags differ by more than the expected provenance markers")

    # The original launcher also specified -1/None. Levanter resolved these on
    # v5p-8 before logging the W&B config; canonicalize the replay accordingly.
    runtime_resolutions = {
        "per_device_parallelism": (
            replay_config["trainer"]["per_device_parallelism"],
            original["trainer"]["per_device_parallelism"],
        ),
        "per_device_eval_parallelism": (
            replay_config["trainer"]["per_device_eval_parallelism"],
            original["trainer"]["per_device_eval_parallelism"],
        ),
        "require_accelerator": (
            replay_config["trainer"]["require_accelerator"],
            original["trainer"]["require_accelerator"],
        ),
    }
    if runtime_resolutions != {
        "per_device_parallelism": (-1, 32),
        "per_device_eval_parallelism": (-1, 32),
        "require_accelerator": (None, True),
    }:
        raise ValueError(f"Unexpected v5p-8 runtime resolutions: {runtime_resolutions}")
    for key in runtime_resolutions:
        replay_config["trainer"][key] = original["trainer"][key]

    # NoAdaptorConfig is a new explicit dataclass field in this source tree but
    # was omitted from the W&B encoding of the original default behavior.
    adapter = replay_config.pop("adapter")
    if adapter:
        raise ValueError(f"Replay unexpectedly configures an adapter: {adapter}")

    unexpected_differences = _diff(original, replay_config)
    result = {
        "status": "pass" if not unexpected_differences else "fail",
        "original_wandb_run": ORIGINAL_WANDB_RUN,
        "original_code_commit": replay.ORIGINAL_CODE_COMMIT,
        "replay_code_commit": replay_code_commit,
        "source_panel_sha256": replay.base.SOURCE_PANEL_SHA256,
        "source_coordinate_hash": replay.EXPECTED_SOURCE_COORDINATE_HASH,
        "execution_horizons": execution_horizons,
        "optimizer_schedule_num_train_steps": optimizer_horizon,
        "allowed_non_state_differences": {
            "tracker_tags_added": sorted(added_tags),
            "post_update_2400": "forced final checkpoint, HF export, and smooth evaluation",
            "live_output_paths": "new executor output root",
        },
        "runtime_resolutions": runtime_resolutions,
        "unexpected_differences": unexpected_differences,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    result = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU-only `--plan` preflight for every loop-guard stage preset (no cluster submission).

Builds a fixture panel selection locally (the real selection is published in region by
``selection.py``), then runs the RL CLI's plan path for each stage and saves every launch
document plus a receipt. The S3 arms bind the declared start checkpoint standing in for the
S2x exports before those trained model artifacts exist:

    uv run --package marin-core python -m experiments.post_training.mochi_loop_guards.preflight \\
        --output <scratch dir>
"""

import argparse
import json
import logging
from pathlib import Path

import yaml
from marin.execution.lazy import materialized_config
from marin.rl.skyrl import ResolvedModelLocator
from rolloutengine.models import TASK_ROLLOUT_COLUMN, TaskRolloutSpec
from zephyr.writers import write_parquet_file

from experiments.post_training import rl
from experiments.post_training import rl_compose as recipe
from experiments.post_training.mochi_loop_guards import presets
from experiments.post_training.mochi_loop_guards.fixture_tasks import fixture_rollout
from experiments.post_training.mochi_loop_guards.presets import (
    MODEL,
    S0,
    S1_RP,
    S1_STOP,
    S2_CONTROL,
    S2_TREATMENT,
    VERSION,
)
from experiments.post_training.rl_presets import Generation, ModelPresets

logger = logging.getLogger(__name__)

FIXTURE_TASKS = {"tb2": 89, "tblite": 100, "swe100": 100, "swesmith_ho": 128}
STAGES = (
    "s0-baseline",
    "s1-stop",
    "s1-rp",
    "s2-antithrash-control",
    "s2-antithrash-treatment",
    "s3-control-panel",
    "s3-treatment-panel",
)


def fixture_pool(root: Path) -> Path:
    """Write a fixture RL task pool with the agentic half-mix into ``root``."""
    selection = root / "users/benfeuer/mochi-loop-guards/snowball-100k-agentic-ho-excluded" / VERSION
    selection.mkdir(parents=True, exist_ok=True)
    families = ["agentic"] * 128 + ["nonagentic"] * 128
    write_parquet_file([{"family": family} for family in families], str(Path(selection) / "train.parquet"))
    summary = {
        "training_rows": len(families),
        "training_family_counts": {"agentic": 128, "nonagentic": 128},
        "grader_kind_counts": {"train": {}},
        "recipe": {"heldout": [{"version": VERSION, "rows": 823}], "exclusions": []},
        "selection_version": VERSION,
        "fixture": True,
    }
    (selection / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return selection


def fixture_selection(root: Path) -> Path:
    """Write a fixture panel selection with the frozen task counts into ``root``.

    The rows are minimal native rollout specs with the frozen task IDs; the plan path reads
    the selection summary, never the task bodies, so this validates stage wiring only.
    """
    selection = root / "users/benfeuer/mochi-loop-guards/panel" / VERSION
    selection.mkdir(parents=True, exist_ok=True)
    summary = {
        "grader_kind_counts": {"train": {}, "validation": {}, "full_validation": {}},
        "selection_version": VERSION,
        "fixture": True,
    }
    for cohort, tasks in FIXTURE_TASKS.items():
        rows = [{TASK_ROLLOUT_COLUMN: fixture_rollout(f"{cohort}/{index}").model_dump_json()} for index in range(tasks)]
        write_parquet_file(rows, str(selection / f"{cohort}.parquet"))
        stop_rows = []
        for row in rows:
            spec = TaskRolloutSpec.model_validate_json(row[TASK_ROLLOUT_COLUMN])
            spec = spec.model_copy(update={"limits": spec.limits.model_copy(update={"repeat_stall_batches": 3})})
            stop_rows.append({TASK_ROLLOUT_COLUMN: spec.model_dump_json()})
        write_parquet_file(stop_rows, str(selection / f"{cohort}_stop.parquet"))
        summary[f"{cohort}_rows"] = tasks
        summary[f"{cohort}_stop_rows"] = tasks
        summary["grader_kind_counts"][cohort] = {}
        summary["grader_kind_counts"][f"{cohort}_stop"] = {}
    (selection / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return selection


def stage_preset(stage: str):
    """The preset of one stage, with the S3 export stand-in bound to the start checkpoint."""
    if stage.startswith("s3-"):
        return presets.s3_panel(stage, MODEL)
    return {
        "s0-baseline": S0,
        "s1-stop": S1_STOP,
        "s1-rp": S1_RP,
        "s2-antithrash-control": S2_CONTROL,
        "s2-antithrash-treatment": S2_TREATMENT,
    }[stage]


def _stage_gpus(preset) -> int:
    layout = preset.recipe.layout
    workload = layout.workload
    if isinstance(workload, Generation):
        plan = workload.plan
        return plan.num_inference_engines * plan.inference_engine_data_parallel_size
    plan = workload.plan
    return plan.policy_num_nodes * plan.policy_num_gpus_per_node + (
        plan.num_inference_engines * plan.inference_engine_data_parallel_size
    )


def plan_stage(stage: str, output: Path, *, pool_selection: Path | None = None) -> dict:
    """Run the RL CLI's plan path for one stage and save its launch document."""
    preset = stage_preset(stage)
    table = ModelPresets(
        model="mochi",
        presets=(preset,),
        namespace="mochi-loop-guards-preflight",
        control_pins={},
        needs_completed_default_export=True,
    )
    argv = ["--version", "dev", "--preset", preset.name]
    if pool_selection is not None:
        argv += ["--tasks", f"users/benfeuer/mochi-loop-guards/snowball-100k-agentic-ho-excluded@{VERSION}"]
    code = rl.main(table, argv=argv)
    if code != 0:
        raise SystemExit(f"{stage}: the plan path exited {code}")
    spec = recipe.run_spec(preset, name=f"preflight-{preset.name}", model_name=table.model)
    step = recipe.run_step(preset, spec)
    launch = materialized_config(step, str(output)).launch_config_yaml
    path = output / f"{stage}.yaml"
    path.write_text(launch)
    document = yaml.safe_load(launch)
    return {
        "stage": stage,
        "preset": preset.name,
        "workload": type(preset.recipe.layout.workload).__name__,
        "gpus": _stage_gpus(preset),
        "layout": preset.recipe.layout.name,
        "host_memory": document["iris"]["allocation"]["memory"],
        "excluded_nvlink_domains": list(preset.recipe.layout.excluded_nvlink_domains),
        "coordinator_timeout_hours": preset.recipe.layout.coordinator_timeout_hours,
        "job_timeout_seconds": preset.recipe.layout.job_timeout_seconds,
        "iris": {
            "cluster": document["iris"]["cluster"],
            "excluded_nvlink_domains": document["iris"]["excluded_nvlink_domains"],
            "priority": document["iris"]["priority"],
        },
        "launch_yaml": str(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Directory for launch documents and the receipt")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    original_root = recipe.SELECTION_ROOT
    stages = []
    # The preflight runs the CLI's plan path entirely on local fixtures: the selection root
    # and the artifact store both resolve into the output directory (rl_tests/fixtures.py
    # uses the same patching for its plan-path tests).
    import experiments.post_training.rl as rl_module  # noqa: PLC0415
    import experiments.post_training.rl_command as rl_command  # noqa: PLC0415

    original_prefix = recipe.artifact_prefix
    original_configure = rl_command.configure_coreweave_s3
    original_cache = rl_module.check_cache
    try:
        recipe.SELECTION_ROOT = str(output / "selection-root")
        recipe.artifact_prefix = lambda cluster: str(output / "artifact-root")
        rl_command.configure_coreweave_s3 = lambda: None
        # The plan path checks the start checkpoint's export cache on the artifact store;
        # the preflight assumes the declared export exists.
        rl_module.check_cache = lambda path: True
        selection = fixture_selection(Path(recipe.SELECTION_ROOT))
        pool = fixture_pool(Path(recipe.SELECTION_ROOT))
        logger.info("fixture selection at %s", selection)
        for stage in STAGES:
            stages.append(plan_stage(stage, output, pool_selection=pool if "s2-" in stage else None))
            logger.info("planned %s", stage)
    finally:
        recipe.SELECTION_ROOT = original_root
        recipe.artifact_prefix = original_prefix
        rl_command.configure_coreweave_s3 = original_configure
        rl_module.check_cache = original_cache
    launch_config = output / "s2-antithrash-treatment.yaml"
    document = yaml.safe_load(launch_config.read_text())
    model = document["inputs"]["model"]
    command = presets.export_command(
        document["artifacts"]["checkpoint_root"],
        S2_TREATMENT.updates,
        str(launch_config),
        model=ResolvedModelLocator(
            uri=model["uri"],
            identity=model["identity"],
            local_path=model["local_path"],
            tokenizer_uri=model["tokenizer_uri"],
            tokenizer_revision=model["tokenizer_revision"],
        ),
        export_path=f"{presets.PANEL_ROOT}/exports/s2-antithrash-treatment-preflight",
    )
    receipt = {"stages": stages, "s2x_export_command": command}
    (output / "preflight-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()

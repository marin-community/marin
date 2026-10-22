# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare pinned Datakit SFT and Dr.Doom initializers on the fixed Snowball inventory.

The two arms share the dedicated evaluation account's 16-task budget, eight each.
Do not overlap another evaluation job using that account with the comparison.
Each arm starts in its own artifact/checkpoint root and resumes only its own state.
Periodic arms persist paired reports under evaluation/step-UPDATE/report.json.
"""

import json
from dataclasses import replace
from enum import StrEnum

import click
import yaml
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import (
    ArtifactDataSource,
    ArtifactHfModel,
    SkyRLRetentionPolicy,
    SkyRLRun,
    SkyRLRuntime,
    SkyRLSpec,
    SkyRLTopology,
    skyrl_step,
)
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.snowball_inventory_evaluate import (
    EvaluationOutcome,
    PanelTask,
    runtime_operation,
    summarize_panel,
)
from experiments.post_training.snowball_inventory_panel import FAMILY_COUNTS
from experiments.post_training.snowball_inventory_smoke import (
    SEED,
    InventoryRunConfig,
    ModelSourceIdentity,
    inventory_training_execution,
    run_inventory,
)
from experiments.post_training.snowball_inventory_train import CHECKPOINT_INTERVAL, ROLE_PLAN, training_recipe

MAX_STEPS = 50
EVAL_CONCURRENCY = 8
REQUEST_WINDOW_TOKENS = 24_576
MAX_NEW_TOKENS = 8_192
EVALUATION_STEPS = tuple(range(0, MAX_STEPS + 1, CHECKPOINT_INTERVAL))


class Initializer(StrEnum):
    SFT = "sft"
    DRDOOM = "drdoom"


INITIALIZERS = {
    Initializer.SFT: ModelSourceIdentity(
        "open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21", "fe666478454e79fe6027a0bab0388d9abd68f51a"
    ),
    Initializer.DRDOOM: ModelSourceIdentity(
        "open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom", "5b5705e8345bb111de03b3f0f9a00c8041e57db4"
    ),
}


def summarize_periodic_evaluation(launch_document: str, probe_report: dict, step: int) -> dict:
    """Pair an in-run evaluation dump with the CPU-admitted fixed panel."""
    if step not in EVALUATION_STEPS:
        raise ValueError(f"The comparison evaluates only updates {EVALUATION_STEPS}")
    if probe_report["status"] != "passed" or probe_report["validation_family_counts"] != FAMILY_COUNTS:
        raise ValueError("The comparison requires the complete CPU-admitted original development panel")
    tasks = [
        PanelTask(row["task_id"], row["evaluation_family"], row["source_contract_id"])
        for row in probe_report["validation_tasks"]
    ]
    if len(tasks) != sum(FAMILY_COUNTS.values()):
        raise ValueError("The admitted development panel is incomplete")
    records = runtime_operation(
        "results",
        {
            "launch_config_yaml": launch_document,
            "evaluation_step": step,
            "dataset_uids": {row["dataset_uid"]: row["task_id"] for row in probe_report["validation_tasks"]},
        },
    )
    report = summarize_panel(tasks, [EvaluationOutcome(**row) for row in records["outcomes"]])
    report.update(
        step=step,
        panel_sha256=probe_report["validation_sha256"],
        source_dumps=records["source_dumps"],
        protocol={"max_new_tokens": MAX_NEW_TOKENS, "request_window_tokens": REQUEST_WINDOW_TOKENS, "temperature": 0.0},
    )
    return report


def run_comparison(config: InventoryRunConfig) -> SkyRLRun:
    """Persist every planned paired evaluation before completing the comparison artifact."""
    completed = run_inventory(config)
    root = StoragePath(config.training.output.terminal_manifest_uri).parent
    probe = json.loads((root / "probe" / "report.json").read_text())
    max_steps = yaml.safe_load(config.training.launch_config_yaml)["skyrl"]["trainer"]["max_steps"]
    incomplete = []
    for step in range(0, max_steps + 1, CHECKPOINT_INTERVAL):
        report = summarize_periodic_evaluation(config.training.launch_config_yaml, probe, step)
        destination = root / "evaluation" / f"step-{step}" / "report.json"
        destination.parent.mkdirs()
        destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        if not report["complete"]:
            incomplete.append(step)
    if incomplete:
        raise ValueError(f"Comparison evaluation coverage is incomplete at updates {incomplete}; reports: {root}")
    return completed


def comparison_recipe(model_uri: str, *, judge_base_url: str, judge_model: str) -> dict:
    """Keep the current optimizer/mixture recipe and evaluate every original panel row."""
    recipe = training_recipe(
        MAX_STEPS,
        request_window_tokens=REQUEST_WINDOW_TOKENS,
        max_new_tokens=MAX_NEW_TOKENS,
        judge_base_url=judge_base_url,
        judge_model=judge_model,
    )
    recipe["environment"]["task_sessions"].update(
        daytona={"api_key_env": "DAYTONA_EVAL_API_KEY", "ttl_minutes": 60},
    )
    recipe.setdefault("trajectory_runner", {}).update(eval_max_concurrent_tasks=EVAL_CONCURRENCY)
    trainer = recipe["trainer"]
    trainer.update(
        evaluation_runner="task",
        eval_num_prompts=None,
        eval_batch_size=sum(FAMILY_COUNTS.values()),
        eval_required_score_coverage=1.0,
        dump_eval_results=True,
    )
    trainer.setdefault("step_phase_budgets", {}).update(
        group_admission=7200,
        training_preparation=1800,
        policy_training=7200,
        evaluation=10800,
    )
    # Explicit callbacks retain ordinary checkpoint/export/telemetry behavior and
    # protect the best fully scored pooled checkpoint from rolling deletion.
    trainer["callbacks"] = [
        {"type": "checkpoint", "save_steps": CHECKPOINT_INTERVAL},
        {"type": "evaluation", "eval_steps": CHECKPOINT_INTERVAL, "eval_before_train": True},
        {
            "type": "best_checkpoint",
            "initial_model_uri": model_uri,
            "expected_tasks": sum(FAMILY_COUNTS.values()),
            "minimum_scored_tasks": sum(FAMILY_COUNTS.values()),
        },
        {"type": "hf_model_save", "save_steps": MAX_STEPS},
        {"type": "inference_stats"},
        {"type": "database_registration"},
    ]
    recipe["generator"].update(
        eval_n_samples_per_prompt=1,
        eval_sampling_params={
            "max_generate_length": MAX_NEW_TOKENS,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "logprobs": None,
            "stop": None,
        },
    )
    return recipe


def comparison_step(
    initializer: Initializer,
    *,
    version: str,
    model_uri: str,
    pool_uri: str,
    pool_version: str,
    pool_sha256: str,
    validation_uri: str,
    validation_version: str,
    validation_sha256: str,
    judge_base_url: str,
    judge_model: str,
) -> ArtifactStep[SkyRLRun]:
    """Build one isolated 50-update arm, with full-pool and original-panel CPU admission."""
    identity = INITIALIZERS[initializer]
    name = user_owned_name(f"checkpoints/snowball-inventory-compare/{initializer.value}")
    model = ArtifactStep.adopt(
        f"models/snowball-inventory-compare/{initializer.value}",
        version,
        model_uri,
        kind=LevanterCheckpoint,
        config={"repository": identity.repository, "revision": identity.revision},
    )
    pool = ArtifactStep.adopt(f"documents/snowball-inventory-compare/{initializer.value}/train", pool_version, pool_uri)
    validation = ArtifactStep.adopt(
        f"documents/snowball-inventory-compare/{initializer.value}/validation", validation_version, validation_uri
    )
    rl = skyrl_step(
        SkyRLSpec(
            name=name,
            version=version,
            config_yaml=yaml.safe_dump(
                comparison_recipe(model_uri, judge_base_url=judge_base_url, judge_model=judge_model), sort_keys=False
            ),
            runtime=SkyRLRuntime(),
            model=ArtifactHfModel(model, identity.repository, identity.revision, relative_path=""),
            train_data=(ArtifactDataSource(pool, relative_path="train.parquet"),),
            validation_data=(ArtifactDataSource(validation, relative_path="validation.parquet"),),
            topology=SkyRLTopology(
                num_nodes=ROLE_PLAN.num_nodes,
                gpus_per_node=ROLE_PLAN.policy_num_gpus_per_node,
                gpu_variant="H100",
                role_plan=ROLE_PLAN,
            ),
            retention=SkyRLRetentionPolicy(resume_checkpoint_count=2),
            seed=SEED,
        ),
        inventory_training_execution(),
        export_hf=True,
    )

    def build_config(ctx: StepContext) -> InventoryRunConfig:
        return InventoryRunConfig(rl.build_config(ctx), pool_sha256, validation_sha256, dict(FAMILY_COUNTS), identity)

    return replace(rl, run=run_comparison, build_config=build_config)


@click.command(help=__doc__)
@click.option("--initializer", type=click.Choice([item.value for item in Initializer]), required=True)
@click.option("--model-uri", required=True)
@click.option("--pool-uri", required=True)
@click.option("--pool-version", required=True)
@click.option("--pool-sha256", required=True)
@click.option("--validation-uri", required=True)
@click.option("--validation-version", required=True)
@click.option("--validation-sha256", required=True)
@click.option("--judge-base-url", required=True)
@click.option("--judge-model", required=True)
@rl_build_options
def main(
    initializer: str,
    model_uri: str,
    pool_uri: str,
    pool_version: str,
    pool_sha256: str,
    validation_uri: str,
    validation_version: str,
    validation_sha256: str,
    judge_base_url: str,
    judge_model: str,
) -> ArtifactStep[SkyRLRun]:
    choice = Initializer(initializer)
    name = user_owned_name(f"checkpoints/snowball-inventory-compare/{choice.value}")
    return comparison_step(
        choice,
        version=resolve_version(name, None),
        model_uri=model_uri,
        pool_uri=pool_uri,
        pool_version=pool_version,
        pool_sha256=pool_sha256,
        validation_uri=validation_uri,
        validation_version=validation_version,
        validation_sha256=validation_sha256,
        judge_base_url=judge_base_url,
        judge_model=judge_model,
    )


if __name__ == "__main__":
    main()

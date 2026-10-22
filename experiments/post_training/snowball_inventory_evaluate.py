# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate a frozen Snowball panel independently of its training run.

The default uses 4096 reply tokens. An 8192-token evaluation requires a separately
frozen panel with the approved 1800/2400/3600-second model/turn/attempt limits.
Prepare it with ``prepare_reference_panel`` in the runtime module; retain the
original panel and pass the derived artifact's URI and hash to this normal main.
"""

import hashlib
import json
import math
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from tempfile import TemporaryDirectory

import click
import yaml
from fray.types import ResourceConfig
from iris.rpc import job_pb2
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.remote import remote
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import (
    IRIS_HUB_CLUSTER_CONFIG,
    ArtifactDataSource,
    ArtifactHfModel,
    IrisSkyRLExecution,
    ResolvedModelLocator,
    SkyRLOutputPaths,
    SkyRLRetentionPolicy,
    SkyRLRun,
    SkyRLRunConfig,
    SkyRLRuntime,
    SkyRLSpec,
    SkyRLTopology,
    run_skyrl,
    skyrl_step,
)
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.conditional_object import conditional_object
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.post_training.async_rl import SNOWBALL_RECIPE
from experiments.post_training.snowball_inventory_panel import FAMILY_COUNTS, PROMPT_TOKEN_BUDGET, pairing_identity
from experiments.post_training.snowball_inventory_smoke import (
    CLUSTER,
    CONTAINER_PROFILE,
    SEED,
    TOKENIZER_MODEL,
    TOKENIZER_REVISION,
    inventory_training_execution,
    preflight_environment,
    runtime_environment,
    smoke_recipe,
)

EVAL_ROLE_PLAN = replace(
    SNOWBALL_RECIPE.role_plan,
    colocate_all=True,
    policy_num_nodes=1,
    reference_num_nodes=None,
    train_batch_size=8,
    policy_mini_batch_size=8,
    n_samples_per_prompt=1,
)
WORKER_MODULE = "experiments.post_training.snowball_inventory_evaluate_runtime"


@dataclass(frozen=True)
class PanelTask:
    task_id: str
    evaluation_family: str
    contract_id: str


@dataclass(frozen=True)
class EvaluationOutcome:
    task_id: str
    score: float | None
    stop_reason: str | None
    verification_status: str | None = None
    failure_type: str | None = None


@dataclass(frozen=True)
class FamilyResult:
    expected: int
    attempted: int
    scored: int
    score_sum: float
    fixed_denominator_mean: float
    coverage: float
    length_stops: int
    failure_categories: dict[str, int]


@dataclass(frozen=True)
class SavedAnswerRecovery:
    per_answer_timeout: float
    total_timeout: float

    def __post_init__(self) -> None:
        if not 0 < self.per_answer_timeout <= self.total_timeout <= 25 * 60:
            raise ValueError("Recovery needs positive per-answer/total limits within the 30-minute worker envelope")


@dataclass(frozen=True)
class EvaluationConfig:
    launch: SkyRLRunConfig
    panel_sha256: str
    output_path: str
    max_new_tokens: int = 4096
    family_counts: dict[str, int] | None = None
    expected_model_manifest_identity: str | None = None
    recovery: SavedAnswerRecovery | None = None
    expected_tokenizer_template_sha256: str | None = None


@dataclass(frozen=True)
class CheckpointExportConfig:
    training_config_uri: str
    step: int
    output_path: str


class InventoryEvaluation(Artifact):
    report_uri: str
    pairing_identity: str
    model_uri: str
    complete: bool


def summarize_panel(tasks: list[PanelTask], outcomes: list[EvaluationOutcome]) -> dict:
    """Keep missing grades in each family's fixed denominator and preserve pairing IDs."""
    if not tasks:
        raise ValueError("The evaluation panel must not be empty")
    by_id = {task.task_id: task for task in tasks}
    if len(by_id) != len(tasks):
        raise ValueError("Panel task IDs must be unique, including aliases")
    observed = {}
    for outcome in outcomes:
        if outcome.task_id not in by_id or outcome.task_id in observed:
            raise ValueError(f"Unexpected or repeated evaluation task: {outcome.task_id}")
        if outcome.score is not None and (not math.isfinite(outcome.score) or not 0 <= outcome.score <= 1):
            raise ValueError("Evaluation requires normalized finite verifier scores")
        observed[outcome.task_id] = outcome
    families = {}
    for family in sorted({task.evaluation_family for task in tasks}):
        selected = [task for task in tasks if task.evaluation_family == family]
        completed = [observed[task.task_id] for task in selected if task.task_id in observed]
        scored = [outcome.score for outcome in completed if outcome.score is not None]
        families[family] = asdict(
            FamilyResult(
                expected=len(selected),
                attempted=len(completed),
                scored=len(scored),
                score_sum=sum(scored),
                fixed_denominator_mean=sum(scored) / len(selected),
                coverage=len(scored) / len(selected),
                length_stops=sum(outcome.stop_reason == "length" for outcome in completed),
                failure_categories=dict(
                    Counter(
                        outcome.failure_type or outcome.verification_status or "unscored"
                        for outcome in completed
                        if outcome.score is None
                    )
                ),
            )
        )
    pairing = [(task.task_id, task.evaluation_family, task.contract_id) for task in tasks]
    return {
        "complete": len(observed) == len(tasks) and all(outcome.score is not None for outcome in outcomes),
        "pairing_identity": pairing_identity(pairing),
        "families": families,
        "aggregate_fixed_denominator_mean": sum(outcome.score or 0.0 for outcome in outcomes) / len(tasks),
        "macro_family_mean": sum(value["fixed_denominator_mean"] for value in families.values()) / len(families),
        "tasks": [
            {
                **asdict(task),
                "attempted": task.task_id in observed,
                "score": observed[task.task_id].score if task.task_id in observed else None,
                "stop_reason": observed[task.task_id].stop_reason if task.task_id in observed else None,
                "verification_status": observed[task.task_id].verification_status if task.task_id in observed else None,
                "failure_type": observed[task.task_id].failure_type if task.task_id in observed else None,
            }
            for task in tasks
        ],
    }


def runtime_operation(action: str, request: dict, *, timeout: float = 30 * 60) -> dict:
    """Run optional runtime code in its frozen dependency environment."""
    with TemporaryDirectory(prefix="snowball-evaluation-") as directory:
        source = Path(directory) / "request.json"
        result = Path(directory) / "result.json"
        source.write_text(json.dumps(request))
        subprocess.run(
            [
                "uv",
                "run",
                "--project",
                "vendor/skyrl",
                "--frozen",
                "--extra",
                "cpu",
                "--no-group",
                "dev",
                "python",
                "-m",
                WORKER_MODULE,
                action,
                "--request",
                str(source),
                "--result",
                str(result),
            ],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            timeout=timeout,
            env=runtime_environment(),
        )
        return json.loads(result.read_text())


def preflight_evaluation(config: EvaluationConfig) -> None:
    report = runtime_operation(
        "preflight",
        {
            "launch_config_yaml": config.launch.launch_config_yaml,
            "panel_sha256": config.panel_sha256,
            "family_counts": config.family_counts if config.family_counts is not None else FAMILY_COUNTS,
            "max_new_tokens": config.max_new_tokens,
            "expected_model_manifest_identity": config.expected_model_manifest_identity,
            "expected_tokenizer_template_sha256": config.expected_tokenizer_template_sha256,
            "recover_saved_answers": config.recovery is not None,
        },
    )
    uri = prefix_join(config.output_path, "preflight.json")
    if config.recovery is not None:
        conditional_object(uri).write(json.dumps(report, allow_nan=False).encode(), expected_version=None)
        return
    with StoragePath(uri).open("wt") as target:
        json.dump(report, target, indent=2)


def run_evaluation(config: EvaluationConfig) -> InventoryEvaluation:
    remote(
        preflight_evaluation,
        env_vars=preflight_environment(config.launch.launch_config_yaml),
        resources=ResourceConfig.with_cpu(cpu=4, ram="32GB", disk="32GB", container_profile=CONTAINER_PROFILE),
    )(config)
    with StoragePath(prefix_join(config.output_path, "preflight.json")).open("rt") as source:
        preflight = json.load(source)
    prepared = replace(config.launch, launch_config_yaml=preflight["launch_config_yaml"])
    run_skyrl(prepared)
    records = runtime_operation(
        "results",
        {
            "launch_config_yaml": prepared.launch_config_yaml,
            "dataset_uids": preflight["dataset_uids"],
        },
    )
    tasks = [PanelTask(**row) for row in preflight["tasks"]]
    outcomes = [EvaluationOutcome(**row) for row in records["outcomes"]]
    summary = summarize_panel(tasks, outcomes)
    summary.update(
        model=asdict(config.launch.model),
        panel_sha256=config.panel_sha256,
        launch_config_sha256=hashlib.sha256(prepared.launch_config_yaml.encode()).hexdigest(),
        model_manifest_identity=preflight["model_manifest_identity"],
        protocol=preflight["protocol"],
        pairing_contract_field=preflight["pairing_contract_field"],
    )
    report_uri = prefix_join(config.output_path, "report.json")
    with StoragePath(report_uri).open("wt") as target:
        json.dump(summary, target, indent=2, allow_nan=False)
    if not summary["complete"]:
        raise ValueError(f"Evaluation coverage is incomplete; diagnostic report: {report_uri}")
    return InventoryEvaluation(
        path=config.output_path,
        report_uri=report_uri,
        pairing_identity=summary["pairing_identity"],
        model_uri=config.launch.model.uri,
        complete=True,
    )


def evaluation_contract(config: EvaluationConfig) -> dict:
    """Bind every scoring input while excluding only the launcher's fresh submission identity."""
    contract = asdict(config)
    launch = contract["launch"]
    document = yaml.safe_load(launch["launch_config_yaml"])
    document["run"]["attempt_id"] = "<submission-attempt>"
    document["iris"]["job_name"] = "<submission-job>"
    launch["attempt_id"] = "<submission-attempt>"
    launch["launch_config_yaml"] = document
    return contract


def admitted_evaluation(config: EvaluationConfig) -> EvaluationConfig:
    """Reuse the exact first launch on retries of this scoring artifact."""
    request = StoragePath(prefix_join(config.output_path, "admitted-request.json"))
    if not request.exists():
        conditional_object(str(request)).write(
            json.dumps(asdict(config), allow_nan=False).encode(), expected_version=None
        )
        return config
    original = json.loads(request.read_text())
    original_launch = original["launch"]["launch_config_yaml"]
    admitted = replace(
        config,
        launch=replace(
            config.launch,
            launch_config_yaml=original_launch,
            attempt_id=original["launch"]["attempt_id"],
        ),
    )
    if evaluation_contract(admitted) != evaluation_contract(config) or asdict(admitted) != original:
        raise ValueError("Existing scoring admission has different source, model, panel, protocol or recovery policy")
    return admitted


def run_recoverable_evaluation(config: EvaluationConfig) -> InventoryEvaluation:
    """Score an exported model; grader failures cannot invalidate its upstream training artifact."""
    config = admitted_evaluation(config)
    assert config.recovery is not None
    preflight_path = StoragePath(prefix_join(config.output_path, "preflight.json"))
    if not preflight_path.exists():
        remote(
            preflight_evaluation,
            resources=ResourceConfig.with_cpu(cpu=4, ram="32GB", disk="32GB", container_profile=CONTAINER_PROFILE),
        )(config)
    preflight = json.loads(preflight_path.read_text())
    prepared = replace(config.launch, launch_config_yaml=preflight["launch_config_yaml"])
    completed = runtime_operation("generation-state", {"launch_config_yaml": prepared.launch_config_yaml})
    if not completed["complete"]:
        # A coordinator crash cannot silently spend another inference job on this artifact.
        conditional_object(prefix_join(config.output_path, "generation-started.json")).write(
            json.dumps(
                {"launch_config_sha256": hashlib.sha256(prepared.launch_config_yaml.encode()).hexdigest()}
            ).encode(),
            expected_version=None,
        )
        run_skyrl(prepared)
    records = runtime_operation(
        "recover-results",
        {
            "launch_config_yaml": prepared.launch_config_yaml,
            "dataset_uids": preflight["dataset_uids"],
            "panel_sha256": config.panel_sha256,
            "output_path": config.output_path,
            **asdict(config.recovery),
        },
        timeout=config.recovery.total_timeout + 5 * 60,
    )
    summary = summarize_panel(
        [PanelTask(**row) for row in preflight["tasks"]],
        [EvaluationOutcome(**row) for row in records["outcomes"]],
    )
    summary.update(
        model=asdict(config.launch.model),
        panel_sha256=config.panel_sha256,
        launch_config_sha256=hashlib.sha256(prepared.launch_config_yaml.encode()).hexdigest(),
        model_manifest_identity=preflight["model_manifest_identity"],
        protocol=preflight["protocol"],
        pairing_contract_field=preflight["pairing_contract_field"],
        recovery=records["recovery"],
        required_score_coverage=1.0,
    )
    report_uri = prefix_join(config.output_path, "report.json")
    with StoragePath(report_uri).open("wt") as target:
        json.dump(summary, target, indent=2, allow_nan=False)
    return InventoryEvaluation(
        path=config.output_path,
        report_uri=report_uri,
        pairing_identity=summary["pairing_identity"],
        model_uri=config.launch.model.uri,
        complete=summary["complete"],
    )


def recoverable_evaluation_recipe(max_new_tokens: int = 8192, *, judge_base_url: str, judge_model: str) -> dict:
    recipe = evaluation_recipe(max_new_tokens)
    # The post-generation scorer enforces full coverage after its one saved-answer attempt.
    recipe["trainer"]["eval_required_score_coverage"] = None
    recipe["environment"]["task_sessions"]["remote_judge"] = {
        "base_url": judge_base_url,
        "api_key_env": "JUDGE_API_KEY",
    }
    recipe["extra_env"]["VERIFYIT_JUDGE_MODEL"] = judge_model
    return recipe


def recoverable_evaluation_step(
    *,
    name: str,
    version: str,
    model: ArtifactStep[LevanterCheckpoint] | ArtifactStep[SkyRLRun],
    panel: ArtifactStep,
    panel_sha256: str,
    family_counts: dict[str, int],
    tokenizer_uri: str,
    tokenizer_revision: str,
    recovery: SavedAnswerRecovery,
    judge_base_url: str,
    judge_model: str,
    expected_model_manifest_identity: str | None = None,
    expected_tokenizer_template_sha256: str | None = None,
    relative_path: str | None = None,
    max_new_tokens: int = 8192,
) -> ArtifactStep[InventoryEvaluation]:
    """Bind independent scoring to a completed export or an exact adopted baseline."""
    execution = replace(
        inventory_training_execution(), memory="256GB", coordinator_timeout_hours=8, job_timeout_seconds=4 * 60 * 60
    )
    launch = skyrl_step(
        SkyRLSpec(
            name=name,
            version=version,
            config_yaml=yaml.safe_dump(
                recoverable_evaluation_recipe(max_new_tokens, judge_base_url=judge_base_url, judge_model=judge_model),
                sort_keys=False,
            ),
            runtime=SkyRLRuntime(),
            model=ArtifactHfModel(
                model,
                tokenizer_uri,
                tokenizer_revision,
                relative_path=relative_path,
            ),
            train_data=(),
            validation_data=(ArtifactDataSource(panel, relative_path="validation.parquet"),),
            topology=SkyRLTopology(num_nodes=1, gpus_per_node=8, gpu_variant="H100", role_plan=EVAL_ROLE_PLAN),
            retention=SkyRLRetentionPolicy(resume_checkpoint_count=1),
            seed=SEED,
        ),
        execution,
        export_hf=False,
    )

    def build_config(ctx: StepContext) -> EvaluationConfig:
        launch_config = launch.build_config(ctx)
        document = yaml.safe_load(launch_config.launch_config_yaml)
        document["skyrl"]["trajectory_runner"]["eval_saved_answers_root"] = prefix_join(
            launch_config.output.export_root, "saved_answers"
        )
        launch_config = replace(launch_config, launch_config_yaml=yaml.safe_dump(document, sort_keys=False))
        return EvaluationConfig(
            launch_config,
            panel_sha256,
            ctx.output_path,
            max_new_tokens,
            dict(family_counts),
            expected_model_manifest_identity,
            recovery,
            expected_tokenizer_template_sha256,
        )

    return ArtifactStep(
        name=launch.name,
        version=launch.version,
        artifact_type=InventoryEvaluation,
        run=run_recoverable_evaluation,
        build_config=build_config,
        deps=launch.deps,
        runtime_args=launch.runtime_args,
    )


def launch_from_document(document: str) -> SkyRLRunConfig:
    root = yaml.safe_load(document)
    return SkyRLRunConfig(
        launch_config_yaml=document,
        run_id=root["run"]["id"],
        attempt_id=root["run"]["attempt_id"],
        model=ResolvedModelLocator(
            **{field.name: root["inputs"]["model"][field.name] for field in fields(ResolvedModelLocator)}
        ),
        output=SkyRLOutputPaths(**{field.name: root["artifacts"][field.name] for field in fields(SkyRLOutputPaths)}),
        export_hf=False,
        draft_checkpoint_root=None,
    )


def run_checkpoint_export(config: CheckpointExportConfig) -> LevanterCheckpoint:
    prepared = runtime_operation("export-config", asdict(config))
    with StoragePath(prefix_join(config.output_path, "preflight.json")).open("wt") as target:
        json.dump(prepared, target, indent=2)
    run_skyrl(launch_from_document(prepared["launch_config_yaml"]))
    runtime_operation("verify-model", {"model_uri": prepared["policy_export_uri"]})
    return LevanterCheckpoint(path=config.output_path)


def checkpoint_export_step(training_config_uri: str, step: int, version: str) -> ArtifactStep[LevanterCheckpoint]:
    source = ArtifactStep.adopt("documents/snowball-evaluation-training-config", version, training_config_uri)
    return ArtifactStep(
        name=user_owned_name(f"checkpoints/snowball-evaluation-export/step-{step}"),
        version=version,
        artifact_type=LevanterCheckpoint,
        run=run_checkpoint_export,
        build_config=lambda ctx: CheckpointExportConfig(ctx.artifact_path(source), step, ctx.output_path),
        deps=(source,),
    )


def evaluation_recipe(max_new_tokens: int = 4096) -> dict:
    recipe = smoke_recipe(PROMPT_TOKEN_BUDGET + max_new_tokens, max_new_tokens, role_plan=EVAL_ROLE_PLAN, max_steps=1)
    recipe["entrypoint"] = "generate"
    recipe["trainer"].update(
        eval_interval=1,
        eval_before_train=False,
        eval_num_prompts=None,
        eval_batch_size=64,
        evaluation_runner="task",
        dump_eval_results=True,
        ckpt_interval=-1,
        hf_save_interval=-1,
        project_name="marin-snowball-inventory-evaluation",
        resume_mode="none",
    )
    recipe["trainer"]["placement"].update(colocate_all=True, policy_num_nodes=1, ref_num_nodes=1)
    # Generate loads only inference engines. Keep the unused learner declarations
    # internally valid for the evaluator's single eight-GPU allocation.
    for role in ("policy", "ref"):
        recipe["trainer"][role]["megatron_config"]["pipeline_model_parallel_size"] = 1
    recipe["generator"].update(weight_sync_transport="broadcast", eval_n_samples_per_prompt=1)
    recipe["trajectory_runner"] = {
        "max_concurrent_tasks": 16,
        "rollout_workers": {"num_workers": 1, "cpus_per_worker": 8},
    }
    recipe["environment"] = {
        "task_sessions": {
            "daytona": {"api_key_env": "DAYTONA_EVAL_API_KEY", "ttl_minutes": 60},
        }
    }
    return recipe


@click.command(help=__doc__)
@click.option("--model-uri", required=True)
@click.option("--model-version", required=True)
@click.option("--panel-uri", required=True)
@click.option("--panel-version", required=True)
@click.option("--panel-sha256", required=True)
@click.option("--training-config-uri")
@click.option("--checkpoint-step", type=click.IntRange(min=1))
@click.option("--max-new-tokens", type=click.Choice([4096, 8192]), default=4096, show_default=True)
@rl_build_options
def main(
    model_uri: str,
    model_version: str,
    panel_uri: str,
    panel_version: str,
    panel_sha256: str,
    training_config_uri: str | None,
    checkpoint_step: int | None,
    max_new_tokens: int = 4096,
) -> ArtifactStep[InventoryEvaluation]:
    if (training_config_uri is None) != (checkpoint_step is None):
        raise click.BadParameter("training-config-uri and checkpoint-step must be supplied together")
    name = user_owned_name("evals/snowball-inventory")
    version = resolve_version(name, None)
    if training_config_uri is not None and checkpoint_step is not None:
        model = checkpoint_export_step(training_config_uri, checkpoint_step, version)
        relative_path = f"exports/global_step_{checkpoint_step}/policy"
    else:
        model = ArtifactStep.adopt("models/snowball-evaluation-input", model_version, model_uri, kind=LevanterCheckpoint)
        relative_path = ""
    panel = ArtifactStep.adopt("documents/snowball-evaluation-panel", panel_version, panel_uri)
    launch = skyrl_step(
        SkyRLSpec(
            name=name,
            version=version,
            config_yaml=yaml.safe_dump(evaluation_recipe(max_new_tokens), sort_keys=False),
            runtime=SkyRLRuntime(),
            model=ArtifactHfModel(model, TOKENIZER_MODEL, TOKENIZER_REVISION, relative_path=relative_path),
            train_data=(),
            validation_data=(ArtifactDataSource(panel, relative_path="validation.parquet"),),
            topology=SkyRLTopology(num_nodes=1, gpus_per_node=8, gpu_variant="H100", role_plan=EVAL_ROLE_PLAN),
            retention=SkyRLRetentionPolicy(resume_checkpoint_count=1),
            seed=SEED,
        ),
        IrisSkyRLExecution(
            cluster=CLUSTER,
            cluster_config=f"lib/iris/config/{CLUSTER}.yaml",
            cpu=32,
            memory="256GB",
            disk="2TB",
            priority="interactive",
            max_retries=0,
            target_cluster=CLUSTER,
            parent_cluster_config=IRIS_HUB_CLUSTER_CONFIG,
            coordinator_timeout_hours=8 if max_new_tokens == 8192 else 6,
            job_timeout_seconds=(4 if max_new_tokens == 8192 else 2) * 60 * 60,
            container_profile=job_pb2.ContainerProfile.Value(CONTAINER_PROFILE),
            wandb_entity="marin-community",
        ),
        export_hf=False,
    )

    def build_config(ctx: StepContext) -> EvaluationConfig:
        return EvaluationConfig(launch.build_config(ctx), panel_sha256, ctx.output_path, max_new_tokens)

    return ArtifactStep(
        name=launch.name,
        version=launch.version,
        artifact_type=InventoryEvaluation,
        run=run_evaluation,
        build_config=build_config,
        deps=launch.deps,
        runtime_args=launch.runtime_args,
    )


if __name__ == "__main__":
    main()

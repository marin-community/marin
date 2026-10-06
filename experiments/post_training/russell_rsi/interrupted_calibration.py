# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate qualified SFT after an interrupted calibration without authorizing RL."""

import asyncio
import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

from fray.current_client import current_client
from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment
from iris.client.client import IrisClient, iris_ctx
from iris.cluster.constraints import CLUSTER_CONSTRAINT_KEY, ConstraintOp, strip_cluster_constraints
from iris.rpc import job_pb2
from marin.evaluation.evalchemy.runner import EvalchemyExecutor
from marin.evaluation.hardware import default_platform
from marin.evaluation.records import read_record, record_path
from marin.evaluation.runner import EndpointRoute, EvaluationBatch, run_evaluation_batch
from marin.execution.lazy import ArtifactStep, StepContext, artifact_identity
from marin.external_dependencies import MARIN_SKYRL
from marin.training.run_environment import dependency_groups_for_resources
from rigging.filesystem.storage_path import StoragePath
from rigging.timing import Duration
from taskcompendium.parquet import read_tasks

from experiments.evaluation.launch import LaunchSpec, prepare_evaluation_batch
from experiments.evaluation.pipeline import EvalStepConfig, EvaluationResult
from experiments.post_training.russell_rsi.bootstrap_loop import write_once
from experiments.post_training.russell_rsi.contract_tasks import digest
from experiments.post_training.russell_rsi.evaluation_journal import AttemptJournal, EvaluationJournal
from experiments.post_training.russell_rsi.launch import CLUSTER
from experiments.post_training.russell_rsi.launch_post_teacher_sft import (
    StudyBaseline,
    StudySelectionConfig,
    post_sft_evaluation_stages,
    seal_study_selection,
)
from experiments.post_training.russell_rsi.launch_rsi_continuation import BASELINE_RETENTION, INCUMBENT_CODING
from experiments.post_training.russell_rsi.repair_tasks import pinned_bytes
from experiments.post_training.russell_rsi.rollout_eval import (
    DevelopmentEvaluationConfig,
    development_worker_provenance,
    require_journal_submission,
    run_development_evaluation,
)
from experiments.post_training.russell_rsi.sources import compact_json_sha256
from experiments.post_training.russell_rsi.token_preflight import PREFLIGHT_PROBES, preflight_task

PROTOCOL = "russell-rsi-calibration-interruption-v1"
OUTPUT_PROTOCOL = "russell-rsi-interrupted-calibration-sft-only-v1"
WORKER_TIMEOUT_HOURS = 6
CODING_TRANSPORT_RETRY_BUDGET = 900
BRANCH_PACKAGES = (
    "marin",
    "rigging",
    "fray",
    "iris",
    "levanter",
    "haliax",
    "rolloutengine",
    "taskcompendium",
    "shellbox",
)
PACKAGED_PYTHONPATH = ":".join(f"/app/lib/{name}/src" for name in BRANCH_PACKAGES) + ":/app"


def require_interruption(config: dict) -> dict:
    """Read the sealed interruption and reject changes to the original study."""
    amendment = json.loads(
        pinned_bytes(config["calibration_interruption_uri"], config["calibration_interruption_sha256"])
    )
    source = amendment["source"]
    original = json.loads(pinned_bytes(source["config"]["uri"], source["config"]["sha256"]))
    unchanged = {
        key: value
        for key, value in config.items()
        if key not in ("version", "calibration_interruption_uri", "calibration_interruption_sha256")
    }
    if unchanged != {key: value for key, value in original.items() if key != "version"}:
        raise ValueError("Interrupted evaluation changed the original qualified study")
    if (
        amendment["protocol"] != PROTOCOL
        or amendment["calibration_status"] != "incomplete_infrastructure"
        or amendment["signal_gate_passed"] is not None
        or amendment["rl_authorized"] is not False
        or amendment["whole_cohort_replacements_remaining"] != 0
        or amendment["repeated_issued_samples"] != 0
        or amendment["evaluation"]
        != {"version": config["version"], "conditions": ["sft"], "coding_limit": 32, "retention_limit": 3}
        or config["version"] == original["version"]
        or source["qualification_sha256"] != config["qualification_sha256"]
    ):
        raise ValueError("Interrupted calibration requires the separate bounded SFT-only amendment")
    if amendment["execution"] != {
        "child_detachment": False,
        "cluster_changes": False,
        "coordinator": "foreground local artifact main",
        "priority": "batch",
        "remote_worker_cluster": CLUSTER,
    }:
        raise ValueError("Interrupted evaluation changed its foreground batch execution contract")
    journal = json.loads(pinned_bytes(source["journal_binding"]["uri"], source["journal_binding"]["sha256"]))
    if journal["config"]["model_identity"] != source["model_identity"]:
        raise ValueError("Interruption model differs from the sealed calibration journal")
    if sum(len(slots) for kind, slots in journal["attempts"].items() if kind == "task") != 256:
        raise ValueError("Interruption must retain the original 256-slot calibration")
    expected_evidence = {
        "terminal_root",
        "terminal_tree",
        "terminal_inventory",
        "issuance_census",
        "completion_issuance",
        "slot_dispositions",
        "grade_intake",
        "terminal_inventory_data",
    }
    if set(amendment["evidence"]) != expected_evidence:
        raise ValueError("Interruption has incomplete terminal evidence")
    for pin in amendment["evidence"].values():
        pinned_bytes(pin["uri"], pin["sha256"])
    return {"amendment": amendment, "source_config": original}


def retention_journal(
    config: DevelopmentEvaluationConfig, *, continuation_binding: dict | None = None
) -> EvaluationJournal:
    """Reserve the same three retention tasks before any model startup."""
    tasks = list(read_tasks(config.tasks_path))
    if (
        len(tasks) != 3
        or len({task.id for task in tasks}) != 3
        or config.limit != 3
        or config.samples_per_task != 1
        or config.startup_attempts != 1
        or config.temperature != 0.0
    ):
        raise ValueError("Interrupted SFT retention differs from its fixed three-task cohort")
    for task in tasks:
        require_journal_submission(task)
    probes = [
        preflight_task(index, instruction, value) for index, (instruction, value) in enumerate(PREFLIGHT_PROBES, 1)
    ]
    provenance = development_worker_provenance()
    binding = {
        "protocol": OUTPUT_PROTOCOL,
        "config": json.loads(json.dumps(asdict(config))),
        "parquet_sha256": hashlib.sha256(StoragePath(config.tasks_path).read_bytes()).hexdigest(),
        "worker_provenance": provenance,
        "worker_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "job_policy": {"failure_retries": 0, "preemption_retries": 0, "timeout_hours": WORKER_TIMEOUT_HOURS},
        "attempts": {
            "task": {f"{task.id}/0": digest(task.model_dump(mode="json")) for task in tasks},
            "preflight": {
                str(index): compact_json_sha256(task.model_dump(mode="json")) for index, task in enumerate(probes, 1)
            },
        },
    }
    if continuation_binding is not None:
        binding["continuation"] = continuation_binding
    journal = EvaluationJournal(StoragePath(config.output_path) / "journal", binding)
    journal.seal()
    return journal


def run_interrupted_retention(config: DevelopmentEvaluationConfig) -> None:
    run_development_evaluation(config, journal=retention_journal(config))


def retention_request(config: DevelopmentEvaluationConfig) -> JobRequest:
    """Return the bounded worker request for the existing local-server evaluator."""
    resources = ResourceConfig.with_gpu("H100", 8, cpu=32, ram="512GB", disk="2TB", target_cluster=CLUSTER)
    return JobRequest(
        name=f"russell-interrupted-retention-{hashlib.sha256(config.output_path.encode()).hexdigest()[:12]}",
        entrypoint=Entrypoint.from_callable(lambda: run_interrupted_retention(config)),
        resources=resources,
        environment=create_environment(
            extras=dependency_groups_for_resources(resources, None),
            pip_packages=[MARIN_SKYRL.requirement()],
            env_vars={"UV_PRERELEASE": "allow", "PYTHONPATH": PACKAGED_PYTHONPATH},
        ),
        max_retries_failure=0,
        max_retries_preemption=0,
        max_task_failures=0,
        priority=job_pb2.PRIORITY_BAND_BATCH,
        timeout=Duration.from_hours(WORKER_TIMEOUT_HOURS),
    )


def submit_retention(config: DevelopmentEvaluationConfig) -> None:
    current_client().submit(retention_request(config), adopt_existing=True).wait(raise_on_failure=True)


class BoundedIrisClient:
    """Bound direct CW02 submissions and translate its federation pin to local placement."""

    def __init__(self, client: IrisClient):
        self.client = client

    def __getattr__(self, name: str) -> Any:
        return getattr(self.client, name)

    def submit(self, **kwargs: Any) -> Any:
        constraints = kwargs.get("constraints") or []
        for constraint in constraints:
            if constraint.key == CLUSTER_CONSTRAINT_KEY and (
                constraint.op != ConstraintOp.EQ or constraint.values[0].value != CLUSTER
            ):
                raise ValueError(f"Unsupported cluster constraint {constraint} on direct {CLUSTER} client")
        kwargs["constraints"] = strip_cluster_constraints(constraints)
        kwargs.update(
            max_retries_failure=0,
            max_retries_preemption=0,
            max_task_failures=0,
            priority_band=job_pb2.PRIORITY_BAND_BATCH,
            timeout=Duration.from_hours(WORKER_TIMEOUT_HOURS),
        )
        return self.client.submit(**kwargs)


def foreground_coding_batch(config: EvalStepConfig, endpoint_route: EndpointRoute) -> EvaluationBatch:
    """Build the unchanged coding panels with an explicit worker endpoint route."""
    batch = prepare_evaluation_batch(
        LaunchSpec(
            model=config.model,
            evals=("humanevalplus", "mbppplus"),
            evalchemy_definitions=(),
            harbor_definitions=(),
            platform=default_platform(config.model),
            accelerator=config.accelerator,
            limit=config.limit,
            records_prefix=None,
            submission_cluster=CLUSTER,
            federated_cluster=CLUSTER,
            priority_band=job_pb2.PRIORITY_BAND_BATCH,
            version=config.version,
        )
    )
    evaluations = []
    for evaluation in batch.evaluations:
        if not isinstance(evaluation.executor, EvalchemyExecutor):
            raise ValueError("Interrupted coding requires the original Evalchemy panels")
        executor = replace(
            evaluation.executor,
            config=replace(
                evaluation.executor.config,
                # Local coding adapters retain their separate transport retry budget.
                extra_model_args={**evaluation.executor.config.extra_model_args, "max_retries": 1},
            ),
        )
        if endpoint_route is EndpointRoute.CAPABILITY:
            executor = replace(
                executor,
                config=replace(
                    executor.config,
                    extra_model_args={
                        **executor.config.extra_model_args,
                        "transport_retry_budget": CODING_TRANSPORT_RETRY_BUDGET,
                    },
                ),
            )
        evaluations.append(replace(evaluation, executor=executor, endpoint_route=endpoint_route))
    return replace(batch, evaluations=tuple(evaluations))


def coding_attempt(config: EvalStepConfig, *, transport_binding: dict | None = None) -> AttemptJournal:
    """Bind the single coding batch before server or worker startup."""
    binding = json.loads(json.dumps(asdict(config)))
    journal_binding = {
        "protocol": OUTPUT_PROTOCOL,
        "config": binding,
        "job_policy": {"failure_retries": 0, "preemption_retries": 0, "model_retries": 0},
    }
    if transport_binding is not None:
        batch = foreground_coding_batch(config, EndpointRoute.CAPABILITY)
        journal_binding["transport_replacement"] = {
            **transport_binding,
            "evaluations": [
                {
                    "route": item.endpoint_route.value,
                    "executor_config": asdict(cast(EvalchemyExecutor, item.executor).config),
                }
                for item in batch.evaluations
            ],
        }
        journal_binding["job_policy"] = {"failure_retries": 0, "preemption_retries": 0}
    return AttemptJournal(
        StoragePath(config.artifact_path) / "journal" / "coding", json.loads(json.dumps(journal_binding))
    )


def run_foreground_coding(config: EvalStepConfig, *, transport_binding: dict | None = None) -> EvaluationResult:
    """Run the existing coding evaluation from the foreground coordinator."""
    attempt = coding_attempt(config, transport_binding=transport_binding)
    saved = attempt.saved_result()
    if saved is not None:
        return EvaluationResult(**saved)

    route = EndpointRoute.CAPABILITY if transport_binding is not None else EndpointRoute.DIRECT
    batch = foreground_coding_batch(config, route)
    if not isinstance(iris_ctx().client, BoundedIrisClient):
        raise ValueError("Foreground coding requires the bounded CW02 client")

    async def operation() -> dict:
        write_once(attempt.directory / "batch.json", json.loads(json.dumps(asdict(batch), default=str)))
        run_evaluation_batch(batch, coordinator_identity=f"foreground-local:{config.artifact_path}")
        run_ids = tuple(evaluation.identity.run_id for evaluation in batch.evaluations)
        return {
            "path": config.artifact_path,
            "group_id": batch.group_id,
            "records_prefix": batch.records_prefix,
            "run_ids": list(run_ids),
            "results_paths": [read_record(record_path(batch.records_prefix, run_id)).results_path for run_id in run_ids],
        }

    return EvaluationResult(**asyncio.run(attempt.run(operation)))


@dataclass(frozen=True)
class InterruptedSelectionConfig:
    selection: StudySelectionConfig
    interruption_uri: str
    interruption_sha256: str


def seal_interrupted_selection(config: InterruptedSelectionConfig) -> None:
    amendment = json.loads(pinned_bytes(config.interruption_uri, config.interruption_sha256))
    write_once(
        StoragePath(config.selection.record.output_path) / "calibration-interruption.json",
        {
            "amendment_uri": config.interruption_uri,
            "amendment_sha256": config.interruption_sha256,
            "calibration_status": amendment["calibration_status"],
            "signal_gate_passed": None,
            "rl_authorized": False,
            "source": amendment["source"],
        },
    )
    seal_study_selection(config.selection)


def interrupted_evaluation_stages(
    config: dict,
    *,
    model: ArtifactStep,
    retention: ArtifactStep,
    export_uri: str,
    study: StudyBaseline,
    sealed: dict,
) -> dict[str, ArtifactStep]:
    if artifact_identity(model) != sealed["amendment"]["source"]["model_identity"]:
        raise ValueError("Interrupted evaluation substituted a different qualified model")
    if study.incumbent.development != INCUMBENT_CODING or study.incumbent.retention != BASELINE_RETENTION:
        raise ValueError("Interrupted evaluation changed its incumbent baseline")
    outputs = post_sft_evaluation_stages(
        config,
        model=model,
        retention=retention,
        export_uri=export_uri,
        checkpoints=[("sft", model)],
        barriers=(),
        outputs={},
        study=replace(study, protocol=OUTPUT_PROTOCOL),
        coding_runner=run_foreground_coding,
        retention_runner=submit_retention,
    )
    selection = outputs["selection"]

    def selection_config(ctx: StepContext) -> InterruptedSelectionConfig:
        assert selection.build_config is not None
        return InterruptedSelectionConfig(
            selection.build_config(ctx),
            config["calibration_interruption_uri"],
            config["calibration_interruption_sha256"],
        )

    terminal = replace(selection, build_config=selection_config, run=seal_interrupted_selection)
    return {**outputs, "selection": terminal, "terminal": terminal}

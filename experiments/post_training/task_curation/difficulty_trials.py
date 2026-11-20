# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execute prepared difficulty tasks in Harbor's locked production runtime."""

import asyncio
import dataclasses
import hashlib
import json
import logging
import math
import os
import tarfile
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from finestore.rl.curation import ContentEvidenceReference
from finestore.rl.difficulty import (
    PROTOCOL_ID,
    PROTOCOLS,
    DifficultyAttempt,
    DifficultyGap,
    difficulty_gap_reason,
)
from harbor.environments.docker_translate import MissingPreparedImageError
from harbor.host_capabilities import HostCapabilities, host_capabilities_from_environment
from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig
from harbor.trial.trial import Trial
from harbor.utils.artifact_writer import get_artifact_writer
from harbor_config.errors import ErrorCategory, error_category
from pydantic import ValidationError
from rigging.filesystem.storage_path import StoragePath
from rigging.filesystem.transfer import copy
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.harbor.runtime import TaskNetworkPolicyReceipt, network_policy_qualified, read_source_runtime
from verifyit.modes.grade_math import MATH_DEADLINE_REFERENCE_PARSE, MATH_VERIFIER_DEADLINE

from experiments.post_training.task_curation.difficulty.launch import DATA_ADMISSION_COMMAND
from experiments.post_training.task_curation.difficulty_config import CandidateRun
from experiments.post_training.task_curation.difficulty_evidence import write_json_evidence, write_same_json_evidence
from experiments.post_training.task_curation.difficulty_prepared import PreparedTask
from experiments.post_training.task_curation.difficulty_sampling import request_parameters

LOGGER = logging.getLogger(__name__)
ATTEMPT_TIMEOUT = 259200
JUDGE_TEMPERATURE = PROTOCOLS[PROTOCOL_ID].temperature
INPUT_BUDGET_REASON = difficulty_gap_reason(DifficultyGap.PROTOCOL_INPUT_BUDGET_EXCEEDED)
PROVIDER_WAIT_REASON = difficulty_gap_reason(DifficultyGap.PROVIDER_CAPACITY_WAIT)
MISSING_PREPARED_IMAGE_REASON = difficulty_gap_reason(DifficultyGap.ENVIRONMENT_ARTIFACT_MISSING, "prepared_image")
UNSUPPORTED_COMPOSE_DOCUMENT_REASON = difficulty_gap_reason(DifficultyGap.UNSUPPORTED_COMPOSE_DOCUMENT)
NETWORK_POLICY_REASON = difficulty_gap_reason(DifficultyGap.NETWORK_POLICY_VIOLATION)
MATH_DEADLINE_LIMIT = 3


@dataclass(frozen=True)
class TaskOutcome:
    attempts: list[DifficultyAttempt]
    pending_reason: str | None


@dataclass(frozen=True)
class ModelTraffic:
    receipt: ContentEvidenceReference
    input_tokens: int
    output_tokens: int


IRIS_TASK_ID_ENV = "IRIS_TASK_ID"


def iris_root_job_id() -> str | None:
    """This root task's Iris job id from the runtime environment, when running under Iris.

    The task wire form is ``/user/job/<ordinal>:<attempt>``; stripping the task
    segment yields the job id every run record should carry so the report never
    has to guess it from job-name conventions.
    """
    wire = os.environ.get(IRIS_TASK_ID_ENV, "")
    if not wire:
        return None
    job_id = wire.rsplit("/", 1)[0]
    return job_id if job_id.startswith("/") else None


@dataclass
class RunSealState:
    """The checkpointed per-run seal: this attempt's own windows and token deltas.

    Rewritten after every aggregated source, so a killed worker still leaves a
    usable seal (own GPU-hours and token deltas survive even when the worker
    dies before completion.json); the in-flight source's window is excluded.
    """

    solver: str
    selection_source_ids: tuple[str, ...]
    started_at: str

    sources_aggregated: int = 0
    own_gpu_hours: float = 0.0
    own_input_tokens: int = 0
    own_output_tokens: int = 0
    judge_own_input_tokens: int = 0
    judge_own_output_tokens: int = 0
    sources: dict[str, dict] = dataclasses.field(default_factory=dict)

    def record(
        self,
        source_key: str,
        source_id: str,
        own_gpu_hours: float,
        traffic: ModelTraffic,
        judge_traffic: ModelTraffic | None,
    ) -> None:
        self.sources_aggregated += 1
        self.own_gpu_hours += own_gpu_hours
        self.own_input_tokens += traffic.input_tokens
        self.own_output_tokens += traffic.output_tokens
        if judge_traffic is not None:
            self.judge_own_input_tokens += judge_traffic.input_tokens
            self.judge_own_output_tokens += judge_traffic.output_tokens
        row = {
            "source_id": source_id,
            "own_gpu_hours": round(own_gpu_hours, 6),
            "own_input_tokens": traffic.input_tokens,
            "own_output_tokens": traffic.output_tokens,
        }
        if judge_traffic is not None:
            row["judge_own_input_tokens"] = judge_traffic.input_tokens
            row["judge_own_output_tokens"] = judge_traffic.output_tokens
        self.sources[source_key] = row

    def payload(self) -> dict:
        return {
            "solver": self.solver,
            "iris_job_id": iris_root_job_id(),
            "started_at": self.started_at,
            "updated_at": datetime.now(UTC).isoformat(),
            "selection_source_ids": list(self.selection_source_ids),
            "sources_aggregated": self.sources_aggregated,
            "own_gpu_hours": round(self.own_gpu_hours, 6),
            "own_input_tokens": self.own_input_tokens,
            "own_output_tokens": self.own_output_tokens,
            "judge_own_input_tokens": self.judge_own_input_tokens,
            "judge_own_output_tokens": self.judge_own_output_tokens,
            "gpu_hour_basis": (
                "sum of this attempt's own recorded source trial windows; startup, teardown "
                "and interrupted source windows excluded"
            ),
            "token_basis": (
                "this attempt's candidate and judge relay traffic; totals inherited from the "
                "campaign chain remain in the per-source ledgers"
            ),
            "sources": self.sources,
        }


def archive_model_traffic(
    requests: Path,
    tasks: list[PreparedTask],
    attempts_per_task: int,
    source_key: str,
    archive: Path,
    destination: StoragePath,
) -> ModelTraffic:
    """Archive one source's known model replies and count their reported usage."""
    input_tokens = output_tokens = 0
    archive.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    with tarfile.open(archive, "w:gz", compresslevel=1) as stream:
        for task in tasks:
            task_key = hashlib.sha256(task.task_id.encode()).hexdigest()
            for number in range(attempts_per_task):
                route = f"/{source_key}-{task_key}-{number}"
                folder = requests / hashlib.sha256(route.encode()).hexdigest()
                if not folder.exists():
                    continue
                stream.add(folder, arcname=folder.name)
                for response_path in folder.glob("*/response.json"):
                    response = json.loads(response_path.read_text())
                    usage = response.get("upstream_usage") or response["body"].get("usage") or {}
                    input_tokens += usage.get("prompt_tokens", 0)
                    output_tokens += usage.get("completion_tokens", 0)
    copy(str(archive), str(destination), no_clobber=True)
    return ModelTraffic(
        ContentEvidenceReference(uri=str(destination), sha256=file_sha256(StoragePath(str(archive)))),
        input_tokens,
        output_tokens,
    )


def seal_model_traffic(root: Path, output: StoragePath) -> None:
    """Archive the drained traffic and its usage totals; safe to call repeatedly."""
    requests = root / "model-requests"
    archive = root / "model-requests-final.tar.gz"
    with tarfile.open(archive, "w:gz", compresslevel=1) as stream:
        if requests.exists():
            stream.add(requests, arcname="model-requests")
    destination = output / "model-requests-final.tar.gz"
    copy(str(archive), str(destination))
    input_tokens = output_tokens = responses_with_usage = 0
    for path in requests.glob("*/*/response.json"):
        response = json.loads(path.read_text())
        usage = response.get("upstream_usage") or response["body"].get("usage") or {}
        responses_with_usage += bool(usage)
        input_tokens += usage.get("prompt_tokens", 0)
        output_tokens += usage.get("completion_tokens", 0)
    write_replaced_json_evidence(
        output / "model-requests-final.json",
        {
            "archive": str(destination),
            "archive_sha256": file_sha256(StoragePath(str(archive))),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "received_responses": len(list(requests.glob("*/*/upstream-response.body"))),
            "responses_with_usage": responses_with_usage,
            "basis": "sum of reported usage after relay shutdown; raw replies without usage remain archived",
        },
    )


@contextmanager
def preserved_model_traffic(root: Path, output: StoragePath):
    """Archive final model traffic after the relay drains, including exceptional exits."""
    try:
        yield
    finally:
        seal_model_traffic(root, output)


def write_replaced_json_evidence(path: StoragePath, value: dict) -> ContentEvidenceReference:
    """Replace this campaign's own rolling seal; the newest seal is authoritative."""
    if path.exists():
        path.rm()
    return write_json_evidence(path, value)


def scored_reward(result, trial: Trial) -> float | None:
    """Accept a completed source grade only when no infrastructure failure invalidates it."""
    if result.finished_at is None or result.completion is None or result.verifier_result is None:
        return None
    if result.exception_info is not None and error_category(result.exception_info.exception_type) not in {
        ErrorCategory.AGENT,
        ErrorCategory.PASSTHROUGH,
    }:
        return None
    reward = (result.verifier_result.rewards or {}).get("reward")
    if type(reward) not in {int, float} or not math.isfinite(reward):
        return None
    verdict_path = trial.paths.verifier_dir / "verdict.json"
    if verdict_path.exists():
        verdict = json.loads(verdict_path.read_text())
        if verdict.get("status") != "scored" or verdict.get("reward") != reward:
            return None
    return float(reward)


def math_deadline_phase(trial: Trial) -> str | None:
    """The grading phase whose deadline fired, or None when this is not a math deadline.

    Reference-parse deadlines are a task property; candidate-parse and comparison
    deadlines follow the reply (the #11511 replay), so only the reference phase may
    drive the terminal block.
    """
    path = trial.paths.verifier_dir / "verdict.json"
    if not path.exists():
        return None
    verdict = json.loads(path.read_text())
    detail = verdict.get("detail")
    if verdict.get("status") != "infra_error" or not isinstance(detail, dict):
        return None
    if detail.get("code") != MATH_VERIFIER_DEADLINE:
        return None
    return detail.get("phase")


def worker_host_capabilities(
    authentication: dict[str, str], environment: Mapping[str, str] | None = None
) -> HostCapabilities:
    """Bind Difficulty's Daytona authority exclusively to the DATA credential."""
    source = dict(os.environ if environment is None else environment)
    source.pop("DAYTONA_API_KEY", None)
    if source.get("DAYTONA_DATA_API_KEY"):
        source["DAYTONA_API_KEY"] = source["DAYTONA_DATA_API_KEY"]
    return host_capabilities_from_environment(
        source,
        model_authentication=authentication,
    )


async def execute_trial(configuration: TrialConfig, authentication: dict[str, str]):
    trial = await Trial.create(
        configuration,
        host_capabilities=worker_host_capabilities(authentication),
    )
    return trial, await trial.run()


def run_task(
    task: PreparedTask,
    config: CandidateRun,
    base_url: str,
    root: Path,
    output: str,
    judge_url: str | None,
    prior_math_deadlines: int,
    record_attempt: Callable[[StoragePath], None] | None = None,
) -> TaskOutcome:
    key = hashlib.sha256(task.task_id.encode()).hexdigest()
    destination = StoragePath(output) / key
    if task.reason is not None or task.path is None:
        write_same_json_evidence(destination / "pending.json", {"task_id": task.task_id, "reason": task.reason})
        return TaskOutcome(
            [DifficultyAttempt(task_id=task.task_id, status="pending", reward=None, reason=task.reason)], task.reason
        )
    attempts = []
    pending_reason = None
    math_deadlines = prior_math_deadlines
    for number in range(config.attempts_per_task):
        local = root / key / f"{number:02d}"
        local.mkdir(parents=True, exist_ok=False)
        endpoint = f"{base_url}/{root.name}-{key}-{number}/v1"
        parameters = request_parameters(config.settings)
        common = {"api_base": endpoint}
        if task.workload == "task_spec":
            kwargs = {
                **common,
                "max_turns": 50,
                "max_tokens": config.settings.output_tokens,
                "request_parameters": parameters,
            }
            agent_name = "task-spec-chat"
        else:
            kwargs = {
                **common,
                "max_turns": 50,
                "temperature": config.settings.temperature,
                "model_info": {
                    "max_input_tokens": config.settings.context_tokens,
                    "max_output_tokens": config.settings.output_tokens,
                    "input_cost_per_token": 0.0,
                    "output_cost_per_token": 0.0,
                },
                "llm_call_kwargs": parameters,
            }
            agent_name = "terminus-2"

        options = dict(config.environment_options)
        authentication = {endpoint: "EMPTY"}
        if task.backend == "daytona":
            # Daytona tasks ignore the worker-local options; their environment
            # takes the provisioning mode and the attempt timeout. The judge
            # and candidate relay endpoints are added below when judged.
            options = {
                "provisioning": config.daytona_provisioning,
                "trial_attempt_timeout_sec": ATTEMPT_TIMEOUT,
                "create_admission_command": list(DATA_ADMISSION_COMMAND),
            }
        if task.judge_model is not None:
            if judge_url is None or config.judge is None:
                raise ValueError("A task judge requires the protocol's host-side relay")
            judge_endpoint = f"{judge_url}/{root.name}-{key}-{number}/v1"
            options.update(model_endpoint=judge_endpoint, model_name=task.judge_model or config.judge.api_model)
            authentication[judge_endpoint] = "EMPTY"
        configuration = TrialConfig(
            task=TaskConfig(path=Path(task.path)),
            trial_name="candidate",
            trials_dir=local,
            trial_attempt_timeout_sec=ATTEMPT_TIMEOUT,
            agent=AgentConfig(
                name=agent_name,
                model_name=f"openai/{config.identity.model}@{config.identity.revision}",
                model_alias=f"openai/{config.api_model}",
                kwargs=kwargs,
            ),
            environment=EnvironmentConfig(type=task.backend, kwargs=options),
        )

        # Framework exceptions propagate and stop dispatch; they are never fabricated as solver zeros.
        construction_reason = None
        try:
            trial, result = asyncio.run(execute_trial(configuration, authentication))
        except MissingPreparedImageError as error:
            construction_reason = MISSING_PREPARED_IMAGE_REASON
            exception_type = type(error).__name__
            message = str(error)
        except ValidationError as error:
            # Only a task's own compose document is a task-input condition; our own
            # configuration validation errors are code bugs and must propagate.
            if error.title not in {"ComposeDocument", "ComposeService"}:
                raise
            construction_reason = UNSUPPORTED_COMPOSE_DOCUMENT_REASON
            exception_type = type(error).__name__
            message = (
                f"compose document declares keys Harbor's translator does not support: {error.error_count()} errors"
            )
        if construction_reason is not None:
            write_json_evidence(
                destination / f"{number:02d}" / "construction-error.json",
                {
                    "task_id": task.task_id,
                    "runtime_sha256": task.runtime_sha256,
                    "backend": task.backend,
                    "exception_type": exception_type,
                    "reason": construction_reason,
                    "message": message,
                },
            )
            attempts.append(
                DifficultyAttempt(task_id=task.task_id, status="pending", reward=None, reason=construction_reason)
            )
            return TaskOutcome(attempts, construction_reason)
        get_artifact_writer().flush(timeout=300)
        archive = local / "trial-artifacts.tar.gz"
        with tarfile.open(archive, "w:gz", compresslevel=1) as stream:
            stream.add(local / "candidate", arcname="trial")
        archive_target = destination / f"{number:02d}" / "trial-artifacts.tar.gz"
        if archive_target.exists():
            raise FileExistsError(archive_target)
        copy(str(archive), str(archive_target), no_clobber=True)
        raw_receipt = {
            "task_id": task.task_id,
            "result": result.model_dump(mode="json"),
            "archive": str(archive_target),
            "archive_sha256": file_sha256(StoragePath(str(archive))),
        }
        write_json_evidence(destination / f"{number:02d}" / "trial.json", raw_receipt)
        reward = scored_reward(result, trial)
        unserved = False
        provider_wait = False
        model_request_failed = False
        candidate_served = False
        route = f"/{root.name}-{key}-{number}"
        for traffic in (root.parent / "model-requests", root.parent / "judge" / "model-requests"):
            requests = sorted((traffic / hashlib.sha256(route.encode()).hexdigest()).glob("*"))
            responses = [
                json.loads((request / "response.json").read_text())
                for request in requests
                if (request / "response.json").exists()
            ]
            if traffic == root.parent / "model-requests":
                candidate_served = any(response["status"] == 200 for response in responses)
            if not requests:
                continue
            last = requests[-1]
            if not (last / "response.json").exists():
                model_request_failed = True
                provider_wait |= (
                    (last / "provider-queued.json").exists() and not (last / "provider-admitted.json").exists()
                ) or any((last / "provider-waits").glob("*.json"))
                continue
            response = json.loads((last / "response.json").read_text())
            model_request_failed |= response["status"] != 200
            error = response["body"].get("error")
            if isinstance(error, dict):
                unserved |= error.get("code") == "input_budget_unserved"
                provider_wait |= error.get("code") == "provider_wait"
        if result.exception_info is not None and not candidate_served:
            model_request_failed = True
        if reward is not None and reward != 0 and candidate_served:
            provider_wait = False
            model_request_failed = False
        if unserved:
            reward = None
            pending_reason = INPUT_BUDGET_REASON
        elif provider_wait:
            reward = None
            pending_reason = PROVIDER_WAIT_REASON
        elif model_request_failed:
            reward = None
        if task.workload == "task_spec":
            binding = read_source_runtime(Path(task.path))
            policy = TaskNetworkPolicyReceipt.model_validate_json(
                (trial.paths.trial_dir / "network-policy.json").read_text()
            )
            if binding is None or binding.runtime_sha256 != task.runtime_sha256:
                raise ValueError("Trial network policy is not bound to the selected source runtime")
            if reward is not None and not network_policy_qualified(binding, policy):
                # The task's environment changed its declared policy at runtime: the
                # grade is invalid but the condition belongs to the task, not the fleet.
                reward = None
                pending_reason = NETWORK_POLICY_REASON
        deadline_phase = None if reward is not None or unserved or provider_wait else math_deadline_phase(trial)
        # Only a reference-parse deadline is a task property; candidate-parse and
        # comparison deadlines follow the reply and must not terminal-block (#11511).
        reference_deadline = deadline_phase == MATH_DEADLINE_REFERENCE_PARSE
        math_deadlines = math_deadlines + 1 if reference_deadline else 0
        terminal_deadline = reference_deadline and math_deadlines >= MATH_DEADLINE_LIMIT
        reason = (
            difficulty_gap_reason(DifficultyGap.MATH_VERIFIER_DEADLINE, deadline_phase)
            if deadline_phase is not None
            else pending_reason if reward is None else None
        )
        attempt = DifficultyAttempt(
            task_id=task.task_id,
            status=(
                "pending"
                if unserved or provider_wait
                else "scored" if reward is not None else "blocked" if terminal_deadline else "infrastructure"
            ),
            reward=reward,
            reason=reason,
        )
        write_json_evidence(
            destination / f"{number:02d}" / "attempt.json",
            {
                **attempt.model_dump(mode="json"),
                **raw_receipt,
                "judge_temperature": JUDGE_TEMPERATURE if task.judge_model is not None else None,
            },
        )
        if record_attempt is not None:
            record_attempt(destination / f"{number:02d}" / "attempt.json")
        attempts.append(attempt)
        if reward is not None or unserved or provider_wait or terminal_deadline:
            break
    return TaskOutcome(attempts, pending_reason)

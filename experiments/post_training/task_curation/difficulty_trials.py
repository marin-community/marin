# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execute prepared difficulty tasks in Harbor's locked production runtime."""

import argparse
import asyncio
import dataclasses
import hashlib
import io
import json
import logging
import math
import os
import tarfile
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from finestore.rl.curation import ContentEvidenceReference
from finestore.rl.difficulty import (
    MEASUREMENTS,
    PROTOCOL_ID,
    PROTOCOLS,
    DifficultyAttempt,
    DifficultyGap,
    DifficultyMeasurement,
    difficulty_gap_reason,
    join_difficulty_gap_reasons,
    measurement_from_attempts,
    register_difficulty,
    require_difficulty_gaps,
)
from finestore.store import DataStore
from harbor.environments.docker_translate import MissingPreparedImageError
from harbor.host_capabilities import HostCapabilities, host_capabilities_from_environment
from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig
from harbor.trial.trial import Trial
from harbor.utils.artifact_writer import get_artifact_writer
from harbor_config.errors import ErrorCategory, error_category
from pydantic import ValidationError
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.filesystem.transfer import copy
from rigging.secrets import require_secret
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.harbor.runtime import TaskNetworkPolicyReceipt, network_policy_qualified, read_source_runtime
from taskcompendium.request_budget import RequestBudget
from verifyit.modes.grade_math import MATH_DEADLINE_REFERENCE_PARSE, MATH_VERIFIER_DEADLINE

from experiments.post_training.task_curation.difficulty_config import CandidateRun
from experiments.post_training.task_curation.difficulty_evidence import write_json_evidence, write_same_json_evidence
from experiments.post_training.task_curation.difficulty_prepared import PreparedSource, PreparedTask
from experiments.post_training.task_curation.difficulty_recovery import (
    CampaignParent,
    previous_campaign,
    recovered_source,
)
from experiments.post_training.task_curation.difficulty_relay import budget_relay
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
    # A resumed run's per-source archive continues the killed attempt's archive:
    # its existing entries are carried forward so the destination keeps covering
    # every route this run version served, instead of colliding with it.
    prior_entries = []
    if destination.exists():
        with tarfile.open(fileobj=io.BytesIO(destination.read_bytes()), mode="r:gz") as prior:
            for member in prior.getmembers():
                prior_entries.append((member, None if member.isdir() else prior.extractfile(member)))
    with tarfile.open(archive, "w:gz", compresslevel=1) as stream:
        for member, data in prior_entries:
            stream.addfile(member, data)
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
    copy(str(archive), str(destination), no_clobber=not prior_entries)
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
    """Resolve host authority from the trial worker's own environment.

    The worker inherits DAYTONA_API_KEY/DAYTONA_API_URL/DAYTONA_TARGET from the
    campaign; passing an empty snapshot here would silently drop the Daytona
    connection and fail every Daytona attempt at client creation.
    """
    return host_capabilities_from_environment(
        os.environ if environment is None else environment,
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
    # A resumed run continues after a killed attempt's durable per-attempt
    # evidence instead of colliding with it.
    existing = sorted(int(path.name) for path in (destination / "*").glob() if path.name.isdigit())
    first_number = existing[-1] + 1 if existing else 0
    for number in range(first_number, first_number + config.attempts_per_task):
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
        attempts.append(attempt)
        if reward is not None or unserved or provider_wait or terminal_deadline:
            break
    return TaskOutcome(attempts, pending_reason)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--local-root", required=True, type=Path)
    parser.add_argument("--previous-campaign")
    parser.add_argument("--previous-serving-sha256")
    parser.add_argument("--source-queue-sha256")
    args = parser.parse_args()
    configure_coreweave_s3()
    config = CandidateRun.model_validate_json(Path(args.config).read_text())
    sources = [PreparedSource.model_validate(row) for row in json.loads(Path(args.sources).read_text())]
    recovery_arguments = (args.previous_campaign, args.previous_serving_sha256, args.source_queue_sha256)
    if any(recovery_arguments) and not all(recovery_arguments):
        raise ValueError("Continuation requires the previous campaign, serving digest and current source queue digest")
    previous = previous_campaign(*recovery_arguments) if all(recovery_arguments) else None
    output = StoragePath(args.output_root)
    # A retried root resumes from its own trials output: the campaign binds the
    # root as the recovery parent, so the existing per-source evidence is the
    # chain being continued, not a conflicting prior campaign.
    resumed = previous is not None and str(previous.root) == str(output.parent)
    if output.exists() and not resumed:
        raise FileExistsError("Preserve previous difficulty attempts and select a new version")
    if args.local_root.exists():
        raise FileExistsError("Preserve previous difficulty attempts and select a new version")
    args.local_root.mkdir(parents=True)
    if previous is not None and not resumed:
        write_json_evidence(
            output / "parent-campaign.json",
            CampaignParent(
                root=str(previous.root),
                source_queue_sha256=args.source_queue_sha256,
                serving_receipt={"uri": str(previous.root / "serving.json"), "sha256": args.previous_serving_sha256},
            ).model_dump(mode="json"),
        )
    key = "EMPTY" if config.api_key_secret is None else require_secret(config.api_key_secret)
    run_seal = RunSealState(
        solver=config.solver,
        selection_source_ids=tuple(source.sample.source_id for source in sources),
        started_at=datetime.now(UTC).isoformat(),
    )
    write_replaced_json_evidence(output / "run-seal.json", run_seal.payload())
    has_judge = any(task.judge_model is not None and task.reason is None for source in sources for task in source.tasks)
    judge = config.judge if has_judge else None
    if has_judge and judge is None:
        raise ValueError("Eligible judges require an explicit production judge configuration")
    judge_root = args.local_root / "judge"
    if has_judge:
        judge_root.mkdir()
    with (
        preserved_model_traffic(args.local_root, output),
        preserved_model_traffic(judge_root, output / "judge") if has_judge else nullcontext(),
        budget_relay(
            endpoint=config.endpoint,
            model=config.api_model,
            api_key=key,
            tokenizer_model=config.identity.tokenizer,
            tokenizer_revision=config.identity.tokenizer_revision,
            parameters=request_parameters(config.settings),
            budget=RequestBudget(
                config.settings.context_tokens, config.settings.input_tokens, config.settings.output_tokens
            ),
            provider_limits=config.provider_limits,
            receipts=args.local_root / "model-requests",
        ) as relay,
        (
            budget_relay(
                endpoint=judge.endpoint,
                model=judge.api_model,
                api_key=require_secret(judge.api_key_secret) if judge.api_key_secret is not None else "EMPTY",
                tokenizer_model=judge.identity.tokenizer,
                tokenizer_revision=judge.identity.tokenizer_revision,
                parameters={"temperature": JUDGE_TEMPERATURE},
                budget=None,
                provider_limits=judge.provider_limits,
                receipts=judge_root / "model-requests",
            )
            if judge is not None
            else nullcontext()
        ) as judge_relay,
    ):
        with ThreadPoolExecutor(max_workers=config.workers) as executor:
            for source in sources:
                started = time.monotonic()
                source_key = hashlib.sha256(source.sample.source_id.encode()).hexdigest()
                if resumed and (output / source_key / "measurement.json").exists():
                    # The killed attempt already published this source's durable
                    # measurement and catalog row; re-processing it would replay
                    # scored tasks and duplicate the row.
                    LOGGER.info(
                        "%s %s: already measured by the retried attempt; skipping",
                        config.solver,
                        source.sample.source_id,
                    )
                    continue
                recovered = (
                    recovered_source(
                        previous,
                        source,
                        config.solver,
                        config.identity,
                        config.settings,
                        request_parameters(config.settings),
                        None if config.judge is None else config.judge.identity,
                    )
                    if previous is not None
                    else None
                )
                old_attempts = recovered.attempts if recovered is not None else []
                terminal = {attempt.task_id for attempt in old_attempts if attempt.status in {"scored", "blocked"}}
                selected_tasks = source.tasks if config.task_limit is None else source.tasks[: config.task_limit]
                tasks = [task for task in selected_tasks if task.task_id not in terminal]
                futures = [
                    executor.submit(
                        run_task,
                        task,
                        config,
                        relay,
                        args.local_root / source_key,
                        str(output / source_key / "attempts"),
                        judge_relay,
                        0 if recovered is None else recovered.math_deadline_streaks.get(task.task_id, 0),
                    )
                    for task in tasks
                ]
                try:
                    outcomes = [future.result() for future in futures]
                except BaseException:
                    for future in futures:
                        future.cancel()
                    raise
                attempts = old_attempts + [attempt for outcome in outcomes for attempt in outcome.attempts]
                pending_reasons = {
                    task.task_id: outcome.pending_reason
                    for task, outcome in zip(tasks, outcomes, strict=True)
                    if outcome.pending_reason is not None
                }
                previous_measurement = None if recovered is None else recovered.measurement
                traffic = archive_model_traffic(
                    args.local_root / "model-requests",
                    tasks,
                    config.attempts_per_task,
                    source_key,
                    args.local_root / source_key / "model-requests.tar.gz",
                    output / source_key / "model-requests.tar.gz",
                )
                source_has_judge = any(task.judge_model is not None and task.reason is None for task in tasks)
                judge_traffic = (
                    archive_model_traffic(
                        judge_root / "model-requests",
                        tasks,
                        config.attempts_per_task,
                        source_key,
                        args.local_root / source_key / "judge-requests.tar.gz",
                        output / source_key / "judge-requests.tar.gz",
                    )
                    if source_has_judge
                    else None
                )
                input_tokens = (
                    traffic.input_tokens
                    + (0 if previous_measurement is None else previous_measurement.input_tokens)
                    + (0 if recovered is None else recovered.prior_input_tokens)
                )
                output_tokens = (
                    traffic.output_tokens
                    + (0 if previous_measurement is None else previous_measurement.output_tokens)
                    + (0 if recovered is None else recovered.prior_output_tokens)
                )
                judge_input_tokens = (
                    0
                    if judge_traffic is None
                    else judge_traffic.input_tokens + (0 if recovered is None else recovered.prior_judge_input_tokens)
                )
                judge_output_tokens = (
                    0
                    if judge_traffic is None
                    else judge_traffic.output_tokens + (0 if recovered is None else recovered.prior_judge_output_tokens)
                )
                attempt_receipts = [] if recovered is None else list(recovered.attempt_receipts)
                for path in sorted((output / source_key / "attempts" / "*" / "*" / "attempt.json").glob(), key=str):
                    ref = ContentEvidenceReference(uri=str(path), sha256=file_sha256(path))
                    # A resumed run's recovery already gathered these receipts;
                    # duplicates would replay the same attempt twice downstream.
                    if ref not in attempt_receipts:
                        attempt_receipts.append(ref)
                # This campaign's own rolling per-source ledger: a resumed run
                # replaces the killed attempt's interrupted aggregate, which its
                # attempts and token totals are fully carried into.
                receipt = write_replaced_json_evidence(
                    output / source_key / "attempts.json",
                    {
                        "attempts": [item.model_dump(mode="json") for item in attempts],
                        "pending_reasons": pending_reasons,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "gpu_hour_basis": (
                            "sum of recorded source trial windows; "
                            "startup, teardown and interrupted source windows excluded"
                        ),
                        "model_requests": traffic.receipt.uri,
                        "model_requests_sha256": traffic.receipt.sha256,
                        "judge_model_requests": (
                            None if judge_traffic is None else judge_traffic.receipt.model_dump(mode="json")
                        ),
                        "judge_input_tokens": None if judge_traffic is None else judge_input_tokens,
                        "judge_output_tokens": None if judge_traffic is None else judge_output_tokens,
                        "judge_temperature": JUDGE_TEMPERATURE if source_has_judge else None,
                        "judge_token_basis": (
                            "this source window; prior campaign traffic remains in linked receipts"
                            if source_has_judge
                            else None
                        ),
                        "attempt_receipts": [ref.model_dump(mode="json") for ref in attempt_receipts],
                        "previous_campaign": None if previous is None else str(previous.root),
                    },
                )
                # Durably seal the traffic drained so far after each source, so a
                # killed worker loses at most its in-flight source's traffic.
                seal_model_traffic(args.local_root, output)
                if has_judge:
                    seal_model_traffic(judge_root, output / "judge")
                own_window_gpu_hours = config.gpu_count * (time.monotonic() - started) / 3600
                payload = dict(
                    measurement_id=uuid4().hex,
                    protocol_id=PROTOCOL_ID,
                    created_at=datetime.now(UTC).isoformat(),
                    source_id=source.sample.source_id,
                    release=source.sample.release,
                    converter_version=source.sample.converter_version,
                    solver=config.solver,
                    status="pending",
                    reason=difficulty_gap_reason(DifficultyGap.AWAITING_SCORED_COVERAGE),
                    sample=source.sample,
                    sample_receipt=source.sample_receipt,
                    model=config.identity,
                    settings=config.settings,
                    n=0,
                    solved=0,
                    wilson_95=None,
                    infrastructure_attempts=0,
                    pending_tasks=len(source.sample.tasks),
                    attempts_receipt=receipt,
                    gpu_hours=(
                        own_window_gpu_hours + (0 if previous_measurement is None else previous_measurement.gpu_hours)
                    ),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
                measurement = measurement_from_attempts(DifficultyMeasurement(**payload), attempts)
                environment_blocked_tasks = {task.task_id for task in tasks if task.reason is not None}
                environment_blocked_tasks.update(
                    task.task_id
                    for task, outcome in zip(tasks, outcomes, strict=True)
                    if outcome.pending_reason
                    in {
                        MISSING_PREPARED_IMAGE_REASON,
                        UNSUPPORTED_COMPOSE_DOCUMENT_REASON,
                        NETWORK_POLICY_REASON,
                    }
                )
                environment_blockers = sorted({pending_reasons[task_id] for task_id in environment_blocked_tasks})
                if not terminal and tasks and len(environment_blocked_tasks) == len(tasks):
                    blockers_reason = join_difficulty_gap_reasons(environment_blockers)
                    require_difficulty_gaps(blockers_reason)
                    measurement = DifficultyMeasurement.model_validate(
                        {
                            **measurement.model_dump(mode="json"),
                            "status": "not_measured",
                            "reason": blockers_reason,
                        }
                    )
                elif pending_reasons:
                    pending_reason = join_difficulty_gap_reasons(list(pending_reasons.values()))
                    require_difficulty_gaps(pending_reason)
                    measurement = measurement.model_copy(update={"reason": pending_reason})
                with DataStore.open(str(output / "catalog")) as store:
                    register_difficulty(store)
                    with store.transaction() as transaction:
                        transaction.table(MEASUREMENTS).add(measurement.model_dump(mode="json"))
                write_json_evidence(output / source_key / "measurement.json", measurement.model_dump(mode="json"))
                run_seal.record(
                    source_key,
                    source.sample.source_id,
                    own_window_gpu_hours,
                    traffic,
                    judge_traffic,
                )
                write_replaced_json_evidence(output / "run-seal.json", run_seal.payload())
                LOGGER.info(
                    "%s %s: %s, scored=%d, solved=%d, pending=%d",
                    config.solver,
                    source.sample.source_id,
                    measurement.status,
                    measurement.n,
                    measurement.solved,
                    measurement.pending_tasks,
                )


if __name__ == "__main__":
    main()

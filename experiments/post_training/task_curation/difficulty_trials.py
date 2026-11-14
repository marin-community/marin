# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execute prepared difficulty tasks in Harbor's locked production runtime."""

import argparse
import asyncio
import hashlib
import json
import logging
import math
import tarfile
import time
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
    DifficultyMeasurement,
    SamplingSettings,
    Solver,
    SolverIdentity,
    measurement_from_attempts,
    register_difficulty,
)
from finestore.store import DataStore
from harbor.host_capabilities import host_capabilities_from_environment
from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig
from harbor.trial.trial import Trial
from harbor.utils.artifact_writer import get_artifact_writer
from harbor_config.errors import ErrorCategory, error_category
from pydantic import BaseModel, ConfigDict
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.filesystem.transfer import copy
from rigging.secrets import require_secret
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.harbor.runtime import TaskNetworkPolicyReceipt, network_policy_qualified, read_source_runtime

from experiments.post_training.task_curation.difficulty_evidence import write_json_evidence
from experiments.post_training.task_curation.difficulty_prepared import PreparedSource, PreparedTask
from experiments.post_training.task_curation.difficulty_recovery import previous_campaign, recovered_source
from experiments.post_training.task_curation.difficulty_relay import CANDIDATE_BUDGET, budget_relay

LOGGER = logging.getLogger(__name__)
ATTEMPT_TIMEOUT = 259200
JUDGE_TEMPERATURE = PROTOCOLS[PROTOCOL_ID].temperature
INPUT_BUDGET_REASON = (
    "A candidate prompt exceeded the protocol input budget before dispatch. No solver score was assigned."
)


@dataclass(frozen=True)
class TaskOutcome:
    attempts: list[DifficultyAttempt]
    pending_reason: str | None


@dataclass(frozen=True)
class ModelTraffic:
    receipt: ContentEvidenceReference
    input_tokens: int
    output_tokens: int


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
                    usage = json.loads(response_path.read_text())["body"].get("usage") or {}
                    input_tokens += usage.get("prompt_tokens", 0)
                    output_tokens += usage.get("completion_tokens", 0)
    copy(str(archive), str(destination), no_clobber=True)
    return ModelTraffic(
        ContentEvidenceReference(uri=str(destination), sha256=file_sha256(StoragePath(str(archive)))),
        input_tokens,
        output_tokens,
    )


class CandidateRun(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    solver: Solver
    identity: SolverIdentity
    settings: SamplingSettings
    endpoint: str
    api_model: str
    api_key_secret: str | None
    request_parameters: dict
    environment_options: dict
    judge_endpoint: str
    judge_api_key_secret: str
    judge_api_model: str
    judge_identity: SolverIdentity
    workers: int
    gpu_count: int
    attempts_per_task: int
    task_limit: int | None


@contextmanager
def preserved_model_traffic(root: Path, output: StoragePath):
    """Archive final model traffic after the relay drains, including exceptional exits."""
    try:
        yield
    finally:
        requests = root / "model-requests"
        archive = root / "model-requests-final.tar.gz"
        with tarfile.open(archive, "w:gz", compresslevel=1) as stream:
            if requests.exists():
                stream.add(requests, arcname="model-requests")
        destination = output / "model-requests-final.tar.gz"
        copy(str(archive), str(destination), no_clobber=True)
        input_tokens = output_tokens = responses_with_usage = 0
        for path in requests.glob("*/*/response.json"):
            usage = json.loads(path.read_text())["body"].get("usage") or {}
            responses_with_usage += bool(usage)
            input_tokens += usage.get("prompt_tokens", 0)
            output_tokens += usage.get("completion_tokens", 0)
        write_json_evidence(
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


async def execute_trial(configuration: TrialConfig, authentication: dict[str, str]):
    trial = await Trial.create(
        configuration,
        host_capabilities=host_capabilities_from_environment(
            {},
            model_authentication=authentication,
        ),
    )
    return trial, await trial.run()


def run_task(
    task: PreparedTask, config: CandidateRun, base_url: str, root: Path, output: str, judge_url: str | None
) -> TaskOutcome:
    key = hashlib.sha256(task.task_id.encode()).hexdigest()
    destination = StoragePath(output) / key
    if task.reason is not None or task.path is None:
        write_json_evidence(destination / "pending.json", {"task_id": task.task_id, "reason": task.reason})
        return TaskOutcome([DifficultyAttempt(task_id=task.task_id, status="pending", reward=None)], task.reason)
    attempts = []
    pending_reason = None
    for number in range(config.attempts_per_task):
        local = root / key / f"{number:02d}"
        local.mkdir(parents=True, exist_ok=False)
        endpoint = f"{base_url}/{root.name}-{key}-{number}/v1"
        parameters = config.request_parameters
        common = {"api_base": endpoint}
        if task.workload == "task_spec":
            kwargs = {
                **common,
                "max_turns": 50,
                "max_tokens": config.settings.output_tokens,
                "context_window": config.settings.context_tokens,
                "max_input_tokens": config.settings.input_tokens,
                "request_parameters": parameters,
                "tokenizer_model": config.identity.tokenizer,
                "tokenizer_revision": config.identity.tokenizer_revision,
            }
            agent_name = "task-spec-chat"
        else:
            kwargs = {
                **common,
                "max_turns": 50,
                "temperature": config.settings.temperature,
                "model_info": {
                    "max_input_tokens": config.settings.input_tokens,
                    "max_output_tokens": config.settings.output_tokens,
                    "input_cost_per_token": 0.0,
                    "output_cost_per_token": 0.0,
                },
                "llm_call_kwargs": parameters,
            }
            agent_name = "terminus-2"

        options = dict(config.environment_options)
        authentication = {endpoint: "EMPTY"}
        if task.judge_model is not None:
            if judge_url is None:
                raise ValueError("A task judge requires the protocol's host-side relay")
            judge_endpoint = f"{judge_url}/{root.name}-{key}-{number}/v1"
            options.update(model_endpoint=judge_endpoint, model_name=task.judge_model or config.judge_api_model)
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
        trial, result = asyncio.run(execute_trial(configuration, authentication))
        reward = scored_reward(result, trial)
        unserved = (trial.paths.agent_dir / "candidate-budget-rejection.json").exists()
        route = f"/{root.name}-{key}-{number}"
        requests = root.parent / "model-requests" / hashlib.sha256(route.encode()).hexdigest()
        for response_path in requests.glob("*/response.json"):
            response = json.loads(response_path.read_text())
            if response["body"].get("error", {}).get("code") == "input_budget_unserved":
                unserved = True
        if unserved:
            reward = None
            pending_reason = INPUT_BUDGET_REASON
        get_artifact_writer().flush(timeout=300)
        archive = local / "trial-artifacts.tar.gz"
        with tarfile.open(archive, "w:gz", compresslevel=1) as stream:
            stream.add(local / "candidate", arcname="trial")
        archive_target = destination / f"{number:02d}" / "trial-artifacts.tar.gz"
        if archive_target.exists():
            raise FileExistsError(archive_target)
        copy(str(archive), str(archive_target), no_clobber=True)
        if task.workload == "task_spec":
            binding = read_source_runtime(Path(task.path))
            policy = TaskNetworkPolicyReceipt.model_validate_json(
                (trial.paths.trial_dir / "network-policy.json").read_text()
            )
            if binding is None or binding.runtime_sha256 != task.runtime_sha256:
                raise ValueError("Trial network policy is not bound to the selected source runtime")
            if reward is not None and not network_policy_qualified(binding, policy):
                raise ValueError("Trial changed an original declared network policy")
        attempt = DifficultyAttempt(
            task_id=task.task_id,
            status="pending" if unserved else ("scored" if reward is not None else "infrastructure"),
            reward=reward,
        )
        write_json_evidence(
            destination / f"{number:02d}" / "attempt.json",
            {
                **attempt.model_dump(mode="json"),
                "reason": "input_budget_unserved" if unserved else None,
                "result": result.model_dump(mode="json"),
                "archive": str(archive_target),
                "archive_sha256": file_sha256(StoragePath(str(archive))),
                "judge_temperature": JUDGE_TEMPERATURE if task.judge_model is not None else None,
            },
        )
        attempts.append(attempt)
        if reward is not None or unserved:
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
    if output.exists() or args.local_root.exists():
        raise FileExistsError("Preserve previous difficulty attempts and select a new version")
    args.local_root.mkdir(parents=True)
    key = "EMPTY" if config.api_key_secret is None else require_secret(config.api_key_secret)
    has_judge = any(task.judge_model is not None for source in sources for task in source.tasks)
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
            parameters=config.request_parameters,
            budget=CANDIDATE_BUDGET,
            receipts=args.local_root / "model-requests",
        ) as relay,
        (
            budget_relay(
                endpoint=config.judge_endpoint,
                model=config.judge_api_model,
                api_key=require_secret(config.judge_api_key_secret),
                tokenizer_model=config.judge_identity.tokenizer,
                tokenizer_revision=config.judge_identity.tokenizer_revision,
                parameters={"temperature": JUDGE_TEMPERATURE},
                budget=None,
                receipts=judge_root / "model-requests",
            )
            if has_judge
            else nullcontext()
        ) as judge_relay,
    ):
        with ThreadPoolExecutor(max_workers=config.workers) as executor:
            for source in sources:
                started = time.monotonic()
                source_key = hashlib.sha256(source.sample.source_id.encode()).hexdigest()
                recovered = (
                    recovered_source(
                        previous,
                        source,
                        config.solver,
                        config.identity,
                        config.settings,
                        config.request_parameters,
                        config.judge_identity,
                    )
                    if previous is not None
                    else None
                )
                old_attempts = recovered.attempts if recovered is not None else []
                scored = {attempt.task_id for attempt in old_attempts if attempt.status == "scored"}
                selected_tasks = source.tasks if config.task_limit is None else source.tasks[: config.task_limit]
                tasks = [task for task in selected_tasks if task.task_id not in scored]
                futures = [
                    executor.submit(
                        run_task,
                        task,
                        config,
                        relay,
                        args.local_root / source_key,
                        str(output / source_key / "attempts"),
                        judge_relay,
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
                judge_traffic = archive_model_traffic(
                    judge_root / "model-requests",
                    tasks,
                    config.attempts_per_task,
                    source_key,
                    args.local_root / source_key / "judge-requests.tar.gz",
                    output / source_key / "judge-requests.tar.gz",
                )
                input_tokens = traffic.input_tokens + (
                    0 if previous_measurement is None else previous_measurement.input_tokens
                )
                output_tokens = traffic.output_tokens + (
                    0 if previous_measurement is None else previous_measurement.output_tokens
                )
                attempt_receipts = [] if recovered is None else list(recovered.attempt_receipts)
                for path in sorted((output / source_key / "attempts" / "*" / "*" / "attempt.json").glob(), key=str):
                    attempt_receipts.append(ContentEvidenceReference(uri=str(path), sha256=file_sha256(path)))
                receipt = write_json_evidence(
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
                        "judge_model_requests": judge_traffic.receipt.model_dump(mode="json"),
                        "judge_input_tokens": judge_traffic.input_tokens,
                        "judge_output_tokens": judge_traffic.output_tokens,
                        "judge_temperature": JUDGE_TEMPERATURE,
                        "judge_token_basis": "this source window; prior campaign traffic remains in linked receipts",
                        "attempt_receipts": [ref.model_dump(mode="json") for ref in attempt_receipts],
                        "previous_campaign": None if previous is None else str(previous.root),
                    },
                )
                payload = dict(
                    measurement_id=uuid4().hex,
                    protocol_id=PROTOCOL_ID,
                    created_at=datetime.now(UTC).isoformat(),
                    source_id=source.sample.source_id,
                    release=source.sample.release,
                    converter_version=source.sample.converter_version,
                    solver=config.solver,
                    status="pending",
                    reason="Awaiting scored coverage",
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
                        config.gpu_count * (time.monotonic() - started) / 3600
                        + (0 if previous_measurement is None else previous_measurement.gpu_hours)
                    ),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
                measurement = measurement_from_attempts(DifficultyMeasurement(**payload), attempts)
                blocked = sorted({task.reason for task in tasks if task.reason is not None})
                if not scored and tasks and all(task.reason is not None for task in tasks):
                    measurement = DifficultyMeasurement.model_validate(
                        {
                            **measurement.model_dump(mode="json"),
                            "status": "not_measured",
                            "reason": "; ".join(blocked),
                        }
                    )
                elif pending_reasons:
                    measurement = measurement.model_copy(
                        update={"reason": "; ".join(sorted(set(pending_reasons.values())))}
                    )
                with DataStore.open(str(output / "catalog")) as store:
                    register_difficulty(store)
                    with store.transaction() as transaction:
                        transaction.table(MEASUREMENTS).add(measurement.model_dump(mode="json"))
                write_json_evidence(output / source_key / "measurement.json", measurement.model_dump(mode="json"))
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

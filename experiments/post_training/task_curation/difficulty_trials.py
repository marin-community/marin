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
import re
import subprocess
import tarfile
from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from finestore.rl.curation import ContentEvidenceReference
from finestore.rl.difficulty import (
    PROTOCOL_ID,
    PROTOCOLS,
    DifficultyAttempt,
    DifficultyGap,
    difficulty_gap_reason,
)
from harbor.environments.docker_translate import MissingPreparedImageError
from harbor.host_capabilities import (
    MODEL_CREDENTIALS,
    MODEL_RELAY_PORT,
    HostCapabilities,
    host_capabilities_from_environment,
    model_guest_credentials,
)
from harbor.models.task.task import Task
from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig
from harbor.trial.trial import Trial
from harbor.utils.artifact_writer import get_artifact_writer
from harbor.utils.env import get_required_host_vars
from harbor_config.errors import ErrorCategory, error_category
from pydantic import ValidationError
from rigging.filesystem.storage_path import StoragePath
from rigging.filesystem.transfer import copy
from rigging.secrets import require_secret
from rigging.timing import ExponentialBackoff, retry_with_backoff
from shellbox.machine import UnsupportedMachineSpec
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.grading_result import GradingFailure
from taskcompendium.harbor.runtime import TaskNetworkPolicyReceipt, network_policy_qualified, read_source_runtime
from verifyit.modes.grade_math import MATH_DEADLINE_REFERENCE_PARSE, MATH_VERIFIER_DEADLINE

from experiments.post_training.task_curation.datasets.environments import BUILDABLE_ENVIRONMENTS
from experiments.post_training.task_curation.difficulty.launch import DATA_ADMISSION_COMMAND
from experiments.post_training.task_curation.difficulty_config import (
    DECLARED_JUDGE_PROVIDER_PREFIX,
    SUBSTITUTE_JUDGE_MODEL,
    TASK_PROVIDER_SECRETS,
    TOGETHER_AI_API_BASE_ENV,
    CandidateRun,
)
from experiments.post_training.task_curation.difficulty_evidence import write_json_evidence, write_same_json_evidence
from experiments.post_training.task_curation.difficulty_prepared import (
    PreparedTask,
    attempt_judge_model,
    bundle_judge_template_var,
    grader_requires_multi_uid,
    task_declared_judge_model,
)
from experiments.post_training.task_curation.difficulty_sampling import request_parameters
from experiments.post_training.task_curation.environment import Environment

LOGGER = logging.getLogger(__name__)
ATTEMPT_TIMEOUT = 259200

HARBOR_DEFAULT_BUILD_TIMEOUT_SEC = 600.0
"""Harbor's environment build allowance when the original task declares none."""
LOCAL_IMAGE_START_ALLOWANCE_SEC = 7200.0
"""A LOCAL actor machine may pull tens of GB before its first attempt starts."""
JUDGE_TEMPERATURE = PROTOCOLS[PROTOCOL_ID].temperature
INPUT_BUDGET_REASON = difficulty_gap_reason(DifficultyGap.PROTOCOL_INPUT_BUDGET_EXCEEDED)
PROVIDER_WAIT_REASON = difficulty_gap_reason(DifficultyGap.PROVIDER_CAPACITY_WAIT)
MISSING_PREPARED_IMAGE_REASON = difficulty_gap_reason(DifficultyGap.ENVIRONMENT_ARTIFACT_MISSING, "prepared_image")
UNSUPPORTED_COMPOSE_DOCUMENT_REASON = difficulty_gap_reason(DifficultyGap.UNSUPPORTED_COMPOSE_DOCUMENT)
NETWORK_POLICY_REASON = difficulty_gap_reason(DifficultyGap.NETWORK_POLICY_VIOLATION)
MATH_DEADLINE_LIMIT = 3
REGISTRY_LOOKUP_ATTEMPTS = 3
REGISTRY_LOOKUP_TIMEOUT_SEC = 30


def task_required_provider_secrets(task_dirs: Iterable[Path]) -> frozenset[str]:
    """Provider credential env names the selected task bundles resolve from the host."""
    required: set[str] = set()
    for directory in task_dirs:
        task = Task(task_dir=directory)
        for env in (task.config.environment.env, task.config.verifier.env, task.config.solution.env):
            for name, _default in get_required_host_vars(dict(env)):
                if name in MODEL_CREDENTIALS:
                    required.add(name)
    return frozenset(required)


def resolve_provider_secrets(required: Iterable[str]) -> dict[str, str]:
    """Resolve the wave's provider credentials from the GSM-backed secret bundle.

    Raises before any attempt runs when a credential cannot resolve, so a wave
    never fails mid-verifier.
    """
    unknown = frozenset(required) - TASK_PROVIDER_SECRETS
    if unknown:
        raise ValueError(f"Task bundles require provider credentials this lane cannot grant: {sorted(unknown)}")
    return {name: require_secret(name) for name in sorted(required)}


def substitute_judge_credentials(tasks: Iterable[PreparedTask]) -> frozenset[str]:
    """Provider credential templates a standing-ruling substitute judge covers.

    The re-declared judge authenticates through the wave's Together grant, so the
    bundle's own judge-credential templates resolve to Harbor's guest dummy: the
    templates never name a provider credential this wave must actually hold.
    """
    covered: set[str] = set()
    for task in tasks:
        if task.path is None or task.judge_model != SUBSTITUTE_JUDGE_MODEL:
            continue
        covered |= task_required_provider_secrets((Path(task.path),))
    return frozenset(covered)


def task_guest_environment(task: PreparedTask, provider_secrets: Mapping[str, str]) -> dict[str, str]:
    """The guest environment for one task's provider credential templates.

    Harbor refuses real provider credentials at the sandbox boundary (the guest
    model relay is the sanctioned channel), so the guest receives Harbor's dummy
    projection for each declared template; the resolved values stay host-side and
    never enter the sandbox. Raises before the task's first attempt when a
    bundle-referenced credential was not resolved for the wave, so a task never
    fails mid-verifier. A task whose judge rides the standing-ruling substitute
    covers its own credential templates with that dummy: the re-declared judge
    authenticates through the wave's Together grant instead.
    """
    if task.path is None:
        return {}
    required = task_required_provider_secrets((Path(task.path),))
    if task.judge_model == SUBSTITUTE_JUDGE_MODEL and required:
        return model_guest_credentials({name: "" for name in sorted(required)})
    missing = required - provider_secrets.keys()
    if missing:
        raise ValueError(
            f"Task {task.task_id} requires provider credentials the wave did not resolve: {sorted(missing)}"
        )
    granted = model_guest_credentials({name: provider_secrets[name] for name in sorted(required)})
    return granted


def trial_persistent_environment(task: PreparedTask) -> dict[str, str]:
    """The persistent sandbox environment one attempt's TrialConfig carries.

    Harbor resolves only declared templates from the guest environment, so a
    declared judge's base-URL override cannot reach the sandbox that way: the
    verifier's LiteLLM Together client reads TOGETHER_AI_API_BASE from the
    sandbox environment, and this value points it at Harbor's in-guest relay so
    the judge's calls authenticate host-side. A re-declared judge-route repair
    also pins the bundle's own `${VAR:-default}` template variable to the served
    substitute, so the source's declaration names the served judge. A task
    without a Together-route judge carries no persistent environment.
    """
    if task.path is None or task.workload != "native":
        return {}
    if task.judge_model == SUBSTITUTE_JUDGE_MODEL:
        template_var = bundle_judge_template_var(task)
        if template_var is None:
            return {TOGETHER_AI_API_BASE_ENV: f"http://127.0.0.1:{MODEL_RELAY_PORT}/v1"}
        if template_var is not None:
            return {
                TOGETHER_AI_API_BASE_ENV: f"http://127.0.0.1:{MODEL_RELAY_PORT}/v1",
                template_var: f"{DECLARED_JUDGE_PROVIDER_PREFIX}{SUBSTITUTE_JUDGE_MODEL}",
            }
    if task_declared_judge_model(task) is None:
        return {}
    return {TOGETHER_AI_API_BASE_ENV: f"http://127.0.0.1:{MODEL_RELAY_PORT}/v1"}


def task_spec_actor_image(task: PreparedTask) -> str | None:
    """The actor image a frozen task-spec declares, when its grader environment names one."""
    if task.path is None or task.workload != "task_spec":
        return None
    spec_path = Path(task.path) / "task-spec.json"
    if not spec_path.is_file():
        return None
    spec = json.loads(spec_path.read_text())
    return ((spec.get("environment_requirements") or {}).get("docker_image")) or None


def _registry_lookup_retryable(error: Exception) -> bool:
    if isinstance(error, subprocess.TimeoutExpired):
        return True
    return isinstance(error, subprocess.CalledProcessError) and bool(
        re.search(r"HTTP status: (?:429|5\d\d)\b", error.stderr or error.stdout or "")
    )


@cache
def pinned_registry_reference(reference: str, authfile: str | None = None) -> str:
    """Resolve a registry tag to its immutable digest reference.

    The LOCAL machine spec refuses registry images that are not pinned by
    sha256 digest. Composed toolchain images carry content-addressed tags but
    no digest, so the wave resolves the tag once and pins the manifest digest
    it verified. An already-pinned reference returns unchanged.
    """
    if "@sha256:" in reference:
        return reference
    command = ["skopeo", "inspect", "--format", "{{.Digest}}"]
    if authfile is not None:
        command.extend(("--authfile", authfile))
    # The wave's skopeo rejects a bare registry reference as an unknown
    # transport; the explicit docker:// transport parses everywhere.
    command.append(f"docker://{reference}")

    def lookup() -> str:
        completed = subprocess.run(
            command, check=False, capture_output=True, text=True, timeout=REGISTRY_LOOKUP_TIMEOUT_SEC
        )
        if completed.returncode != 0:
            raise subprocess.CalledProcessError(completed.returncode, command, completed.stdout, completed.stderr)
        digest = completed.stdout.strip()
        if re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
            raise ValueError(f"Registry digest lookup returned a malformed digest for {reference}: {digest[:400]}")
        return f"{reference}@{digest}"

    try:
        return retry_with_backoff(
            lookup,
            retryable=_registry_lookup_retryable,
            max_attempts=REGISTRY_LOOKUP_ATTEMPTS,
            backoff=ExponentialBackoff(initial=1, maximum=2, factor=2, jitter=0),
            operation=f"Registry digest lookup for {reference}",
        )
    except subprocess.TimeoutExpired as error:
        raise ValueError(f"Registry digest lookup timed out for {reference}") from error
    except subprocess.CalledProcessError as error:
        diagnostic = (error.stderr or error.stdout or "").strip()
        raise ValueError(
            f"Registry digest lookup failed for {reference} (rc={error.returncode}): {diagnostic[:400]}"
        ) from error


def actor_phase_gap_reason(task: PreparedTask, result) -> str | None:
    """The typed cause when a trial dies before its verifier started.

    A trial whose exception fired while the actor machine was still starting
    (image pull or environment start) is an infrastructure retry keyed to the
    declared actor image, never a coverage shortfall. A pin that cannot run its
    source at all stays the source-image defect the repair replaces.
    """
    if result.exception_info is None:
        return None
    if result.verifier is not None and result.verifier.started_at is not None:
        return None
    pin = task_spec_actor_image(task)
    if pin in DEFECTIVE_ACTOR_IMAGES:
        return difficulty_gap_reason(DifficultyGap.ENVIRONMENT_ARTIFACT_MISSING, pin)
    if result.environment_setup is not None and result.environment_setup.finished_at is None:
        return difficulty_gap_reason(DifficultyGap.ENVIRONMENT_START_FAILURE, pin)
    return None


def read_verdict(trial: Trial) -> dict | None:
    """The trial's structured verdict, when the verifier produced one."""
    path = trial.paths.verifier_dir / "verdict.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


_LOCK_TEMPDIR_COMMENT = re.compile(r"^\s*#.*requirements\.in")


def _normalized_lock(lock: str) -> str:
    """The lock's pinned content, without the compile comment lines that name the staging directory."""
    return "".join(line for line in lock.splitlines(keepends=True) if not _LOCK_TEMPDIR_COMMENT.match(line))


def _declared_lock(environment: Environment) -> str | None:
    """The declared environment's compiled lock content, or None when its lock cannot compile."""
    # The image-build module pulls in the Marin execution graph, which the Harbor runtime
    # environment does not carry; resolve it only when a grader lock actually needs matching.
    from experiments.post_training.task_curation.images.build import (  # noqa: PLC0415
        LOCK_FILE,
        compile_lock,
    )

    try:
        with TemporaryDirectory() as directory:
            compile_lock(environment.requirements, Path(directory) / LOCK_FILE)
            return _normalized_lock((Path(directory) / LOCK_FILE).read_text())
    except Exception as error:
        LOGGER.warning(
            "Declared environment lock compile failed; its grader env cannot be matched: %s: %s",
            type(error).__name__,
            error,
        )
        return None


@cache
def declared_grader_machine_image(lock_url: str) -> str | None:
    """The composed toolchain image for a frozen lock-only grader environment, when one is declared.

    The frozen panel's grader environment records a packages lock whose machine lacked the
    toolchain. The declared environment whose compiled lock carries the same pinned packages and a
    Dockerfile fragment owns the composed image that delivers both, so the grader machine runs from
    it instead of the host. Returns None when no declaration matches, keeping the existing
    lock-built machine; callers that need the toolchain fail closed at the toolchain probe.
    """

    from experiments.post_training.task_curation.difficulty_artifacts import (  # noqa: PLC0415
        REGIONAL_ENVIRONMENT_ROOT,
    )
    from experiments.post_training.task_curation.images.build import (  # noqa: PLC0415  # noqa: PLC0415
        DEFAULT_REPOSITORY,
        ENVIRONMENT_ARTIFACT_VERSION,
        IDENTITY_CHARS,
        identity_digest,
        image_tag,
    )

    lock = _normalized_lock(StoragePath(lock_url).read_text())
    for environment in BUILDABLE_ENVIRONMENTS:
        if environment.dockerfile is None:
            continue
        declared = _declared_lock(environment)
        if declared is not None and declared == lock:
            identity = identity_digest(environment)
            # The LOCAL backend refuses tag references: deliver the digest-pinned image the
            # in-region build recorded, falling back to the identity tag when no record exists.
            try:
                from marin.execution.artifact import read_record  # noqa: PLC0415

                record = read_record(
                    f"{REGIONAL_ENVIRONMENT_ROOT}/env-{identity[:IDENTITY_CHARS]}/{ENVIRONMENT_ARTIFACT_VERSION}"
                )
                image = (record.result or {}).get("image") if record is not None else None
            except Exception:
                image = None
            return image or image_tag(DEFAULT_REPOSITORY, identity)
    return None


DEFECTIVE_ACTOR_IMAGES: frozenset[str] = frozenset(
    {"ghcr.io/marin-community/rl-data-openresearcher@sha256:6ccb79c34d3c1da34c6bd947b9a0206e852b2e00d79b671efade9af1"}
)
"""Actor image pins that cannot run their source: the lightweight OpenResearcher base lacks its
retrieval index and the full-index image exceeds Daytona's snapshot limit. Attempts that die in the
actor phase under these pins type as ENVIRONMENT_ARTIFACT_MISSING with the pin, never as grader
failures."""


def task_spec_grader_machine_image(task: PreparedTask) -> str | None:
    """The composed grader machine image for one prepared task-spec task, when its lock resolves."""
    if task.path is None or task.workload != "task_spec":
        return None
    spec_path = Path(task.path) / "task-spec.json"
    if not spec_path.is_file():
        return None
    spec = json.loads(spec_path.read_text())
    lock_url = ((spec.get("grader") or {}).get("environment") or {}).get("packages_lock")
    if not lock_url:
        return None
    return declared_grader_machine_image(lock_url)


def grader_infrastructure_reason(*, verifier_reached: bool, verdict: dict | None) -> str | None:
    """Typed reason when the grading infrastructure consumed the attempt.

    The verifier phase ran and either produced an ``infra_error`` verdict or
    died before producing one. run_task only calls this for attempts that
    already excluded task-input conditions, provider waits and solver-side
    failures; the verdict decides. The identity is one of the judge transport's
    typed tokens (for example ``http_401``) or a grading-failure value; any
    other evidence stays out of the ledger.
    """
    from verifyit.grade import TRANSPORT_REASON  # noqa: PLC0415

    if not verifier_reached:
        return None
    if verdict is None:
        return DifficultyGap.GRADER_INFRASTRUCTURE_FAILURE.value
    if verdict.get("status") != "infra_error":
        return None
    detail = verdict.get("detail")
    identity = None
    if isinstance(detail, dict):
        attempts = detail.get("transport_attempts")
        if isinstance(attempts, (list, tuple)):
            for record in reversed(attempts):
                if (
                    isinstance(record, dict)
                    and record.get("outcome") == "failed"
                    and isinstance(record.get("reason"), str)
                ):
                    candidate = record["reason"]
                    if TRANSPORT_REASON.fullmatch(candidate):
                        identity = candidate
                    break
        failure = detail.get("failure")
        if identity is None and failure in {item.value for item in GradingFailure}:
            identity = failure
    return difficulty_gap_reason(DifficultyGap.GRADER_INFRASTRUCTURE_FAILURE, identity)


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
    verdict = read_verdict(trial)
    if verdict is None:
        return None
    detail = verdict.get("detail")
    if verdict.get("status") != "infra_error" or not isinstance(detail, dict):
        return None
    if detail.get("code") != MATH_VERIFIER_DEADLINE:
        return None
    return detail.get("phase")


def worker_host_capabilities(authentication: dict[str, str], environment: Mapping[str, str]) -> HostCapabilities:
    """Bind Difficulty's Daytona authority exclusively to the DATA credential."""
    source = dict(environment)
    source.pop("DAYTONA_API_KEY", None)
    if source.get("DAYTONA_DATA_API_KEY"):
        source["DAYTONA_API_KEY"] = source["DAYTONA_DATA_API_KEY"]
    capabilities = host_capabilities_from_environment(
        source,
        model_authentication=authentication,
    )
    connection = capabilities.daytona
    if connection is not None and not (connection.api_key and connection.api_url and connection.target):
        # client_factory's error names no field; name the missing one and whether the DATA
        # credential reached this process, so a wave-side wiring gap is visible at once.
        missing = [
            name
            for name, value in (
                ("api_key", connection.api_key),
                ("api_url", connection.api_url),
                ("target", connection.target),
            )
            if not value
        ]
        raise UnsupportedMachineSpec(
            "Daytona connection is incomplete for a Difficulty trial: missing "
            f"{', '.join(missing)}; DAYTONA_DATA_API_KEY "
            f"{'present' if source.get('DAYTONA_DATA_API_KEY') else 'ABSENT'} in the worker environment"
        )
    return capabilities


async def execute_trial(
    configuration: TrialConfig,
    authentication: dict[str, str],
    guest_environment: Mapping[str, str],
    host_capabilities: HostCapabilities,
):
    trial = await Trial.create(
        configuration,
        host_capabilities=dataclasses.replace(host_capabilities, model_authentication=authentication),
        guest_environment=guest_environment,
    )
    return trial, await trial.run()


def trial_environment_options(
    task: PreparedTask,
    environment_options: Mapping[str, Any],
    daytona_provisioning: str,
    machine_image: str | None,
) -> dict[str, Any]:
    """The environment kwargs for one attempt's TrialConfig.

    Daytona tasks ignore the worker-local options; their environment takes the
    provisioning mode and the attempt timeout. The composed grader machine image
    is not worker-local: a tier-routed task's grader must still run from its
    digest-pinned composed image (the Daytona grader machine refuses to start
    from a bare host image, so dropping the binding fails every attempt).
    """
    if task.backend == "daytona":
        options: dict[str, Any] = {
            "provisioning": daytona_provisioning,
            "trial_attempt_timeout_sec": ATTEMPT_TIMEOUT,
            "create_admission_command": list(DATA_ADMISSION_COMMAND),
        }
    else:
        options = dict(environment_options)
    if machine_image is not None:
        options["grader_machine_image"] = machine_image
    return options


def trial_environment_config(task: PreparedTask, options: dict) -> EnvironmentConfig:
    """Keep frozen native bytes while applying the authorized Daytona resource condition."""
    overrides = {}
    if task.resource_deviation is not None:
        if task.path is None:
            raise ValueError("A resource-deviated trial requires its original native package")
        overrides = task.resource_deviation.environment_overrides(
            Path(task.path), backend=task.backend, workload=task.workload
        )
    return EnvironmentConfig(type=task.backend, kwargs=options, env=trial_persistent_environment(task), **overrides)


def run_task(
    task: PreparedTask,
    config: CandidateRun,
    base_url: str,
    root: Path,
    output: str,
    judge_url: str | None,
    prior_math_deadlines: int,
    *,
    host_capabilities: HostCapabilities,
    record_attempt: Callable[[StoragePath], None] | None = None,
    guest_environment: Mapping[str, str] = {},
) -> TaskOutcome:
    key = hashlib.sha256(task.task_id.encode()).hexdigest()
    destination = StoragePath(output) / key
    if task.reason is not None or task.path is None:
        write_same_json_evidence(destination / "pending.json", {"task_id": task.task_id, "reason": task.reason})
        return TaskOutcome(
            [DifficultyAttempt(task_id=task.task_id, status="pending", reward=None, reason=task.reason)], task.reason
        )
    attempts = []
    machine_image = task_spec_grader_machine_image(task)
    if machine_image is not None:
        # The composed toolchain image is the declared environment's own artifact: it bakes
        # the task's packages lock and, for the Lean graders, the Lean toolchain the grader
        # probes for. The harness marks the machine lock-satisfied, so both backends run the
        # grader from the digest-pinned image instead of mounting a lock runtime.
        machine_image = pinned_registry_reference(machine_image, config.environment_options.get("registry_authfile"))
        LOGGER.info("Grader machine for %s runs from the composed toolchain image %s", task.task_id, machine_image)
    if (
        task.backend == "local"
        and task.workload == "task_spec"
        and task.path is not None
        and grader_requires_multi_uid(json.loads((Path(task.path) / "task-spec.json").read_text()))
    ):
        # A privilege-dropping grader cannot run on the LOCAL bubblewrap tier (single-uid user
        # namespace); preparation routes such tasks up to Daytona, and this guard fails loudly
        # if a routing regression ever serves one locally - it must never compile as root.
        raise ValueError(
            f"Task {task.task_id} grades with a privilege-dropping sandbox; preparation must route it up to Daytona"
        )
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
                "proactive_summarization_threshold": config.settings.context_tokens - config.settings.input_tokens,
                "model_info": {
                    "max_input_tokens": config.settings.context_tokens,
                    "max_output_tokens": config.settings.output_tokens,
                    "input_cost_per_token": 0.0,
                    "output_cost_per_token": 0.0,
                },
                "llm_call_kwargs": parameters,
            }
            agent_name = "terminus-2"

        options = trial_environment_options(task, config.environment_options, config.daytona_provisioning, machine_image)
        authentication = {endpoint: "EMPTY"}
        judge_model = attempt_judge_model(task)
        if judge_model is not None:
            if judge_url is None or config.judge is None:
                raise ValueError("A task judge requires the protocol's host-side relay")
            judge_endpoint = f"{judge_url}/{root.name}-{key}-{number}/v1"
            options.update(model_endpoint=judge_endpoint, model_name=judge_model or config.judge.api_model)
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
            environment=trial_environment_config(task, options),
            environment_build_timeout_multiplier=(
                LOCAL_IMAGE_START_ALLOWANCE_SEC / HARBOR_DEFAULT_BUILD_TIMEOUT_SEC if task.backend == "local" else None
            ),
        )

        # Framework exceptions propagate and stop dispatch; they are never fabricated as solver zeros.
        construction_reason = None
        try:
            trial, result = asyncio.run(
                execute_trial(configuration, authentication, guest_environment, host_capabilities)
            )
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
            **({"solver_protocol_id": config.solver_protocol_id} if config.solver_protocol_id != PROTOCOL_ID else {}),
            **(
                {"resource_deviation": dataclasses.asdict(task.resource_deviation)}
                if task.resource_deviation is not None
                else {}
            ),
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
        grader_infra_reason = None
        if reward is None and not unserved and not provider_wait and deadline_phase is None:
            grader_infra_reason = grader_infrastructure_reason(
                verifier_reached=result.verifier is not None and result.verifier.started_at is not None,
                verdict=read_verdict(trial),
            )
            if grader_infra_reason is None:
                grader_infra_reason = actor_phase_gap_reason(task, result)
        if deadline_phase is not None:
            reason = difficulty_gap_reason(DifficultyGap.MATH_VERIFIER_DEADLINE, deadline_phase)
        elif reward is not None:
            reason = None
        elif grader_infra_reason is not None:
            reason = grader_infra_reason
        else:
            reason = pending_reason
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

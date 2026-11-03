from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path
from typing import Any, Literal, Protocol
from urllib.parse import urlparse
from urllib.request import url2pathname

from pydantic import BaseModel, Field, field_validator

from harbor.bridges.base import bridge_input_files
from harbor.models.job.config import JobConfig, RetryConfig
from harbor.models.task.id import GitTaskId, LocalTaskId, PackageTaskId
from harbor.models.trial.config import (
    AgentConfig,
    DEFAULT_TRIAL_CLEANUP_GRACE_SEC,
    EnvironmentConfig,
    TaskConfig,
    TrialConfig,
    UserAgentConfig,
    VerifierConfig,
)
from harbor.tasks.packager import Packager
from harbor.skills import compute_skill_digest, resolve_skills
from harbor.utils.env import sanitize_env_assignment

_DIGEST_PREFIX = "sha256:"
_ENV_ARG_FLAGS = {"--ae", "--agent-env", "--ve", "--verifier-env"}
TaskIdType = GitTaskId | LocalTaskId | PackageTaskId

# Agent-kwarg keys that are per-ALLOCATION runtime endpoints, not replay-affecting
# trial identity: the served-model URL + auth key. A legitimate resume re-runs the
# SAME trials against a NEW serve node (a fresh SLURM allocation gets a different
# MASTER_ADDR, and the eval sbatch rewrites config.json's `api_base` to it before
# `harbor jobs resume`). Including these in the lock equality key made every
# cross-allocation resume spuriously fail `_write_job_lock` with FileExistsError
# ("lock.json does not match the resolved job lock") — same class of false-reject the
# created_at/harbor/invocation/n_concurrent_trials carry-overs in Job._write_job_lock
# already guard against, but at the per-trial agent.kwargs level. Excluded from the
# equality key ONLY.
_VOLATILE_AGENT_KWARG_KEYS = frozenset({"api_base", "key"})
# Agent environment variables that carry the same per-allocation endpoint as
# ``api_base``. Marin rewrites these when a resume lands on a new Iris serve
# node, so they must not make the otherwise-identical resolved lock conflict.
_VOLATILE_AGENT_ENV_KEYS = frozenset(
    {"HARBOR_PI_API_BASE", "OPENAI_API_BASE", "OPENAI_BASE_URL"}
)
# Environment cache-acquisition kwargs (the Daytona snapshot trio) are defined
# alongside the resume identity fingerprint they must stay in sync with: see
# ``TrialConfig._PROVISIONING_ENV_KWARG_KEYS`` in harbor_config.


class TaskDownloadResolution(Protocol):
    content_hash: str | None
    resolved_git_commit_id: str | None


def _validate_digest(value: str) -> str:
    if not value.startswith(_DIGEST_PREFIX):
        raise ValueError(f"Digest must start with '{_DIGEST_PREFIX}'. Got: {value}")
    hex_digest = value.removeprefix(_DIGEST_PREFIX)
    if len(hex_digest) != 64 or any(c not in "0123456789abcdef" for c in hex_digest):
        raise ValueError(
            f"Digest must be in 'sha256:<64 hex chars>' format. Got: {value}"
        )
    return value


def _prefixed_digest(value: str) -> str:
    return value if value.startswith(_DIGEST_PREFIX) else f"{_DIGEST_PREFIX}{value}"


def sanitize_cli_invocation(argv: Sequence[str]) -> list[str]:
    sanitized: list[str] = []
    redact_next = False

    for arg in argv:
        if redact_next:
            sanitized.append(sanitize_env_assignment(arg))
            redact_next = False
            continue

        if arg in _ENV_ARG_FLAGS:
            sanitized.append(arg)
            redact_next = True
            continue

        flag, sep, value = arg.partition("=")
        if sep and flag in _ENV_ARG_FLAGS:
            sanitized.append(f"{flag}={sanitize_env_assignment(value)}")
        else:
            sanitized.append(arg)

    return sanitized


class HarborLockInfo(BaseModel):
    version: str | None = None
    git_commit_hash: str | None = None
    is_editable: bool | None = None


class TaskLock(BaseModel):
    name: str
    type: Literal["local", "git", "package"]
    digest: str
    source: str | None = None
    path: Path | None = None
    git_url: str | None = None
    git_commit_id: str | None = None

    @field_validator("digest")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        return _validate_digest(value)

    def __eq__(self, other):
        if not isinstance(other, TaskLock):
            return NotImplemented
        return self._equality_key() == other._equality_key()

    def _equality_key(self) -> tuple[str]:
        return (self.digest,)


class ExtraInstructionLock(BaseModel):
    path: Path
    digest: str

    @field_validator("digest")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        return _validate_digest(value)

    def __eq__(self, other):
        if not isinstance(other, ExtraInstructionLock):
            return NotImplemented
        return self._equality_key() == other._equality_key()

    def _equality_key(self) -> tuple[str]:
        return (self.digest,)


class AgentSkillLock(BaseModel):
    name: str
    source: Path
    digest: str

    @field_validator("digest")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        return _validate_digest(value)

    def __eq__(self, other):
        if not isinstance(other, AgentSkillLock):
            return NotImplemented
        return self._equality_key() == other._equality_key()

    def _equality_key(self) -> tuple[str, str]:
        return (self.name, self.digest)


class TrialLock(BaseModel):
    task: TaskLock
    timeout_multiplier: float = 1.0
    agent_timeout_multiplier: float | None = None
    verifier_timeout_multiplier: float | None = None
    agent_setup_timeout_multiplier: float | None = None
    environment_build_timeout_multiplier: float | None = None
    trial_attempt_timeout_sec: float | None = None
    trial_cleanup_grace_sec: float = DEFAULT_TRIAL_CLEANUP_GRACE_SEC
    extra_instructions: list[ExtraInstructionLock] | None = None
    agent: AgentConfig
    user_agent: UserAgentConfig | None = None
    user_persona: ExtraInstructionLock | None = None
    user_prompt_template: ExtraInstructionLock | None = None
    bridge_prompt: ExtraInstructionLock | None = None
    bridge_inputs: dict[str, ExtraInstructionLock] = Field(default_factory=dict)
    skills: list[AgentSkillLock] = Field(default_factory=list)
    environment: EnvironmentConfig
    extra_docker_compose: list["ExtraDockerComposeLock"] | None = None
    verifier: VerifierConfig

    def __eq__(self, other):
        if not isinstance(other, TrialLock):
            return NotImplemented
        return self._equality_key() == other._equality_key()

    def _equality_key(self) -> tuple[Any, ...]:
        return (
            self.task._equality_key(),
            self.timeout_multiplier,
            self.agent_timeout_multiplier,
            self.verifier_timeout_multiplier,
            self.agent_setup_timeout_multiplier,
            self.environment_build_timeout_multiplier,
            self.trial_attempt_timeout_sec,
            self.trial_cleanup_grace_sec,
            _lock_list_equality_key(self.extra_instructions),
            _agent_equality_key(self.agent),
            _user_agent_equality_key(
                self.user_agent, bridge_input_names=set(self.bridge_inputs)
            ),
            self.user_persona._equality_key() if self.user_persona else None,
            (
                self.user_prompt_template._equality_key()
                if self.user_prompt_template
                else None
            ),
            self.bridge_prompt._equality_key() if self.bridge_prompt else None,
            tuple(
                sorted(
                    (name, lock._equality_key())
                    for name, lock in self.bridge_inputs.items()
                )
            ),
            tuple(skill._equality_key() for skill in self.skills),
            _environment_equality_key(self.environment),
            _lock_list_equality_key(self.extra_docker_compose),
            _frozen_value(self.verifier.identity_fingerprint()),
        )


class ExtraDockerComposeLock(BaseModel):
    path: Path
    digest: str

    @field_validator("digest")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        return _validate_digest(value)

    def __eq__(self, other):
        if not isinstance(other, ExtraDockerComposeLock):
            return NotImplemented
        return self._equality_key() == other._equality_key()

    def _equality_key(self) -> tuple[str]:
        return (self.digest,)


class JobLock(BaseModel):
    # If replay-affecting fields are added here, make sure JobConfig/TrialConfig
    # expose the requested inputs and update the equality tests.
    schema_version: int = 1
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    harbor: HarborLockInfo = Field(default_factory=HarborLockInfo)
    invocation: list[str] | None = None
    n_concurrent_trials: int
    retry: RetryConfig
    trials: list[TrialLock] = Field(default_factory=list)

    def __eq__(self, other):
        if not isinstance(other, JobLock):
            return NotImplemented
        return self._equality_key() == other._equality_key()

    def _equality_key(self) -> tuple[Any, ...]:
        return (
            self.schema_version,
            self.n_concurrent_trials,
            _frozen_value(self.retry),
            _unordered_lock_list_equality_key(self.trials),
        )


def _unordered_lock_list_equality_key(locks: Sequence[Any]) -> tuple[Any, ...]:
    return tuple(sorted((lock._equality_key() for lock in locks), key=repr))


def _lock_list_equality_key(locks: Sequence[Any] | None) -> tuple[Any, ...] | None:
    if locks is None:
        return None
    return tuple(lock._equality_key() for lock in locks)


def _frozen_value(value: Any, exclude: set[str] | None = None) -> Any:
    if isinstance(value, BaseModel):
        return (
            value.__class__,
            _frozen_value(
                value.model_dump(
                    mode="python",
                    exclude=exclude or set(),
                    exclude_none=True,
                )
            ),
        )
    if isinstance(value, dict):
        return tuple(
            sorted(
                (_frozen_value(key), _frozen_value(item)) for key, item in value.items()
            )
        )
    if isinstance(value, (list, tuple)):
        return tuple(_frozen_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted(_frozen_value(item) for item in value))
    return value


def _kwargs_stripped_frozen_key(
    model: AgentConfig | EnvironmentConfig,
    strip_keys: frozenset[str],
    exclude: set[str],
) -> Any:
    """Frozen equality key for a config model, ignoring selected ``kwargs`` keys
    that are non-replay runtime/provisioning knobs (``_VOLATILE_AGENT_KWARG_KEYS``
    for the agent, ``TrialConfig._PROVISIONING_ENV_KWARG_KEYS`` for the
    environment). Those keys change how a trial is served or provisioned, not
    what it evaluates, so a resume must re-resolve the same lock. The real
    values still serialize into lock.json; only the comparison ignores them.
    """
    if not (model.kwargs and strip_keys.intersection(model.kwargs)):
        return _frozen_value(model, exclude=exclude)
    sanitized = model.model_copy(
        update={
            "kwargs": {k: v for k, v in model.kwargs.items() if k not in strip_keys}
        }
    )
    return _frozen_value(sanitized, exclude=exclude)


def _agent_equality_key(agent: AgentConfig) -> Any:
    sanitized = agent.model_copy(
        update={
            "kwargs": {
                key: value
                for key, value in agent.kwargs.items()
                if key not in _VOLATILE_AGENT_KWARG_KEYS
            },
            "env": {
                key: value
                for key, value in agent.env.items()
                if key not in _VOLATILE_AGENT_ENV_KEYS
            },
        }
    )
    return _frozen_value(sanitized, exclude={"skills", "model_alias"})


def _environment_equality_key(environment: EnvironmentConfig) -> Any:
    return _kwargs_stripped_frozen_key(
        environment,
        TrialConfig._PROVISIONING_ENV_KWARG_KEYS,
        {"extra_docker_compose"},
    )


def _user_agent_equality_key(
    user_agent: UserAgentConfig | None,
    *,
    bridge_input_names: set[str],
) -> Any:
    if user_agent is None:
        return None
    value = user_agent.model_dump(
        mode="python",
        exclude={
            "skills",
            "model_alias",
            "user_persona_path",
            "user_prompt_template_path",
        },
        exclude_none=True,
    )
    bridge = value.get("bridge")
    if isinstance(bridge, dict):
        bridge.pop("prompt_path", None)
        kwargs = bridge.get("kwargs")
        if isinstance(kwargs, dict):
            for name in bridge_input_names:
                kwargs.pop(name, None)
    return _frozen_value(value)


def build_job_lock(
    *,
    config: JobConfig,
    trial_configs: Sequence[TrialConfig],
    invocation: list[str] | None = None,
    task_download_results: Mapping[TaskIdType, TaskDownloadResolution] | None = None,
) -> JobLock:
    trials = [
        _build_lock_trial(
            trial_config,
            _get_task_download_result(trial_config.task, task_download_results),
        )
        for trial_config in trial_configs
    ]

    return JobLock(
        harbor=_get_harbor_info(),
        invocation=sanitize_cli_invocation(
            invocation if invocation is not None else list(sys.argv)
        ),
        n_concurrent_trials=config.n_concurrent_trials,
        retry=config.retry,
        trials=trials,
    )


def _build_lock_trial(
    trial_config: TrialConfig,
    task_download_result: TaskDownloadResolution | None = None,
) -> TrialLock:
    return TrialLock(
        task=_build_lock_trial_task(trial_config.task, task_download_result),
        timeout_multiplier=trial_config.timeout_multiplier,
        agent_timeout_multiplier=trial_config.agent_timeout_multiplier,
        verifier_timeout_multiplier=trial_config.verifier_timeout_multiplier,
        agent_setup_timeout_multiplier=trial_config.agent_setup_timeout_multiplier,
        environment_build_timeout_multiplier=(
            trial_config.environment_build_timeout_multiplier
        ),
        trial_attempt_timeout_sec=trial_config.trial_attempt_timeout_sec,
        trial_cleanup_grace_sec=trial_config.trial_cleanup_grace_sec,
        extra_instructions=(
            _build_extra_instruction_locks(trial_config.extra_instruction_paths)
            if trial_config.extra_instruction_paths
            else None
        ),
        agent=trial_config.agent,
        user_agent=trial_config.user_agent,
        user_persona=(
            ExtraInstructionLock(
                path=trial_config.user_agent.user_persona_path,
                digest=_file_sha256_digest(trial_config.user_agent.user_persona_path),
            )
            if trial_config.user_agent is not None
            and trial_config.user_agent.user_persona_path is not None
            else None
        ),
        user_prompt_template=(
            ExtraInstructionLock(
                path=trial_config.user_agent.user_prompt_template_path,
                digest=_file_sha256_digest(
                    trial_config.user_agent.user_prompt_template_path
                ),
            )
            if trial_config.user_agent is not None
            and trial_config.user_agent.user_prompt_template_path is not None
            else None
        ),
        bridge_prompt=(
            ExtraInstructionLock(
                path=trial_config.user_agent.bridge.prompt_path,
                digest=_file_sha256_digest(trial_config.user_agent.bridge.prompt_path),
            )
            if trial_config.user_agent is not None
            and trial_config.user_agent.bridge.prompt_path is not None
            else None
        ),
        bridge_inputs=_build_bridge_input_locks(trial_config),
        skills=_build_agent_skill_locks(trial_config.agent.skills),
        environment=trial_config.environment,
        extra_docker_compose=_build_extra_docker_compose_locks(
            trial_config.environment.extra_docker_compose
        ),
        verifier=trial_config.verifier,
    )


def _build_agent_skill_locks(skills: list[Path]) -> list[AgentSkillLock]:
    return [
        AgentSkillLock(
            name=skill.name,
            source=skill.source,
            digest=compute_skill_digest(skill.source),
        )
        for skill in resolve_skills(skills)
    ]


def _build_bridge_input_locks(
    trial_config: TrialConfig,
) -> dict[str, ExtraInstructionLock]:
    if trial_config.user_agent is None:
        return {}
    return {
        name: ExtraInstructionLock(path=path, digest=_file_sha256_digest(path))
        for name, path in bridge_input_files(trial_config.user_agent.bridge).items()
    }


def _build_extra_docker_compose_locks(
    paths: Sequence[Path],
) -> list[ExtraDockerComposeLock] | None:
    if not paths:
        return None
    return [
        ExtraDockerComposeLock(path=path, digest=_file_sha256_digest(path))
        for path in paths
    ]


def _file_sha256_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.expanduser().open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return _prefixed_digest(h.hexdigest())


def _build_lock_trial_task(
    task_config: TaskConfig,
    task_download_result: TaskDownloadResolution | None = None,
) -> TaskLock:
    task_id = task_config.get_task_id()
    name = task_id.get_name()

    if task_config.is_package_task():
        task_type: Literal["local", "git", "package"] = "package"
        if task_download_result is not None and task_download_result.content_hash:
            digest = _prefixed_digest(task_download_result.content_hash)
        elif task_config.ref is None or not task_config.ref.startswith(_DIGEST_PREFIX):
            raise ValueError(
                f"Package task {name!r} was not resolved to a sha256 digest."
            )
        else:
            digest = task_config.ref
    else:
        task_type = "git" if task_config.is_git_task() else "local"
        content_hash, _ = Packager.compute_content_hash(task_config.get_local_path())
        digest = _prefixed_digest(content_hash)

    git_commit_id = task_config.git_commit_id
    if (
        task_config.is_git_task()
        and task_download_result is not None
        and task_download_result.resolved_git_commit_id
    ):
        git_commit_id = task_download_result.resolved_git_commit_id

    return TaskLock(
        name=name,
        type=task_type,
        digest=digest,
        source=task_config.source,
        path=task_config.path,
        git_url=task_config.git_url,
        git_commit_id=git_commit_id,
    )


def _build_extra_instruction_locks(paths: Sequence[Path]) -> list[ExtraInstructionLock]:
    extra_instructions: list[ExtraInstructionLock] = []
    for path in paths:
        resolved_path = path.expanduser()
        if not resolved_path.exists():
            raise FileNotFoundError(f"Extra instruction file not found: {path}")
        digest = _file_sha256_digest(path)
        extra_instructions.append(ExtraInstructionLock(path=path, digest=digest))
    return extra_instructions


def _get_task_download_result(
    task_config: TaskConfig,
    task_download_results: Mapping[TaskIdType, TaskDownloadResolution] | None,
) -> TaskDownloadResolution | None:
    if task_download_results is None:
        return None
    return task_download_results.get(task_config.get_task_id())


def _get_harbor_info() -> HarborLockInfo:
    return HarborLockInfo(
        version=_get_harbor_version(),
        git_commit_hash=_get_harbor_git_commit_hash(),
        is_editable=_get_harbor_is_editable_install(),
    )


def _get_harbor_version() -> str | None:
    try:
        return version("harbor")
    except PackageNotFoundError:
        return None


def _get_harbor_git_commit_hash() -> str | None:
    direct_url_data = _get_harbor_direct_url_data()
    if direct_url_data is None:
        return None

    vcs_info = direct_url_data.get("vcs_info")
    if isinstance(vcs_info, dict) and vcs_info.get("vcs") == "git":
        commit_id = vcs_info.get("commit_id")
        if isinstance(commit_id, str) and commit_id:
            return commit_id

    if not _is_harbor_editable_install(direct_url_data):
        return None

    repo_path = _get_file_path_from_direct_url(direct_url_data.get("url"))
    if repo_path is None:
        return None

    return _get_git_commit_hash(repo_path)


def _get_git_commit_hash(repo_path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            check=False,
            text=True,
            timeout=2,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _get_harbor_is_editable_install() -> bool | None:
    direct_url_data = _get_harbor_direct_url_data()
    if direct_url_data is None:
        return None
    return _is_harbor_editable_install(direct_url_data)


def _get_harbor_direct_url_data() -> dict | None:
    try:
        dist = distribution("harbor")
    except PackageNotFoundError:
        return None

    direct_url = dist.read_text("direct_url.json")
    if direct_url is None:
        return {}

    try:
        direct_url_data = json.loads(direct_url)
    except json.JSONDecodeError:
        return None

    return direct_url_data if isinstance(direct_url_data, dict) else None


def _is_harbor_editable_install(direct_url_data: dict) -> bool:
    dir_info = direct_url_data.get("dir_info")
    if not isinstance(dir_info, dict):
        return False
    return bool(dir_info.get("editable", False))


def _get_file_path_from_direct_url(url: object) -> Path | None:
    if not isinstance(url, str):
        return None

    parsed_url = urlparse(url)
    if parsed_url.scheme != "file" or parsed_url.netloc not in ("", "localhost"):
        return None

    return Path(url2pathname(parsed_url.path))

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
import os

from harbor.agents.utils import PROVIDER_KEYS
from harbor_config.env import get_required_host_vars as get_required_host_vars
from harbor_config.env import is_env_template as is_env_template
from harbor_config.env import is_sensitive_env_key as is_sensitive_env_key
from harbor_config.env import parse_bool_env_value as parse_bool_env_value
from harbor_config.env import redact_sensitive_value as redact_sensitive_value
from harbor_config.env import resolve_env_vars as _resolve_env_vars
from harbor_config.env import sanitize_env_assignment as sanitize_env_assignment
from harbor_config.env import templatize_sensitive_env as templatize_sensitive_env

_TASK_ENVIRONMENT: ContextVar[Mapping[str, str]] = ContextVar(
    "harbor_task_environment", default={}
)
_MODEL_CREDENTIAL_KEYS = frozenset(
    key
    for names in PROVIDER_KEYS.values()
    for key in ([names] if isinstance(names, str) else names)
    if is_sensitive_env_key(key)
) | {
    "TOGETHER_API_KEY",
    "CUSTOM_API_KEY",
    "MSWEA_API_KEY",
    "LLM_API_KEY",
    "API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "GOOGLE_API_KEY",
    "CURSOR_API_KEY",
    "WINDSURF_API_KEY",
    "DATABRICKS_TOKEN",
    "TETRATE_API_KEY",
    "OPENCODE_API_KEY",
    "COPILOT_GITHUB_TOKEN",
    "ROVODEV_USER_API_TOKEN",
}
_HOST_CREDENTIAL_PREFIXES = (
    "AWS_",
    "S3_",
    "CW_KEY_",
    "AZURE_",
    "CLOUDSDK_AUTH_",
    "IRIS_",
    "RIGGING_",
)
_HOST_CREDENTIAL_KEYS = frozenset(
    {
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_CREDENTIALS",
        "GOOGLE_CLOUD_KEYFILE_JSON",
        "GCP_CREDENTIALS",
        "GCP_SERVICE_ACCOUNT_KEY",
        "HF_TOKEN",
        "HF_TOKEN_PATH",
        "HUGGING_FACE_HUB_TOKEN",
        "HUGGINGFACE_API_KEY",
        "HUGGINGFACE_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
        "WANDB_API_KEY",
        "MARIN_CLUSTER_TOKEN",
        "KUBECONFIG",
        "SSH_AUTH_SOCK",
        "FSSPEC_S3",
        "CODEX_AUTH_JSON_PATH",
        "CODEX_FORCE_AUTH_JSON",
    }
)


def validate_task_environment(values: Mapping[str, str]) -> None:
    """Reject host authentication material at the sandbox environment boundary."""
    for key, value in values.items():
        normalized = key.upper()
        if normalized in _HOST_CREDENTIAL_KEYS or normalized.startswith(
            _HOST_CREDENTIAL_PREFIXES
        ):
            raise ValueError(
                f"Host storage or control-plane credential variable is forbidden in task environment: {key}"
            )
        if (
            normalized in _MODEL_CREDENTIAL_KEYS
            and value not in ("", "EMPTY")
            and not is_env_template(value)
        ):
            raise ValueError(
                f"Provider credentials must remain on the host; use a Shellbox model relay: {key}"
            )
    protected_values = {
        value
        for key, value in os.environ.items()
        if value
        and value != "EMPTY"
        and (
            key.upper() in _HOST_CREDENTIAL_KEYS
            or key.upper() in _MODEL_CREDENTIAL_KEYS
            or (
                key.upper().startswith(_HOST_CREDENTIAL_PREFIXES)
                and is_sensitive_env_key(key)
            )
        )
    }
    if any(value in protected_values for value in values.values()):
        raise ValueError("Task environment contains host credential material")


@contextmanager
def task_environment(values: Mapping[str, str]) -> Iterator[None]:
    """Grant explicit task capabilities without exposing the harness environment."""
    validate_task_environment(values)
    token = _TASK_ENVIRONMENT.set(dict(values))
    try:
        yield
    finally:
        _TASK_ENVIRONMENT.reset(token)


def task_environment_values() -> dict[str, str]:
    """Return the explicitly granted guest environment for installed agents."""
    return dict(_TASK_ENVIRONMENT.get())


def resolve_env_vars(env_dict: dict[str, str]) -> dict[str, str]:
    """Resolve task templates only from capabilities explicitly granted by its harness."""
    validate_task_environment(env_dict)
    resolved = _resolve_env_vars(env_dict, source=_TASK_ENVIRONMENT.get())
    validate_task_environment(resolved)
    return resolved

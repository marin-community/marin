# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve host credentials and project model access into guest environments."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from harbor.utils.env import protected_environment_values
from harbor_config.env import get_required_host_vars, resolve_env_vars

DAYTONA_API_URL = "https://app.daytona.io/api"
OPENAI_API_URL = "https://api.openai.com/v1"
MODEL_RELAY_PORT = 8787
CANDIDATE_MODEL_RELAY_PORT = 8788
VERIFIER_MODEL_ROLES = (
    "MODEL_NAME",
    "JUDGE_MODEL",
    "REFUSAL_JUDGE_MODEL",
    "EXPANSION_MODEL",
)
_MODEL_ROLES = (*VERIFIER_MODEL_ROLES, "TAU2_USER_MODEL", "TAU2_NL_ASSERTIONS_MODEL")
_MODEL_CREDENTIALS = ("OPENAI_API_KEY", "TOGETHER_API_KEY", "CUSTOM_API_KEY")


def judge_model(values: Mapping[str, str]) -> str | None:
    """Return the shared model identity declared by verifier roles."""
    models = {
        (
            value
            if name == "MODEL_NAME"
            else (
                value.removeprefix("together_ai/")
                if value.startswith("together_ai/")
                else value.removeprefix("openai/")
            )
        )
        for name in VERIFIER_MODEL_ROLES
        if (value := values.get(name))
    }
    if len(models) > 1:
        raise ValueError(
            "Distinct verifier model roles require separate host capabilities"
        )
    return next(iter(models), None)


def endpoint_authentication(
    endpoint: str,
    declarations: Mapping[str, str],
    *,
    source: Mapping[str, str],
) -> dict[str, str]:
    """Bind an endpoint to explicitly declared model credentials from a host snapshot."""
    declared = {
        name: value
        for name, value in declarations.items()
        if name in _MODEL_CREDENTIALS
    }
    sources = {name for name, _ in get_required_host_vars(declared)}
    if sources - set(_MODEL_CREDENTIALS):
        raise ValueError(
            "Model credentials must name an explicit model-provider source"
        )
    resolved = resolve_env_vars(
        declared, source={name: source[name] for name in sources if name in source}
    )
    keys = {value for value in resolved.values() if value and value != "EMPTY"}
    if len(keys) > 1:
        raise ValueError("A model endpoint requires one explicit credential")
    key = next(iter(keys), None)
    if key is None and not declared and endpoint.rstrip("/") == OPENAI_API_URL:
        key = source.get("OPENAI_API_KEY")
    return {endpoint: key or "EMPTY"}


def model_guest_credentials(values: Mapping[str, str]) -> dict[str, str]:
    """Replace declared provider credentials with guest dummy values."""
    return {
        name: (
            "EMPTY"
            if name.endswith("API_KEY") or name in {"ANTHROPIC_AUTH_TOKEN", "API_KEY"}
            else value
        )
        for name, value in values.items()
    }


def model_guest_environment(values: Mapping[str, str], model: str) -> dict[str, str]:
    """Project model aliases and dummy credentials onto the guest judge relay."""
    result = model_guest_credentials(values)
    result.update(
        OPENAI_BASE_URL=f"http://127.0.0.1:{MODEL_RELAY_PORT}/v1",
        OPENAI_API_BASE=f"http://127.0.0.1:{MODEL_RELAY_PORT}/v1",
        OPENAI_API_KEY="EMPTY",
    )
    for name in _MODEL_ROLES:
        if name in result:
            result[name] = model if name == "MODEL_NAME" else "openai/" + model
    return result


def model_api_url(values: Mapping[str, str]) -> str | None:
    """Read either OpenAI SDK endpoint spelling and reject conflicting aliases."""
    urls = {
        value.rstrip("/")
        for name in ("OPENAI_BASE_URL", "OPENAI_API_BASE")
        if (value := values.get(name))
    }
    if len(urls) > 1:
        raise ValueError("Conflicting OpenAI model endpoint aliases")
    return next(iter(urls), None)


def task_model_environment(model: str) -> dict[str, str]:
    """Grant task model roles access through the guest judge relay."""
    return model_guest_environment(
        {
            "TOGETHER_API_KEY": "EMPTY",
            "CUSTOM_API_KEY": "EMPTY",
            "TAU2_USER_MODEL": model,
            "TAU2_NL_ASSERTIONS_MODEL": model,
        },
        model,
    )


@dataclass(frozen=True)
class DaytonaConnection:
    api_key: str | None = field(repr=False)
    api_url: str
    target: str | None


@dataclass(frozen=True)
class HostCapabilities:
    """Host authority passed outside serialized job and task configuration."""

    daytona: DaytonaConnection | None = field(default=None, repr=False)
    model_authentication: Mapping[str, str] = field(default_factory=dict, repr=False)
    protected_values: frozenset[str] = field(default_factory=frozenset, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_authentication",
            MappingProxyType(dict(self.model_authentication)),
        )


def host_capabilities_from_environment(
    environment: Mapping[str, str],
    *,
    model_authentication: Mapping[str, str],
) -> HostCapabilities:
    """Resolve backend and model authority from one executable-owned snapshot."""
    return HostCapabilities(
        daytona=DaytonaConnection(
            api_key=environment.get("DAYTONA_API_KEY"),
            api_url=environment.get("DAYTONA_API_URL", DAYTONA_API_URL),
            target=environment.get("DAYTONA_TARGET"),
        ),
        model_authentication=model_authentication,
        protected_values=protected_environment_values(environment)
        | frozenset(
            value
            for value in model_authentication.values()
            if value and value != "EMPTY"
        ),
    )

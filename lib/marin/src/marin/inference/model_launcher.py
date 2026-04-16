# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared contract between model backends and evaluators.

- `ModelDeployment`: how to stand up a server (backend-owned).
- `RunningModel` / `OpenAIEndpoint`: the HTTP handle that crosses the boundary.
- Run configs (evaluator-owned) live in `marin.evaluation.api`.

A `ModelLauncher` stands up a server for a `ModelDeployment` and yields a
`RunningModel`. Callers are free to skip the launcher entirely and construct a
`RunningModel` directly (e.g. for an already-running endpoint, a hosted API,
or a fake server in tests).

Concrete launchers live in sibling modules (e.g. `marin.inference.vllm_launcher`).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Protocol

# Sentinel URL for Harbor's external-API mode: LiteLLM resolves the real
# endpoint from the model name (e.g. "claude-opus-4"). Not a real wire URL —
# consumers that speak OpenAI HTTP directly must reject it.
LITELLM_PROVIDER_URL = "litellm://provider"


@dataclass(frozen=True)
class OpenAIEndpoint:
    """Pure OpenAI wire contract: base URL + model id. No tokenizer."""

    url: str
    model: str


@dataclass(frozen=True)
class ModelDeployment:
    """Model-only: how to stand up a server for a particular model."""

    path: str | None
    engine_kwargs: dict = field(default_factory=dict)
    tokenizer_override: str | None = None


@dataclass(frozen=True)
class RunningModel:
    """Shared handoff: an endpoint plus a tokenizer reference for clients that need one.

    `tokenizer_ref` is a local/HF tokenizer id. Some OpenAI-compatible clients
    (lm-eval's `local-completions`) need one; others (LiteLLM-routed Harbor)
    do not — they can pass the empty string.
    """

    endpoint: OpenAIEndpoint
    tokenizer_ref: str


class ModelLauncher(Protocol):
    """Brings up a server for a `ModelDeployment` and yields a `RunningModel`."""

    @contextmanager
    def launch(self, deployment: ModelDeployment) -> Iterator[RunningModel]: ...

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class OpenAIEndpoint:
    """OpenAI-compatible HTTP endpoint for a served model."""

    base_url: str
    model: str
    api_key: str | None = None

    def url(self, path: str) -> str:
        """Return an endpoint URL under the API root."""
        return f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"


@dataclass(frozen=True)
class RunningModel:
    """A model that is already being served by a launcher-owned backend."""

    endpoint: OpenAIEndpoint
    tokenizer: str | None = None


@dataclass(frozen=True)
class ModelDeployment:
    """Configuration needed by a launcher to serve a model artifact."""

    model_name: str
    model_path: str
    tokenizer: str | None = None
    engine_kwargs: Mapping[str, object] = field(default_factory=dict)


class ModelLauncher(Protocol):
    """Launch a model and own its serving lifecycle."""

    def launch(self, deployment: ModelDeployment) -> AbstractContextManager[RunningModel]:
        """Return a context manager that yields a running served model."""
        ...

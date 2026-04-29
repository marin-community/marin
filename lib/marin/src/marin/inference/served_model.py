# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from typing import Literal, Protocol

from marin.evaluation.evaluators.evaluator import ModelConfig
from marin.inference.vllm_server import VllmEnvironment

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OpenAIEndpoint:
    """OpenAI-compatible HTTP endpoint for a served model."""

    base_url: str
    model: str
    api_key: str | None = None


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


class VllmModelLauncher:
    """Direct vLLM implementation of the served-model launcher boundary."""

    def __init__(
        self,
        *,
        mode: Literal["native", "docker"] | None = None,
        host: str = "127.0.0.1",
        port: int | None = None,
        timeout_seconds: int = 3600,
        docker_image: str | None = None,
        docker_run_args: list[str] | None = None,
        extra_args: list[str] | None = None,
    ) -> None:
        self.mode = mode
        self.host = host
        self.port = port
        self.timeout_seconds = timeout_seconds
        self.docker_image = docker_image
        self.docker_run_args = docker_run_args
        self.extra_args = extra_args

    @contextmanager
    def launch(self, deployment: ModelDeployment) -> Iterator[RunningModel]:
        """Start vLLM for the deployment and yield a backend-neutral handle."""
        model = ModelConfig(
            name=deployment.model_name,
            path=deployment.model_path,
            engine_kwargs=dict(deployment.engine_kwargs),
        )
        with VllmEnvironment(
            model=model,
            mode=self.mode,
            host=self.host,
            port=self.port,
            timeout_seconds=self.timeout_seconds,
            docker_image=self.docker_image,
            docker_run_args=self.docker_run_args,
            extra_args=self.extra_args,
        ) as env:
            if env.model_id is None:
                raise RuntimeError("vLLM server did not report a model id.")
            try:
                yield RunningModel(
                    endpoint=OpenAIEndpoint(base_url=env.server_url, model=env.model_id),
                    tokenizer=deployment.tokenizer,
                )
            except Exception:
                for label, value in env.diagnostics(max_lines=300).items():
                    logger.error("%s:\n%s", label, value)
                raise

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""vLLM local inference backend."""

import contextlib
import os
import socket
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass

from marin.evaluation.evaluators.evaluator import ModelConfig
from marin.inference.backend import OPENAI_API_SUFFIX, ModelSpec
from marin.inference.config import DEFAULT_CUDA_VLLM_VERSION, VllmEngineConfig, VllmLauncherType, VllmSource
from marin.inference.tpu_vllm_pins import tpu_inference_fork_ref, vllm_fork_ref
from marin.inference.vllm_server import (
    IsolatedCudaVllm,
    IsolatedTpuVllm,
    VllmEnvironment,
    VllmLauncher,
    VllmType,
    WorkspaceVllm,
)


def vllm_launcher(config: VllmEngineConfig) -> VllmLauncher:
    if config.launcher is VllmLauncherType.WORKSPACE:
        return WorkspaceVllm()
    if config.launcher is VllmLauncherType.TPU:
        return IsolatedTpuVllm(vllm_ref=vllm_fork_ref(), tpu_inference_ref=tpu_inference_fork_ref())
    source = VllmType.MARIN_FORK if config.source is VllmSource.MARIN_FORK else VllmType.UPSTREAM
    version = config.version if source is VllmType.UPSTREAM else None
    if source is VllmType.UPSTREAM and version is None:
        version = DEFAULT_CUDA_VLLM_VERSION
    return IsolatedCudaVllm(source=source, version=version)


def _reserve_localhost_port(host: str) -> int:
    with socket.socket() as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


@contextlib.contextmanager
def _chat_template_argument(content: str | None) -> Iterator[tuple[str, ...]]:
    if content is None:
        yield ()
        return
    with tempfile.NamedTemporaryFile("w", suffix=".jinja", prefix="marin_serve_chat_", delete=False) as handle:
        handle.write(content)
        path = handle.name
    try:
        yield ("--chat-template", path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(path)


@dataclass(frozen=True)
class VllmServedModel:
    base_url: str
    model_id: str
    environment: VllmEnvironment

    def check_alive(self) -> None:
        server = self.environment.vllm_server
        if server is not None and server.process.poll() is not None:
            raise RuntimeError(f"vLLM server exited unexpectedly with code {server.process.returncode}")


@dataclass(frozen=True)
class VllmBackend:
    config: VllmEngineConfig
    host: str = "127.0.0.1"
    port: int | None = None
    name: str = "vllm"

    @contextlib.contextmanager
    def serve(self, spec: ModelSpec) -> Iterator[VllmServedModel]:
        resolved_port = _reserve_localhost_port(self.host) if self.port is None else self.port
        engine_kwargs: dict[str, object] = {"dtype": spec.dtype}
        if spec.max_model_len is not None:
            engine_kwargs["max_model_len"] = spec.max_model_len
        if self.config.max_num_batched_tokens is not None:
            engine_kwargs["max_num_batched_tokens"] = self.config.max_num_batched_tokens
        model = ModelConfig(name=spec.model, path=spec.model_path, engine_kwargs=engine_kwargs)
        with _chat_template_argument(spec.chat_template_content) as chat_template_args:
            with VllmEnvironment(
                model=model,
                host=self.host,
                port=resolved_port,
                timeout_seconds=self.config.startup_timeout_seconds,
                extra_args=[
                    *(
                        ("--tensor-parallel-size", str(spec.tensor_parallel_size))
                        if spec.tensor_parallel_size is not None
                        else ()
                    ),
                    "--served-model-name",
                    spec.model,
                    *chat_template_args,
                    *self.config.extra_args,
                ],
                launcher=vllm_launcher(self.config),
            ) as environment:
                if environment.model_id is None:
                    raise RuntimeError("vLLM server did not report a model id")
                yield VllmServedModel(
                    base_url=environment.server_url.removesuffix(OPENAI_API_SUFFIX),
                    model_id=environment.model_id,
                    environment=environment,
                )

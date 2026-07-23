# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Start an inference backend on the current host."""

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass

from marin.inference.backend import ModelSpec, ServedModel
from marin.inference.config import LevanterEngineConfig, ServedModelConfig, VllmEngineConfig
from marin.inference.types import OpenAIEndpoint, RunningModel


@dataclass(frozen=True)
class LocalInferenceSession:
    model: RunningModel
    backend_name: str
    tensor_parallel_size: int | None
    _served: ServedModel

    def check_alive(self) -> None:
        """Raise when the local backend has stopped serving."""

        self._served.check_alive()


@contextlib.contextmanager
def local_inference(
    model: ServedModelConfig,
    engine: VllmEngineConfig | LevanterEngineConfig,
    *,
    host: str = "127.0.0.1",
    port: int | None = None,
    num_chips: int | None = None,
) -> Iterator[LocalInferenceSession]:
    """Start one inference server in this process and yield its OpenAI endpoint."""

    spec = ModelSpec(
        model=model.model,
        model_path=model.model_path or model.model,
        num_chips=num_chips,
        tensor_parallel_size=model.tensor_parallel_size,
        dtype=model.dtype,
        max_model_len=model.max_model_len,
        chat_template_content=model.chat_template_content,
    )
    if isinstance(engine, VllmEngineConfig):
        # Import only the selected implementation; Levanter pulls in JAX and Transformers.
        from marin.inference.vllm_backend import VllmBackend  # noqa: PLC0415

        backend = VllmBackend(engine, host=host, port=port)
    elif isinstance(engine, LevanterEngineConfig):
        from marin.inference.levanter_backend import LevanterBackend  # noqa: PLC0415

        backend = LevanterBackend(engine, host=host, port=port or 0)
    else:
        raise TypeError(f"Unsupported inference engine config {type(engine).__name__}")

    with backend.serve(spec) as served:
        yield LocalInferenceSession(
            model=RunningModel(
                endpoint=OpenAIEndpoint(base_url=f"{served.base_url.rstrip('/')}/v1", model=served.model_id),
                tokenizer=model.tokenizer,
            ),
            backend_name=backend.name,
            tensor_parallel_size=model.tensor_parallel_size,
            _served=served,
        )

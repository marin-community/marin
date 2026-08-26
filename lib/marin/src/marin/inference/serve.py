# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Start an inference backend on the current host."""

import contextlib
from collections.abc import Iterator, Mapping, Sequence
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
    vllm_extra_args: Sequence[str] = (),
    vllm_subprocess_env: Mapping[str, str] | None = None,
    wait_until_ready: bool = True,
    render_tensor_parallel_size: bool = True,
) -> Iterator[LocalInferenceSession]:
    """Start one inference server in this process and yield its OpenAI endpoint."""

    spec = ModelSpec(
        weights=model.weights,
        api_model=model.model_id,
        num_chips=num_chips,
        tensor_parallel_size=model.tensor_parallel_size if render_tensor_parallel_size else None,
        dtype=model.dtype,
        max_model_len=model.max_model_len,
        chat_template_content=model.chat_template_content,
        revision=model.revision,
    )
    if isinstance(engine, VllmEngineConfig):
        # Import only the selected implementation; Levanter pulls in JAX and Transformers.
        from marin.inference.vllm_backend import VllmBackend  # noqa: PLC0415

        backend = VllmBackend(engine, host=host, port=port)
    elif isinstance(engine, LevanterEngineConfig):
        if vllm_extra_args or vllm_subprocess_env is not None or not wait_until_ready or not render_tensor_parallel_size:
            raise ValueError("native vLLM launch settings require the vLLM backend")
        from marin.inference.levanter_backend import LevanterBackend  # noqa: PLC0415

        backend = LevanterBackend(engine, host=host, port=port or 0)
    else:
        raise TypeError(f"Unsupported inference engine config {type(engine).__name__}")

    serve_kwargs = (
        {
            "extra_args": vllm_extra_args,
            "subprocess_env": vllm_subprocess_env,
            "wait_until_ready": wait_until_ready,
        }
        if isinstance(engine, VllmEngineConfig)
        else {}
    )
    with backend.serve(spec, **serve_kwargs) as served:
        yield LocalInferenceSession(
            model=RunningModel(
                endpoint=OpenAIEndpoint(base_url=f"{served.base_url.rstrip('/')}/v1", model=served.model_id),
                tokenizer=model.tokenizer,
            ),
            backend_name=backend.name,
            tensor_parallel_size=model.tensor_parallel_size,
            _served=served,
        )

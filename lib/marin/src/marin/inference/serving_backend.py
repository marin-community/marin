# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Serving backends for quick-serve: the stack that actually answers OpenAI requests.

A backend boots one OpenAI-compatible HTTP server for one model on the local slice and hands
back the URL it listens on. vLLM runs as a subprocess; Levanter's inference engine runs
in-process under uvicorn. Both speak OpenAI over localhost HTTP, so the quick-serve dashboard
and its ``/v1`` reverse proxy front either one without knowing which is behind it — which is what
makes the two stacks comparable on the same model, the same slice, and the same API.

A backend *is* its config: :class:`VllmBackend` and :class:`LevanterBackend` are frozen
dataclasses carrying their own knobs and their own ``serve()``, and the launcher cloudpickles the
chosen one into the Iris job.
"""

import logging
import socket
import tempfile
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from typing import Protocol

import jax
import jax.numpy as jnp
import jmp
from levanter.compat.hf_checkpoints import HFCheckpointConverter, load_tokenizer
from levanter.inference.engine import InferenceEngineConfig
from levanter.inference.openai import InferenceServer, InferenceServerConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig

from marin.evaluation.evaluators.evaluator import ModelConfig
from marin.inference.quick_serve_dashboard import BackgroundServer, bind_serving_socket, serve_app_background
from marin.inference.vllm_server import (
    JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECONDS,
    JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES,
    IsolatedCudaVllm,
    IsolatedTpuVllm,
    VllmEnvironment,
    VllmLauncher,
    WorkspaceVllm,
    default_jax_compilation_cache_dir,
)

logger = logging.getLogger(__name__)

# Every OpenAI-compatible server mounts its routes under /v1; a backend reports the root above it.
OPENAI_API_SUFFIX = "/v1"
# Levanter sizes its KV page table from max_seq_len, so — unlike vLLM, which derives the window
# from the model — it needs a concrete number. Serve a modest window by default and let
# --max-model-len raise it, rather than reserving a KV cache for a model's full 128k claim.
DEFAULT_LEVANTER_MAX_SEQ_LEN = 4096
# The engine requires max_queued_tokens >= max_seqs (and >= the prefill/decode pack sizes).
_MIN_QUEUED_TOKENS = 512
# Levanter loads weights at one dtype; vLLM's `auto`/`half`/`float` aliases have no meaning here.
LEVANTER_DTYPES = ("bfloat16", "float16", "float32")


@dataclass(frozen=True)
class ModelSpec:
    """What to serve, and on what slice: the inputs every backend needs."""

    model: str
    """Friendly model id (HF repo id or object-store path); the id the OpenAI API reports."""
    model_path: str
    """Resolved path the weights load from: an HF repo id, or an HF-format snapshot directory."""
    num_chips: int
    tensor_parallel_size: int
    dtype: str
    max_model_len: int | None
    chat_template_content: str | None


class ServedModel(Protocol):
    """A running OpenAI-compatible server for one model on this slice."""

    @property
    def base_url(self) -> str:
        """Root URL of the local server, without the ``/v1`` suffix."""
        ...

    @property
    def model_id(self) -> str:
        """Model id the server reports over the OpenAI API."""
        ...

    def check_alive(self) -> None:
        """Raise if the server has died."""
        ...


class ServingBackend(Protocol):
    """Boots an OpenAI-compatible server for one model on the local slice."""

    @property
    def name(self) -> str:
        """Backend id, surfaced on the dashboard and in the endpoint's metadata."""
        ...

    def serve(self, spec: ModelSpec) -> AbstractContextManager[ServedModel]:
        """Serve ``spec`` for the duration of the context, yielding the running server."""
        ...


@dataclass(frozen=True)
class VllmServedModel:
    """A ``vllm serve`` subprocess."""

    base_url: str
    model_id: str
    environment: VllmEnvironment

    def check_alive(self) -> None:
        server = self.environment.vllm_server
        if server is not None and server.process.poll() is not None:
            raise RuntimeError(f"vLLM server exited unexpectedly with code {server.process.returncode}.")


@dataclass(frozen=True)
class VllmBackend:
    """Serve the model with vLLM, launched as a subprocess.

    The launcher decides which vLLM runs: the one already on the venv/image ``PATH``, or a stock
    CUDA / Marin-forked TPU vLLM provisioned in a throwaway uv-tool env (see
    :meth:`select_launcher`).
    """

    name: str = "vllm"
    vllm_version: str | None = None
    """When set, provision stock CUDA vLLM at this exact version in an isolated uv-tool env — the
    GPU serving path. ``None`` serves from the vLLM already on the venv/image ``PATH``: the
    workspace TPU-vLLM stack, or a prebuilt ``--task-image``."""
    tpu_vllm_ref: str | None = None
    """When set (with ``tpu_inference_ref``), provision Marin's forked TPU vLLM in an isolated
    uv-tool env — the checkout-free TPU serving path."""
    tpu_inference_ref: str | None = None
    """``uvx --with`` requirement for the tpu-inference fork; paired with ``tpu_vllm_ref``."""
    max_num_batched_tokens: int = 512
    """Prefill batch size. Kept modest because the TPU paged-attention kernel's on-chip (VMEM)
    scratch grows with this; large values overflow VMEM at compile."""
    startup_timeout_seconds: int = 1800
    """How long ``vllm serve`` may take to answer ``/v1/models`` before the job gives up."""
    extra_args: tuple[str, ...] = field(default_factory=tuple)
    """Raw flags forwarded verbatim to ``vllm serve``."""

    def select_launcher(self) -> VllmLauncher:
        if self.vllm_version:
            return IsolatedCudaVllm(version=self.vllm_version)
        if self.tpu_vllm_ref:
            if not self.tpu_inference_ref:
                raise ValueError("tpu_vllm_ref requires tpu_inference_ref (the tpu-inference fork).")
            return IsolatedTpuVllm(vllm_ref=self.tpu_vllm_ref, tpu_inference_ref=self.tpu_inference_ref)
        return WorkspaceVllm()

    @contextmanager
    def serve(self, spec: ModelSpec) -> Iterator[VllmServedModel]:
        engine_kwargs: dict[str, object] = {"max_num_batched_tokens": self.max_num_batched_tokens}
        if spec.max_model_len is not None:
            engine_kwargs["max_model_len"] = spec.max_model_len
        model = ModelConfig(name="quick-serve", path=spec.model_path, engine_kwargs=engine_kwargs)

        with VllmEnvironment(
            model,
            host="127.0.0.1",
            port=_reserve_localhost_port(),
            timeout_seconds=self.startup_timeout_seconds,
            extra_args=self._cli_args(spec),
            launcher=self.select_launcher(),
        ) as environment:
            if environment.model_id is None:
                raise RuntimeError("vLLM server did not report a model id.")
            yield VllmServedModel(
                base_url=environment.server_url.removesuffix(OPENAI_API_SUFFIX),
                model_id=environment.model_id,
                environment=environment,
            )

    def _cli_args(self, spec: ModelSpec) -> list[str]:
        # Pin the served model name to the requested model so the OpenAI API id stays the friendly
        # HF id regardless of whether the backing path is local or gs://.
        args = [
            "--tensor-parallel-size",
            str(spec.tensor_parallel_size),
            "--dtype",
            spec.dtype,
            "--served-model-name",
            spec.model,
        ]
        chat_template_path = _write_chat_template(spec.chat_template_content)
        if chat_template_path is not None:
            args += ["--chat-template", chat_template_path]
        return args + list(self.extra_args)


@dataclass(frozen=True)
class LevanterServedModel:
    """Levanter's inference engine, serving in-process under uvicorn."""

    base_url: str
    model_id: str
    uvicorn: BackgroundServer

    def check_alive(self) -> None:
        if not self.uvicorn.is_alive():
            raise RuntimeError("The Levanter inference server stopped serving.")


@dataclass(frozen=True)
class LevanterBackend:
    """Serve the model with Levanter's inference engine, in-process on the slice's chips.

    Weights load through :class:`~levanter.compat.hf_checkpoints.HFCheckpointConverter`, which
    infers the Levanter model class from the checkpoint's HF architecture and reads config and
    weights over fsspec, so an HF repo id and an object-store HF-format snapshot both work. A
    native Levanter checkpoint tree (not an HF export) is not servable this way.
    """

    name: str = "levanter"
    max_seqs: int = 16
    """Concurrent sequences the engine holds slots for."""
    page_size: int = 128
    """Tokens per KV-cache page."""
    hbm_utilization: float = 0.8
    """Fraction of device HBM the KV cache may claim."""

    @contextmanager
    def serve(self, spec: ModelSpec) -> Iterator[LevanterServedModel]:
        # Levanter compiles on the first request; write to the cache the vLLM path already uses so
        # a re-serve of the same model on the same slice skips the compile.
        jax.config.update("jax_compilation_cache_dir", default_jax_compilation_cache_dir())
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECONDS)

        dtype = validate_levanter_dtype(spec.dtype)
        trainer = TrainerConfig(
            mp=jmp.get_policy(f"p={dtype},c={dtype}"),
            mesh=inference_mesh(spec.num_chips, spec.tensor_parallel_size),
        )
        tokenizer = load_tokenizer(spec.model_path)
        if spec.chat_template_content is not None:
            tokenizer.chat_template = spec.chat_template_content

        converter = HFCheckpointConverter.from_hf(spec.model_path)
        model_config = converter.default_config
        max_seq_len = levanter_max_seq_len(spec.max_model_len, model_config.max_seq_len)
        logger.info(
            "Loading %s into Levanter (%s, dtype=%s, max_seq_len=%d, chips=%d, model_axis=%d)",
            spec.model_path,
            type(model_config).__name__,
            dtype,
            max_seq_len,
            spec.num_chips,
            spec.tensor_parallel_size,
        )

        with trainer.use_device_mesh():
            model = converter.load_pretrained(
                model_config.model_type,
                ref=spec.model_path,
                dtype=trainer.mp.compute_dtype,
                axis_mapping=trainer.parameter_axis_mapping,
            )
            server = InferenceServer.create(
                InferenceServerConfig(
                    trainer=trainer,
                    tokenizer=spec.model_path,
                    model_name=spec.model,
                    service=InferenceEngineConfig(
                        max_seq_len=max_seq_len,
                        max_seqs=self.max_seqs,
                        page_size=self.page_size,
                        hbm_utilization=self.hbm_utilization,
                        max_queued_tokens=max(_MIN_QUEUED_TOKENS, self.max_seqs),
                        compute_dtype=jnp.dtype(dtype),
                    ),
                ),
                model=model,
                tokenizer=tokenizer,
            )

        # Levanter's own InferenceServer.serve() owns a uvicorn it never signals to stop, so serve
        # its app under the same helper the dashboard uses and keep teardown in our hands.
        sock = bind_serving_socket("127.0.0.1", 0)
        port = sock.getsockname()[1]
        try:
            with serve_app_background(server.app, sock, name="levanter-inference") as background:
                yield LevanterServedModel(base_url=f"http://127.0.0.1:{port}", model_id=spec.model, uvicorn=background)
        finally:
            server.shutdown()


def inference_mesh(num_chips: int, tensor_parallel_size: int) -> MeshConfig:
    """Build the serving mesh: ``model`` shards the model, ``data`` absorbs the rest of the slice.

    A mesh must cover every chip, and Levanter shards attention heads and MLPs across the ``model``
    axis, so the model axis is the tensor-parallel width and whatever chips remain land on
    ``data``. When the chip count divides the model's head count the whole slice goes on the model
    axis (``data`` is 1); a head count the slice cannot divide leaves chips over, and they
    replicate the model rather than shard it — correct, but duplicated work.
    """
    data, remainder = divmod(num_chips, tensor_parallel_size)
    if remainder:
        raise ValueError(f"tensor_parallel_size={tensor_parallel_size} does not divide the {num_chips}-chip slice.")
    if data > 1:
        logger.warning(
            "tensor_parallel_size=%d does not span all %d chips; the model replicates %d ways.",
            tensor_parallel_size,
            num_chips,
            data,
        )
    return MeshConfig(axes={"replica": 1, "data": data, "model": tensor_parallel_size})


def validate_levanter_dtype(dtype: str) -> str:
    """Check a quick-serve ``--dtype`` names a dtype Levanter can load weights at."""
    if dtype not in LEVANTER_DTYPES:
        raise ValueError(f"--dtype {dtype!r} is not supported by the levanter backend; use one of {LEVANTER_DTYPES}.")
    return dtype


def levanter_max_seq_len(max_model_len: int | None, model_max_seq_len: int) -> int:
    """Pick the KV-cache window: the requested length, or a default clamped to the model's."""
    if max_model_len is None:
        return min(DEFAULT_LEVANTER_MAX_SEQ_LEN, model_max_seq_len)
    if max_model_len > model_max_seq_len:
        raise ValueError(
            f"--max-model-len {max_model_len} exceeds the model's maximum sequence length {model_max_seq_len}."
        )
    return max_model_len


def _reserve_localhost_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _write_chat_template(content: str | None) -> str | None:
    """Write an inline chat template to a file, for backends that take one by path."""
    if content is None:
        return None
    with tempfile.NamedTemporaryFile("w", suffix=".jinja", prefix="quick_serve_chat_", delete=False) as handle:
        handle.write(content)
        return handle.name

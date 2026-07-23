# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Levanter local inference backend."""

import dataclasses
import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import jmp
from levanter.compat.hf_checkpoints import HFCheckpointConverter, load_tokenizer
from levanter.inference.engine import InferenceEngineConfig
from levanter.inference.openai import InferenceServer, InferenceServerConfig
from levanter.models.lm_model import LmHeadModel
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from transformers import PreTrainedTokenizerBase

from marin.inference.backend import ModelSpec
from marin.inference.config import LevanterEngineConfig
from marin.inference.dashboard_server import BackgroundServer, bind_serving_socket, serve_app_background
from marin.inference.model_preparation import read_attention_heads, select_tensor_parallel_size
from marin.inference.vllm_server import (
    JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECONDS,
    JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES,
    default_jax_compilation_cache_dir,
)

logger = logging.getLogger(__name__)

# Levanter sizes its KV page table from max_seq_len, so — unlike vLLM, which derives the window
# from the model — it needs a concrete number. Serve a modest window by default and let
# --max-model-len raise it, rather than reserving a KV cache for a model's full 128k claim.
DEFAULT_LEVANTER_MAX_SEQ_LEN = 4096
# The engine requires max_queued_tokens >= max_seqs (and >= the prefill/decode pack sizes).
_MIN_QUEUED_TOKENS = 512
# Levanter loads weights at one dtype; vLLM's `auto`/`half`/`float` aliases have no meaning here.
LEVANTER_DTYPES = ("bfloat16", "float16", "float32")


@runtime_checkable
class SupportsPagedGeneration(Protocol):
    """A Levanter model the paged-decode inference engine can drive.

    :meth:`LevanterBackend.serve` builds an ``InferenceEngine`` that calls ``initial_cache`` and
    ``decode``. A model that implements only the full-forward ``LmHeadModel`` interface -- e.g.
    Snowball, which has no paged decode for grug attention yet -- cannot be served through it, so
    ``serve`` rejects it up front rather than after the (large) weight load.
    """

    def initial_cache(self, spec, *, dtype): ...

    def decode(self, tokens, cache, binfo, pos_ids): ...


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
class LoadedLevanterModel:
    """A Levanter model loaded on the slice's device mesh, ready to score or wrap in an engine.

    Yielded from :meth:`LevanterBackend.load_model` *inside* the device-mesh context: the model's
    arrays are already committed and sharded on ``trainer.device_mesh``, and Grug-style forwards
    reshard against the ambient mesh, so it must be used within the yielding ``with`` block.
    """

    model: LmHeadModel
    trainer: TrainerConfig
    tokenizer: PreTrainedTokenizerBase
    max_seq_len: int
    compute_dtype: str


@dataclass(frozen=True)
class LevanterBackend:
    """Serve the model with Levanter's inference engine, in-process on the slice's chips.

    Weights load through :class:`~levanter.compat.hf_checkpoints.HFCheckpointConverter`, which
    infers the Levanter model class from the checkpoint's HF architecture and reads config and
    weights over fsspec, so an HF repo id and an object-store HF-format snapshot both work. A
    native Levanter checkpoint tree (not an HF export) is not servable this way.
    """

    config: LevanterEngineConfig
    host: str = "127.0.0.1"
    port: int = 0
    name: str = "levanter"

    def _resolved_spec(self, spec: ModelSpec) -> ModelSpec:
        num_chips = spec.num_chips or jax.device_count()
        tensor_parallel_size = spec.tensor_parallel_size
        if tensor_parallel_size is None:
            attention_heads, key_value_heads = read_attention_heads(spec.model_path)
            tensor_parallel_size = select_tensor_parallel_size(attention_heads, num_chips, key_value_heads)
        return replace(spec, num_chips=num_chips, tensor_parallel_size=tensor_parallel_size)

    @contextmanager
    def load_model(
        self, spec: ModelSpec, config_overrides: Mapping[str, Any] | None = None
    ) -> Iterator[LoadedLevanterModel]:
        """Load model weights onto the serving mesh.

        Args:
            spec: Model, dtype, and sharding settings to load.
            config_overrides: Fields to replace on the HF-derived Levanter model config.

        Yields:
            The loaded model and serving metadata inside its active device-mesh context.
        """
        spec = self._resolved_spec(spec)
        assert spec.num_chips is not None
        assert spec.tensor_parallel_size is not None

        # Levanter compiles on the first request; write to the cache the vLLM path already uses so
        # a re-serve of the same model on the same slice skips the compile.
        jax.config.update("jax_compilation_cache_dir", default_jax_compilation_cache_dir())
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECONDS)

        dtype = validate_levanter_dtype(spec.dtype)
        # Resolve the model class first: whether the mesh needs explicit axes is a model property.
        converter = HFCheckpointConverter.from_hf(spec.model_path)
        model_config = converter.default_config
        if config_overrides:
            model_config = dataclasses.replace(model_config, **dict(config_overrides))
        trainer = TrainerConfig(
            mp=jmp.get_policy(f"p={dtype},c={dtype}"),
            mesh=inference_mesh(spec.num_chips, spec.tensor_parallel_size),
            use_explicit_mesh_axes=model_config.requires_explicit_mesh_axes,
        )
        tokenizer = load_tokenizer(spec.model_path)
        if spec.chat_template_content is not None:
            tokenizer.chat_template = spec.chat_template_content

        max_seq_len = levanter_max_seq_len(spec.max_model_len, model_config.max_seq_len)
        logger.info(
            "Loading %s into Levanter (%s, dtype=%s, max_seq_len=%d, chips=%d, model_axis=%d, explicit_mesh=%s)",
            spec.model_path,
            type(model_config).__name__,
            dtype,
            max_seq_len,
            spec.num_chips,
            spec.tensor_parallel_size,
            trainer.use_explicit_mesh_axes,
        )

        with trainer.use_device_mesh():
            model = converter.load_pretrained(
                model_config.model_type,
                ref=spec.model_path,
                config=model_config,
                dtype=trainer.mp.compute_dtype,
                axis_mapping=trainer.parameter_axis_mapping,
            )
            yield LoadedLevanterModel(
                model=model,
                trainer=trainer,
                tokenizer=tokenizer,
                max_seq_len=max_seq_len,
                compute_dtype=dtype,
            )

    @contextmanager
    def serve(self, spec: ModelSpec) -> Iterator[LevanterServedModel]:
        # Reject models the inference engine cannot drive before the weight load: resolving the model
        # class from the HF config is cheap, loading the weights is not. (Forward-only scoring via
        # load_model has no such requirement.)
        model_type = HFCheckpointConverter.from_hf(spec.model_path).default_config.model_type
        if not issubclass(model_type, SupportsPagedGeneration):
            raise NotImplementedError(
                f"{model_type.__name__} implements only the full-forward LmHeadModel interface (no "
                "paged initial_cache/decode), so it cannot be served through the Levanter inference "
                "engine. Score it with LevanterBackend.load_model, or serve it with vLLM."
            )
        with self.load_model(spec) as loaded:
            # InferenceServer.create must build the engine on-mesh, so it runs inside load_model's
            # device-mesh context; the serve loop below then runs off-mesh, as before.
            server = InferenceServer.create(
                InferenceServerConfig(
                    trainer=loaded.trainer,
                    tokenizer=spec.model_path,
                    model_name=spec.model,
                    service=InferenceEngineConfig(
                        max_seq_len=loaded.max_seq_len,
                        max_seqs=self.config.max_seqs,
                        page_size=self.config.page_size,
                        hbm_utilization=self.config.hbm_utilization,
                        max_queued_tokens=max(_MIN_QUEUED_TOKENS, self.config.max_seqs),
                        compute_dtype=jnp.dtype(loaded.compute_dtype),
                    ),
                ),
                model=loaded.model,
                tokenizer=loaded.tokenizer,
            )

        # Levanter's own InferenceServer.serve() owns a uvicorn it never signals to stop, so serve
        # its app under the same helper the dashboard uses and keep teardown in our hands.
        sock = bind_serving_socket(self.host, self.port)
        port = sock.getsockname()[1]
        try:
            with serve_app_background(server.app, sock, name="levanter-inference") as background:
                yield LevanterServedModel(base_url=f"http://{self.host}:{port}", model_id=spec.model, uvicorn=background)
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
    """Check that ``--dtype`` names a dtype Levanter can load weights at."""
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

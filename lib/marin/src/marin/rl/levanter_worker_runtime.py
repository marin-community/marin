# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Native Snowball policy learner, isolated from the SkyRL Torch runtime."""

import argparse
import asyncio
import atexit
import contextlib
import ctypes
import dataclasses
import faulthandler
import gc
import hashlib
import json
import math
import os
import shlex
import signal
import sys
import tempfile
import threading
import time
import traceback
from collections.abc import Callable
from functools import partial
from pathlib import Path

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import jaxlib
import jmp
import numpy as np
import optax
from haliax._src.compile_utils import clear_caches as clear_haliax_caches
from jax.experimental import multihost_utils
from jax.extend.backend import clear_backends
from jax.sharding import NamedSharding, PartitionSpec
from jaxlib import xla_client
from levanter.checkpoint import load_checkpoint, load_checkpoint_metadata, save_checkpoint
from levanter.compat.hf_checkpoints import RepoRef
from levanter.distributed import DistributedConfig
from levanter.inference.snowball_engine_replay import align_snowball_policy_routes
from levanter.inference.snowball_replay_payload import SnowballReplayPayload
from levanter.kernels.pallas.fused_cross_entropy_loss.api import FixedCEPlan, FixedCESignature, prepare_fixed_ce_plan
from levanter.models.snowball import SnowballConfig, SnowballLMHeadModel
from levanter.policy import ClippedPolicyConfig
from levanter.policy_learner import SpmdPolicyLearner
from levanter.policy_model import PolicyBatch, PolicyRouteCaptures, policy_model_loss
from levanter.tensorstore_serialization import _transfer_shard_to_pageable_host
from levanter.tracker.tracker import NoopConfig
from levanter.trainer import Trainer, TrainerConfig
from levanter.utils.jax_utils import multihost_allgather_sync
from levanter.utils.mesh import MeshConfig
from numpy._core import multiarray as np_multiarray

from marin.rl.levanter_compile_diagnostics import _memory_stats, capture_compile, capture_failure, scientific_xla_flags
from marin.rl.levanter_publication import SnowballPublication
from marin.rl.levanter_shutdown_observer import publish_ack_marker, validated_ack_marker
from marin.rl.levanter_worker_protocol import read_frame, write_frame

_DIAGNOSTIC_STALL_SECONDS = 120.0
_FINALIZATION_TRACE_SECONDS = 3.0
_SHUTDOWN_PHASE_MAX_EVENTS = 64
_SHUTDOWN_PHASE_MAX_BYTES = 32 << 10
_SHUTDOWN_PHASE_EVENTS = frozenset(
    {"cleanup_phase_begin", "cleanup_phase_end", "cleanup_phase_error", "process_exit_begin"}
)
_SHUTDOWN_PHASE_NAMES = frozenset(
    {
        "synchronize",
        "trainer_exit",
        "release_owned_resources",
        "clear_haliax_caches",
        "clear_jax_caches",
        "collect_python_resources",
        "release_backend_clients",
        "distributed_shutdown",
    }
)
_FINITE_CHUNK_ELEMENTS = 1 << 20
_FINITE_COMPONENTS = ("model", "opt_state", "training_key", "step", "model_averaging")


def _diagnostic_receipt(event: str, **fields) -> None:
    print(
        "native_phase_receipt "
        + json.dumps(
            {
                "event": event,
                "pid": os.getpid(),
                "process_id": jax.process_index(),
                "thread_id": threading.get_ident(),
                "main_thread_id": threading.main_thread().ident,
                "time_ns": time.time_ns(),
                "monotonic_ns": time.monotonic_ns(),
                **fields,
            }
        ),
        file=sys.stderr,
        flush=True,
    )


def _trainable_filter(model):
    mask = jax.tree.map(lambda _: True, model)
    return eqx.tree_at(
        lambda tree: tuple(block.mlp.router_bias for block in tree.transformer.blocks),
        mask,
        tuple(False for _ in model.transformer.blocks),
    )


def _payload_fingerprint(payload: dict) -> str:
    digest = hashlib.sha256()
    for name in ("old_logprobs", "advantages", "policy_weights"):
        value = payload["arrays"][name]
        digest.update(name.encode())
        digest.update(np.asarray(value).tobytes())
    return digest.hexdigest()


def _router_bias_fingerprint(model: SnowballLMHeadModel) -> str:
    return _state_fingerprint(tuple(block.mlp.router_bias for block in model.transformer.blocks))["sha256"]


def _state_fingerprint(tree) -> dict:
    """Hash this process's state shards in global-index order with bounded host memory."""
    digest = hashlib.sha256()
    layout_digest = hashlib.sha256()
    addressable_bytes = 0
    schema_digest = hashlib.sha256()
    elements = nonfinite = arrays = unique_shards = 0
    bad_shards = []
    for path, value in jax.tree_util.tree_flatten_with_path(tree)[0]:
        if not eqx.is_array(value):
            continue
        arrays += 1
        schema_digest.update(json.dumps((jax.tree_util.keystr(path), value.shape, str(value.dtype))).encode())
        shards = (
            [
                (repr(shard.index), shard)
                for shard in sorted(value.addressable_shards, key=lambda shard: repr(shard.index))
            ]
            if isinstance(value, jax.Array)
            else [("host", None)]
        )
        seen_indices = set()
        for index, shard in shards:
            if index in seen_indices:
                continue
            seen_indices.add(index)
            descriptor = (jax.tree_util.keystr(path), value.shape, str(value.dtype), index)
            encoded_descriptor = json.dumps(descriptor).encode()
            digest.update(encoded_descriptor)
            layout_digest.update(encoded_descriptor)
            data = value if shard is None else shard.data
            # NumPy conversion caches the host copy on a GPU JAX array. Reuse the
            # checkpoint staging path so the original model/Adam shard retains none.
            snapshot = (
                asyncio.run(_transfer_shard_to_pageable_host(shard))
                if shard is not None and data.device.platform == "gpu"
                else np.asarray(data)
            )
            array = np.ascontiguousarray(snapshot)
            digest.update(memoryview(array.view(np.uint8)))
            addressable_bytes += array.nbytes
            unique_shards += 1
            flat = array.reshape(-1)
            bad = 0
            for start in range(0, flat.size, _FINITE_CHUNK_ELEMENTS):
                block = flat[start : start + _FINITE_CHUNK_ELEMENTS]
                finite = np.isfinite(block)
                bad += int(block.size) - int(np.count_nonzero(finite))
                del finite, block
            elements += int(array.size)
            nonfinite += bad
            if bad and len(bad_shards) < 8:
                bad_shards.append({"path": jax.tree_util.keystr(path)[:256], "index": index[:256], "nonfinite": bad})
            del flat
            del array, snapshot
    return {
        "sha256": digest.hexdigest(),
        "layout_sha256": layout_digest.hexdigest(),
        "addressable_bytes": addressable_bytes,
        "schema_sha256": schema_digest.hexdigest(),
        "arrays": arrays,
        "unique_local_shards": unique_shards,
        "elements": elements,
        "finite_elements": elements - nonfinite,
        "nonfinite_elements": nonfinite,
        "first_nonfinite_shards": bad_shards,
    }


def _persist_finite_proof(receipt: dict, *, suffix: str) -> dict:
    """Persist bounded host evidence without replacing an existing failure."""
    paths, errors = [], []
    phase = receipt.get("phase", "invalid")
    safe_phase = phase if isinstance(phase, str) and phase.isalnum() and len(phase) < 32 else "invalid"
    directories = dict.fromkeys(
        value for name in ("SKYRL_DEBUG_ARTIFACT_DIR", "IRIS_OUTPUT_DIR") if (value := os.environ.get(name))
    )
    encoded = json.dumps(receipt, sort_keys=True, allow_nan=False)
    for directory in directories:
        path = (
            Path(directory)
            / "native-finite"
            / f"process-{receipt['process_id']}-pid-{os.getpid()}"
            / f"{safe_phase}-{suffix}.json"
        )
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(encoded + "\n")
            paths.append(str(path))
        except Exception as error:
            errors.append(type(error).__name__)
    try:
        if sys.stderr is not None:
            print("NATIVE_STATE_FINITE " + encoded, file=sys.stderr, flush=True)
    except Exception:
        pass
    return {"persisted": bool(paths), "paths": paths, "error_types": errors}


def _initialize_distributed(config: dict) -> None:
    distributed = config["distributed"]
    if distributed["num_processes"] > 1:
        jax.distributed.initialize(
            coordinator_address=distributed["coordinator_address"],
            num_processes=distributed["num_processes"],
            process_id=distributed["process_id"],
            local_device_ids=list(range(distributed["local_device_count"])),
            initialization_timeout=config["initialization_timeout_seconds"],
        )
    backend = jax.default_backend()
    requires_gpu = config["require_accelerator"] and backend != "gpu"
    if requires_gpu or jax.local_device_count() != distributed["local_device_count"]:
        observed = {
            "expected_local_devices": distributed["local_device_count"],
            "local_devices": jax.local_device_count(),
            "global_devices": jax.device_count(),
            "backend": backend,
            "device_kinds": sorted({device.device_kind for device in jax.local_devices()}),
            "selectors": {
                name: os.environ.get(name)
                for name in ("JAX_PLATFORMS", "JAX_PLATFORM_NAME", "JAX_CUDA_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
            },
        }
        reason = (
            "Native learner requires CUDA GPU devices"
            if requires_gpu
            else "JAX local device count differs from the reserved learner GPUs"
        )
        raise ValueError(f"{reason}: {json.dumps(observed)}")


def _projection_dtype(runtime: dict) -> jnp.dtype | None:
    return {"float32": None, "bfloat16": jnp.dtype("bfloat16")}[runtime["projection_logits_dtype"]]


def _numpy_allocation_policy(requested: bool | None) -> dict:
    policy = {
        "requested_madvise_hugepage": requested,
        "observed_madvise_hugepage": bool(np_multiarray._get_madvise_hugepage()),
        "environment_value": os.environ.get("NUMPY_MADVISE_HUGEPAGE"),
        "numpy_version": np.__version__,
        "platform": sys.platform,
    }
    if requested is not None and policy["observed_madvise_hugepage"] != requested:
        raise ValueError(f"Requested NumPy madvise-hugepage policy is not effective: {json.dumps(policy)}")
    return policy


def _allocator_profile() -> dict:
    if os.environ.get("TF_FORCE_UNIFIED_MEMORY", "false").lower() not in ("false", "0"):
        raise ValueError("Native learner requires disabled TF_FORCE_UNIFIED_MEMORY to preserve its memory cap")
    options = xla_client.generate_pjrt_gpu_plugin_options()
    return {
        "allocator": options["allocator"],
        "memory_fraction": options.get("memory_fraction", 0.75),
        "preallocate": options.get("preallocate", True),
        "unified_memory": False,
    }


def _validate_allocator(config: dict) -> None:
    flags = shlex.split(os.environ.get("XLA_FLAGS", ""))
    for index, token in enumerate(flags):
        name, separator, value = token.partition("=")
        if name not in ("--xla_gpu_command_buffer_update_mode", "--xla_gpu_temp_buffer_use_separate_color"):
            continue
        if not separator:
            value = flags[index + 1] if index + 1 < len(flags) else ""
        if name == "--xla_gpu_command_buffer_update_mode":
            # SKIP_TEMP makes pinned PJRT silently replace BFC with VMM.
            if value != "ALWAYS_UPDATE":
                raise ValueError("Native learner requires xla_gpu_command_buffer_update_mode=ALWAYS_UPDATE")
        elif value not in ("false", "0"):
            # This otherwise installs an uncapped CUDA-async scratch allocator.
            raise ValueError("Native learner requires disabled xla_gpu_temp_buffer_use_separate_color")
    runtime = config["runtime"]
    if runtime["allocator"] not in ("bfc", "vmm") or (runtime["allocator"] == "vmm" and runtime["preallocate"]):
        raise ValueError("Native learner requires bfc, or vmm with preallocate=false")
    expected = {
        "allocator": runtime["allocator"],
        "memory_fraction": runtime["device_memory_fraction"],
        "preallocate": runtime["preallocate"],
        "unified_memory": False,
    }
    if jax.config.jax_pjrt_client_create_options or _allocator_profile() != expected:
        raise ValueError("Native learner PJRT allocator options differ from the configured allocation policy")


def _allocator_limits() -> list[dict]:
    rows = []
    for device in jax.local_devices():
        row: dict = {"device_id": device.id}
        try:
            stats = device.memory_stats()
            row["bytes_limit"] = None if stats is None else stats.get("bytes_limit")
            row["statistics_available"] = stats is not None
        except Exception as error:
            row["error_type"] = type(error).__name__
            row["statistics_available"] = False
        rows.append(row)
    return rows


def _create_trainer(config: dict) -> Trainer:
    distributed = config["distributed"]
    runtime = config["runtime"]
    mesh_axes = dict(runtime["mesh_axes"])
    dcn = {"replica_dcn": mesh_axes.pop("replica_dcn")}
    train_batch_size = config["mini_batch_size"]
    compute_dtype = "bfloat16" if config["bf16"] else "float32"
    trainer_config = TrainerConfig(
        id=config["run_id"],
        seed=config["seed"],
        train_batch_size=train_batch_size,
        per_device_parallelism=config["micro_batch_size_per_device"],
        num_train_steps=config["num_training_steps"],
        mesh=MeshConfig(
            axes=mesh_axes,
            dcn_axes=dcn,
            compute_mapping={"batch": ["replica_dcn", "data", "expert"], "position": "context"},
            param_mapping={},
        ),
        use_explicit_mesh_axes=True,
        mp=jmp.get_policy(f"params=float32,compute={compute_dtype},output=float32"),
        field_dtype_overrides={"router": jnp.dtype("float32"), "router_bias": jnp.dtype("float32")},
        distributed=DistributedConfig(initialize_jax_distributed=False),
        tracker=NoopConfig(),
        require_accelerator=config["require_accelerator"],
        log_dir=f"{tempfile.gettempdir()}/levanter-policy/{config['run_id']}/process-{distributed['process_id']}",
        log_jaxprs=False,
        log_xla_hlo=False,
        load_checkpoint=False,
    )
    trainer_config.initialize()
    microbatch = trainer_config.microbatch_size
    if microbatch is None or train_batch_size % microbatch:
        raise ValueError("Policy optimizer batch must divide the native execution microbatch")
    optimizer = config["optimizer"]
    optimizer_transform = optax.chain(
        optax.clip_by_global_norm(optimizer["max_grad_norm"]),
        optax.adamw(
            optimizer["lr"],
            b1=optimizer["adam_betas"][0],
            b2=optimizer["adam_betas"][1],
            eps=optimizer["epsilon"],
            weight_decay=optimizer["weight_decay"],
        ),
    )
    projection_dtype = _projection_dtype(runtime)
    loss = partial(
        policy_model_loss,
        config=ClippedPolicyConfig(config["eps_clip_low"], config["eps_clip_high"], 0.0),
        accumulation_steps=train_batch_size // microbatch,
        block_size=runtime["projection_block_size"],
        logits_dtype=projection_dtype,
    )
    return Trainer(trainer_config, optimizer_transform, loss, add_default_hooks=False)


def _load_model(config: dict, trainer: Trainer):
    runtime = config["runtime"]
    model_config = SnowballConfig()
    source = config["model"]
    reference = (
        RepoRef(source["source_uri"], None) if source.get("source_uri") else RepoRef(source["path"], source["revision"])
    )
    converter = model_config.hf_checkpoint_converter(ref_checkpoint=reference)
    # Streamed SkyRL model paths contain the exact staged metadata/tokenizer;
    # the immutable object URI remains the authority for model tensor bytes.
    converter = converter.replaced(tokenizer=source["path"])
    model_config = converter.config_from_hf_checkpoint()
    model_config = dataclasses.replace(
        model_config,
        **{
            name: runtime[name]
            for name in (
                "attention_implementation",
                "moe_implementation",
                "router_dot_fp32",
                "moe_capacity_factor",
                "fp32_activations",
                "head_gate_fp32",
                "xsa_fp32",
            )
        },
    )
    if model_config.moe_capacity_factor < runtime["mesh_axes"]["expert"]:
        raise ValueError("The SkyRL learner requires dropless expert dispatch")
    # Load FP32 master parameters directly onto the native model's mesh. In
    # particular, routing weights must not be rounded through a BF16 load.
    model = converter.load_pretrained(
        SnowballLMHeadModel,
        config=model_config,
        axis_mapping=trainer.parameter_axis_mapping,
        dtype=jnp.float32,
    )
    return model


class NativePolicyRuntime:
    """Own the native model, Adam state, RNG and publication snapshot."""

    def __init__(self):
        self.settings: dict = {}
        self.trainer = None
        self.state = None
        self.learner: SpmdPolicyLearner | None = None
        self.publisher = None
        self.version = 0
        self.source_identity = None
        self._compiler_evidence = None
        self.numpy_allocation_policy = None
        self._closed = False
        self._shutdown_observer_marker = None
        self._shutdown_phase_events = 0
        self._shutdown_phase_bytes = 0
        self._shutdown_phase_sink_failed = False

    def initialize(self, config: dict) -> dict:
        if self.trainer is not None:
            raise RuntimeError("Learner is already initialized")
        self.settings = config
        self._closed = False
        self.numpy_allocation_policy = _numpy_allocation_policy(config["runtime"].get("numpy_madvise_hugepage"))
        _validate_allocator(config)
        if config["runtime"].get("validate_checkpoint_roundtrip", False):
            os.environ["LEVANTER_CE_DIAGNOSTICS"] = "1"
        else:
            os.environ.pop("LEVANTER_CE_DIAGNOSTICS", None)
        _initialize_distributed(config)
        self.trainer = _create_trainer(config)
        self.trainer.__enter__()
        self._load_policy(config)
        self.publisher = SnowballPublication(self.trainer.mp.cast_to_compute(self.state.model), self.trainer.device_mesh)
        return self._initialization_receipt()

    def _load_policy(self, config: dict) -> None:
        runtime = config["runtime"]
        source = config["model"]
        projection_dtype = _projection_dtype(runtime)
        model = _load_model(config, self.trainer)
        # Finish import resharding before allocating the full Adam state.
        jax.block_until_ready(model)
        self.state = self.trainer.initial_state(
            jax.random.PRNGKey(config["seed"]), model=model, is_trainable=_trainable_filter(model)
        )
        self.learner = SpmdPolicyLearner(
            self.trainer,
            block_size=runtime["projection_block_size"],
            logits_dtype=projection_dtype,
        )
        self.source_identity = (
            {"source_uri": source["source_uri"], "source_identity": source["source_identity"]}
            if source.get("source_uri")
            else {"path": source["path"], "revision": source["revision"]}
        )
        self.frozen_router_bias = _router_bias_fingerprint(self.state.model)

    def _initialization_receipt(self) -> dict:
        model_config = self.state.model.config
        source = self.settings["model"]
        train_batch_size = self.settings["mini_batch_size"]
        microbatch = self.trainer.config.microbatch_size
        assert microbatch is not None
        return {
            **self.runtime_identity(),
            "process_id": jax.process_index(),
            "global_devices": jax.device_count(),
            "local_devices": jax.local_device_count(),
            "mesh": dict(self.trainer.device_mesh.shape),
            "device_layout": [
                {
                    "global_device_id": device.id,
                    "process_id": device.process_index,
                    "coordinates": dict(zip(self.trainer.device_mesh.axis_names, coordinates, strict=True)),
                }
                for coordinates, device in np.ndenumerate(self.trainer.device_mesh.devices)
            ],
            "model_config": dataclasses.asdict(model_config),
            "source": source,
            "step": int(self.state.step),
            "version": self.version,
            "microbatches": train_batch_size // microbatch,
            "frozen_router_bias_sha256": self.frozen_router_bias,
        }

    def _batch(self, payload: dict) -> PolicyBatch:
        assert self.trainer is not None
        source = payload["arrays"]
        batch_size, sequence_length = source["tokens"].shape
        diagnostics = self.settings["runtime"].get("validate_checkpoint_roundtrip", False)
        if diagnostics:
            _diagnostic_receipt(
                "batch_prepare_begin",
                tokens_shape=[batch_size, sequence_length],
                response_length=payload["response_length"],
            )
        axes = (hax.Axis("batch", batch_size), hax.Axis("position", sequence_length))
        response_axes = (axes[0], hax.Axis("response", payload["response_length"]))
        sharding = NamedSharding(self.trainer.device_mesh, PartitionSpec(("replica_dcn", "data", "expert"), None))

        def named(name: str, *, response: bool):
            values = source.get(name)
            if values is None:
                values = np.zeros(tuple(axis.size for axis in response_axes), dtype=np.float32)
            values = np.asarray(values)
            array = jax.make_array_from_callback(values.shape, sharding, lambda index: values[index])
            return hax.named(array, response_axes if response else axes)

        replay = None
        if (route_payloads := payload.get("route_capture_payloads")) is not None:
            assert self.state is not None
            selected, alpha = align_snowball_policy_routes(
                self.state.model,
                tuple(SnowballReplayPayload.from_bytes(data) for data in route_payloads),
                policy_id=payload["rollout_policy_id"],
                tokens=source["tokens"],
                token_valid=source["attention_mask"],
                position_ids=source["position_ids"],
            )

            def named_routes(values, suffix):
                return hax.named(
                    jax.make_array_from_callback(values.shape, sharding, lambda index: values[index]),
                    (*axes, *suffix),
                )

            replay = PolicyRouteCaptures(
                named_routes(selected, ("layer", "top_k")),
                named_routes(alpha, ("layer",)),
            )
        batch = PolicyBatch(
            tokens=named("tokens", response=False),
            attention_mask=named("attention_mask", response=False),
            position_ids=named("position_ids", response=False),
            response_mask=named("response_mask", response=True),
            advantages=named("advantages", response=True),
            policy_weights=named("policy_weights", response=True),
            kl_weights=named("kl_weights", response=True),
            old_logprobs=named("old_logprobs", response=True),
            reference_logprobs=None,
            route_captures=replay,
        )
        if diagnostics:
            _diagnostic_receipt("batch_prepare_end", tokens_shape=[batch_size, sequence_length])
        return batch

    @contextlib.contextmanager
    def request_diagnostics(self, method: str):
        enabled = method in ("model_fingerprint", "train", "shutdown", "eof") and self.settings.get("runtime", {}).get(
            "validate_checkpoint_roundtrip", False
        )
        if not enabled:
            yield
            return
        main_ident = threading.main_thread().ident
        receipt = self.shutdown_receipt if method in ("shutdown", "eof") else _diagnostic_receipt
        finished = threading.Event()
        guard = threading.Lock()
        request_id = f"{os.getpid()}:{time.monotonic_ns()}"

        def stalled():
            with guard:
                frame = sys._current_frames().get(main_ident) if main_ident is not None else None
                if finished.is_set() or frame is None:
                    return
                stack = traceback.format_stack(frame, limit=40)
                captured_wall, captured_monotonic = time.time_ns(), time.monotonic_ns()
            # Logging may finish after request_end; the request ID and capture time
            # identify this snapshot without making the main thread wait on stderr.
            receipt(
                "request_stalled",
                method=method,
                request_id=request_id,
                main_thread_stack=stack,
                time_ns=captured_wall,
                monotonic_ns=captured_monotonic,
            )

        timer = threading.Timer(_DIAGNOSTIC_STALL_SECONDS, stalled)
        timer.daemon = True
        receipt("request_begin", method=method, request_id=request_id)
        timer.start()
        try:
            yield
        finally:
            with guard:
                finished.set()
            timer.cancel()
            receipt("request_end", method=method, request_id=request_id)

    def _projection_profile(self, payload: dict) -> tuple[dict, FixedCEPlan]:
        """Describe the closed Snowball/BF16/XLA execution, without tracing attention or CE."""
        runtime = self.settings["runtime"]
        if "LEVANTER_CE_XLA_FAST_BWD" in os.environ:
            raise ValueError("Native prepared projection requires LEVANTER_CE_XLA_FAST_BWD to be unset")
        if type(self.state.model) is not SnowballLMHeadModel or _projection_dtype(runtime) != jnp.bfloat16:
            raise ValueError("Native prepared projection requires Snowball with explicit BF16 logits")
        if self.state.model.config.fp32_activations or self.state.model.config.attention_implementation not in (
            "reference",
            "gpu_fa4_cute",
        ):
            raise ValueError("Native prepared projection requires the covered Snowball activation/attention profile")
        overrides = {name: str(dtype) for name, dtype in self.trainer.config.field_dtype_overrides.items()}
        if overrides != {"router": "float32", "router_bias": "float32"}:
            raise ValueError("Native prepared projection requires the matched router-only FP32 overrides")
        mesh = self.trainer.device_mesh
        if any(mesh.shape[name] != 1 for name in ("context", "replica", "model")):
            raise ValueError("Native prepared projection requires context=replica=model=1")
        mapping = self.trainer.compute_axis_mapping
        canonical_mapping = MeshConfig(
            compute_mapping={"batch": ["replica_dcn", "data", "expert"], "position": "context"}, param_mapping={}
        )
        if (
            mapping != canonical_mapping.resolved_compute_mapping
            or self.trainer.parameter_axis_mapping != canonical_mapping.resolved_param_mapping
            or self.trainer.hooks.jit_hooks
        ):
            raise ValueError("Native prepared projection requires the fixed learner mappings without JIT hooks")
        arrays = payload["arrays"]
        batch_size, sequence_length = arrays["tokens"].shape
        response_length = payload["response_length"]
        microbatch = self.trainer.config.microbatch_size
        if (
            batch_size != self.settings["mini_batch_size"]
            or microbatch is None
            or batch_size % microbatch
            or not 0 < response_length < sequence_length
        ):
            raise ValueError("Native prepared projection requires a complete divisible optimizer batch")
        for name in ("tokens", "attention_mask", "position_ids", "response_mask"):
            if arrays.get(name) is None:
                raise ValueError(f"Missing native policy array {name}")
        input_dtypes = {
            "tokens": "int32",
            "attention_mask": "bool",
            "position_ids": "int32",
            "response_mask": "bool",
            **{
                name: "float32"
                for name in (
                    "advantages",
                    "old_logprobs",
                    "policy_weights",
                    "kl_weights",
                    "reference_logprobs",
                    "loss_mask",
                )
            },
        }
        for name, array in arrays.items():
            if name not in input_dtypes or (array is not None and str(array.dtype) != input_dtypes[name]):
                raise ValueError(f"Unsupported native policy array or dtype: {name}")
            if array is not None:
                expected = (
                    batch_size,
                    sequence_length if name in ("tokens", "attention_mask", "position_ids") else response_length,
                )
                if array.shape != expected:
                    raise ValueError(f"Native policy array {name} does not have shape {expected}")
        if arrays.get("reference_logprobs") is not None or arrays.get("segment_ids") is not None:
            raise ValueError("Native prepared projection currently supports the unpacked KL-off recipe")
        compute = eqx.filter_eval_shape(self.trainer.mp.cast_to_compute, self.state.model)
        for path, value in jax.tree_util.tree_flatten_with_path(compute)[0]:
            if isinstance(value, jax.ShapeDtypeStruct) and jnp.issubdtype(value.dtype, jnp.floating):
                expected_dtype = overrides.get(getattr(path[-1], "name", None), "bfloat16")
                if str(value.dtype) != expected_dtype:
                    raise ValueError(f"Unsupported native compute dtype at {jax.tree_util.keystr(path)}")
        head = eqx.filter_eval_shape(lambda model: model.get_lm_head(), compute)
        embedding_dtype = compute.transformer.token_embed.dtype
        normalized = eqx.filter_eval_shape(
            lambda norm, gate, x: gate(norm(x)),
            compute.transformer.final_norm,
            compute.transformer.final_gated_norm,
            jax.ShapeDtypeStruct(
                (compute.Embed.size,), embedding_dtype, sharding=NamedSharding(mesh, PartitionSpec(None))
            ),
        )
        if embedding_dtype != jnp.bfloat16 or head.array.dtype != jnp.bfloat16 or normalized.dtype != jnp.bfloat16:
            raise ValueError("Native prepared projection requires a proven BF16 Snowball compute view")
        loss_mapping = {name: mapping[name] for name in ("batch", "response") if name in mapping}

        def partition_factor(name):
            axes = loss_mapping.get(name, ())
            axes = (axes,) if isinstance(axes, str) else axes
            return math.prod(mesh.shape[axis] for axis in axes)

        batch_factor, response_factor = partition_factor("batch"), partition_factor("response")
        if batch_size % batch_factor or microbatch % batch_factor or response_length % response_factor:
            raise ValueError("Fixed CE batch axes must divide their actual loss mesh partitions")
        h, v = head.resolve_axis(self.state.model.Embed).size, head.resolve_axis(self.state.model.Vocab).size
        rows = tuple(
            sorted({size // batch_factor * (response_length // response_factor) for size in (batch_size, microbatch)})
        )
        signatures = tuple(FixedCESignature((row, h), (row,), (h, v), "bfloat16", "int32", "bfloat16") for row in rows)
        profile = {
            "model_config": dataclasses.asdict(self.state.model.config),
            "source": self.source_identity,
            "versions": {"jax": jax.__version__, "jaxlib": jaxlib.__version__, "python": sys.version},
            "jax_numerics": {
                "default_matmul_precision": jax.config.jax_default_matmul_precision,
                "enable_x64": jax.config.jax_enable_x64,
                "default_prng_impl": jax.config.jax_default_prng_impl,
            },
            "xla_flags": scientific_xla_flags(self.settings),
            "allocation_policy": _allocator_profile(),
            "allocator_bytes_limits": sorted(
                {int(row["bytes_limit"]) for row in _allocator_limits() if row.get("bytes_limit") is not None}
            ),
            "mesh": {
                "axes": list(mesh.axis_names),
                "shape": dict(mesh.shape),
                "devices": [
                    {
                        "id": device.id,
                        "process": device.process_index,
                        "kind": device.device_kind,
                        "coordinates": list(coordinates),
                    }
                    for coordinates, device in np.ndenumerate(mesh.devices)
                ],
            },
            "compute_mapping": dict(mapping),
            "parameter_mapping": dict(self.trainer.parameter_axis_mapping),
            "loss_mapping": loss_mapping,
            "mp": {
                name: str(getattr(self.trainer.mp, name)) for name in ("param_dtype", "compute_dtype", "output_dtype")
            },
            "field_dtype_overrides": {
                name: str(dtype) for name, dtype in self.trainer.config.field_dtype_overrides.items()
            },
            "compute_leaves": [
                (jax.tree_util.keystr(path), value.shape, str(value.dtype), str(value.sharding.spec))
                for path, value in jax.tree_util.tree_flatten_with_path(compute)[0]
                if isinstance(value, jax.ShapeDtypeStruct)
            ],
            "head": {"axes": [(axis.name, axis.size) for axis in head.axes], "dtype": str(head.array.dtype)},
            "activation_dtype": str(normalized.dtype),
            "arrays": {
                name: None if array is None else {"shape": array.shape, "dtype": str(array.dtype)}
                for name, array in sorted(arrays.items())
            },
            "response_length": response_length,
            "microbatch": microbatch,
            "optimizer_batch": batch_size,
            "accumulation_steps": batch_size // microbatch,
            "optimizer": self.settings["optimizer"],
            "optimizer_kind": "optax.chain(clip_by_global_norm,adamw)",
            "policy": {"eps_clip_low": self.settings["eps_clip_low"], "eps_clip_high": self.settings["eps_clip_high"]},
            "projection": {
                "block_size": runtime["projection_block_size"],
                "logits_dtype": "bfloat16",
                "dtype": "float32",
                "implementation": "xla",
                "fast_backward": False,
            },
        }
        profile_hash = hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest()
        return profile, FixedCEPlan(
            signatures, runtime["projection_block_size"], logits_dtype="bfloat16", execution_profile_sha256=profile_hash
        )

    def _prepare_projection(self, payload: dict, method: str) -> FixedCEPlan | None:
        profile, plan, error = {}, None, None
        command = {"method": method, "version": self.version, "global_step": payload.get("global_step")}
        try:
            command["optimizer_step"] = int(self.state.step)
            digest = hashlib.sha256()
            for name, array in sorted(payload["arrays"].items()):
                digest.update(name.encode())
                if array is not None:
                    digest.update(str((array.shape, array.dtype)).encode())
                    digest.update(np.asarray(array).tobytes())
            command["payload_sha256"] = digest.hexdigest()
            command["response_length"] = payload["response_length"]
            mode = "fixed-xla" if _projection_dtype(self.settings["runtime"]) == jnp.bfloat16 else "legacy"
            mode_envelope = {"ok": True, "mode": mode, "command": command}
            json.dumps(mode_envelope)
        except Exception as exc:
            mode_envelope = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        modes = multihost_allgather_sync(mode_envelope)
        if any(not peer.get("ok") for peer in modes):
            raise ValueError(f"Native projection command rejected before tracing: {modes}")
        if any(peer != modes[0] for peer in modes[1:]):
            raise ValueError("Native projection mode or command differs across JAX processes")
        if modes[0]["mode"] == "legacy":
            # FP32/default GPU projection keeps its existing kernel selection. The
            # generic unprepared tracing-coordination hazard is a separate gate.
            if method == "train" and payload["arrays"]["tokens"].shape[0] != self.settings["mini_batch_size"]:
                raise ValueError("Every native update requires a complete optimizer minibatch")
            return None
        try:
            profile, plan = self._projection_profile(payload)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        diagnostics = self.settings.get("runtime", {}).get("validate_checkpoint_roundtrip", False)
        if diagnostics:
            _diagnostic_receipt("projection_prepare_begin", method=method)
        prepared = prepare_fixed_ce_plan(
            plan,
            execution_profile=profile,
            command=command,
            process_ids=tuple(range(jax.process_count())),
            local_error=error,
        )
        if diagnostics:
            _diagnostic_receipt(
                "projection_prepare_end",
                method=method,
                profile_sha256=prepared.execution_profile_sha256,
                signatures=[dataclasses.asdict(signature) for signature in prepared.signatures],
            )
        return prepared

    def forward(self, payload: dict) -> dict:
        plan = self._prepare_projection(payload, "forward")
        assert self.learner is not None
        assert self.state is not None
        learner = dataclasses.replace(self.learner, projection_plan=plan)
        batch = self._batch(payload)
        scored = learner.score(self.state, batch)
        values = np.asarray(multihost_utils.process_allgather(scored.old_logprobs.array, tiled=True))
        return {"output": values if jax.process_index() == 0 else None, "version": self.version}

    def train(self, payload: dict) -> dict:
        plan = self._prepare_projection(payload, "train")
        assert self.learner is not None
        assert self.state is not None
        learner = dataclasses.replace(self.learner, projection_plan=plan)
        frozen_fingerprint = _payload_fingerprint(payload)
        diagnostics = self.settings["runtime"].get("validate_checkpoint_roundtrip", False)
        if diagnostics:
            _diagnostic_receipt(
                "train_received",
                tokens_shape=list(payload["arrays"]["tokens"].shape),
                frozen_batch_sha256=frozen_fingerprint,
                global_step=payload["global_step"],
            )
        batch = self._batch(payload)
        if diagnostics:
            _diagnostic_receipt("update_begin")
        if diagnostics and self._compiler_evidence is None:
            self._compiler_evidence = capture_compile(self.trainer, self.state, batch, plan, self.settings)
            # One fast rank must not trigger an execution OOM and tear down peers
            # while their pre-execution evidence is still being uploaded.
            multihost_allgather_sync({"compiler_evidence_ready": True})
        try:
            result = learner.update(self.state, batch)
        except BaseException as error:
            if self._compiler_evidence is not None:
                capture_failure(self._compiler_evidence, self.settings, error)
            raise
        if diagnostics:
            _diagnostic_receipt("update_end")
        self.state = result.state
        if not np.isfinite(result.loss):
            raise FloatingPointError("Native policy loss is nonfinite")
        if frozen_fingerprint != _payload_fingerprint(payload):
            raise RuntimeError("Policy update mutated frozen scores or objective weights")
        if _router_bias_fingerprint(self.state.model) != self.frozen_router_bias:
            raise RuntimeError("Policy update mutated frozen QB router biases")
        self.version = int(payload["global_step"])
        return {
            "policy_loss": float(result.loss),
            "policy_lr": self.settings["optimizer"]["lr"],
            "policy_update_steps": 1.0,
            "optimizer_step": int(self.state.step),
            "policy_version": self.version,
            "frozen_batch_sha256": frozen_fingerprint,
            "router_bias_unchanged": 1.0,
        }

    def checkpoint(self, method: str, payload: dict) -> dict:
        path = payload["ckpt_dir"]
        if method == "save_checkpoint":
            save_checkpoint(
                self.state,
                int(self.state.step),
                path,
                metadata={"policy_version": self.version, "source": self.source_identity},
            )
        else:
            metadata = load_checkpoint_metadata(path)
            if metadata["source"] != self.source_identity:
                raise ValueError("Policy checkpoint initializer identity differs from the active recipe")
            self.state = load_checkpoint(
                self.state,
                path,
                axis_mapping=self.trainer.parameter_axis_mapping,
                mesh=self.trainer.device_mesh,
            )
            self.version = metadata["policy_version"]
            if _router_bias_fingerprint(self.state.model) != self.frozen_router_bias:
                raise ValueError("Policy checkpoint changed the initializer's frozen QB router biases")
        receipt = {"step": int(self.state.step), "version": self.version, "checkpoint": path}
        if payload.get("include_fingerprints", False):
            receipt["state_fingerprints"] = (
                self.validate_state_finite(payload["finite_validation"])
                if "finite_validation" in payload
                else self.state_fingerprints()
            )
        return receipt

    def state_fingerprints(self) -> dict:
        return {
            "process_id": jax.process_index(),
            "components": {name: _state_fingerprint(getattr(self.state, name)) for name in _FINITE_COMPONENTS},
        }

    def validate_state_finite(self, request: dict) -> dict:
        """Collect local host counts once, then jointly reject bad or missing proof."""
        rank = self.settings["distributed"]["process_id"]
        local: dict = {"process_id": rank, "phase": request.get("phase"), "ok": False, "complete": False}
        fingerprints = None
        stage = "request"
        try:
            phase, expected_step, expected_version = (request[name] for name in ("phase", "step", "version"))
            valid = phase in ("initial", "restored") or (
                isinstance(phase, str) and phase.startswith("step") and phase[4:].isdecimal() and len(phase) < 32
            )
            if not valid or type(expected_step) is not int or type(expected_version) is not int:
                raise ValueError("Invalid finite-validation phase or step/version")
            if (
                (phase == "initial" and (expected_step, expected_version) != (0, 0))
                or (phase == "restored" and (expected_step, expected_version) != (1, 1))
                or (phase.startswith("step") and expected_step != int(phase[4:]))
            ):
                raise ValueError("Finite-validation phase does not match its requested boundary")
            stage = "fingerprint"
            fingerprints = self.state_fingerprints()
            actual_step = int(self.state.step)
            stage = "schema"
            if (actual_step, self.version) != (expected_step, expected_version):
                raise ValueError("Finite-validation state differs from the requested boundary")
            config = {
                key: value
                for key, value in self.settings.items()
                if key not in ("distributed", "compiler_dump_dir", "model")
            }
            config["distributed"] = {
                key: self.settings["distributed"][key] for key in ("num_processes", "local_device_count")
            }
            config["model_source"] = self.source_identity
            local.update(
                ok=True,
                complete=True,
                step=actual_step,
                version=self.version,
                model_source=self.source_identity,
                common_config_sha256=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                fingerprints=fingerprints,
                averaging_is_none=self.state.model_averaging is None,
                schema="local-unique-shards-finite-v1",
            )
            stage = "serialization"
            json.dumps(local, allow_nan=False)
        except Exception as error:
            local = {
                "process_id": rank,
                "phase": request.get("phase") if isinstance(request.get("phase"), str) else "invalid",
                "ok": False,
                "complete": False,
                "error_stage": stage,
                "error_type": type(error).__name__,
            }
        try:
            local["persistence"] = _persist_finite_proof(local, suffix="candidate")
        except Exception as error:
            local["persistence"] = {"persisted": False, "paths": [], "error_types": [type(error).__name__]}
        peers = multihost_allgather_sync(local)
        expected_ranks = list(range(self.settings["distributed"]["num_processes"]))
        kind = None
        if any(not peer["ok"] for peer in peers):
            kind = (
                "state_scan_failed"
                if any(peer.get("error_stage") == "fingerprint" for peer in peers)
                else "validation_mismatch"
            )
        elif sorted(peer["process_id"] for peer in peers) != expected_ranks:
            kind = "validation_mismatch"
        else:
            schemas = []
            for peer in peers:
                components = peer["fingerprints"]["components"]
                schemas.append(
                    (
                        peer["phase"],
                        peer["step"],
                        peer["version"],
                        peer["common_config_sha256"],
                        peer["averaging_is_none"],
                        {name: value["schema_sha256"] for name, value in components.items()},
                    )
                )
                if set(components) != set(_FINITE_COMPONENTS) or any(
                    value["arrays"] <= 0 or value["unique_local_shards"] <= 0 or value["elements"] <= 0
                    for name, value in components.items()
                    if name != "model_averaging" or not peer["averaging_is_none"]
                ):
                    kind = "validation_mismatch"
            if any(schema != schemas[0] for schema in schemas[1:]):
                kind = "validation_mismatch"
            if any(
                value["nonfinite_elements"] for peer in peers for value in peer["fingerprints"]["components"].values()
            ):
                kind = "nonfinite_state"
        if kind is None and any(not peer["persistence"]["persisted"] for peer in peers):
            kind = "proof_persistence_failed"
        verdict = {
            "status": "passed" if kind is None else "failed",
            "failure_kind": kind,
            "phase": local["phase"],
            "process_id": rank,
            "peers": peers,
            "coverage_scope": "unique addressable indices per process; replicas across processes may repeat",
        }
        try:
            _persist_finite_proof(verdict, suffix="verdict")
        except Exception:
            pass  # The gathered candidate persistence is the mandatory proof.
        if kind is not None:
            raise RuntimeError(f"Native finite validation failed: {kind}; phase={local['phase']}")
        assert fingerprints is not None
        return fingerprints

    def runtime_identity(self) -> dict:
        return {
            "compiler_environment": {
                "XLA_FLAGS": os.environ.get("XLA_FLAGS", ""),
                "normalized_xla_flags": scientific_xla_flags(self.settings),
                "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
            },
            "pid": os.getpid(),
            "jax_version": jax.__version__,
            "jaxlib_version": jaxlib.__version__,
            "process_id": jax.process_index(),
            "process_count": jax.process_count(),
            "backend": jax.default_backend(),
            "device_kinds": sorted({device.device_kind for device in jax.devices()}),
            "global_devices": jax.device_count(),
            "local_devices": jax.local_device_count(),
            "global_model_parameters": sum(
                math.prod(value.shape) for value in jax.tree.leaves(self.state.model) if eqx.is_array(value)
            ),
            "source": self.source_identity,
            "allocation_policy": _allocator_profile(),
            "allocator_limits": _allocator_limits(),
            "numpy_allocation_policy": self.numpy_allocation_policy,
        }

    def memory_statistics(self) -> dict:
        try:
            return {
                "pid": os.getpid(),
                "process_id": jax.process_index(),
                "backend": jax.default_backend(),
                "optimizer_step": int(self.state.step),
                "version": self.version,
                "source": self.source_identity,
                "allocation_policy": _allocator_profile(),
                "numpy_allocation_policy": self.numpy_allocation_policy,
                "statistics_api": "jax.Device.memory_stats",
                "devices": [row | {"statistics_available": row.get("stats") is not None} for row in _memory_stats()],
            }
        except Exception as error:
            return {"pid": os.getpid(), "observer_error_type": type(error).__name__}

    def validate_model_change(self, payload: dict) -> dict:
        before, after = payload["before"], payload["after"]
        changed = before["sha256"] != after["sha256"]
        same_layout = before["layout_sha256"] == after["layout_sha256"]
        flags = np.asarray(multihost_utils.process_allgather(np.array([changed, same_layout], dtype=np.int32))).reshape(
            -1, 2
        )
        if not np.all(flags[:, 1]):
            raise RuntimeError("Native model shard layout changed during the optimizer update")
        if not np.any(flags[:, 0]):
            raise RuntimeError("Native optimizer update did not change model parameters on any process")
        return {"local_model_changed": changed, "global_model_changed": True, "layout_unchanged": True}

    def publication(self, method: str, payload: dict):
        if method == "inventory":
            return self.publisher.inventory(self.version)
        if method == "begin":
            if payload["version"] != self.version:
                raise ValueError(f"Requested policy version {payload['version']} differs from learner {self.version}")
            return self.publisher.begin(self.trainer.mp.cast_to_compute(self.state.model), self.version)
        if method == "next":
            return self.publisher.next()
        if method == "end":
            return self.publisher.end()
        raise ValueError(f"Unknown publication operation {method}")

    def close(self) -> None:
        """Finish owned device work before acknowledging shutdown or returning on EOF."""
        if self._closed:
            return
        self._closed = True
        had_device_state = self.state is not None or self.trainer is not None
        problems = []
        # Haliax's named_jit caches own shardings and meshes independently of
        # JAX's caches; release them before destroying the backend client.
        for phase, operation in (
            ("synchronize", self._synchronize),
            ("trainer_exit", self._finish_trainer),
            ("release_owned_resources", self._release_owned_resources),
            ("clear_haliax_caches", clear_haliax_caches if had_device_state else lambda: None),
            ("clear_jax_caches", jax.clear_caches if had_device_state else lambda: None),
            ("collect_python_resources", gc.collect),
            ("release_backend_clients", self._release_backend_clients if had_device_state else lambda: None),
            ("distributed_shutdown", self._shutdown_distributed),
        ):
            try:
                with self._shutdown_phase(phase):
                    operation()
            except Exception as error:
                problems.append(error)
        if problems:
            raise ExceptionGroup("Native learner cleanup failed", problems)

    def _synchronize(self) -> None:
        # Rejected initialization must not turn cleanup into backend creation.
        if self.state is not None:
            jax.block_until_ready(self.state)
        if self.state is not None or self.trainer is not None:
            jax.effects_barrier()

    def _finish_trainer(self) -> None:
        if self.trainer is not None:
            self.trainer.__exit__(None, None, None)

    def _release_owned_resources(self) -> None:
        self.publisher = None
        self.learner = None
        self.state = None
        self.trainer = None
        self._compiler_evidence = None

    def _shutdown_distributed(self) -> None:
        if jax.distributed.is_initialized():
            jax.distributed.shutdown()

    def _release_backend_clients(self) -> None:
        # Pinned JAX0.11 exports this extension API. Release PJRT ownership while
        # distributed coordination is alive, rather than in late C finalization.
        clear_backends()
        gc.collect()

    def shutdown_receipt(self, event: str, **fields) -> None:
        marker_error = None
        if event == "shutdown_ack_written" and self._shutdown_observer_marker is not None:
            try:
                publish_ack_marker(
                    self._shutdown_observer_marker,
                    process_id=self.settings.get("distributed", {}).get("process_id"),
                )
            except Exception as error:
                # The binary ACK has already been flushed. A missing marker
                # fails observation, but must not change the original exit path.
                marker_error = type(error).__name__
                fields["observer_marker_error_type"] = marker_error
        if not self.settings.get("runtime", {}).get("validate_checkpoint_roundtrip", False) and marker_error is None:
            return
        receipt = {
            "event": event,
            "pid": os.getpid(),
            "process_id": self.settings.get("distributed", {}).get("process_id"),
            "thread_id": threading.get_ident(),
            "time_ns": time.time_ns(),
            "monotonic_ns": time.monotonic_ns(),
            **fields,
        }
        self._persist_shutdown_phase(receipt)
        if sys.stderr is None:
            return
        # No JAX calls here: this also runs after distributed shutdown and on
        # rejected initialization, when querying the backend would be unsafe.
        try:
            print(
                "native_shutdown_receipt " + json.dumps(receipt),
                file=sys.stderr,
                flush=True,
            )
        except Exception:
            # A broken diagnostic stream must not skip cleanup or mask its error.
            pass

    def _persist_shutdown_phase(self, receipt: dict) -> None:
        # These synchronous events precede the binary ACK. ACK/atexit/watchdog
        # events must never add filesystem work to the original exit window.
        event = receipt["event"]
        if (
            event not in _SHUTDOWN_PHASE_EVENTS
            or self._shutdown_phase_sink_failed
            or self._shutdown_phase_events >= _SHUTDOWN_PHASE_MAX_EVENTS
        ):
            return
        try:
            row = {
                "event": event,
                "pid": os.getpid(),
                "thread_id": threading.get_ident(),
                "time_ns": time.time_ns(),
                "monotonic_ns": time.monotonic_ns(),
            }
            rank = self.settings.get("distributed", {}).get("process_id")
            row["process_id"] = rank if type(rank) is int and rank >= 0 else None
            if event == "process_exit_begin":
                if receipt.get("reason") not in ("shutdown", "eof"):
                    return
                row["reason"] = receipt["reason"]
            else:
                if receipt.get("phase") not in _SHUTDOWN_PHASE_NAMES:
                    return
                row["phase"] = receipt["phase"]
                error_type = receipt.get("error_type")
                if isinstance(error_type, str) and len(error_type) <= 128 and error_type.isidentifier():
                    row["error_type"] = error_type
            encoded = (json.dumps(row, sort_keys=True) + "\n").encode()
            if self._shutdown_phase_bytes + len(encoded) > _SHUTDOWN_PHASE_MAX_BYTES:
                return
            directory = os.environ.get("SKYRL_DEBUG_ARTIFACT_DIR") or os.environ.get("IRIS_OUTPUT_DIR")
            if not directory:
                return
            path = Path(directory) / "native-finalization" / str(os.getpid()) / "shutdown-phases.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb" if self._shutdown_phase_events == 0 else "r+b") as output:
                if output.seek(0, os.SEEK_END) != self._shutdown_phase_bytes:
                    raise OSError("Shutdown phase artifact changed outside its owner")
                output.write(encoded)
            # Closing flushes each bounded record before proceeding. Avoid fsync
            # latency; the existing artifact collector handles delivery.
            self._shutdown_phase_events += 1
            self._shutdown_phase_bytes += len(encoded)
        except Exception:
            # Evidence loss is nonqualifying, but must not replace a cleanup
            # error, skip later phases, or change protocol/observer behavior.
            self._shutdown_phase_sink_failed = True

    @contextlib.contextmanager
    def _shutdown_phase(self, phase: str):
        self.shutdown_receipt("cleanup_phase_begin", phase=phase)
        try:
            yield
        except BaseException as error:
            self.shutdown_receipt("cleanup_phase_error", phase=phase, error_type=type(error).__name__)
            raise
        else:
            self.shutdown_receipt("cleanup_phase_end", phase=phase)

    def prepare_process_exit(self, reason: str) -> None:
        """Trace Python atexit stalls; CPython cancels this before late C teardown."""
        if not self.settings.get("runtime", {}).get("validate_checkpoint_roundtrip", False):
            return
        scope = "watched_debug_artifacts"
        destination = os.environ.get("SKYRL_DEBUG_ARTIFACT_DIR")
        if not destination:
            scope = "iris_output_fallback"
            destination = os.environ.get("IRIS_OUTPUT_DIR")
        if not destination:
            scope = "local_temporary_fallback"
            destination = tempfile.gettempdir()
        try:
            directory = os.path.join(destination, "native-finalization", str(os.getpid()))
            os.makedirs(directory, exist_ok=True)
            path = os.path.join(directory, "atexit-stack.txt")
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        except (OSError, ValueError) as error:
            path, scope = "stderr", "stderr_fallback"
            self.shutdown_receipt("finalization_trace_open_error", error_type=type(error).__name__)
            try:
                descriptor = sys.stderr.fileno()
            except Exception:
                # No usable diagnostic sink remains. Cleanup and the actual
                # process-exit contract still apply without an observer.
                return
        # Deliberately keep the low-level descriptor until real process exit.
        # A local file object would close before atexit. The native timer does
        # not keep a healthy process alive or change the parent's exit deadline.
        try:
            faulthandler.dump_traceback_later(_FINALIZATION_TRACE_SECONDS, file=descriptor, repeat=False, exit=False)
        except Exception as error:
            if path != "stderr":
                with contextlib.suppress(OSError):
                    os.close(descriptor)
            self.shutdown_receipt("finalization_trace_arm_error", error_type=type(error).__name__)
            return
        atexit.register(self.shutdown_receipt, "python_atexit_begin")
        self.shutdown_receipt(
            "process_exit_begin",
            reason=reason,
            trace_path=path,
            trace_scope=scope,
            trace_after_seconds=_FINALIZATION_TRACE_SECONDS,
            trace_threads="faulthandler maximum 100 threads, 100 frames per thread; no locals",
        )

    def dispatch(self, method: str, payload: dict):
        if method == "initialize":
            return self.initialize(payload)
        if method == "forward":
            return self.forward(payload)
        if method == "train":
            return self.train(payload)
        if method in ("save_checkpoint", "load_checkpoint"):
            return self.checkpoint(method, payload)
        if method == "state_fingerprints":
            return self.state_fingerprints()
        if method == "validate_state_finite":
            return self.validate_state_finite(payload)
        if method == "model_fingerprint":
            return _state_fingerprint(self.state.model)
        if method == "runtime_identity":
            return self.runtime_identity()
        if method == "memory_statistics":
            return self.memory_statistics()
        if method == "validate_model_change":
            return self.validate_model_change(payload)
        if method == "publication":
            return self.publication(payload["method"], payload["payload"])
        if method == "barrier":
            multihost_utils.sync_global_devices("skyrl-policy-barrier")
            return {"version": self.version}
        if method == "shutdown":
            if payload.get("diagnostic_shutdown_observer") is not None:
                if not self.settings.get("runtime", {}).get("diagnostic_shutdown_observer", False):
                    raise ValueError("Shutdown observer was not explicitly enabled")
                self._shutdown_observer_marker = validated_ack_marker(payload)
            self.close()
            return None
        raise ValueError(f"Unknown learner operation {method}")


def main(runtime_factory: Callable[[], NativePolicyRuntime] = NativePolicyRuntime) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response-fd", type=int, required=True)
    args = parser.parse_args()
    incoming = sys.stdin.buffer
    # stdout is logging from process start, including native/import-time writes.
    # Only framed responses use the explicitly inherited, separate descriptor.
    with os.fdopen(args.response_fd, "wb") as outgoing, contextlib.redirect_stdout(sys.stderr):
        os.set_inheritable(outgoing.fileno(), False)
        if sys.platform == "linux":
            parent = os.getppid()
            # Ray kills failed actors without running their cleanup hooks. Kernel
            # delivery prevents the learner from retaining GPUs after its owner dies.
            if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM) != 0:
                raise OSError(ctypes.get_errno(), "Could not bind learner lifetime to its Ray worker")
            if os.getppid() != parent:
                return
        runtime = runtime_factory()
        while True:
            try:
                request = read_frame(incoming)
            except EOFError:
                with runtime.request_diagnostics("eof"):
                    runtime.close()
                runtime.prepare_process_exit("eof")
                return
            try:
                with runtime.request_diagnostics(request["method"]):
                    result = runtime.dispatch(request["method"], request["payload"])
            except Exception as error:
                try:
                    runtime.close()
                except Exception as cleanup_error:
                    error.add_note(f"Additional learner cleanup failed: {cleanup_error!r}")
                write_frame(outgoing, {"result": None, "error": traceback.format_exc()})
                raise
            if request["method"] == "shutdown":
                runtime.prepare_process_exit("shutdown")
            write_frame(outgoing, {"result": result, "error": None})
            if request["method"] == "shutdown":
                runtime.shutdown_receipt("shutdown_ack_written")
                return


if __name__ == "__main__":
    main()

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Native Snowball policy learner, isolated from the SkyRL Torch runtime."""

import asyncio
import contextlib
import ctypes
import dataclasses
import hashlib
import json
import math
import os
import signal
import sys
import tempfile
import traceback
from functools import partial

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import optax
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding, PartitionSpec
from levanter.checkpoint import load_checkpoint, load_checkpoint_metadata, save_checkpoint
from levanter.compat.hf_checkpoints import RepoRef
from levanter.distributed import DistributedConfig
from levanter.models.snowball import SnowballConfig, SnowballLMHeadModel
from levanter.policy import ClippedPolicyConfig
from levanter.policy_learner import SpmdPolicyLearner
from levanter.policy_model import PolicyBatch, policy_model_loss
from levanter.tensorstore_serialization import _transfer_shard_to_pageable_host
from levanter.tracker.tracker import NoopConfig
from levanter.trainer import Trainer, TrainerConfig
from levanter.utils.mesh import MeshConfig

from marin.rl.levanter_publication import SnowballPublication
from marin.rl.levanter_worker_protocol import read_frame, write_frame


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
    for path, value in jax.tree_util.tree_flatten_with_path(tree)[0]:
        if not eqx.is_array(value):
            continue
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
            del array, snapshot
    return {
        "sha256": digest.hexdigest(),
        "layout_sha256": layout_digest.hexdigest(),
        "addressable_bytes": addressable_bytes,
    }


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
        self.trainer = None
        self.state = None
        self.learner = None
        self.publisher = None
        self.version = 0
        self.source_identity = None

    def initialize(self, config: dict) -> dict:
        if self.trainer is not None:
            raise RuntimeError("Learner is already initialized")
        self.settings = config
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

        return PolicyBatch(
            tokens=named("tokens", response=False),
            attention_mask=named("attention_mask", response=False),
            position_ids=named("position_ids", response=False),
            response_mask=named("response_mask", response=True),
            advantages=named("advantages", response=True),
            policy_weights=named("policy_weights", response=True),
            kl_weights=named("kl_weights", response=True),
            old_logprobs=named("old_logprobs", response=True),
            reference_logprobs=None,
        )

    def forward(self, payload: dict) -> dict:
        batch = self._batch(payload)
        scored = self.learner.score(self.state, batch)
        values = np.asarray(multihost_utils.process_allgather(scored.old_logprobs.array, tiled=True))
        return {"output": values if jax.process_index() == 0 else None, "version": self.version}

    def train(self, payload: dict) -> dict:
        if payload["arrays"]["tokens"].shape[0] != self.settings["mini_batch_size"]:
            raise ValueError("Every native update requires a complete optimizer minibatch")
        frozen_fingerprint = _payload_fingerprint(payload)
        result = self.learner.update(self.state, self._batch(payload))
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
            receipt["state_fingerprints"] = self.state_fingerprints()
        return receipt

    def state_fingerprints(self) -> dict:
        return {
            "process_id": jax.process_index(),
            "components": {
                name: _state_fingerprint(getattr(self.state, name))
                for name in ("model", "opt_state", "training_key", "step", "model_averaging")
            },
        }

    def runtime_identity(self) -> dict:
        return {
            "pid": os.getpid(),
            "process_id": jax.process_index(),
            "backend": jax.default_backend(),
            "device_kinds": sorted({device.device_kind for device in jax.devices()}),
            "global_devices": jax.device_count(),
            "local_devices": jax.local_device_count(),
            "global_model_parameters": sum(
                math.prod(value.shape) for value in jax.tree.leaves(self.state.model) if eqx.is_array(value)
            ),
            "source": self.source_identity,
        }

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
        if self.trainer is not None:
            self.trainer.__exit__(None, None, None)
            self.trainer = None
        if jax.distributed.is_initialized():
            jax.distributed.shutdown()

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
        if method == "model_fingerprint":
            return _state_fingerprint(self.state.model)
        if method == "runtime_identity":
            return self.runtime_identity()
        if method == "validate_model_change":
            return self.validate_model_change(payload)
        if method == "publication":
            return self.publication(payload["method"], payload["payload"])
        if method == "barrier":
            multihost_utils.sync_global_devices("skyrl-policy-barrier")
            return {"version": self.version}
        if method == "shutdown":
            self.close()
            return None
        raise ValueError(f"Unknown learner operation {method}")


def main() -> None:
    # Preserve a binary protocol stream while third-party logging/prints go to
    # the host worker's stderr. A failed operation terminates this learner.
    if sys.platform == "linux":
        parent = os.getppid()
        # Ray kills failed actors without running their cleanup hooks. Kernel
        # delivery prevents the learner from retaining GPUs after its owner dies.
        if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM) != 0:
            raise OSError(ctypes.get_errno(), "Could not bind learner lifetime to its Ray worker")
        if os.getppid() != parent:
            return
    incoming, outgoing = sys.stdin.buffer, sys.stdout.buffer
    with contextlib.redirect_stdout(sys.stderr):
        runtime = NativePolicyRuntime()
        while True:
            try:
                request = read_frame(incoming)
            except EOFError:
                runtime.close()
                return
            try:
                result = runtime.dispatch(request["method"], request["payload"])
            except Exception:
                write_frame(outgoing, {"result": None, "error": traceback.format_exc()})
                raise
            write_frame(outgoing, {"result": result, "error": None})
            if request["method"] == "shutdown":
                return


if __name__ == "__main__":
    main()

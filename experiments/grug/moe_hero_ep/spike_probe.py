# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Gradient probe for the hero-ctx16k grad spike at step 325710 (#10677).

Restores a hero checkpoint on one GB200 rack (EP64, the hero's per-rack layout) and computes
full-batch hero gradients as eleven 256-sequence rack slices. The hero's global batch of 2816
sequences is the concatenation of its eleven racks' slices (``replica_dcn`` is the outermost batch
axis), so slice ``r`` of step ``N`` is dataset rows ``N * 2816 + r * 256`` onward, the same rows
and device placement the hero's rack ``r`` trained on. The full-batch gradient is the
token-weighted sum of the slice gradients.

Nothing is trained and nothing is written next to the checkpoint. Results go to
``--output-path``. The job runs an initial command list, then polls ``<output>/cmd/`` for more
JSON commands until a ``quit`` command or ``--idle-timeout`` seconds without one. The ``exec``
payloads used for #10677 are in ``spike_probe_snippets/``; each expects its inputs (``STEP``,
``RACK``, ``SEQ``, ...) prepended as assignments.

Launch as an Iris coordinator job:

    iris --config lib/iris/config/marin.yaml job run --no-wait --enable-extra-resources \\
        --target-cluster cw-us-east-08a --priority production --cpu 2 --memory 8GB --disk 32GB \\
        --job-name hero-spike-probe-coord -- python -m experiments.grug.moe_hero_ep.spike_probe \\
        --run-id hero-spike-probe1
"""

import dataclasses
import functools
import io
import json
import logging
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import click
import draccus
import fsspec
import jax
import jax.numpy as jnp
import jmp
import numpy as np
from fray.cluster import ResourceConfig
from haliax.partitioning import set_mesh
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding, reshard
from jax.sharding import PartitionSpec as P
from levanter.callbacks.watch import compute_watch_stats
from levanter.checkpoint import load_checkpoint
from levanter.data.loader import DataLoader
from levanter.data.text.datasets import LmDataConfig
from levanter.grug.sharding import compact_grug_mesh
from levanter.optim.config import OptimizerConfig
from levanter.schedule import BatchSchedule
from levanter.store.jagged_array import set_jagged_array_read_cache_bytes
from levanter.tracker.telemetry import TelemetryConfig
from levanter.trainer import TrainerConfig
from rigging.filesystem.storage_path import StoragePath

from experiments.grug.checkpointing import restore_grug_state_from_checkpoint
from experiments.grug.dispatch import dispatch_grug_training_run
from experiments.grug.moe_hero_ep.hero_recipe import (
    HERO_EP_EXPERT_AXIS_SIZE,
    HERO_EP_NODES,
    HERO_GPUS_PER_NODE,
    HERO_MASTER_PARAM_MODE,
    HERO_MIXED_PRECISION_BY_MASTER_PARAM_MODE,
    HERO_MODEL_CONFIG,
    HERO_NODE_CPU,
    HERO_NODE_DISK,
    HERO_NODE_RAM,
    HERO_TENSORSTORE_CACHE_BYTES,
    with_transport_remat_mode,
)
from experiments.grug.moe_hero_ep.heuristic import MoeHeuristic
from experiments.grug.moe_hero_ep.model import GrugModelConfig, apply_qb_betas
from experiments.grug.moe_hero_ep.train import (
    _BATCH_AXES,
    GrugTrainState,
    _apply_hero_ep_runtime_defaults,
    _loss_and_grads,
    build_train_dataset,
    initial_state,
    restore_template_from,
    take_master_as_params,
    template_for_candidate_layout,
)

logger = logging.getLogger(__name__)

HERO_CHECKPOINT = "s3://marin-us-east-02a/marin/grug/hero-ctx16k-step257k/2026.08.19.2/checkpoints/step-324000"
SCRATCH_PREFIX = "s3://marin-us-east-08a/tmp/ttl="
OUTPUT_ROOT = f"{SCRATCH_PREFIX}30d/hero-spike-replay"
HERO_DATA_CONFIG_JSON = Path(__file__).with_name("spike_probe_hero_data_config.json")
HERO_SCHEDULE_STEPS = 390_251
HERO_GLOBAL_BATCH = 2816
RACK_BATCH = 256
HERO_RACKS = HERO_GLOBAL_BATCH // RACK_BATCH
HERO_SEED = 0
HERO_Z_LOSS_WEIGHT = 1e-4
HERO_GATE_ROUTER_WEIGHT_DECAY = 0.02
COMMAND_BYTES = 1 << 20
POLL_SECONDS = 15

DEFAULT_COMMANDS = (
    {"op": "full", "step": 324000, "seqstats": True},
    {"op": "full", "step": 325710, "seqstats": True},
    {"op": "bisect", "step": 325710, "slice": "auto"},
    {"op": "full", "step": 325700},
    {"op": "full", "step": 325720},
    {"op": "full", "step": 325940, "seqstats": True},
    {"op": "bisect", "step": 325940, "slice": "auto"},
    {"op": "bisect", "step": 324000, "slice": "auto"},
)


@dataclass(frozen=True)
class SpikeProbeConfig:
    run_id: str
    checkpoint_path: str
    output_path: str
    model: GrugModelConfig
    data: LmDataConfig
    optimizer: OptimizerConfig
    trainer: TrainerConfig
    initial_commands: tuple[str, ...]
    idle_timeout: float


def _to_float(x) -> float:
    return float(np.asarray(jax.device_get(x)))


def _gather(x) -> np.ndarray:
    return np.asarray(multihost_utils.process_allgather(x, tiled=True))


class Probe:
    """Holds the restored state, the data loader, and the compiled probe functions."""

    def __init__(self, config: SpikeProbeConfig, mesh):
        self.config = config
        self.mesh = mesh
        self.mp = config.trainer.mp
        optimizer = config.optimizer.build(HERO_SCHEDULE_STEPS)
        data_key, model_key = jax.random.split(jax.random.PRNGKey(HERO_SEED), 2)

        @jax.jit
        def _init_state(model_rng):
            return initial_state(
                config.model,
                optimizer=optimizer,
                mp=self.mp,
                key=model_rng,
                ema_beta=None,
                offload_opt_state=True,
                master_param_mode=HERO_MASTER_PARAM_MODE,
            )

        template = restore_template_from(_init_state(model_key))
        state = restore_grug_state_from_checkpoint(
            template,
            checkpoint_search_paths=[config.checkpoint_path],
            load_checkpoint_setting=True,
            mesh=mesh,
            allow_partial=False,
            template_for_candidate=lambda candidate: template_for_candidate_layout(
                template, candidate, HERO_MASTER_PARAM_MODE
            ),
        )
        state = take_master_as_params(state)
        self.restored_step = int(state.step)
        logger.info("Restored step %d from %s", self.restored_step, config.checkpoint_path)
        self.params = state.params
        self.pending_qb_betas = state.pending_qb_betas
        self.opt_state = state.opt_state
        del state

        dataset = build_train_dataset(
            config.data,
            max_seq_len=config.model.max_seq_len,
            batch_schedule=BatchSchedule(HERO_GLOBAL_BATCH),
            key=data_key,
        )
        # A 256-row loader over the hero's dataset: its batch k is rows k*256.., so batch
        # step*11 + r is the hero's rack r slice of step `step`.
        self.loader = DataLoader(
            dataset,
            RACK_BATCH,
            max_buffered_batches=0,
            mesh=mesh,
            axis_resources={"__BATCH__": _BATCH_AXES},
            fetch_batch_size=4,
            batch_axis_name="__BATCH__",
            allow_nondivisible_batch_size=False,
        )
        self.seq_sharding = NamedSharding(mesh, P(_BATCH_AXES))
        z_loss = HERO_Z_LOSS_WEIGHT
        mp = self.mp

        @jax.jit
        def grad_fn(params, pending, batch, seq_mask):
            qb_params = apply_qb_betas(params, pending)
            weight = batch.loss_weight * seq_mask[:, None].astype(batch.loss_weight.dtype)
            batch = dataclasses.replace(batch, loss_weight=weight)
            (loss, metrics), grads = _loss_and_grads(qb_params, batch, mp, z_loss)
            return loss, jnp.sum(weight), metrics, grads

        @functools.partial(jax.jit, donate_argnums=(0,))
        def accumulate(acc, grads, weight):
            return jax.tree.map(lambda a, g: a + g * weight, acc, grads)

        @jax.jit
        def scale(grads, weight):
            return jax.tree.map(lambda g: g * weight, grads)

        @jax.jit
        def norms_fn(grads, weight):
            scaled = jax.tree.map(lambda g: g * weight, grads)
            return compute_watch_stats(
                watch_targets=["grads"],
                include_norms=True,
                include_per_parameter_norms=True,
                include_histogram=False,
                split_scan_layers=True,
                grads=scaled,
            )

        @jax.jit
        def param_norms_fn(params, pending):
            return compute_watch_stats(
                watch_targets=["params"],
                include_norms=True,
                include_per_parameter_norms=True,
                include_histogram=False,
                split_scan_layers=True,
                params=apply_qb_betas(params, pending),
            )

        @jax.jit
        def token_loss_fn(params, pending, batch):
            compute_params = mp.cast_to_compute(apply_qb_betas(params, pending))
            return compute_params.next_token_loss(
                batch.tokens, batch.loss_weight, mask=batch.attn_mask, reduction="none", logsumexp_weight=None
            )

        self.grad_fn = grad_fn
        self.accumulate = accumulate
        self.scale = scale
        self.norms_fn = norms_fn
        self.param_norms_fn = param_norms_fn
        self.token_loss_fn = token_loss_fn
        self.full_results: dict[int, dict] = {}
        self._batch_cache: dict[tuple[int, int], object] = {}

    # ---- data ----
    def batch(self, step: int, rack: int):
        key = (step, rack)
        if key not in self._batch_cache:
            if len(self._batch_cache) >= 4:
                self._batch_cache.pop(next(iter(self._batch_cache)))
            iterator = self.loader.iter_from_step(step * HERO_RACKS + rack)
            self._batch_cache[key] = next(iterator)
        return self._batch_cache[key]

    def seq_mask(self, indices=None):
        mask = np.zeros((RACK_BATCH,), dtype=np.float32)
        if indices is None:
            mask[:] = 1
        else:
            mask[np.asarray(indices, dtype=np.int64)] = 1
        return jax.make_array_from_callback(mask.shape, self.seq_sharding, lambda index: mask[index])

    # ---- primitives ----
    def slice_grad(self, step: int, rack: int, indices=None):
        loss, denom, metrics, grads = self.grad_fn(
            self.params, self.pending_qb_betas, self.batch(step, rack), self.seq_mask(indices)
        )
        return _to_float(loss), _to_float(denom), metrics, grads

    def norms(self, grads, weight: float) -> dict[str, float]:
        stats = self.norms_fn(grads, jnp.float32(weight))
        return {k: _to_float(v) for k, v in stats.items()}

    def seq_stats(self, step: int, rack: int) -> dict:
        batch = self.batch(step, rack)
        token_loss = _gather(self.token_loss_fn(self.params, self.pending_qb_betas, batch))
        weight = _gather(batch.loss_weight)
        tokens = _gather(batch.tokens)
        seq_loss_sum = (token_loss * weight).sum(axis=1)
        seq_tokens = weight.sum(axis=1)
        return {
            "seq_mean_loss": (seq_loss_sum / np.maximum(seq_tokens, 1)).tolist(),
            "seq_max_token_loss": (token_loss * (weight > 0)).max(axis=1).tolist(),
            "seq_tokens": seq_tokens.tolist(),
            "seq_unique_tokens": [len(np.unique(row)) for row in tokens],
            "seq_top_token_frac": [float(np.bincount(row).max() / len(row)) for row in tokens],
        }

    def dump(self, step: int, rack: int) -> dict:
        batch = self.batch(step, rack)
        token_loss = _gather(self.token_loss_fn(self.params, self.pending_qb_betas, batch))
        arrays = {
            "tokens": _gather(batch.tokens),
            "loss_weight": _gather(batch.loss_weight),
            "token_loss": token_loss,
        }
        if batch.attn_mask.segment_ids is not None:
            arrays["segment_ids"] = _gather(batch.attn_mask.segment_ids[0])
        return arrays


def _shape_template(tree):
    return jax.tree.map(
        lambda leaf: (
            jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=leaf.sharding) if isinstance(leaf, jax.Array) else leaf
        ),
        tree,
    )


def _delete_arrays(tree) -> None:
    jax.tree.map(lambda leaf: leaf.delete() if isinstance(leaf, jax.Array) else None, tree)


def _layer_summary(norms: dict[str, float]) -> dict[str, float]:
    per_layer: dict[str, float] = {}
    for key, value in norms.items():
        parts = key.split("/")[-1].split(".")
        if len(parts) > 2 and parts[0] == "stacked_blocks" and parts[1].isdigit():
            name = f"layer{int(parts[1]):02d}"
        elif "stacked_blocks.stacked" in key or key.endswith("/total"):
            continue
        else:
            name = parts[0]
        per_layer[name] = per_layer.get(name, 0.0) + value * value
    return {k: v**0.5 for k, v in sorted(per_layer.items())}


class Runner:
    def __init__(self, probe: Probe, output_path: str):
        self.probe = probe
        self.output = output_path.rstrip("/")
        self.is_writer = jax.process_index() == 0

    def write_json(self, name: str, payload: dict) -> None:
        if self.is_writer:
            StoragePath(f"{self.output}/results/{name}.json").write_text(json.dumps(payload, indent=1))
            logger.info("RESULT %s %s", name, json.dumps(_short(payload))[:2000])

    def write_npz(self, name: str, arrays: dict[str, np.ndarray]) -> None:
        if self.is_writer:
            buffer = io.BytesIO()
            np.savez_compressed(buffer, **arrays)
            with StoragePath(f"{self.output}/results/{name}.npz").open("wb") as f:
                f.write(buffer.getvalue())

    # ---- ops ----
    def op_full(self, cmd: dict) -> dict:
        probe = self.probe
        step = int(cmd["step"])
        slices = []
        acc = None
        total_denom = 0.0
        loss_sum = 0.0
        for rack in range(HERO_RACKS):
            started = time.time()
            loss, denom, metrics, grads = probe.slice_grad(step, rack)
            slice_norms = probe.norms(grads, 1.0)
            entry = {
                "rack": rack,
                "loss": loss,
                "denom": denom,
                "grad_total": slice_norms["grad/norm/total"],
                "layers": _layer_summary(slice_norms),
                "z_loss_logging_only": _to_float(metrics["train/router/z_loss_logging_only"]),
                "seconds": time.time() - started,
            }
            if cmd.get("seqstats"):
                entry.update(probe.seq_stats(step, rack))
            slices.append(entry)
            logger.info("full step=%d rack=%d loss=%.6f grad=%.4f", step, rack, loss, entry["grad_total"])
            acc = (
                probe.scale(grads, jnp.float32(denom))
                if acc is None
                else probe.accumulate(acc, grads, jnp.float32(denom))
            )
            del grads
            total_denom += denom
            loss_sum += loss * denom
        full_norms = probe.norms(acc, 1.0 / total_denom)
        del acc
        for entry in slices:
            # Each slice's share of the full mean gradient.
            entry["contribution"] = entry["grad_total"] * entry["denom"] / total_denom
        result = {
            "step": step,
            "restored_step": probe.restored_step,
            "loss": loss_sum / total_denom,
            "grad_norm_total": full_norms["grad/norm/total"],
            "grad_norms": full_norms,
            "slices": slices,
        }
        if cmd.get("params"):
            result["param_norms"] = {
                k: _to_float(v) for k, v in probe.param_norms_fn(probe.params, probe.pending_qb_betas).items()
            }
        probe.full_results[step] = result
        return result

    def _resolve_slice(self, cmd: dict) -> int:
        if cmd.get("slice", "auto") != "auto":
            return int(cmd["slice"])
        step = int(cmd["step"])
        if step not in self.probe.full_results:
            self.op_full({"op": "full", "step": step})
        slices = self.probe.full_results[step]["slices"]
        return int(max(slices, key=lambda s: s["grad_total"])["rack"])

    def _group_entry(self, step: int, rack: int, indices, slice_denom: float) -> dict:
        loss, denom, _, grads = self.probe.slice_grad(step, rack, indices)
        norms = self.probe.norms(grads, 1.0)
        del grads
        return {
            "indices": [int(i) for i in indices],
            "loss": loss,
            "denom": denom,
            "grad_total": norms["grad/norm/total"],
            "contribution": norms["grad/norm/total"] * denom / max(slice_denom, 1.0),
            "layers": _layer_summary(norms),
        }

    def op_groups(self, cmd: dict) -> dict:
        step = int(cmd["step"])
        rack = self._resolve_slice(cmd)
        _, slice_denom, _, grads = self.probe.slice_grad(step, rack)
        del grads
        groups = [self._group_entry(step, rack, g, slice_denom) for g in cmd["groups"]]
        return {"step": step, "rack": rack, "slice_denom": slice_denom, "groups": groups}

    def op_bisect(self, cmd: dict) -> dict:
        step = int(cmd["step"])
        rack = self._resolve_slice(cmd)
        group_size = int(cmd.get("group_size", 16))
        _, slice_denom, _, grads = self.probe.slice_grad(step, rack)
        slice_norm = self.probe.norms(grads, 1.0)["grad/norm/total"]
        del grads
        coarse = [
            self._group_entry(step, rack, list(range(start, start + group_size)), slice_denom)
            for start in range(0, RACK_BATCH, group_size)
        ]
        top = max(coarse, key=lambda g: g["contribution"])
        singles = [self._group_entry(step, rack, [i], slice_denom) for i in top["indices"]]
        top_single = max(singles, key=lambda g: g["contribution"])
        without = [i for i in range(RACK_BATCH) if i != top_single["indices"][0]]
        leave_out = self._group_entry(step, rack, without, slice_denom)
        self.write_npz(f"dump-step{step}-rack{rack}", self.probe.dump(step, rack))
        return {
            "step": step,
            "rack": rack,
            "slice_denom": slice_denom,
            "slice_grad_total": slice_norm,
            "coarse": coarse,
            "singles": singles,
            "top_sequence": top_single["indices"][0],
            "leave_top_out": leave_out,
        }

    def op_load_weights(self, cmd: dict) -> dict:
        """Replace the probe's params and pending QB betas with another checkpoint's (read only)."""
        probe = self.probe
        path = cmd["checkpoint"]
        # The checkpoint is one OCDBT store keyed by train-state field, so read it with a train-state
        # template that names only the leaves we want; None fields are skipped.
        template = GrugTrainState(
            step=jax.ShapeDtypeStruct((), jnp.int32, sharding=NamedSharding(probe.mesh, P())),
            params=_shape_template(probe.params),
            master_params=None,
            opt_state=None,
            ema_params=None,
            pending_qb_betas=_shape_template(probe.pending_qb_betas),
        )
        # Keep the current weights alive until the new ones are read, so a missing checkpoint
        # (temp checkpoints rotate hourly) leaves the probe usable.
        loaded = load_checkpoint(template, path, mesh=probe.mesh)
        _delete_arrays((probe.params, probe.pending_qb_betas))
        probe.params = loaded.params
        probe.pending_qb_betas = loaded.pending_qb_betas
        probe.restored_step = int(loaded.step)
        probe.full_results.clear()
        return {"checkpoint": path, "restored_step": probe.restored_step}

    def op_swap(self, cmd: dict) -> dict:
        """Slice gradient of (step, slice) with row ``seq`` replaced by row ``src_seq`` of (src_step, src_slice).

        ``src_step`` of None leaves the batch unchanged and masks row ``seq`` out of the loss instead.
        """
        probe = self.probe
        step, rack, seq = int(cmd["step"]), int(cmd["slice"]), int(cmd["seq"])
        target = probe.batch(step, rack)
        _, slice_denom, _, grads = probe.slice_grad(step, rack)
        base = probe.norms(grads, 1.0)
        del grads
        out = {"step": step, "rack": rack, "seq": seq, "base_grad_total": base["grad/norm/total"]}
        if cmd.get("src_step") is None:
            indices = [i for i in range(RACK_BATCH) if i != seq]
            out["removed"] = self._group_entry(step, rack, indices, slice_denom)
            return out
        source = probe.batch(int(cmd["src_step"]), int(cmd["src_slice"]))
        src_seq = int(cmd["src_seq"])
        rows = jnp.arange(RACK_BATCH)

        @jax.jit
        def splice(dst, src):
            def leaf(x, y):
                # The batch axis is sharded; replicate before taking one row.
                row = reshard(y, P())[src_seq]
                mask = (rows == seq).reshape((RACK_BATCH,) + (1,) * (x.ndim - 1))
                return jnp.where(mask, row[None], x)

            return jax.tree.map(leaf, dst, src)

        key = (step, rack)
        probe._batch_cache[key] = splice(target, source)
        try:
            out["swapped"] = self._group_entry(step, rack, list(range(RACK_BATCH)), slice_denom)
            out["swapped_only"] = self._group_entry(step, rack, [seq], slice_denom)
        finally:
            probe._batch_cache[key] = target
        return out

    def op_dump(self, cmd: dict) -> dict:
        step, rack = int(cmd["step"]), int(cmd["slice"])
        self.write_npz(f"dump-step{step}-rack{rack}", self.probe.dump(step, rack))
        return {"step": step, "rack": rack}

    def op_exec(self, cmd: dict) -> dict:
        namespace = {
            "probe": self.probe,
            "runner": self,
            "jax": jax,
            "jnp": jnp,
            "np": np,
            "P": P,
            "result": {},
        }
        exec(cmd["code"], namespace)
        return namespace["result"]

    def run(self, cmd: dict, name: str) -> None:
        op = cmd["op"]
        started = time.time()
        logger.info("COMMAND %s %s", name, json.dumps(cmd)[:500])
        try:
            result = getattr(self, f"op_{op}")(cmd)
        except Exception:
            # One failed probe command must not discard the restored state; record it and continue.
            logger.exception("command %s failed", name)
            result = {"error": traceback.format_exc()}
        result["command"] = cmd
        result["seconds"] = time.time() - started
        self.write_json(name, result)

    # ---- command queue ----
    def next_command(self, done: set[str]) -> tuple[str, dict] | None:
        payload = np.zeros((COMMAND_BYTES,), dtype=np.uint8)
        if self.is_writer:
            cmd_dir = f"{self.output}/cmd"
            fs, _ = fsspec.core.url_to_fs(cmd_dir)
            fs.invalidate_cache()
            names = sorted(p.name for p in StoragePath(f"{cmd_dir}/*.json").glob())
            pending = [n for n in names if n not in done]
            if pending:
                name = pending[0]
                text = StoragePath(f"{cmd_dir}/{name}").read_text()
                encoded = json.dumps({"name": name, "cmd": json.loads(text)}).encode()
                payload[: len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)
        payload = np.asarray(multihost_utils.broadcast_one_to_all(payload))
        raw = payload.tobytes().rstrip(b"\x00")
        if not raw:
            return None
        message = json.loads(raw)
        return message["name"], message["cmd"]


def _short(payload: dict) -> dict:
    return {k: v for k, v in payload.items() if k not in ("grad_norms", "param_norms", "slices", "coarse", "singles")}


def _run_probe_local(config: SpikeProbeConfig) -> None:
    set_jagged_array_read_cache_bytes(HERO_TENSORSTORE_CACHE_BYTES)
    config.trainer.initialize()
    mesh = compact_grug_mesh(
        expert_axis_size=HERO_EP_EXPERT_AXIS_SIZE,
        replica_axis_size=1,
        context_axis_size=1,
    )
    with set_mesh(mesh):
        probe = Probe(config, mesh)
        runner = Runner(probe, config.output_path)
        for i, text in enumerate(config.initial_commands):
            cmd = json.loads(text)
            runner.run(cmd, f"init{i:02d}-{cmd['op']}-{cmd.get('step', '')}")
        done: set[str] = set()
        last_command = time.time()
        while time.time() - last_command < config.idle_timeout:
            message = runner.next_command(done)
            if message is None:
                time.sleep(POLL_SECONDS)
                continue
            name, cmd = message
            done.add(name)
            last_command = time.time()
            if cmd["op"] == "quit":
                break
            runner.run(cmd, name.removesuffix(".json"))
    logger.info("spike probe finished")


@click.command()
@click.option("--run-id", required=True, help="Probe run id; results go under the output root.")
@click.option("--checkpoint", default=HERO_CHECKPOINT, show_default=True, help="Hero checkpoint to read.")
@click.option("--commands", default=None, help="JSON list of initial commands; default runs the spike plan.")
@click.option("--output-root", default=OUTPUT_ROOT, show_default=True, help="Scratch prefix for results.")
@click.option("--idle-timeout", default=3 * 3600.0, show_default=True, help="Seconds to wait for queued commands.")
def main(run_id: str, checkpoint: str, commands: str | None, output_root: str, idle_timeout: float) -> None:
    output_path = f"{output_root}/{run_id}"
    if checkpoint.startswith(output_path) or not output_root.startswith(SCRATCH_PREFIX):
        raise ValueError("output path must be our own scratch prefix")
    model = with_transport_remat_mode(HERO_MODEL_CONFIG)
    data = draccus.decode(LmDataConfig, json.loads(HERO_DATA_CONFIG_JSON.read_text()))
    optimizer = MoeHeuristic().build_optimizer_config(
        num_train_steps=HERO_SCHEDULE_STEPS,
        batch_size=HERO_GLOBAL_BATCH,
        hidden_dim=model.hidden_dim,
        seq_len=model.max_seq_len,
    )
    optimizer = dataclasses.replace(optimizer, gate_router_weight_decay=HERO_GATE_ROUTER_WEIGHT_DECAY)
    initial = json.loads(commands) if commands is not None else list(DEFAULT_COMMANDS)
    config = SpikeProbeConfig(
        run_id=run_id,
        checkpoint_path=checkpoint,
        output_path=output_path,
        model=model,
        data=data,
        optimizer=optimizer,
        trainer=TrainerConfig(
            id=run_id,
            seed=HERO_SEED,
            train_batch_size=RACK_BATCH,
            num_train_steps=HERO_SCHEDULE_STEPS,
            mp=jmp.get_policy(HERO_MIXED_PRECISION_BY_MASTER_PARAM_MODE[HERO_MASTER_PARAM_MODE]),
            tracker=TelemetryConfig(),
            use_explicit_mesh_axes=True,
            require_accelerator=True,
        ),
        initial_commands=tuple(json.dumps(c) for c in initial),
        idle_timeout=idle_timeout,
    )
    logger.info("spike probe output: %s (checkpoint %s, read only)", output_path, checkpoint)
    _apply_hero_ep_runtime_defaults(
        inline_watch_enabled=True,
        moe_implementation=model.moe_implementation,
        remat_mode=model.remat_mode,
        processes_per_task=HERO_GPUS_PER_NODE,
    )
    dispatch_grug_training_run(
        run_id=run_id,
        config=config,
        local_entrypoint=_run_probe_local,
        resources=ResourceConfig.with_gpu(
            "GB200",
            count=HERO_GPUS_PER_NODE,
            cpu=HERO_NODE_CPU,
            ram=HERO_NODE_RAM,
            disk=HERO_NODE_DISK,
            replicas=HERO_EP_NODES,
        ),
        processes_per_task=HERO_GPUS_PER_NODE,
        max_retries_failure=3,
        max_task_failures=3,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

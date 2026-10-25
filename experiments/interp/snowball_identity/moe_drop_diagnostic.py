# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""#310 diagnostic after the lr0-r5 gate failure: where the Levanter Snowball forward departs from the torch reference.

One H100x8 job on cw-rno2a; nothing is trained, saved as a model or exported. Three arms of the same compute view
(the gate's: BF16 compute, FP32 router bias, FP32 router dot), each differing from (a) in exactly one thing:

(a) ``ep4-cap1``: data 2 x expert 4, every block's ``capacity_factor`` 1.0, as loaded. The r5 gate's MoE path
    (ring, expert parallel 4, silent drops), without the cross-host context axis.
(b) ``ep4-dropless``: the same mesh and model with every block's ``mlp.expert_mlp.capacity_factor`` set to the
    expert-parallel size (4.0), so the ring path's per-shard capacity covers every assignment. Dropless by
    construction.
(c) ``ep1``: data 8 x expert 1. Grug's MoE falls back to the local scatter path. Dropless by construction.

Per arm, under ``<root>/<arm>/``:

- ``positions.pt`` and ``gate.json``: the 465 pinned sequences scored by ``levanter_positions.positions_records`` and
  judged by ``precision_gate.evaluate``, the gate's own code and thresholds (batch 8, pad multiple 64).
- per-device memory after the arm (inference only; it does not measure training at 8 x 8192).
- the drop count per layer, recomputed on the host from that arm's observed routes with the ring path's capacity
  rule (``ring_drops``), over the same batches including their padding.
- sublayer dumps of layers 0-2 for the 11 ``check`` sequences (one batch of 16, padded to 64): embed, attn-norm,
  attn-out, mlp-norm, router logits, moe-out, shared-out, resid; compared with the torch reference at every real
  position.

The torch reference is the gate's: the vendored torch GrugMoE (``harness.import_grug``) with eager attention and
eager experts, on the same checkpoint. A subprocess loads only the embedding and layers 0-2 on one GPU, before JAX
starts, and writes ``<root>/torch-dumps.pt``; its routes must equal the gate's torch ``routes.pt`` for those layers.
For each check sequence the receipt also holds the layer-0 router logits at position 4 from both stacks, with the
top-5 biased logits and the 4th/5th margin.

Everything lands in ``<root>/receipt.json``. Production code is imported, never changed; the arm (b) change is made
on the loaded model object.
"""

import argparse
import dataclasses
import gc
import io
import json
import logging
import math
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import asdict, replace
from pathlib import Path

import haliax as hax
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import torch
from fray.iris_backend import convert_constraints, convert_resources
from fray.types import ResourceConfig
from haliax import Axis
from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.cluster.types import Entrypoint, EnvironmentSpec
from jax.sharding import PartitionSpec as P
from jax.sharding import reshard
from levanter.compute_dtype import cast_model_for_compute
from levanter.grug.attention import AttentionMask
from levanter.main.train_lm import preserve_hf_runtime_settings
from levanter.models.snowball import _BATCH_AXES, GrugMoeHfConfig, _activation_spec, _context_axis, _token_spec
from levanter.utils.mesh import MeshConfig
from marin.merging.checkpoint import CheckpointReader, CheckpointSource
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.secrets import SecretBehavior
from rigging.timing import Duration

from experiments.interp.snowball_identity.harness import import_grug
from experiments.interp.snowball_identity.levanter_positions import (
    PAD_ID,
    positions_records,
    router_dot_dtypes,
    save_positions,
)
from experiments.interp.snowball_identity.logit_check import read, stored_sequences
from experiments.interp.snowball_identity.multihost import gather_global
from experiments.interp.snowball_identity.precision_gate import evaluate, write_json
from experiments.interp.snowball_identity.sequences import Sequence
from experiments.post_training.identity_context_distill.lr0_probe import (
    PRIORITY_BANDS,
    approved_receipt,
    base_config,
    check_pod_storage,
    gate_inputs,
)
from experiments.post_training.identity_context_distill.runtime_recipe import configure_precision_runtime
from experiments.post_training.identity_context_distill.sft import BASE_MODEL

logger = logging.getLogger(__name__)

T310 = "s3://marin-rno2a/marin/experiments/snowball-identity/t310"
DUMP_LAYERS = 3
LAYERS = 26
BATCH = 8
PAD_MULTIPLE = 64
DUMP_ROWS = 16
MARGIN_POSITION = 4
OPS = ("attn_norm", "attn_out", "mlp_norm", "router_logits", "moe_out", "shared_out", "resid")
ARMS = {
    "ep4-cap1": {"expert": 4, "capacity_factor": None, "dropless": "no: capacity_factor 1.0 as loaded"},
    "ep4-dropless": {"expert": 4, "capacity_factor": 4.0, "dropless": "by construction: capacity = all assignments"},
    "ep1": {"expert": 1, "capacity_factor": None, "dropless": "by construction: local scatter path, no expert axis"},
}
"""Each arm differs from ep4-cap1 in exactly one thing: the capacity factor (ep4-dropless) or the mesh (ep1)."""
POLICY = "p=f32,c=bfloat16,o=f32"
MODULE = "experiments.interp.snowball_identity.moe_drop_diagnostic"
INPUTS = gate_inputs("unused", f"{T310}/torch-ref-queue-r1/results/t310-router-rounding-r1/routing.json")
"""The r5 gate's inputs; each arm writes under its own ``output``."""


# ---- host-side capacity rule -------------------------------------------------------------------------------------


def ring_drops(routes: np.ndarray, num_experts: int, expert_axis: int, capacity_factor: float) -> int:
    """Assignments the ring path drops for one layer's routes [B, S, K] of a batch laid out on Snowball's batch axes.

    Mirrors ``ep_ring._ring_ep`` for a mesh of ``(replica_dcn, data, expert)`` batch axes with no context axis:
    rows are split over the devices in order, the expert axis is minor, and each expert shard gathers the tokens of
    its ``expert_axis`` group. A shard accepts at most ``clip(ceil(cf * A / ep), local_experts, physical)`` of the
    ``A`` gathered assignments, with ``physical = min(A, max(local_experts, ceil(cf * A / ep)))``; every token is
    valid (Snowball passes no ``token_valid``). An expert axis of 1 runs the local path, which never drops."""
    if expert_axis == 1:
        return 0
    batch = routes.shape[0]
    if batch % BATCH:
        raise ValueError(f"batch {batch} does not split over {BATCH} devices")
    per_device = batch // BATCH
    local_experts = num_experts // expert_axis
    dropped = 0
    for group in range(BATCH // expert_axis):
        rows = slice(group * expert_axis * per_device, (group + 1) * expert_axis * per_device)
        experts = routes[rows].reshape(-1)
        assignments = experts.size
        physical = min(assignments, max(local_experts, math.ceil(capacity_factor * assignments / expert_axis)))
        logical = min(max(math.ceil(assignments * capacity_factor / expert_axis), local_experts), physical)
        counts = np.bincount(experts // local_experts, minlength=expert_axis)
        dropped += int(np.maximum(counts - logical, 0).sum())
    return dropped


def gate_batches(sequences: list[Sequence], batch: int, pad_multiple: int) -> Iterator[np.ndarray]:
    """The token grids ``positions_records`` scores, in its order: sorted by length, short steps filled with copies
    of their first sequence, right-padded with ``PAD_ID``."""
    order = sorted(range(len(sequences)), key=lambda i: len(sequences[i].tokens))
    for step in range(0, len(order), batch):
        rows = [sequences[i].tokens for i in order[step : step + batch]]
        rows += [rows[0]] * (batch - len(rows))
        yield padded(rows, pad_multiple)


def padded(rows: list[tuple[int, ...]], pad_multiple: int) -> np.ndarray:
    length = -(-max(len(r) for r in rows) // pad_multiple) * pad_multiple
    grid = np.full((len(rows), length), PAD_ID, dtype=np.int32)
    for row, tokens in enumerate(rows):
        grid[row, : len(tokens)] = tokens
    return grid


# ---- comparison --------------------------------------------------------------------------------------------------


def top_margins(router_logits: np.ndarray, bias: np.ndarray, k: int = 4) -> dict:
    """The top-(k+1) biased router logits at one position and the margin that decides the k-th expert."""
    biased = router_logits.astype(np.float64) + bias.astype(np.float64)
    order = np.argsort(-biased, kind="stable")[: k + 1]
    return {
        "ids": order.tolist(),
        "biased": biased[order].tolist(),
        "margin_k_vs_next": float(biased[order[k - 1]] - biased[order[k]]),
    }


def op_diff(ours: np.ndarray, ref: np.ndarray) -> dict:
    """Max and RMS absolute difference, the reference's max magnitude, and the max difference in units of one BF16
    rounding step at that magnitude (2^-8 x max|ref|)."""
    ours, ref = ours.astype(np.float64), ref.astype(np.float64)
    diff = np.abs(ours - ref)
    scale = float(np.abs(ref).max())
    return {
        "max_abs": float(diff.max()),
        "rms": float(np.sqrt(np.mean(diff**2))),
        "ref_max_abs": scale,
        "max_in_bf16_steps": float(diff.max() / (scale * 2.0**-8)) if scale else float("inf"),
    }


def compare_dumps(levanter: dict, torch_dumps: dict, bias: dict[int, np.ndarray]) -> dict:
    """Per layer and op, over every real position of every check sequence: the Levanter vs torch difference; the
    route agreement for layers 0-2; and the layer-0 router logits at ``MARGIN_POSITION`` from both stacks."""
    ids = [s for s in torch_dumps if not s.startswith("_")]
    table: dict = {}
    routes = {}
    margins = {}
    for layer in range(DUMP_LAYERS):
        names = ("embed",) if layer == 0 else ()
        for op in names + OPS:
            ours = np.concatenate([levanter[s][op if op == "embed" else f"{op}/{layer}"] for s in ids])
            ref = np.concatenate([torch_dumps[s][op if op == "embed" else f"{op}/{layer}"] for s in ids])
            table[op if op == "embed" else f"L{layer}/{op}"] = op_diff(ours, ref)
        ours_ids = np.concatenate([levanter[s][f"routes/{layer}"] for s in ids])
        ref_ids = np.concatenate([torch_dumps[s][f"routes/{layer}"] for s in ids])
        routes[f"L{layer}"] = {
            "positions": len(ref_ids),
            "set_agree": int((np.sort(ours_ids, -1) == np.sort(ref_ids, -1)).all(-1).sum()),
            "ordered_agree": int((ours_ids == ref_ids).all(-1).sum()),
        }
    for sequence_id in ids:
        ours = levanter[sequence_id]["router_logits/0"][MARGIN_POSITION]
        ref = torch_dumps[sequence_id]["router_logits/0"][MARGIN_POSITION]
        margins[sequence_id] = {
            "levanter": top_margins(ours, bias[0]),
            "torch": top_margins(ref, bias[0]),
            "unbiased_max_abs_diff": float(np.abs(ours.astype(np.float64) - ref.astype(np.float64)).max()),
        }
    return {"ops": table, "routes": routes, "layer0_position4": margins}


# ---- torch reference (subprocess) --------------------------------------------------------------------------------


def torch_dumps(weights: str, sequences: list[Sequence]) -> dict:
    """Layers 0-2 of the torch reference on one GPU, one unpadded sequence at a time; float32 numpy per op."""
    grug = import_grug()
    source = json.loads(read(prefix_join(weights, "config.json")))
    if source["num_hidden_layers"] != LAYERS:
        raise ValueError(f"expected {LAYERS} layers, got {source['num_hidden_layers']}")
    config = grug.GrugMoeConfig(**{**source, "num_hidden_layers": DUMP_LAYERS})
    config._attn_implementation = "eager"
    with torch.device("meta"):
        model = grug.GrugMoeForCausalLM(config).eval().requires_grad_(False)
    reader = CheckpointReader(CheckpointSource(path=weights, revision=""))
    expected = model.state_dict()
    missing = set(expected) - set(reader.weight_map)
    if missing:
        raise ValueError(f"checkpoint lacks {sorted(missing)[:5]}")
    device = torch.device("cuda:0")
    model.load_state_dict({name: reader.tensor(name).to(device) for name in expected}, strict=True, assign=True)
    # The 26-layer schedule, not a 3-layer one (whose last layer would be long).
    long_layers = grug.grug_long_layer_flags(LAYERS)

    captured: dict[str, np.ndarray] = {}

    def keep(name):
        def hook(module, args, output):
            if name.startswith("router_logits"):
                captured[name] = output.router_logits.float().cpu().numpy()
                captured[name.replace("router_logits", "routes")] = output.selected_experts.cpu().numpy()
            else:
                captured[name] = output[0].float().cpu().numpy()

        return hook

    inner = model.model
    inner.embed_gated_norm.register_forward_hook(keep("embed"))
    for layer in range(DUMP_LAYERS):
        block = inner.layers[layer]
        for name, module in (
            ("attn_norm", block.attn_gated_norm),
            ("attn_out", block.self_attn),
            ("mlp_norm", block.mlp_gated_norm),
            ("router_logits", block.mlp.router),
            ("moe_out", block.mlp),
            ("shared_out", block.shared_expert),
            ("resid", block),
        ):
            module.register_forward_hook(keep(f"{name}/{layer}"))

    result: dict = {}
    with torch.no_grad():
        for sequence in sequences:
            captured.clear()
            ids = torch.tensor([list(sequence.tokens)], device=device)
            hidden = inner.embed_gated_norm(inner.embed_norm(inner.embed_tokens(ids)))
            positions = torch.arange(ids.shape[1], device=device).unsqueeze(0)
            for layer in range(DUMP_LAYERS):
                hidden = inner.layers[layer](hidden, None, positions, long_layers[layer])
            result[sequence.id] = dict(captured)
    result["_bias"] = {layer: inner.layers[layer].mlp.router.bias.float().cpu().numpy() for layer in range(DUMP_LAYERS)}
    return result


def run_torch_phase(root: str, positions: str) -> None:
    sequences = [s for s in stored_sequences(positions) if s.group == "check"]
    dumps = torch_dumps(BASE_MODEL, sequences)
    buffer = io.BytesIO()
    torch.save(dumps, buffer)
    StoragePath(prefix_join(root, "torch-dumps.pt")).write_bytes(buffer.getvalue())


# ---- Levanter side -----------------------------------------------------------------------------------------------


def mesh_config(expert: int) -> MeshConfig:
    """The production mesh on one host: data x expert = 8, no context axis, same mappings."""
    return MeshConfig(
        # context 1 keeps the parameter mapping's axis names valid; the model treats it as absent.
        axes={"data": BATCH // expert, "expert": expert, "context": 1, "replica": 1, "model": 1},
        dcn_axes={"replica_dcn": 1},
        compute_mapping={"batch": ["replica_dcn", "data", "expert"], "position": "context", "vocab": "model"},
        param_mapping={"embed": ["data", "context"]},
    )


def with_capacity(view, factor: float):
    """``view`` with every block's ``mlp.expert_mlp.capacity_factor`` replaced (a static field, so by
    ``dataclasses.replace``); returns the new view and the recorded change."""
    old = sorted({block.mlp.expert_mlp.capacity_factor for block in view.transformer.blocks})
    blocks = tuple(
        dataclasses.replace(
            block,
            mlp=dataclasses.replace(
                block.mlp, expert_mlp=dataclasses.replace(block.mlp.expert_mlp, capacity_factor=factor)
            ),
        )
        for block in view.transformer.blocks
    )
    new_view = dataclasses.replace(view, transformer=dataclasses.replace(view.transformer, blocks=blocks))
    change = {
        "field": "transformer.blocks[0..25].mlp.expert_mlp.capacity_factor",
        "old": old,
        "new": factor,
        "method": "dataclasses.replace on the loaded compute view (static field; eqx.tree_at reaches leaves only)",
    }
    return new_view, change


def _sublayers(view, tokens: jax.Array) -> dict:
    """Layers 0-2 of the Snowball forward, op by op, with the production modules (``SnowballBlock.forward`` order);
    every output replicated."""
    t = view.transformer
    cfg = t.config
    short_mask = AttentionMask(is_causal=True, sliding_window=cfg.sliding_window, segment_ids=None)
    long_mask = AttentionMask(is_causal=True, sliding_window=None, segment_ids=None)
    tokens = reshard(tokens, P(_BATCH_AXES, _context_axis()))
    hidden = t.embed_gated_norm(t.embed_norm(t.token_embed.at[tokens].get(out_sharding=_activation_spec())))
    out = {"embed": hidden}
    for layer in range(DUMP_LAYERS):
        block = t.blocks[layer]
        is_long = layer % 4 == 3 or layer == LAYERS - 1
        attn_in = block.attn_gated_norm(block.rms_attn(hidden))
        attn_out = block.attn(attn_in, long_mask if is_long else short_mask, disable_rope=is_long)
        x = hidden + attn_out
        mlp_in = block.mlp_gated_norm(block.rms_mlp(x))
        # SnowballMoEMLP._forward's router dot, repeated to expose the unbiased logits.
        b, s, _ = mlp_in.shape
        router_input = jnp.reshape(mlp_in, (b * s, mlp_in.shape[-1]), out_sharding=_token_spec())
        router_weight = reshard(block.mlp.router, P(None, None))
        if cfg.router_dot_fp32:
            router_input, router_weight = router_input.astype(jnp.float32), router_weight.astype(jnp.float32)
        logits = jnp.einsum("td,de->te", router_input, router_weight).astype(jnp.float32)
        routed, routes = block.mlp.with_routes(mlp_in)
        shared = block.shared(mlp_in)
        hidden = x + (routed + shared)
        out |= {
            f"attn_norm/{layer}": attn_in,
            f"attn_out/{layer}": attn_out,
            f"mlp_norm/{layer}": mlp_in,
            f"router_logits/{layer}": jnp.reshape(logits, (b, s, logits.shape[-1])),
            f"routes/{layer}": routes,
            f"moe_out/{layer}": routed,
            f"shared_out/{layer}": shared,
            f"resid/{layer}": hidden,
        }
    return {name: reshard(value, P()) for name, value in out.items()}


def _routes(view, tokens: jax.Array) -> jax.Array:
    Batch, Pos = Axis("batch", tokens.shape[0]), view.Pos.resize(tokens.shape[1])
    _, routes = view.activations_with_routes(hax.named(tokens, (Batch, Pos)))
    return reshard(routes, P())


def levanter_dumps(view, check: list[Sequence]) -> tuple[dict, dict]:
    """Sublayer outputs per check sequence (real positions only, float32 numpy), and the dump batch's drop counts."""
    rows = [s.tokens for s in check] + [check[0].tokens] * (DUMP_ROWS - len(check))
    grid = padded(rows, PAD_MULTIPLE)
    out: dict[str, np.ndarray] = {k: gather_global(v) for k, v in jax.jit(_sublayers)(view, jnp.asarray(grid)).items()}
    dumps = {
        s.id: {
            name: value[row, : len(s.tokens)].astype(np.int64 if name.startswith("routes") else np.float32)
            for name, value in out.items()
        }
        for row, s in enumerate(check)
    }
    return dumps, {"grid": list(grid.shape), "routes": {layer: out[f"routes/{layer}"] for layer in range(DUMP_LAYERS)}}


def load_view(config, receipt: dict, source: dict, expert: int):
    """The trainer's compute view of the pinned source under the one-host mesh: the LR0 probe's loader steps."""
    trainer_config = replace(config.trainer, mesh=mesh_config(expert))
    mesh = trainer_config.device_mesh
    with hax.partitioning.set_mesh(mesh), hax.axis_mapping(trainer_config.parameter_axis_mapping):
        tokenizer = config.data.the_tokenizer
        converter = config.model.hf_checkpoint_converter().replaced(reference_checkpoint=BASE_MODEL, tokenizer=tokenizer)
        reloaded = converter.config_from_hf_config(GrugMoeHfConfig(**source))
        reloaded = preserve_hf_runtime_settings(reloaded, config.model, config.trainer.compute_dtype_overrides)
        model = converter.load_pretrained(
            reloaded.model_type, config=reloaded, axis_mapping=trainer_config.parameter_axis_mapping, dtype=None
        )
        if asdict(model.config) != receipt["runtime_model_config"]:
            raise ValueError("loaded config differs from the approved runtime model config")
        view = cast_model_for_compute(model, jmp.get_policy(POLICY), config.trainer.compute_dtype_overrides)
    return view, mesh, trainer_config


def run_arm(name: str, view, mesh, trainer_config, overrides, root: str, sequences, check, torch_ref) -> dict:
    arm = ARMS[name]
    started = time.time()
    change = None
    with hax.partitioning.set_mesh(mesh), hax.axis_mapping(trainer_config.compute_axis_mapping):
        if arm["capacity_factor"] is not None:
            view, change = with_capacity(view, arm["capacity_factor"])
        factor = view.transformer.blocks[0].mlp.expert_mlp.capacity_factor
        implementation = view.transformer.blocks[0].mlp.expert_mlp.implementation
        out = prefix_join(root, name)
        records = positions_records(view, sequences, BATCH, PAD_MULTIPLE, PAD_ID, routes=True)
        dot_dtypes = router_dot_dtypes(view, jnp.bfloat16)
        positions_path = prefix_join(out, "positions.pt")
        save_positions(records, positions_path)
        del records
        report = evaluate(replace(INPUTS, output=out), overrides, positions_path, dot_dtypes)
        write_json(prefix_join(out, "gate.json"), report)

        routes_fn = jax.jit(_routes)
        drops = np.zeros(LAYERS, dtype=np.int64)
        assignments = 0
        for grid in gate_batches(sequences, BATCH, PAD_MULTIPLE):
            routes = gather_global(routes_fn(view, jnp.asarray(grid)))
            assignments += routes[0].size
            for layer in range(LAYERS):
                drops[layer] += ring_drops(routes[layer], view.config.num_experts, arm["expert"], factor)

        dumps, dump_batch = levanter_dumps(view, check)
    bias = {layer: np.asarray(torch_ref["_bias"][layer]) for layer in range(DUMP_LAYERS)}
    dump_drops = [
        ring_drops(dump_batch["routes"][layer], view.config.num_experts, arm["expert"], factor)
        for layer in range(DUMP_LAYERS)
    ]
    buffer = io.BytesIO()
    torch.save(dumps, buffer)
    StoragePath(prefix_join(out, "levanter-dumps.pt")).write_bytes(buffer.getvalue())
    parity = report["parity"]["positions"]
    return {
        "arm": name,
        "dropless": arm["dropless"],
        "mesh": {"axes": dict(mesh.shape)},
        "change_from_ep4-cap1": change or ({"mesh": "data 8 x expert 1"} if arm["expert"] == 1 else None),
        "moe_implementation": implementation,
        "capacity_factor": factor,
        "router_dot_dtypes": dot_dtypes,
        "gate_failures": report["failures"],
        "parity": {
            group: {
                "positions": s["positions"],
                "top1": (s["top1_agree"] + s["top1_near_tie"]) / s["positions"],
                "p50": s["p50_abs_logprob_diff"],
                "p99": s["p99_abs_logprob_diff"],
                "max": s["max_abs_logprob_diff"],
                "missing": s["missing_from_harness_top50"],
            }
            for group, s in parity.items()
        },
        "contrastive": {k: v for k, v in report["parity"]["contrastive"].items() if k != "row_disagreements"},
        "routes": {
            "totals": report["routes"]["totals"],
            "set_agree_by_layer": [
                sum(v["set_agree"] for g, v in layer.items() if isinstance(v, dict) and g != "long_past_window")
                / sum(v["positions"] for g, v in layer.items() if isinstance(v, dict) and g != "long_past_window")
                for layer in report["routes"]["layers"]
            ],
        },
        "drops_per_layer_gate_batches": drops.tolist(),
        "assignments_per_layer_gate_batches": assignments,
        "drops_per_layer_dump_batch": dump_drops,
        "dump_batch_grid": dump_batch["grid"],
        "sublayers_vs_torch": compare_dumps(dumps, torch_ref, bias),
        "device_memory_after_arm": device_memory(),
        "seconds": round(time.time() - started, 1),
    }


def device_memory() -> list[dict]:
    """Per local device: bytes in use now and the process's peak so far (a running maximum over all arms)."""
    found = []
    for device in jax.local_devices():
        stats = device.memory_stats() or {}
        found.append({k: stats.get(k) for k in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit")})
    return found


def torch_routes_match_gate_reference(torch_ref: dict) -> dict:
    """The subprocess's layer 0-2 routes for the check sequences vs the gate's torch ``routes.pt``."""
    gate = torch.load(io.BytesIO(read(prefix_join(INPUTS.torch_routing, "routes.pt"))), map_location="cpu")
    result = {}
    for sequence_id, dumps in torch_ref.items():
        if sequence_id.startswith("_"):
            continue
        reference = gate[sequence_id].long().numpy()[:, :DUMP_LAYERS]
        ours = np.stack([dumps[f"routes/{layer}"] for layer in range(DUMP_LAYERS)], axis=1)
        result[sequence_id] = bool(np.array_equal(ours, reference))
    return result


def diagnostic_worker(root: str, receipt_bytes: bytes) -> None:
    check_pod_storage()
    started = time.time()
    torch_started = time.time()
    subprocess.run(
        [sys.executable, "-m", MODULE, "torch-phase", "--root", root],
        check=True,
        env=os.environ.copy(),
    )
    torch_seconds = time.time() - torch_started
    torch_ref = torch.load(io.BytesIO(read(prefix_join(root, "torch-dumps.pt"))), map_location="cpu", weights_only=False)

    receipt = approved_receipt(receipt_bytes)
    reference = receipt["routing_reference"]
    config_bytes = (StoragePath(BASE_MODEL) / "config.json").read_bytes()
    generation_bytes = (StoragePath(BASE_MODEL) / "generation_config.json").read_bytes()
    config, observed = configure_precision_runtime(
        base_config(root, "moe-drop-diagnostic", config_bytes, generation_bytes),
        config_bytes=config_bytes,
        generation_bytes=generation_bytes,
        rounding_bytes=StoragePath(reference["path"]).read_bytes(),
        rounding_path=reference["path"],
        rounding_sha256=reference["sha256"],
    )
    if observed != receipt:
        raise ValueError("Derived runtime recipe differs from the reviewed receipt")
    overrides = config.trainer.compute_dtype_overrides
    source = json.loads(config_bytes)
    sequences = stored_sequences(INPUTS.positions)
    check = [s for s in sequences if s.group == "check"]
    result: dict = {
        "torch_seconds": round(torch_seconds, 1),
        "torch_routes_equal_gate_routes_layers_0_2": torch_routes_match_gate_reference(torch_ref),
        "arms": {},
    }
    write_json(prefix_join(root, "receipt.json"), result)
    for expert, names in ((4, ("ep4-cap1", "ep4-dropless")), (1, ("ep1",))):
        load_started = time.time()
        view, mesh, trainer_config = load_view(config, receipt, source, expert)
        result[f"load_seconds_expert{expert}"] = round(time.time() - load_started, 1)
        for name in names:
            result["arms"][name] = run_arm(
                name, view, mesh, trainer_config, overrides, root, sequences, check, torch_ref
            )
            write_json(prefix_join(root, "receipt.json"), result)
        del view
        gc.collect()
        jax.clear_caches()
    result["seconds"] = round(time.time() - started, 1)
    write_json(prefix_join(root, "receipt.json"), result)


# ---- launcher ----------------------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    phase = sub.add_parser("torch-phase", help="inside the job: write <root>/torch-dumps.pt")
    phase.add_argument("--root", required=True)
    launch = sub.add_parser("launch", help="print the plan; with --run, submit one H100x8 job")
    launch.add_argument("--job-name", required=True)
    launch.add_argument("--root", required=True, help=f"fresh prefix under {T310}/")
    launch.add_argument("--runtime-receipt", required=True, help="local runtime-recipe-f37aec3.json")
    launch.add_argument("--timeout-seconds", type=int, default=5400)
    launch.add_argument("--priority", choices=("interactive", "production"), default="production")
    launch.add_argument("--expect-commit")
    launch.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.command == "torch-phase":
        run_torch_phase(args.root, INPUTS.positions)
        return
    launch_job(args)


def launch_job(args) -> None:
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout
    receipt_bytes = Path(args.runtime_receipt).read_bytes()
    approved_receipt(receipt_bytes)
    if not args.root.startswith(f"{T310}/diag-"):
        raise ValueError(f"root must be a fresh {T310}/diag-* prefix")
    resources = ResourceConfig.with_gpu("H100", count=8, cpu=64, ram="768GB", disk="1TB", target_cluster="cw-rno2a")
    plan = {
        "job": args.job_name,
        "commit": head,
        "clean": not dirty,
        "root": args.root,
        "arms": ARMS,
        "gate_inputs": vars(INPUTS),
        "resources": "1 x H100x8 cw-rno2a, 64 cpu, 768GB, 1TB",
        "priority": args.priority,
        "timeout_seconds": args.timeout_seconds,
        "h100_hour_ceiling": 8 * args.timeout_seconds / 3600,
        "retries_failure": 0,
        "retries_preemption": 0,
    }
    print(json.dumps(plan, indent=1))
    if not args.run:
        return
    if dirty or args.expect_commit != head:
        raise ValueError("--run needs a clean tree at --expect-commit")
    if StoragePath(args.root).exists():
        raise FileExistsError(f"{args.root} exists")
    key, secret = os.environ["CW_KEY_ID"], os.environ["CW_KEY_SECRET"]
    secrets = {"CW_KEY_ID": key, "CW_KEY_SECRET": secret, "AWS_ACCESS_KEY_ID": key, "AWS_SECRET_ACCESS_KEY": secret}
    environment = EnvironmentSpec(extras=["gpu"], secrets=secrets, secret_behavior=SecretBehavior.ISOLATE)
    entrypoint = Entrypoint.from_callable(diagnostic_worker, args.root, receipt_bytes)
    with connect_controller(cluster_name="marin") as info:
        with IrisClient.remote(info.url, workspace=Path.cwd(), credentials=info.credentials) as client:
            job = client.submit(
                entrypoint=entrypoint,
                name=args.job_name,
                resources=convert_resources(resources),
                environment=environment,
                constraints=convert_constraints(resources) or None,
                timeout=Duration.from_seconds(args.timeout_seconds),
                max_retries_failure=0,
                max_retries_preemption=0,
                priority_band=PRIORITY_BANDS[args.priority],
            )
    print(json.dumps({"job_id": str(job.job_id)}))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

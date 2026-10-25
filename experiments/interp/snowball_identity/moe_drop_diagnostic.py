# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""#310 two-arm rerun on the integrated pin: the dropless Levanter Snowball forward with and without the FP32 combine.

One H100x8 job on cw-rno2a; nothing is trained, saved as a model or exported. Both arms are the gate's compute
view (BF16 compute, FP32 router bias, FP32 router dot) on data 2 x expert 4 with ``moe_capacity_factor`` 4 (the
expert-axis size: dropless), set through the recipe, not by editing the model. They differ only in
``router_combine_fp32``:

- ``dropless``: ``router_combine_fp32`` False.
- ``dropless-fp32-combine``: ``router_combine_fp32`` True.

Each arm's runtime recipe is derived in the job (``configure_precision_runtime``) and must equal that arm's reviewed
runtime receipt, pinned by sha256 in ``ARMS``; the model is then loaded the LR0 probe's way, so the two fields reach
it through the HF reload.

Per arm, under ``<root>/<arm>/``:

- ``positions.pt`` and ``gate.json``: the 465 pinned sequences scored by ``levanter_positions.positions_records``
  (validity-masked, with the backend's per-layer dispatch drops, gathered at vLLM's stored top-20 ids) and judged
  by ``precision_gate.evaluate``, the gate's own code and thresholds (batch 8, pad multiple 64); ``gate.json`` holds
  the MoE settings, the drops and per-device memory.
- ``g1.json``: the G1 routes check (``route_proxy_check.check``) against the torch floor's pinned batch-1 top-6 and
  receipt.
- ``item5.pt``: for the item-5 rows, the top-6 biased router logits at every position and layer from the same
  validity-masked forward on the gate's own batch, with the routes; the receipt summarises the 4th/5th margins at
  the position-layers where Levanter and torch batch-1 choose different expert sets, from both stacks.
- sublayer dumps of layers 0-2 for the 11 ``check`` sequences, compared with the torch reference (as in r2).

The torch reference is the gate's: the vendored torch GrugMoE (``harness.import_grug``) with eager attention and
eager experts. A subprocess loads only the embedding and layers 0-2 on one GPU, before JAX starts, and writes
``<root>/torch-dumps.pt``; its routes must equal the gate's torch ``routes.pt`` for those layers.

Everything lands in ``<root>/receipt.json``, including each device's memory before and after each arm (the peak is
a running maximum over the process).
"""

import argparse
import gc
import importlib
import io
import json
import logging
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
from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.cluster.types import Entrypoint, EnvironmentSpec
from jax.sharding import PartitionSpec as P
from jax.sharding import reshard
from levanter.compute_dtype import cast_model_for_compute
from levanter.grug.attention import AttentionMask
from levanter.grug.sharding import unshard
from levanter.main.train_lm import preserve_hf_runtime_settings
from levanter.models.snowball import (
    _BATCH_AXES,
    GrugMoeHfConfig,
    _activation_spec,
    _context_axis,
    _token_spec,
    long_attention_layer_mask,
)
from levanter.utils.mesh import MeshConfig
from marin.merging.checkpoint import CheckpointReader, CheckpointSource
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.secrets import SecretBehavior
from rigging.timing import Duration

from experiments.interp.snowball_identity import route_proxy_check
from experiments.interp.snowball_identity.harness import import_grug
from experiments.interp.snowball_identity.levanter_positions import (
    PAD_ID,
    positions_records,
    router_dot_dtypes,
    save_positions,
)
from experiments.interp.snowball_identity.logit_check import read, reference_ids, stored_sequences, write_json
from experiments.interp.snowball_identity.multihost import gather_global, gather_host_rows
from experiments.interp.snowball_identity.precision_gate import (
    device_memory_report,
    evaluate,
    local_device_memory,
)
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
EXPERT_AXIS = 4
CAPACITY_FACTOR = 4.0
TOP = 6
K = 4
MARGIN_POSITION = 4
OPS = ("attn_norm", "attn_out", "mlp_norm", "router_logits", "moe_out", "shared_out", "resid")
ARMS = {
    # margay's metadata-only receipts on 4188e87c: t310/runtime-recipe-4188e87c-combine-{off,on}.json
    "dropless": {
        "router_combine_fp32": False,
        "receipt_sha256": "dcddbf4d59d0863a773e11a97002567dbea622fab52506d9c716fa4b4a30b7d2",
    },
    "dropless-fp32-combine": {
        "router_combine_fp32": True,
        "receipt_sha256": "d123ddbdbeed0c3cc843cd247e23d74cb2b21e7538056150da8139843c81fddd",
    },
}
"""The arms differ only in ``router_combine_fp32``; each is bound to its reviewed runtime receipt by sha256."""
ITEM5_ROWS = ("contrastive|id296-imp-031|no_system|4", "contrastive|id296-imp-031|no_system|7")
"""Item 5: the rows of the ep4-dropless contrastive maximum (4.28) in r2."""
FLOOR = f"{T310}/diag-torch-floor-r1"
POLICY = "p=f32,c=bfloat16,o=f32"
MODULE = "experiments.interp.snowball_identity.moe_drop_diagnostic"
INPUTS = gate_inputs("unused", f"{T310}/torch-ref-queue-r1/results/t310-router-rounding-r1/routing.json")
"""The r5 gate's inputs; each arm writes under its own ``output``."""


# ---- gate grids ----------------------------------------------------------------------------------------------------


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


def gate_batch_of(sequences: list[Sequence], sequence_id: str, batch: int, pad_multiple: int) -> tuple:
    """The gate grid holding ``sequence_id`` (``gate_batches`` order), its validity grid as ``positions_records``
    builds it (real rows' tokens only) and the sequence's row."""
    order = sorted(range(len(sequences)), key=lambda i: len(sequences[i].tokens))
    for step, grid in enumerate(gate_batches(sequences, batch, pad_multiple)):
        indices = order[step * batch : (step + 1) * batch]
        valid = np.zeros(grid.shape, dtype=bool)
        for row, index in enumerate(indices):
            valid[row, : len(sequences[index].tokens)] = True
        for row, index in enumerate(indices):
            if sequences[index].id == sequence_id:
                return grid, valid, row
    raise KeyError(sequence_id)


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


def margin_stats(values: np.ndarray) -> dict:
    if not len(values):
        return {"count": 0}
    return {
        "count": len(values),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "max": float(values.max()),
        "below_0.1": float(np.mean(values < 0.1)),
    }


def item5_margins(levanter: dict, torch_top6: dict, position: int | None = None) -> dict:
    """One row: Levanter's top-6 ``ids``/``biased`` [T, L, 6] vs torch batch-1's (``top6.pt`` layout). At the
    position-layers where the top-4 expert sets differ: the count per layer and both stacks' 4th/5th biased-logit
    margins; at ``position`` (if given), every layer's margins and whether the sets agree."""
    ours_ids, ours = np.asarray(levanter["ids"]), np.asarray(levanter["biased"], dtype=np.float64)
    theirs_ids, theirs = np.asarray(torch_top6["ids"]), np.asarray(torch_top6["biased"], dtype=np.float64)
    if ours_ids.shape != theirs_ids.shape:
        raise ValueError(f"shapes differ: {ours_ids.shape} vs {theirs_ids.shape}")
    differ = (np.sort(ours_ids[..., :K], -1) != np.sort(theirs_ids[..., :K], -1)).any(-1)  # [T, L]
    ours_margin = ours[..., K - 1] - ours[..., K]
    theirs_margin = theirs[..., K - 1] - theirs[..., K]
    found = {
        "positions": int(differ.shape[0]),
        "set_disagreements": int(differ.sum()),
        "set_disagreements_by_layer": differ.sum(0).tolist(),
        "levanter_margin_at_disagreements": margin_stats(ours_margin[differ]),
        "torch_margin_at_disagreements": margin_stats(theirs_margin[differ]),
    }
    if position is not None:
        found[f"position_{position}"] = {
            "set_agree_by_layer": (~differ[position]).tolist(),
            "levanter_margin_by_layer": ours_margin[position].tolist(),
            "torch_margin_by_layer": theirs_margin[position].tolist(),
        }
    return found


# ---- torch reference (subprocess) --------------------------------------------------------------------------------


def first_layers_config(source: dict) -> dict:
    """``source`` cut to the first ``DUMP_LAYERS`` layers; every layer-count alias is set, or GrugMoeConfig refuses."""
    aliases = [key for key in ("num_hidden_layers", "num_layers") if key in source]
    if not aliases or any(source[key] != LAYERS for key in aliases):
        raise ValueError(f"expected {LAYERS} layers, got {[source.get(key) for key in aliases]}")
    return source | {key: DUMP_LAYERS for key in aliases}


def torch_dumps(weights: str, sequences: list[Sequence], device: torch.device) -> dict:
    """Layers 0-2 of the torch reference on one device, one unpadded sequence at a time; float32 numpy per op."""
    grug = import_grug()
    config = grug.GrugMoeConfig(**first_layers_config(json.loads(read(prefix_join(weights, "config.json")))))
    config._attn_implementation = "eager"
    with torch.device("meta"):
        model = grug.GrugMoeForCausalLM(config).eval().requires_grad_(False)
    reader = CheckpointReader(CheckpointSource(path=weights, revision=""))
    expected = model.state_dict()
    missing = set(expected) - set(reader.weight_map)
    if missing:
        raise ValueError(f"checkpoint lacks {sorted(missing)[:5]}")
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
    dumps = torch_dumps(BASE_MODEL, sequences, torch.device("cuda:0"))
    buffer = io.BytesIO()
    torch.save(dumps, buffer)
    StoragePath(prefix_join(root, "torch-dumps.pt")).write_bytes(buffer.getvalue())


# ---- Levanter side -----------------------------------------------------------------------------------------------


def mesh_config() -> MeshConfig:
    """The production mesh on one host: data 2 x expert 4, no context axis, same mappings."""
    return MeshConfig(
        # context 1 keeps the parameter mapping's axis names valid; the model treats it as absent.
        axes={"data": BATCH // EXPERT_AXIS, "expert": EXPERT_AXIS, "context": 1, "replica": 1, "model": 1},
        dcn_axes={"replica_dcn": 1},
        compute_mapping={"batch": ["replica_dcn", "data", "expert"], "position": "context", "vocab": "model"},
        param_mapping={"embed": ["data", "context"]},
    )


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


def _router_top6(view, tokens: jax.Array, valid: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """``SnowballTransformer._forward`` with ``token_valid``, layer by layer in the same scan, also returning every
    layer's top-6 biased router logits: ids and values [L, B, S, 6], and the routes [L, B, S, K] the MoE used."""
    t = view.transformer
    cfg = t.config
    short_mask = AttentionMask(is_causal=True, sliding_window=cfg.sliding_window, segment_ids=None)
    long_mask = AttentionMask(is_causal=True, sliding_window=None, segment_ids=None)
    tokens = reshard(tokens, P(_BATCH_AXES, _context_axis()))
    valid = reshard(valid, P(_BATCH_AXES, _context_axis()))
    hidden = t.embed_gated_norm(t.embed_norm(t.token_embed.at[tokens].get(out_sharding=_activation_spec())))
    stacked = jax.tree_util.tree_map(lambda *layers: jnp.stack(layers), *t.blocks)

    def layer_step(carry, layer_and_flag):
        block, use_long = layer_and_flag
        attn_in = block.attn_gated_norm(block.rms_attn(carry))
        attn_out = jax.lax.cond(
            jnp.asarray(use_long, dtype=jnp.bool_),
            lambda _: block.attn(attn_in, long_mask, disable_rope=True),
            lambda _: block.attn(attn_in, short_mask, disable_rope=False),
            operand=None,
        )
        x = carry + attn_out
        mlp_in = block.mlp_gated_norm(block.rms_mlp(x))
        # SnowballMoEMLP._forward's router dot and bias, repeated to expose the biased logits.
        b, s, _ = mlp_in.shape
        router_input = jnp.reshape(mlp_in, (b * s, mlp_in.shape[-1]), out_sharding=_token_spec())
        router_weight = reshard(block.mlp.router, P(None, None))
        if cfg.router_dot_fp32:
            router_input, router_weight = router_input.astype(jnp.float32), router_weight.astype(jnp.float32)
        biased = jnp.einsum("td,de->te", router_input, router_weight).astype(jnp.float32)
        biased = biased + unshard(block.mlp.router_bias)
        values, ids = jax.lax.top_k(biased, TOP)
        routed, routes, _ = block.mlp._forward(mlp_in, True, valid)
        top6 = (reshard(jnp.reshape(ids, (b, s, TOP)), P()), reshard(jnp.reshape(values, (b, s, TOP)), P()))
        return x + (routed + block.shared(mlp_in)), (*top6, routes)

    _, (ids, values, routes) = jax.lax.scan(layer_step, hidden, (stacked, long_attention_layer_mask(len(t.blocks))))
    return ids, values, routes


def item5_dumps(view, sequences: list[Sequence]) -> dict:
    """Per item-5 row, from a validity-masked forward of the gate batch that holds it: top-6 ``ids`` and ``biased``
    [T, L, 6] and ``routes`` [T, L, K] at the row's real positions."""
    capture = jax.jit(_router_top6)
    found = {}
    for sequence_id in ITEM5_ROWS:
        grid, valid, row = gate_batch_of(sequences, sequence_id, BATCH, PAD_MULTIPLE)
        ids, values, routes = (gather_global(x) for x in capture(view, jnp.asarray(grid), jnp.asarray(valid)))
        n = int(valid[row].sum())
        found[sequence_id] = {
            "ids": torch.from_numpy(ids[:, row, :n].transpose(1, 0, 2).astype(np.int16)),
            "biased": torch.from_numpy(values[:, row, :n].transpose(1, 0, 2).astype(np.float32)),
            "routes": torch.from_numpy(routes[:, row, :n].transpose(1, 0, 2).astype(np.int16)),
        }
    return found


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


def load_view(config, receipt: dict, source: dict):
    """The trainer's compute view of the pinned source under the one-host mesh: the LR0 probe's loader steps."""
    trainer_config = replace(config.trainer, mesh=mesh_config())
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


def device_memory() -> list[dict]:
    """Every device's bytes in use, peak so far (a running maximum over the process) and limit."""
    return device_memory_report(gather_host_rows(local_device_memory()))


def score_positions(view, sequences: list[Sequence]) -> tuple[dict, dict]:
    """The gate's scoring, as the LR0 gate scores: routes, and the log-prob at each of vLLM's stored top-20 ids, so
    every vLLM token is measured."""
    return positions_records(
        view,
        sequences,
        BATCH,
        PAD_MULTIPLE,
        PAD_ID,
        routes=True,
        reference_ids=reference_ids(INPUTS.vllm, sequences),
    )


def run_arm(name: str, view, mesh, trainer_config, overrides, root: str, sequences, check, torch_ref, floor) -> dict:
    """``floor``: the torch floor's pinned ``top6`` (batch 1) and ``receipt``, for G1 and the item-5 margins."""
    started = time.time()
    memory_before = device_memory()
    out = prefix_join(root, name)
    with hax.partitioning.set_mesh(mesh), hax.axis_mapping(trainer_config.compute_axis_mapping):
        mlp = view.transformer.blocks[0].mlp.expert_mlp
        settings = {
            "capacity_factor": view.config.moe_capacity_factor,
            "expert_mlp_capacity_factor": mlp.capacity_factor,
            "expert_axis": dict(mesh.shape)["expert"],
            "router_combine_fp32": view.config.router_combine_fp32,
            "expert_mlp_combine_in_fp32": mlp.combine_in_fp32,
            "moe_implementation": mlp.implementation,
        }
        records, dispatch = score_positions(view, sequences)
        dot_dtypes = router_dot_dtypes(view, jnp.bfloat16)
        g1 = route_proxy_check.check(records, floor["top6"], floor["receipt"])
        positions_path = prefix_join(out, "positions.pt")
        save_positions(records, positions_path)
        moe = {key: settings[key] for key in ("capacity_factor", "expert_axis", "router_combine_fp32")}
        memory = device_memory()
        report = evaluate(
            replace(INPUTS, output=out), overrides, positions_path, dot_dtypes, moe | {"dispatch": dispatch}, memory
        )
        write_json(prefix_join(out, "gate.json"), report)
        g1["inputs"] = {
            "positions": positions_path,
            "torch_top6_sha256": route_proxy_check.TORCH_TOP6_SHA256,
            "floor_receipt_sha256": route_proxy_check.FLOOR_RECEIPT_SHA256,
        }
        write_json(prefix_join(out, "g1.json"), g1)

        item5 = item5_dumps(view, sequences)
        dumps, dump_batch = levanter_dumps(view, check)
    item5_summary = {}
    for sequence_id, found in item5.items():
        item5_summary[sequence_id] = {
            "routes_equal_positions_pt": bool(torch.equal(found["routes"], records[sequence_id]["routes"])),
            "top4_prefix_equal_routes": bool(torch.equal(found["ids"][..., :K], found["routes"])),
            **item5_margins(found, floor["top6"][sequence_id], position=27),
        }
    del records
    buffer = io.BytesIO()
    torch.save(item5, buffer)
    StoragePath(prefix_join(out, "item5.pt")).write_bytes(buffer.getvalue())
    bias = {layer: np.asarray(torch_ref["_bias"][layer]) for layer in range(DUMP_LAYERS)}
    buffer = io.BytesIO()
    torch.save(dumps, buffer)
    StoragePath(prefix_join(out, "levanter-dumps.pt")).write_bytes(buffer.getvalue())
    parity = report["parity"]["positions"]
    return {
        "arm": name,
        "mesh": {"axes": dict(mesh.shape)},
        "settings": settings,
        "router_dot_dtypes": dot_dtypes,
        "gate_failures": report["failures"],
        "dispatch": dispatch,
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
        "g1": {
            "layers_passing": g1["layers_passing"],
            "rate_ok_layers": [r["layer"] for r in g1["layers"] if r["rate_ok"]],
            "failures": g1["failures"],
            "totals": g1["totals"],
        },
        "item5": item5_summary,
        "dump_batch_grid": dump_batch["grid"],
        "sublayers_vs_torch": compare_dumps(dumps, torch_ref, bias),
        "device_memory_before_arm": memory_before,
        "device_memory_after_arm": device_memory(),
        "seconds": round(time.time() - started, 1),
    }


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


def arm_config(name: str, receipt: dict, root: str, config_bytes: bytes, generation_bytes: bytes):
    """The arm's runtime recipe, derived as the LR0 probe derives it, which must equal the arm's reviewed receipt."""
    base = base_config(root, f"diag-{name}", config_bytes, generation_bytes, receipt)
    reference = receipt["routing_reference"]
    config, observed = configure_precision_runtime(
        base,
        config_bytes=config_bytes,
        generation_bytes=generation_bytes,
        rounding_bytes=StoragePath(reference["path"]).read_bytes(),
        rounding_path=reference["path"],
        rounding_sha256=reference["sha256"],
    )
    if observed != receipt:
        raise ValueError(f"{name}: derived runtime recipe differs from the reviewed receipt")
    model = config.model
    if (model.moe_capacity_factor, model.router_combine_fp32) != (CAPACITY_FACTOR, ARMS[name]["router_combine_fp32"]):
        raise ValueError(f"{name}: capacity {model.moe_capacity_factor}, combine {model.router_combine_fp32}")
    return config


def load_floor() -> dict:
    top6 = route_proxy_check.load_pinned(f"{FLOOR}/batch1/top6.pt", route_proxy_check.TORCH_TOP6_SHA256)
    receipt = route_proxy_check.load_pinned(f"{FLOOR}/receipt.json", route_proxy_check.FLOOR_RECEIPT_SHA256)
    return {"top6": torch.load(io.BytesIO(top6), map_location="cpu"), "receipt": json.loads(receipt)}


def diagnostic_worker(root: str, receipts: dict[str, bytes]) -> None:
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
    floor = load_floor()
    config_bytes = (StoragePath(BASE_MODEL) / "config.json").read_bytes()
    generation_bytes = (StoragePath(BASE_MODEL) / "generation_config.json").read_bytes()
    source = json.loads(config_bytes)
    sequences = stored_sequences(INPUTS.positions)
    check = [s for s in sequences if s.group == "check"]
    result: dict = {
        "torch_seconds": round(torch_seconds, 1),
        "torch_routes_equal_gate_routes_layers_0_2": torch_routes_match_gate_reference(torch_ref),
        "receipts": {name: arm["receipt_sha256"] for name, arm in ARMS.items()},
        "arms": {},
    }
    write_json(prefix_join(root, "receipt.json"), result)
    for name, arm in ARMS.items():
        receipt = approved_receipt(receipts[name], arm["receipt_sha256"])
        config = arm_config(name, receipt, root, config_bytes, generation_bytes)
        load_started = time.time()
        view, mesh, trainer_config = load_view(config, receipt, source)
        result[f"load_seconds_{name}"] = round(time.time() - load_started, 1)
        result["arms"][name] = run_arm(
            name,
            view,
            mesh,
            trainer_config,
            config.trainer.compute_dtype_overrides,
            root,
            sequences,
            check,
            torch_ref,
            floor,
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
    for name in ARMS:
        launch.add_argument(f"--receipt-{name}", required=True, help=f"local runtime receipt for the {name} arm")
    launch.add_argument("--timeout-seconds", type=int, default=2700)
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
    receipts = {name: Path(getattr(args, f"receipt_{name.replace('-', '_')}")).read_bytes() for name in ARMS}
    for name, arm in ARMS.items():
        approved_receipt(receipts[name], arm["receipt_sha256"])
    if not args.root.startswith(f"{T310}/diag-"):
        raise ValueError(f"root must be a fresh {T310}/diag-* prefix")
    resources = ResourceConfig.with_gpu("H100", count=8, cpu=64, ram="768GB", disk="1TB", target_cluster="cw-rno2a")
    plan = {
        "job": args.job_name,
        "commit": head,
        "clean": not dirty,
        "root": args.root,
        "arms": ARMS,
        "capacity_factor": CAPACITY_FACTOR,
        "expert_axis": EXPERT_AXIS,
        "item5_rows": ITEM5_ROWS,
        "g1": {
            "floor": FLOOR,
            "torch_top6_sha256": route_proxy_check.TORCH_TOP6_SHA256,
            "floor_receipt_sha256": route_proxy_check.FLOOR_RECEIPT_SHA256,
            "rule": {"k": route_proxy_check.K_RATE, "margin_factor": route_proxy_check.MARGIN_FACTOR},
        },
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
    entrypoint = Entrypoint.from_callable(importlib.import_module(MODULE).diagnostic_worker, args.root, receipts)
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

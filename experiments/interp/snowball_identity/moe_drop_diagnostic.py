# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""#310 diagnostic arms on the integrated pin: the dropless Levanter Snowball forward, three arms in one job.

One H100x8 job on cw-rno2a; nothing is trained, saved as a model or exported. Each arm is the gate's compute view
(BF16 compute, FP32 router bias, FP32 router dot) on data 2 x expert 4 with ``moe_capacity_factor`` 4 (the
expert-axis size: dropless), FP32 combine and FP32 activations, set through the recipe, not by editing the model.
``ARMS`` names the attention implementation and XLA precision setting each arm must have:

- ``torch-order-xla-default`` (a): ``torch_order`` attention, XLA's default flags.
- ``torch-order-no-excess-precision`` (b): ``torch_order`` attention, ``--xla_allow_excess_precision=false``.
- ``fa4-no-excess-precision`` (c): ``gpu_fa4_cute`` attention, ``--xla_allow_excess_precision=false``.

The arm's process resolves the implementation the module selects on its backend and fails before scoring unless it
is ``RESOLVED_ATTENTION``'s; gate.json records it as ``resolved_attention_implementation``.

Each arm runs in its own process, started with the arm's ``XLA_FLAGS``; the process checks the setting before XLA's
backend starts (``require_xla_precision``). Its runtime recipe is derived there (``configure_precision_runtime``) and
must equal the arm's reviewed runtime receipt (schema 6), passed at launch with its sha256; the model is then loaded
the LR0 probe's way, so the fields reach it through the HF reload. Peak device memory is per arm, since each arm has
its own process.

Under ``<root>/<arm>/``:

- ``positions.pt`` and ``gate.json``: the 465 pinned sequences scored by ``levanter_positions.positions_records``
  (validity-masked, with the backend's per-layer dispatch drops, gathered at vLLM's stored top-20 ids) and judged
  by ``precision_gate.evaluate``, the gate's own code and thresholds (batch 8, pad multiple 64); ``gate.json`` holds
  the MoE settings, the drops, per-device memory, the arm's ``recipe`` (attention implementation, XLA precision
  setting, ``XLA_FLAGS``), ``direct_levanter_vs_vllm`` (per layer, both vLLM passes; the L0 rule value is the minimum
  over the passes) and ``decision`` (the coordinator's rule, below).
- ``g1.json``: the G1 routes check (``route_proxy_check.check``) against the torch floor's pinned batch-1 top-6 and
  receipt; ``g1a.json``: G1a (``route_proxy_check.check_vllm_floor``) against the served-stack floor, marten's
  pinned vLLM route capture at ``VLLM_CAPTURE``.
- ``item5.pt``: for the item-5 rows, the top-6 biased router logits at every position and layer from the same
  validity-masked forward on the gate's own batch, with the routes; the receipt summarises the 4th/5th margins at
  the position-layers where Levanter and torch batch-1 choose different expert sets, from both stacks.
- sublayer dumps of layers 0-2 for the 11 ``check`` sequences, compared with the torch reference (as in r2).

The torch reference is the gate's: the vendored torch GrugMoE (``harness.import_grug``) with eager attention and
eager experts. A subprocess loads only the embedding and layers 0-2 on one GPU, before JAX starts, and writes
``<root>/torch-dumps.pt``; its routes must equal the gate's torch ``routes.pt`` for those layers.

Each arm's process writes ``<root>/<arm>/arm.json``; everything lands in ``<root>/receipt.json``, including each
device's memory before and after each arm and the arm chosen by the rule. The rule: an arm meets it when the
log-prob gate passes, G1 or G1a passes, and its L0 Levanter-vs-vLLM set agreement, the minimum over the two vLLM
passes, is at least ``L0_VLLM_FLOOR``; among arms that meet it, ``PREFERENCE`` picks; with none, hold.
"""

import argparse
import datetime
import importlib
import io
import json
import logging
import os
import subprocess
import sys
import tempfile
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
from experiments.interp.snowball_identity.logit_check import (
    read,
    reference_ids,
    require_fresh_root,
    stored_sequences,
    write_json,
)
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
from experiments.post_training.identity_context_distill.runtime_recipe import (
    XLA_CACHE_FLAGS,
    configure_precision_runtime,
    require_xla_precision,
    xla_precision_env,
    xla_precision_in_process,
)
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
_RECIPE = {"router_combine_fp32": True, "fp32_activations": True}
ARMS = {
    "torch-order-xla-default": _RECIPE | {"attention_implementation": "torch_order", "xla_precision": "default"},
    "torch-order-no-excess-precision": (
        _RECIPE | {"attention_implementation": "torch_order", "xla_precision": "no_excess_precision"}
    ),
    "fa4-no-excess-precision": (
        _RECIPE | {"attention_implementation": "gpu_fa4_cute", "xla_precision": "no_excess_precision"}
    ),
}
"""The recipe settings each arm's reviewed runtime receipt must give."""
RESOLVED_ATTENTION = {
    "torch-order-xla-default": "torch_order",
    "torch-order-no-excess-precision": "torch_order",
    "fa4-no-excess-precision": "gpu_fa4_cute",
}
"""The attention implementation each arm's module must select on the job's GPUs (``resolved_attention``)."""
PREFERENCE = ("fa4-no-excess-precision", "torch-order-xla-default", "torch-order-no-excess-precision")
"""The coordinator's order when more than one arm meets the rule."""
L0_VLLM_FLOOR = 0.97
"""Minimum L0 Levanter-vs-vLLM route set agreement, taken as the minimum over the two vLLM passes."""
ITEM5_ROWS = ("contrastive|id296-imp-031|no_system|4", "contrastive|id296-imp-031|no_system|7")
"""Item 5: the rows of the ep4-dropless contrastive maximum (4.28) in r2."""
FLOOR = f"{T310}/diag-torch-floor-r1"
VLLM_CAPTURE = f"{T310}/vllm-route-capture-r1"
"""Byte copies of marten's capture (``marten-task310-vllm-routes-r1/capture``), pinned in ``route_proxy_check``."""
POLICY = "p=f32,c=bfloat16,o=f32"
MODULE = "experiments.interp.snowball_identity.moe_drop_diagnostic"
INPUTS = gate_inputs("unused", f"{T310}/torch-ref-queue-r1/results/t310-router-rounding-r1/routing.json", "G1")
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


def direct_levanter_vs_vllm(g1a: dict) -> dict:
    """Levanter-vs-vLLM route agreement per layer against each vLLM pass, both kept; the L0 rule value is the
    minimum over the two passes of L0's set agreement."""
    layers = [{"layer": r["layer"], **r["levanter_vs_vllm"]} for r in g1a["layers"]]
    l0 = next(r for r in layers if r["layer"] == 0)["set"]
    return {
        "layers": layers,
        "overall_set": g1a["overall"]["levanter_vs_vllm"],
        "l0_set_passes": l0,
        "l0_rule_value": min(l0),
        "l0_rule": "min over the two vLLM passes of L0 set agreement",
        "l0_floor": L0_VLLM_FLOOR,
        "l0_ok": min(l0) >= L0_VLLM_FLOOR,
    }


ROUTE_PROXY_PREFIXES = ("routes-G1:", "routes-G1a:", "routes-G1b-L0:")
"""``evaluate``'s failure rows from the G1 / G1a / G1b L0 route checks, which the rule reads as their own lines."""


def logprob_gate_failures(gate_failures: list[str]) -> list[str]:
    """The gate's failures without the G1 / G1a rows: parity, the cross-stack routes row and dispatch stay."""
    return [f for f in gate_failures if not f.startswith(ROUTE_PROXY_PREFIXES)]


def decision(gate_failures: list[str], g1: dict, g1a: dict, direct: dict) -> dict:
    """The coordinator's rule for one arm: the log-prob gate passes (``gate_failures`` less the G1 / G1a rows), G1
    or G1a passes, and the L0 rule value is at least ``L0_VLLM_FLOOR``. A check passes when it lists no failures."""
    lines = {
        "logprob_gate": not logprob_gate_failures(gate_failures),
        "g1": not g1["failures"],
        "g1a": not g1a["failures"],
        "l0_levanter_vs_vllm": direct["l0_ok"],
    }
    return {
        "lines": lines,
        "meets_rule": lines["logprob_gate"] and (lines["g1"] or lines["g1a"]) and lines["l0_levanter_vs_vllm"],
    }


def choose(arms: dict) -> str:
    """The first arm in ``PREFERENCE`` that ran and meets the rule, else ``hold``."""
    for name in PREFERENCE:
        if name in arms and arms[name]["decision"]["meets_rule"]:
            return name
    return "hold"


def peak_memory(memory: list[dict]) -> list[dict]:
    return [
        {"process": row["process"], "local_device": row["local_device"], "peak_bytes_in_use": row["peak_bytes_in_use"]}
        for row in memory
    ]


def resolved_attention(configured: str | None) -> str | None:
    """The implementation ``SnowballAttention`` passes to ``attention`` for ``configured`` on this backend (the same
    selection as snowball.py:507-509)."""
    if configured is None and jax.default_backend() == "gpu":
        return "gpu_fa4_cute"
    return configured


def check_resolved_attention(name: str, configured: str | None) -> str | None:
    """The arm's resolved implementation; an arm whose module would select another one fails before scoring."""
    resolved = resolved_attention(configured)
    if resolved != RESOLVED_ATTENTION[name]:
        raise ValueError(f"{name}: the module selects {resolved!r}, the arm needs {RESOLVED_ATTENTION[name]!r}")
    return resolved


def recipe_record(view) -> dict:
    return {
        "attention_implementation": view.config.attention_implementation,
        "resolved_attention_implementation": resolved_attention(view.config.attention_implementation),
        "xla_precision": xla_precision_in_process(),
        "XLA_FLAGS": os.environ.get("XLA_FLAGS"),
    }


def run_arm(name: str, view, mesh, trainer_config, overrides, root: str, sequences, check, torch_ref, floor) -> dict:
    """``floor``: the torch floor's pinned ``top6`` (batch 1) and ``receipt``, for G1 and the item-5 margins, and the
    vLLM capture's ``pass1`` / ``pass2`` routes for G1a."""
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
            "fp32_activations": view.config.fp32_activations,
            "attention_implementation": view.config.attention_implementation,
            "expert_mlp_combine_in_fp32": mlp.combine_in_fp32,
            "moe_implementation": mlp.implementation,
        }
        records, dispatch = score_positions(view, sequences)
        dot_dtypes = router_dot_dtypes(view, jnp.bfloat16)
        g1 = route_proxy_check.check(records, floor["top6"], floor["receipt"])
        g1a = route_proxy_check.check_vllm_floor(
            records, floor["top6"], floor["receipt"], floor["pass1"], floor["pass2"]
        )
        positions_path = prefix_join(out, "positions.pt")
        save_positions(records, positions_path)
        moe = {
            key: settings[key] for key in ("capacity_factor", "expert_axis", "router_combine_fp32", "fp32_activations")
        }
        memory = device_memory()
        report = evaluate(
            replace(INPUTS, output=out), overrides, positions_path, dot_dtypes, moe | {"dispatch": dispatch}, memory
        )
        direct = direct_levanter_vs_vllm(g1a)
        report["logprob_gate_failures"] = logprob_gate_failures(report["failures"])
        report["recipe"] = recipe_record(view)
        report["direct_levanter_vs_vllm"] = direct
        report["decision"] = decision(report["failures"], g1, g1a, direct)
        write_json(prefix_join(out, "gate.json"), report)
        g1["inputs"] = {
            "positions": positions_path,
            "torch_top6_sha256": route_proxy_check.TORCH_TOP6_SHA256,
            "floor_receipt_sha256": route_proxy_check.FLOOR_RECEIPT_SHA256,
        }
        write_json(prefix_join(out, "g1.json"), g1)
        g1a["inputs"] = g1["inputs"] | {
            "vllm_capture": VLLM_CAPTURE,
            "vllm_receipt_sha256": route_proxy_check.VLLM_RECEIPT_SHA256,
            "vllm_pass1_sha256": route_proxy_check.VLLM_PASS1_SHA256,
            "vllm_pass2_sha256": route_proxy_check.VLLM_PASS2_SHA256,
        }
        write_json(prefix_join(out, "g1a.json"), g1a)

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
    memory_after = device_memory()
    return {
        "arm": name,
        "mesh": {"axes": dict(mesh.shape)},
        "recipe": report["recipe"],
        "decision": report["decision"],
        "direct_levanter_vs_vllm": report["direct_levanter_vs_vllm"],
        "peak_device_memory": peak_memory(memory_after),
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
        "g1a": {
            "layers_passing": g1a["layers_passing"],
            "lines_passing": g1a["lines_passing"],
            "overall": g1a["overall"],
            "failures": g1a["failures"],
        },
        "item5": item5_summary,
        "dump_batch_grid": dump_batch["grid"],
        "sublayers_vs_torch": compare_dumps(dumps, torch_ref, bias),
        "device_memory_before_arm": memory_before,
        "device_memory_after_arm": memory_after,
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
    """The arm's runtime recipe, derived as the LR0 probe derives it (with this process's XLA precision setting),
    which must equal the reviewed receipt and give the arm's settings."""
    base = base_config(root, f"diag-{name}", config_bytes, generation_bytes, receipt)
    reference = receipt["routing_reference"]
    config, observed = configure_precision_runtime(
        base,
        config_bytes=config_bytes,
        generation_bytes=generation_bytes,
        rounding_bytes=StoragePath(reference["path"]).read_bytes(),
        rounding_path=reference["path"],
        rounding_sha256=reference["sha256"],
        xla_precision=xla_precision_in_process(),
    )
    if observed != receipt:
        raise ValueError(f"{name}: derived runtime recipe differs from the reviewed receipt")
    found = receipt_settings(observed)
    if found != ARMS[name] or config.model.moe_capacity_factor != CAPACITY_FACTOR:
        raise ValueError(f"{name}: capacity {config.model.moe_capacity_factor}, settings {found}")
    return config


def receipt_settings(receipt: dict) -> dict:
    """The ``ARMS`` settings a runtime receipt gives."""
    model = receipt["runtime_model_config"]
    return {
        "router_combine_fp32": model.get("router_combine_fp32"),
        "fp32_activations": model.get("fp32_activations"),
        "attention_implementation": model.get("attention_implementation"),
        "xla_precision": receipt.get("xla_precision"),
    }


def load_floor() -> dict:
    top6 = route_proxy_check.load_pinned(f"{FLOOR}/batch1/top6.pt", route_proxy_check.TORCH_TOP6_SHA256)
    receipt = route_proxy_check.load_pinned(f"{FLOOR}/receipt.json", route_proxy_check.FLOOR_RECEIPT_SHA256)
    route_proxy_check.load_pinned(f"{VLLM_CAPTURE}/receipt.json", route_proxy_check.VLLM_RECEIPT_SHA256)
    passes = {
        f"pass{n}": torch.load(
            io.BytesIO(route_proxy_check.load_pinned(f"{VLLM_CAPTURE}/pass-{n}/routes.pt", sha)), map_location="cpu"
        )
        for n, sha in ((1, route_proxy_check.VLLM_PASS1_SHA256), (2, route_proxy_check.VLLM_PASS2_SHA256))
    }
    return {"top6": torch.load(io.BytesIO(top6), map_location="cpu"), "receipt": json.loads(receipt), **passes}


def arm_env(name: str) -> dict[str, str]:
    """The worker's environment with ``XLA_FLAGS`` set to the arm's XLA precision setting (Iris's cache flags kept)."""
    env = os.environ.copy()
    kept = [f for f in env.pop("XLA_FLAGS", "").split() if f.partition("=")[0] in XLA_CACHE_FLAGS]
    flags = " ".join([*kept, *xla_precision_env(ARMS[name]["xla_precision"]).values()])
    if flags:
        env["XLA_FLAGS"] = flags
    return env


def arm_phase(root: str, name: str, receipt_path: str, receipt_sha256: str) -> None:
    """One arm, in its own process: check the XLA precision setting before anything starts XLA's backend, then run
    the arm and write ``<root>/<arm>/arm.json``."""
    require_xla_precision(ARMS[name]["xla_precision"])
    started = time.time()
    receipt = approved_receipt(Path(receipt_path).read_bytes(), receipt_sha256)
    torch_ref = torch.load(io.BytesIO(read(prefix_join(root, "torch-dumps.pt"))), map_location="cpu", weights_only=False)
    floor = load_floor()
    config_bytes = (StoragePath(BASE_MODEL) / "config.json").read_bytes()
    generation_bytes = (StoragePath(BASE_MODEL) / "generation_config.json").read_bytes()
    sequences = stored_sequences(INPUTS.positions)
    check = [s for s in sequences if s.group == "check"]
    config = arm_config(name, receipt, root, config_bytes, generation_bytes)
    load_started = time.time()
    view, mesh, trainer_config = load_view(config, receipt, json.loads(config_bytes))
    load_seconds = round(time.time() - load_started, 1)
    check_resolved_attention(name, view.config.attention_implementation)
    result = run_arm(
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
    result |= {"runtime_receipt_sha256": receipt_sha256, "load_seconds": load_seconds}
    result["process_seconds"] = round(time.time() - started, 1)
    write_json(prefix_join(root, f"{name}/arm.json"), result)


def diagnostic_worker(root: str, arms: list[tuple[str, bytes, str]]) -> None:
    """``arms``: (name, the reviewed runtime receipt's exact bytes, its sha256), run in this order. The torch phase
    runs once; each arm runs in its own process with its own ``XLA_FLAGS``. A failed arm is recorded and the next
    one still runs; the job fails at the end if any arm failed."""
    check_pod_storage()
    started = time.time()
    torch_started = time.time()
    subprocess.run(
        [sys.executable, "-m", MODULE, "torch-phase", "--root", root],
        check=True,
        env=os.environ.copy(),
    )
    torch_ref = torch.load(io.BytesIO(read(prefix_join(root, "torch-dumps.pt"))), map_location="cpu", weights_only=False)
    result: dict = {
        "torch_seconds": round(time.time() - torch_started, 1),
        "torch_routes_equal_gate_routes_layers_0_2": torch_routes_match_gate_reference(torch_ref),
        "runtime_receipt_sha256": {name: sha for name, _, sha in arms},
        "arms": {},
        "arm_exit_codes": {},
    }
    del torch_ref
    write_json(prefix_join(root, "receipt.json"), result)
    local = Path(tempfile.mkdtemp())
    for name, receipt_bytes, sha in arms:
        receipt_path = local / f"{name}-runtime-receipt.json"
        receipt_path.write_bytes(receipt_bytes)
        command = [sys.executable, "-m", MODULE, "arm-phase", "--root", root, "--arm", name]
        command += ["--receipt-file", str(receipt_path), "--receipt-sha256", sha]
        code = subprocess.run(command, env=arm_env(name)).returncode
        result["arm_exit_codes"][name] = code
        if code == 0:
            result["arms"][name] = json.loads(read(prefix_join(root, f"{name}/arm.json")))
        write_json(prefix_join(root, "receipt.json"), result)
    result["chosen"] = choose(result["arms"])
    result["seconds"] = round(time.time() - started, 1)
    write_json(prefix_join(root, "receipt.json"), result)
    failed = [name for name, code in result["arm_exit_codes"].items() if code]
    if failed:
        raise RuntimeError(f"arms failed: {failed}")


# ---- launcher ----------------------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    phase = sub.add_parser("torch-phase", help="inside the job: write <root>/torch-dumps.pt")
    phase.add_argument("--root", required=True)
    arm = sub.add_parser("arm-phase", help="inside the job: run one arm in this process")
    arm.add_argument("--root", required=True)
    arm.add_argument("--arm", choices=sorted(ARMS), required=True)
    arm.add_argument("--receipt-file", required=True)
    arm.add_argument("--receipt-sha256", required=True)
    launch = sub.add_parser("launch", help="print the plan; with --run, submit one H100x8 job running every arm")
    launch.add_argument("--job-name", required=True)
    launch.add_argument("--root", required=True, help=f"fresh prefix under {T310}/")
    launch.add_argument(
        "--arm-receipt",
        nargs=3,
        action="append",
        required=True,
        metavar=("ARM", "RECEIPT", "SHA256"),
        help="one per arm: the arm, its local reviewed runtime receipt and that receipt's sha256",
    )
    launch.add_argument("--timeout-seconds", type=int, default=2700)
    launch.add_argument("--priority", choices=("interactive", "production"), default="production")
    launch.add_argument("--expect-commit")
    launch.add_argument("--launch-receipt", help="local JSON receipt to write; required with --run")
    launch.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.command == "torch-phase":
        run_torch_phase(args.root, INPUTS.positions)
        return
    if args.command == "arm-phase":
        arm_phase(args.root, args.arm, args.receipt_file, args.receipt_sha256)
        return
    launch_job(args)


def arm_receipts(given: list[list[str]]) -> list[tuple[str, bytes, str]]:
    """Every arm exactly once, in ``ARMS`` order; each receipt's bytes must hash to its sha256 and give the arm's
    settings."""
    names = [name for name, _, _ in given]
    if sorted(names) != sorted(ARMS):
        raise ValueError(f"--arm-receipt must name each of {sorted(ARMS)} once, got {names}")
    by_name = {name: (path, sha) for name, path, sha in given}
    arms = []
    for name in ARMS:
        path, sha = by_name[name]
        data = Path(path).read_bytes()
        receipt = approved_receipt(data, sha)
        if receipt.get("schema_version") != 6:
            raise ValueError(f"{name}: receipt schema {receipt.get('schema_version')!r}, need 6")
        if receipt_settings(receipt) != ARMS[name]:
            raise ValueError(f"{name}: receipt settings {receipt_settings(receipt)} are not the arm's {ARMS[name]}")
        arms.append((name, data, sha))
    return arms


def launch_job(args) -> None:
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout
    arms = arm_receipts(args.arm_receipt)
    paths = {name: path for name, path, _ in args.arm_receipt}
    if not args.root.startswith(f"{T310}/diag-"):
        raise ValueError(f"root must be a fresh {T310}/diag-* prefix")
    resources = ResourceConfig.with_gpu("H100", count=8, cpu=64, ram="768GB", disk="1TB", target_cluster="cw-rno2a")
    plan = {
        "job": args.job_name,
        "commit": head,
        "clean": not dirty,
        "root": args.root,
        "arms": {
            name: {
                **ARMS[name],
                "XLA_FLAGS": xla_precision_env(ARMS[name]["xla_precision"]).get("XLA_FLAGS"),
                "expected_resolved_attention_implementation": RESOLVED_ATTENTION[name],
                "runtime_receipt": paths[name],
                "runtime_receipt_sha256": sha,
            }
            for name, _, sha in arms
        },
        "arm_order": [name for name, _, _ in arms],
        "rule": {
            "meets": (
                "log-prob gate passes, G1 or G1a passes, L0 Levanter-vs-vLLM set agreement (min over the two "
                f"vLLM passes) >= {L0_VLLM_FLOOR}"
            ),
            "preference": PREFERENCE,
            "otherwise": "hold",
        },
        "capacity_factor": CAPACITY_FACTOR,
        "expert_axis": EXPERT_AXIS,
        "item5_rows": ITEM5_ROWS,
        "g1": {
            "floor": FLOOR,
            "torch_top6_sha256": route_proxy_check.TORCH_TOP6_SHA256,
            "floor_receipt_sha256": route_proxy_check.FLOOR_RECEIPT_SHA256,
            "rule": {"k": route_proxy_check.K_RATE, "margin_factor": route_proxy_check.MARGIN_FACTOR},
        },
        "g1a": {
            "vllm_capture": VLLM_CAPTURE,
            "vllm_receipt_sha256": route_proxy_check.VLLM_RECEIPT_SHA256,
            "vllm_pass1_sha256": route_proxy_check.VLLM_PASS1_SHA256,
            "vllm_pass2_sha256": route_proxy_check.VLLM_PASS2_SHA256,
            "rule": {"k": route_proxy_check.K_RATE_VLLM, "margin_factor": route_proxy_check.MARGIN_FACTOR},
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
    if not args.launch_receipt:
        raise ValueError("--run needs --launch-receipt")
    require_fresh_root(args.root)
    key, secret = os.environ["CW_KEY_ID"], os.environ["CW_KEY_SECRET"]
    secrets = {"CW_KEY_ID": key, "CW_KEY_SECRET": secret, "AWS_ACCESS_KEY_ID": key, "AWS_SECRET_ACCESS_KEY": secret}
    # No job-level XLA_FLAGS: each arm's process gets its own (``arm_env``).
    environment = EnvironmentSpec(extras=["gpu"], secrets=secrets, secret_behavior=SecretBehavior.ISOLATE)
    entrypoint = Entrypoint.from_callable(importlib.import_module(MODULE).diagnostic_worker, args.root, arms)
    launch = {**plan, "state": "pre-submit", "submit_utc": datetime.datetime.now(datetime.UTC).isoformat()}
    Path(args.launch_receipt).write_text(json.dumps(launch, indent=1) + "\n")
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
    launch.update(state="submitted", iris_job_id=str(job.job_id))
    Path(args.launch_receipt).write_text(json.dumps(launch, indent=1) + "\n")
    print(json.dumps({"job_id": str(job.job_id)}))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

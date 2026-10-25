# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""#310 L0 table follow-up (jerboa's ruling 2): Levanter's embed path and L0 attention-side norm, step by step.

``diag-l0-ops-r1`` found Levanter already departs from torch at ``embed`` (embed_norm + embed_gated_norm), where torch
is bit-identical across batch shapes. This job runs Levanter's production ``RMSNorm`` and ``GatedNorm`` modules
(``levanter.models.snowball``) with the checkpoint's own tensors, in the BF16 compute view, on the 11 check
sequences, and compares every step with torch batch-1 (``diag-l0-ops-r1/check-ops.pt``, bit-tied to the fix-r1
rerun's torch dumps):

- ``composite``: embed lookup -> embed_norm -> embed_gated_norm, then input RMSNorm -> attn gated norm, in one jitted
  program, as in production. Only ``embed`` and ``L0/attn_norm`` are outputs.
- ``per_op``: every step its own jitted program, so each output is rounded to BF16 as torch eager rounds it.
- ``isolated``: every step its own jitted program fed torch's input for that step, so each op's own error is measured
  without what came before it.
- ``fp32_activations`` / ``composite_fp32_activations``: GatedNorm's silu and sigmoid computed in FP32 and cast to
  BF16 once, per op and in one program.
- ``by_sub_step`` / ``fp32_activations_by_sub_step``: every GatedNorm sub-step fed torch's input for that sub-step,
  so silu and sigmoid are measured as single ops.
- ``two_round`` / ``two_round_isolated``: ``per_op`` / ``isolated`` with both RMSNorms in vLLM's native-path order
  (normalized value cast to BF16, then the weight multiply); ``composite_two_round``: the composite with those norms;
  and both vs ``composite`` at its outputs.

Each mode runs at batch-1, on the gate grid holding the sequence, and on the rerun's 16 x 64 dump grid. The whole
set runs twice, in two processes: with XLA's defaults, and with ``--xla_allow_excess_precision=false``.

The MLP side of block 0 (r3), at batch-1 on the check sequences, with FP32 GatedNorm activations as in production:
the residual add (embed + attn_out), ``post_attention_layernorm``, the MLP GatedNorm sub-steps and ``mlp_norm``, the
L0 router input. Torch's attention output comes from ``check-ops.pt``; Levanter's own embed and attn_out come from the
fp32act-r1 arm's ``levanter-dumps.pt`` (the full model, one H100x8 forward). Residual modes (``RESIDUAL_MODES``):
``native`` is production's ``x + attn_out`` on BF16 arrays; ``bf16`` adds in FP32 and rounds to BF16 once (torch
eager's and the Grug source's BF16 add); ``fp32`` keeps the FP32 sum into the RMSNorm (vLLM's fused residual form).
Per input source and residual mode: ``mlp_per_op`` (every step its own program), ``mlp_composite`` (one program, as
production fuses it) and the two compared directly; ``mlp_isolated`` feeds torch's input to each step. The composite's
optimized HLO (native residual, torch inputs) is written per XLA setting with its add and convert instructions.

One H100 on cw-rno2a; nothing is trained or saved as a model.
"""

import argparse
import functools
import hashlib
import importlib
import io
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch
from fray.iris_backend import convert_constraints, convert_resources
from fray.types import ResourceConfig
from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.cluster.types import Entrypoint, EnvironmentSpec
from jax.sharding import Mesh
from levanter.models.snowball import GatedNorm, RMSNorm
from marin.merging.checkpoint import CheckpointReader, CheckpointSource
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.secrets import SecretBehavior
from rigging.timing import Duration

from experiments.interp.snowball_identity.logit_check import read, stored_sequences
from experiments.interp.snowball_identity.moe_drop_diagnostic import (
    BATCH,
    DUMP_ROWS,
    INPUTS,
    PAD_MULTIPLE,
    T310,
    gate_batch_of,
    op_diff,
    padded,
)
from experiments.interp.snowball_identity.precision_gate import write_json
from experiments.interp.snowball_identity.sequences import Sequence
from experiments.post_training.identity_context_distill.lr0_probe import PRIORITY_BANDS, check_pod_storage
from experiments.post_training.identity_context_distill.sft import BASE_MODEL

logger = logging.getLogger(__name__)

MODULE = "experiments.interp.snowball_identity.levanter_l0_ops"
CHECK_OPS = f"{T310}/diag-l0-ops-r1/check-ops.pt"
CHECK_OPS_SHA256 = "0488704b205f0c9b844877f81418a61e5bf073ab7c9d7f2555ea3f1b20c73fb5"
XLA_SETTINGS = {"default": None, "no_excess_precision": "--xla_allow_excess_precision=false"}
GATED = ("down", "silu", "up", "sigmoid")
MODES = (
    "composite",
    "per_op",
    "isolated",
    "two_round",
    "two_round_isolated",
    "composite_two_round",
    "fp32_activations",
    "composite_fp32_activations",
    "by_sub_step",
    "fp32_activations_by_sub_step",
)
COMPOSITE_OUTPUTS = ("embed", "L0/attn_norm")
MLP = (
    "layers.0.post_attention_layernorm",
    "layers.0.mlp_gated_norm",
    "L0/mlp_rmsnorm",
    "L0/mlp_gated_norm",
    "L0/mlp_norm",
)
"""Block 0's MLP-side norm pair: checkpoint names, then torch's op names."""
RESIDUAL_MODES = ("native", "bf16", "fp32")
LEVANTER_DUMPS = f"{T310}/diag-moe-fp32act-r1/dropless-fp32-combine-fp32-activations/levanter-dumps.pt"
LEVANTER_DUMPS_SHA256 = "f2d116bb1a0ce214fa265b2daea5661b07591e168e2dc75169f64f60e1fdc747"
"""Levanter's own embed and L0 attn_out (``embed``, ``attn_out/0``) from the fp32act-r1 arm, pinned."""
NORMS = {
    "embed": ("embed_norm", "embed_gated_norm", "embed_rmsnorm", "embed_gated", "embed"),
    "attn": (
        "layers.0.input_layernorm",
        "layers.0.attn_gated_norm",
        "L0/attn_rmsnorm",
        "L0/attn_gated_norm",
        "L0/attn_norm",
    ),
}
"""Per norm pair: checkpoint names of the RMSNorm and the GatedNorm, then this job's (= torch's) op names."""


def load_modules(weights: str) -> tuple[jax.Array, dict, float]:
    """The embedding table and the two norm pairs, in the BF16 compute view (checkpoint tensors are BF16)."""
    reader = CheckpointReader(CheckpointSource(path=weights, revision=""))
    eps = json.loads(read(prefix_join(weights, "config.json")))["rms_norm_eps"]

    def tensor(name: str) -> jax.Array:
        value = reader.tensor(f"model.{name}")
        if value.dtype != torch.bfloat16:
            raise ValueError(f"model.{name} is {value.dtype}, expected bfloat16")
        return jnp.asarray(value.float().numpy(), dtype=jnp.bfloat16)

    modules = {}
    for key, (rms, gated, *_) in (*NORMS.items(), ("mlp", MLP)):
        modules[key] = (
            RMSNorm(weight=tensor(f"{rms}.weight"), eps=eps),
            GatedNorm(w_down=tensor(f"{gated}.down_proj.weight").T, w_up=tensor(f"{gated}.up_proj.weight").T),
        )
    return tensor("embed_tokens.weight"), modules, eps


def _activation(fn, fp32: bool):
    """``fn`` on BF16 as JAX does it, or (``fp32``) on the input cast to FP32, rounded to BF16 once."""
    return (lambda x: fn(x.astype(jnp.float32)).astype(x.dtype)) if fp32 else fn


def gated_steps(gated: GatedNorm, x: jax.Array, fp32: bool = False, given: dict | None = None) -> dict[str, jax.Array]:
    """``GatedNorm.__call__`` step by step (snowball.py), each step a separate program. ``fp32``: silu and sigmoid in
    FP32, cast once. ``given`` (sub-step name -> torch's value): each sub-step takes torch's input for it."""

    def take(name: str, ours: jax.Array) -> jax.Array:
        return ours if given is None else given[name]

    down = _jit(lambda x, w: jnp.einsum("...d,dr->...r", x, w))(x, gated.w_down)
    silu = _jit(_activation(jax.nn.silu, fp32))(take("down", down))
    up = _jit(lambda h, w: jnp.einsum("...r,rd->...d", h, w))(take("silu", silu), gated.w_up)
    gate = _jit(_activation(jax.nn.sigmoid, fp32))(take("up", up))
    out = _jit(lambda x, g: x * g.astype(x.dtype))(x, take("sigmoid", gate))
    return {"down": down, "silu": silu, "up": up, "sigmoid": gate, "out": out}


@functools.cache
def _jit(fn):
    return jax.jit(fn)


@jax.jit
def composite(table: jax.Array, modules: dict, ids: jax.Array) -> dict[str, jax.Array]:
    (e_rms, e_gated), (a_rms, a_gated) = modules["embed"], modules["attn"]
    hidden = e_gated(e_rms(table[ids]))
    return {"embed": hidden, "L0/attn_norm": a_gated(a_rms(hidden))}


def _two_round_inline(rms: RMSNorm, x: jax.Array) -> jax.Array:
    xf = x.astype(jnp.float32)
    normalized = (xf * jax.lax.rsqrt(jnp.mean(jnp.square(xf), axis=-1, keepdims=True) + rms.eps)).astype(x.dtype)
    return normalized * rms.weight.astype(x.dtype)


@jax.jit
def composite_fp32_activations(table: jax.Array, modules: dict, ids: jax.Array) -> dict[str, jax.Array]:
    """``composite`` with each GatedNorm's silu and sigmoid in FP32, cast to BF16 once, in one program."""

    def gated(g: GatedNorm, x: jax.Array) -> jax.Array:
        hidden = _activation(jax.nn.silu, True)(jnp.einsum("...d,dr->...r", x, g.w_down))
        return x * _activation(jax.nn.sigmoid, True)(jnp.einsum("...r,rd->...d", hidden, g.w_up)).astype(x.dtype)

    (e_rms, e_gated), (a_rms, a_gated) = modules["embed"], modules["attn"]
    hidden = gated(e_gated, e_rms(table[ids]))
    return {"embed": hidden, "L0/attn_norm": gated(a_gated, a_rms(hidden))}


@jax.jit
def composite_two_round(table: jax.Array, modules: dict, ids: jax.Array) -> dict[str, jax.Array]:
    """``composite`` with both RMSNorms in vLLM's native order, in one program (XLA may fuse the rounding away under
    its default excess-precision setting)."""
    (e_rms, e_gated), (a_rms, a_gated) = modules["embed"], modules["attn"]
    hidden = e_gated(_two_round_inline(e_rms, table[ids]))
    return {"embed": hidden, "L0/attn_norm": a_gated(_two_round_inline(a_rms, hidden))}


def jaxpr_dtypes(fn, *args) -> list[dict]:
    """Each primitive of ``fn``'s jaxpr with its input and output dtypes (constants as their dtype)."""
    return [
        {
            "primitive": eqn.primitive.name,
            "in": [str(getattr(v.aval, "dtype", type(v).__name__)) for v in eqn.invars],
            "out": [str(v.aval.dtype) for v in eqn.outvars],
        }
        for eqn in jax.make_jaxpr(fn)(*args).jaxpr.eqns
    ]


def trace(table: jax.Array, modules: dict, ids: jax.Array) -> dict:
    """The traced dtypes of the embed RMSNorm and GatedNorm, and whether RMSNorm multiplies the weight in FP32
    (the normalized value is not cast to BF16 before the weight multiply)."""
    rms, gated = modules["embed"]
    x = table[ids]
    rms_trace = jaxpr_dtypes(lambda m, x: m(x), rms, x)
    weight_multiply = [eqn for eqn in rms_trace if eqn["primitive"] == "mul"][-1]
    return {
        "embed_norm": rms_trace,
        "embed_norm_weight_multiply_in_fp32": weight_multiply["in"] == ["float32", "float32"],
        "embed_gated_norm": jaxpr_dtypes(lambda m, x: m(x), gated, x),
    }


def two_round_rms(rms: RMSNorm, x: jax.Array) -> jax.Array:
    """vLLM's native (Python IR) RMSNorm order: the FP32-normalized value is cast to BF16, then multiplied by the
    BF16 weight; two programs, so the first rounding cannot be fused away."""

    def normalize(x, eps):
        xf = x.astype(jnp.float32)
        return (xf * jax.lax.rsqrt(jnp.mean(jnp.square(xf), axis=-1, keepdims=True) + eps)).astype(x.dtype)

    normalized = _jit(normalize)(x, rms.eps)
    return _jit(lambda n, w: n * w.astype(n.dtype))(normalized, rms.weight)


def residual(embed: jax.Array, attn_out: jax.Array, mode: str) -> jax.Array:
    """Block 0's residual sum in ``mode`` (``RESIDUAL_MODES``)."""
    if mode == "native":
        return embed + attn_out
    total = embed.astype(jnp.float32) + attn_out.astype(jnp.float32)
    return total.astype(jnp.bfloat16) if mode == "bf16" else total


def _mlp_norm(modules: dict, embed: jax.Array, attn_out: jax.Array, mode: str) -> dict[str, jax.Array]:
    rms, gated = modules["mlp"]
    hidden = residual(embed, attn_out, mode)
    normed = rms(hidden).astype(jnp.bfloat16)
    gate_hidden = _activation(jax.nn.silu, True)(jnp.einsum("...d,dr->...r", normed, gated.w_down))
    gate = _activation(jax.nn.sigmoid, True)(jnp.einsum("...r,rd->...d", gate_hidden, gated.w_up))
    return {"L0/resid_attn": hidden.astype(jnp.bfloat16), "L0/mlp_norm": normed * gate.astype(normed.dtype)}


mlp_composite = jax.jit(_mlp_norm, static_argnums=3)
"""Residual add, post-attention RMSNorm and MLP GatedNorm in one program, as production's block fuses them."""


def mlp_per_op(
    modules: dict, embed: jax.Array, attn_out: jax.Array, mode: str, torch_inputs: dict | None = None
) -> dict[str, jax.Array]:
    """Every MLP-side step its own program. With ``torch_inputs``, each step takes torch's input for it."""
    rms, gated = modules["mlp"]
    _, _, rms_name, gated_name, out_name = MLP
    if torch_inputs is not None:
        embed, attn_out = torch_inputs["embed"], torch_inputs["L0/attn_out"]
    ops = {"L0/resid_attn": _jit(functools.partial(residual, mode=mode))(embed, attn_out)}
    source = ops["L0/resid_attn"] if torch_inputs is None else torch_inputs["L0/resid_attn"]
    ops[rms_name] = _jit(lambda m, x: m(x).astype(jnp.bfloat16))(rms, source)
    source = ops[rms_name] if torch_inputs is None else torch_inputs[rms_name]
    given = None if torch_inputs is None else {step: torch_inputs[f"{gated_name}/{step}"] for step in GATED}
    steps = gated_steps(gated, source, True, given)
    ops |= {f"{gated_name}/{step}": steps[step] for step in GATED}
    ops[out_name] = steps["out"]
    ops["L0/resid_attn"] = ops["L0/resid_attn"].astype(jnp.bfloat16)
    return ops


def hlo_adds(hlo: str) -> list[str]:
    """The optimized HLO's add, convert and fusion-root instructions, to read whether the residual sum is rounded."""
    keep = (" add(", " convert(", "ROOT ")
    return [line.strip()[:200] for line in hlo.splitlines() if any(k in line for k in keep)]


def mlp_side(modules: dict, reference: dict, levanter: dict, root: str, setting: str) -> dict:
    """The r3 MLP-side modes at batch-1 on the check sequences (see the module docstring)."""
    found: dict[str, dict] = {}
    for sid, ops in reference.items():
        torch_inputs = {name: jnp.asarray(value[None], dtype=jnp.bfloat16) for name, value in ops.items()}
        sources = {
            "torch": (torch_inputs["embed"], torch_inputs["L0/attn_out"]),
            "levanter": tuple(jnp.asarray(levanter[sid][k][None], dtype=jnp.bfloat16) for k in ("embed", "attn_out/0")),
        }
        outputs = {"mlp_isolated": mlp_per_op(modules, *sources["torch"], "native", torch_inputs)}
        for source, (embed, attn_out) in sources.items():
            for mode in RESIDUAL_MODES:
                outputs[f"mlp_per_op/{source}/{mode}"] = mlp_per_op(modules, embed, attn_out, mode)
                outputs[f"mlp_composite/{source}/{mode}"] = mlp_composite(modules, embed, attn_out, mode)
        for mode, ops in outputs.items():
            found.setdefault(mode, {})[sid] = {name: np.asarray(v.astype(jnp.float32))[0] for name, v in ops.items()}
    result = {mode: compare(by_sequence, reference) for mode, by_sequence in found.items()}
    result["composite_vs_per_op"] = {
        f"{source}/{mode}": compare(
            {sid: ops for sid, ops in found[f"mlp_composite/{source}/{mode}"].items()},
            found[f"mlp_per_op/{source}/{mode}"],
        )
        for source in ("torch", "levanter")
        for mode in RESIDUAL_MODES
    }
    result["levanter_inputs_vs_torch"] = compare(
        {sid: {"embed": levanter[sid]["embed"], "L0/attn_out": levanter[sid]["attn_out/0"]} for sid in reference},
        reference,
    )
    sid = sorted(reference)[0]
    embed, attn_out = (jnp.asarray(reference[sid][k][None], dtype=jnp.bfloat16) for k in ("embed", "L0/attn_out"))
    result["residual_dtype"] = {mode: str(residual(embed, attn_out, mode).dtype) for mode in RESIDUAL_MODES}
    result["composite_jaxpr_adds"] = [
        str(eqn)
        for eqn in jax.make_jaxpr(_mlp_norm, static_argnums=3)(modules, embed, attn_out, "native").eqns
        if eqn.primitive.name in ("add", "convert_element_type")
    ][:8]
    hlo = mlp_composite.lower(modules, embed, attn_out, "native").compile().as_text()
    StoragePath(prefix_join(root, f"{setting}-mlp-composite-hlo.txt")).write_bytes(hlo.encode())
    result["mlp_composite_hlo_sha256"] = hashlib.sha256(hlo.encode()).hexdigest()
    result["mlp_composite_hlo_adds"] = hlo_adds(hlo)
    return result


def per_op(
    table: jax.Array,
    modules: dict,
    ids: jax.Array,
    torch_inputs: dict | None,
    two_round: bool = False,
    fp32: bool = False,
    by_sub_step: bool = False,
) -> dict[str, jax.Array]:
    """Every step its own program. With ``torch_inputs`` (op name -> torch's value), each norm takes torch's input
    instead of this chain's previous output, and with ``by_sub_step`` so does every GatedNorm sub-step.
    ``two_round``: both RMSNorms round as ``two_round_rms``. ``fp32``: GatedNorm activations in FP32, cast once."""
    ops = {"embed_tokens": _jit(lambda t, i: t[i])(table, ids)}
    previous = "embed_tokens"
    for key, (_, _, rms_name, gated_name, out_name) in NORMS.items():
        rms, gated = modules[key]
        source = ops[previous] if torch_inputs is None else torch_inputs[previous]
        ops[rms_name] = two_round_rms(rms, source) if two_round else _jit(lambda m, x: m(x))(rms, source)
        source = ops[rms_name] if torch_inputs is None else torch_inputs[rms_name]
        given = None
        if by_sub_step:
            given = {step: torch_inputs[f"{gated_name}/{step}"] for step in GATED}
        steps = gated_steps(gated, source, fp32, given)
        ops |= {f"{gated_name}/{step}": steps[step] for step in GATED}
        ops[out_name] = steps["out"]
        previous = out_name
    return ops


def grids(sequences: list[Sequence], check: list[Sequence]) -> dict[str, list[tuple[np.ndarray, dict[str, int]]]]:
    """Per grid kind: the grids to run and, per grid, the row of each check sequence it holds."""
    found: dict = {"batch1": [(np.asarray([s.tokens], dtype=np.int32), {s.id: 0}) for s in check], "gate": []}
    for sequence in check:
        grid, _, row = gate_batch_of(sequences, sequence.id, BATCH, PAD_MULTIPLE)
        found["gate"].append((grid, {sequence.id: row}))
    rows = [s.tokens for s in check] + [check[0].tokens] * (DUMP_ROWS - len(check))
    found["dump"] = [(padded(rows, PAD_MULTIPLE), {s.id: row for row, s in enumerate(check)})]
    return found


def compare(ops_by_sequence: dict[str, dict[str, np.ndarray]], reference: dict) -> dict:
    """Per op over the check sequences: ``op_diff`` vs torch batch-1, and how many elements differ at all."""
    report = {}
    for name in next(iter(ops_by_sequence.values())):
        ours = np.concatenate([ops_by_sequence[sid][name].reshape(-1) for sid in sorted(ops_by_sequence)])
        ref = np.concatenate(
            [np.asarray(reference[sid][name], dtype=np.float32).reshape(-1) for sid in sorted(ops_by_sequence)]
        )
        report[name] = op_diff(ours, ref) | {"differing": int((ours != ref).sum()), "elements": int(ours.size)}
    return report


def phase(root: str, setting: str) -> None:
    """One process: every mode on every grid kind, under this process's XLA flags."""
    started = time.time()
    reference = torch.load(io.BytesIO(read(CHECK_OPS)), map_location="cpu", weights_only=False)
    reference = {sid: {name: value.numpy() for name, value in ops.items()} for sid, ops in reference.items()}
    sequences = stored_sequences(INPUTS.positions)
    check = [s for s in sequences if s.group == "check"]
    if {s.id for s in check} != set(reference):
        raise ValueError("check-ops.pt covers other sequences")
    table, modules, eps = load_modules(BASE_MODEL)
    levanter_bytes = read(LEVANTER_DUMPS)
    if hashlib.sha256(levanter_bytes).hexdigest() != LEVANTER_DUMPS_SHA256:
        raise ValueError(f"{LEVANTER_DUMPS}: sha256 differs from the pinned {LEVANTER_DUMPS_SHA256}")
    levanter = torch.load(io.BytesIO(levanter_bytes), map_location="cpu", weights_only=False)
    result: dict = {"xla_flags": os.environ.get("XLA_FLAGS"), "eps": eps, "backend": jax.default_backend()}
    with jax.sharding.set_mesh(Mesh(np.asarray(jax.devices()[:1]), ("data",))):
        first = jnp.asarray([check[0].tokens], dtype=jnp.int32)
        result["mlp"] = mlp_side(modules, reference, levanter, root, setting)
        result["trace"] = trace(table, modules, first)
        hlo = composite.lower(table, modules, first).compile().as_text()
        StoragePath(prefix_join(root, f"{setting}-composite-hlo.txt")).write_bytes(hlo.encode())
        result["composite_hlo_sha256"] = hashlib.sha256(hlo.encode()).hexdigest()
        for kind, runs in grids(sequences, check).items():
            found: dict[str, dict] = {mode: {} for mode in MODES}
            for grid, rows in runs:
                ids = jnp.asarray(grid)
                torch_inputs = None
                if grid.shape[0] == 1:
                    (sid,) = rows
                    torch_inputs = {
                        name: jnp.asarray(value[None], dtype=jnp.bfloat16) for name, value in reference[sid].items()
                    }
                outputs = {
                    "composite": composite(table, modules, ids),
                    "per_op": per_op(table, modules, ids, None),
                    "two_round": per_op(table, modules, ids, None, two_round=True),
                    "composite_two_round": composite_two_round(table, modules, ids),
                    "fp32_activations": per_op(table, modules, ids, None, fp32=True),
                    "composite_fp32_activations": composite_fp32_activations(table, modules, ids),
                }
                if torch_inputs is not None:
                    outputs["isolated"] = per_op(table, modules, ids, torch_inputs)
                    outputs["two_round_isolated"] = per_op(table, modules, ids, torch_inputs, two_round=True)
                    outputs["by_sub_step"] = per_op(table, modules, ids, torch_inputs, by_sub_step=True)
                    outputs["fp32_activations_by_sub_step"] = per_op(
                        table, modules, ids, torch_inputs, fp32=True, by_sub_step=True
                    )
                for mode, ops in outputs.items():
                    host = {name: np.asarray(value.astype(jnp.float32)) for name, value in ops.items()}
                    for sid, row in rows.items():
                        n = len(reference[sid]["embed"])
                        found[mode][sid] = {name: value[row, :n] for name, value in host.items()}
            result[kind] = {mode: compare(by_sequence, reference) for mode, by_sequence in found.items() if by_sequence}
            # The two-round RMSNorm's own effect at the composite's outputs.
            result[kind]["two_round_vs_composite"] = {
                mode: compare(
                    {sid: {name: ops[name] for name in COMPOSITE_OUTPUTS} for sid, ops in found[mode].items()},
                    found["composite"],
                )
                for mode in ("two_round", "composite_two_round")
            }
            # One program vs separate programs with FP32 activations, at the composite's outputs.
            result[kind]["composite_fp32_vs_per_op_fp32"] = compare(
                {
                    sid: {name: ops[name] for name in COMPOSITE_OUTPUTS}
                    for sid, ops in found["composite_fp32_activations"].items()
                },
                found["fp32_activations"],
            )
    result["seconds"] = round(time.time() - started, 1)
    write_json(prefix_join(root, f"{setting}.json"), result)


def l0_levanter_worker(root: str) -> None:
    check_pod_storage()
    started = time.time()
    check_ops = read(CHECK_OPS)
    receipt: dict = {"check_ops": CHECK_OPS, "check_ops_sha256": hashlib.sha256(check_ops).hexdigest(), "seconds": {}}
    if CHECK_OPS_SHA256 is not None and receipt["check_ops_sha256"] != CHECK_OPS_SHA256:
        raise ValueError(f"check-ops.pt sha256 {receipt['check_ops_sha256']}, pinned {CHECK_OPS_SHA256}")
    for setting, flags in XLA_SETTINGS.items():
        env = os.environ.copy()
        env.pop("XLA_FLAGS", None)
        if flags:
            env["XLA_FLAGS"] = flags
        phase_started = time.time()
        subprocess.run(
            [sys.executable, "-m", MODULE, "phase", "--root", root, "--setting", setting], check=True, env=env
        )
        receipt["seconds"][setting] = round(time.time() - phase_started, 1)
        receipt[setting] = json.loads(read(prefix_join(root, f"{setting}.json")))
    receipt["seconds"]["total"] = round(time.time() - started, 1)
    write_json(prefix_join(root, "receipt.json"), receipt)


def entrypoint_for(root: str) -> Entrypoint:
    """The worker by module reference (see ``torch_route_floor.entrypoint_for``)."""
    return Entrypoint.from_callable(importlib.import_module(MODULE).l0_levanter_worker, root)


RESOURCES = ResourceConfig.with_gpu("H100", count=1, cpu=16, ram="128GB", disk="256GB", target_cluster="cw-rno2a")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")
    phase_args = sub.add_parser("phase")
    phase_args.add_argument("--root", required=True)
    phase_args.add_argument("--setting", choices=sorted(XLA_SETTINGS), required=True)
    launch = sub.add_parser("launch")
    launch.add_argument("--job-name", required=True)
    launch.add_argument("--root", required=True, help=f"fresh prefix under {T310}/diag-")
    launch.add_argument("--timeout-seconds", type=int, default=1800)
    launch.add_argument("--priority", choices=("interactive", "production"), default="production")
    launch.add_argument("--expect-commit")
    launch.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.command == "phase":
        phase(args.root, args.setting)
        return
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout
    if not args.root.startswith(f"{T310}/diag-"):
        raise ValueError(f"root must be a fresh {T310}/diag-* prefix")
    plan = {
        "job": args.job_name,
        "commit": head,
        "clean": not dirty,
        "root": args.root,
        "modes": [
            "composite (one program, as production)",
            "per_op (each step its own program)",
            "isolated (each step fed torch's input; batch-1 only)",
            "two_round (per_op with vLLM's native RMSNorm order: cast to BF16, then the weight multiply)",
            "two_round_isolated (batch-1 only)",
            "composite_two_round (composite with the two-round RMSNorms, one program)",
            "two_round_vs_composite (two_round and composite_two_round vs composite at embed, L0/attn_norm)",
            "fp32_activations (per_op with GatedNorm silu/sigmoid in FP32, cast once)",
            "composite_fp32_activations (the same in one program)",
            "by_sub_step (every GatedNorm sub-step fed torch's input; batch-1 only)",
            "fp32_activations_by_sub_step (batch-1 only)",
            "composite_fp32_vs_per_op_fp32 (composite_fp32_activations vs fp32_activations at embed, L0/attn_norm)",
            "mlp_isolated (r3: residual add, post-attention RMSNorm and every MLP GatedNorm sub-step on torch's input)",
            "mlp_per_op/{torch,levanter}/{native,bf16,fp32} (r3: MLP side step by step from torch's or Levanter's "
            "embed + attn_out, per residual mode)",
            "mlp_composite/{torch,levanter}/{native,bf16,fp32} (r3: the same in one program)",
            "mlp composite_vs_per_op, levanter_inputs_vs_torch, residual dtypes, composite jaxpr adds and HLO",
        ],
        "grids": [
            "batch1",
            f"gate (batch {BATCH}, pad {PAD_MULTIPLE})",
            f"dump ({DUMP_ROWS} x {PAD_MULTIPLE}, the rerun's)",
        ],
        "xla_settings": XLA_SETTINGS,
        "reference": CHECK_OPS,
        "reference_sha256": CHECK_OPS_SHA256,
        "levanter_dumps": LEVANTER_DUMPS,
        "levanter_dumps_sha256": LEVANTER_DUMPS_SHA256,
        "weights": BASE_MODEL,
        "resources": "1 x H100 cw-rno2a, 16 cpu, 128GB, 256GB",
        "priority": args.priority,
        "timeout_seconds": args.timeout_seconds,
        "h100_hour_ceiling": args.timeout_seconds / 3600,
        "retries_failure": 0,
        "retries_preemption": 0,
    }
    print(json.dumps(plan, indent=1))
    if not args.run:
        return
    if dirty or args.expect_commit != head:
        raise ValueError("--run needs a clean tree at --expect-commit")
    fs, path = filesystem_for(args.root)
    if fs.exists(path):
        raise FileExistsError(f"{args.root} exists")
    key, secret = os.environ["CW_KEY_ID"], os.environ["CW_KEY_SECRET"]
    secrets = {"CW_KEY_ID": key, "CW_KEY_SECRET": secret, "AWS_ACCESS_KEY_ID": key, "AWS_SECRET_ACCESS_KEY": secret}
    environment = EnvironmentSpec(extras=["gpu"], secrets=secrets, secret_behavior=SecretBehavior.ISOLATE)
    with connect_controller(cluster_name="marin") as info:
        with IrisClient.remote(info.url, workspace=Path.cwd(), credentials=info.credentials) as client:
            job = client.submit(
                entrypoint=entrypoint_for(args.root),
                name=args.job_name,
                resources=convert_resources(RESOURCES),
                environment=environment,
                constraints=convert_constraints(RESOURCES) or None,
                timeout=Duration.from_seconds(args.timeout_seconds),
                max_retries_failure=0,
                max_retries_preemption=0,
                priority_band=PRIORITY_BANDS[args.priority],
            )
    print(json.dumps({"job_id": str(job.job_id)}))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

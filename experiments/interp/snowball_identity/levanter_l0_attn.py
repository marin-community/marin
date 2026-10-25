# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""#310 L0 attn_out sub-step dump (jerboa's approval after r3; shoveler's lines in the checklist at 15:25Z).

r3 found the L0 router-input excess enters through ``attn_out``. This job runs block 0's production
``SnowballAttention`` (``levanter.models.snowball``, ``fp32_activations`` on as in the recipe) with the checkpoint's own
tensors on one H100, batch-1 on the 11 check sequences, against torch batch-1 (``diag-l0-ops-r1/check-ops.pt``).

Sub-steps are read with taps, not a re-implementation: the helpers ``SnowballAttention.__call__`` calls through
``levanter.models.snowball``'s globals (``rearrange`` for the q/k/v projections, ``rms_norm``,
``apply_rotary_embedding``, ``attention``, ``align_kv_heads``, ``sigmoid_fp32`` and ``jnp.reshape``) are wrapped, so
the module's own code runs unchanged and each tap records the helper's output.

- ``isolated``: every tap also returns torch's value for its step, so each step sees torch's input. The q/k/v
  projections see torch's ``L0/attn_norm``; ``o_proj`` sees torch's ``L0/attn_gated`` (its output is ``L0/attn_out``).
  The head gate is the gate matmul and the sigmoid together (torch does not dump the gate logits). ``L0/attn_gated``
  is the aligned-v (XSA) correction and the gate multiply on torch's core, v and gate.
- ``isolated_xsa``: ``isolated`` with the sigmoid tap returning 0.5, so the module's gate is exactly 1 and the tap
  before ``o_proj`` reads the XSA correction alone.
- ``core/<arm>``: the attention core on torch's ``q_scaled``, ``k_rope`` and ``v_proj``: ``selected`` (the
  implementation the module passes on this backend), ``reference`` and ``xla_flash`` (production ``attention``), and
  ``torch_order`` (a JAX transcription of torch's eager core, grug_moe.py ``_eager_attention``: FP32 scores from
  upcast q and k, FP32 softmax, weights rounded to BF16, BF16 PV; diagnostic only).
- ``chain``: the taps record without substituting, from torch's ``L0/attn_norm``: the errors as they accumulate.
- ``composite/<implementation>``: the untapped module in one program from torch's ``L0/attn_norm``, with the
  production selection and with ``reference`` / ``xla_flash`` set in the config; ``composite_vs_chain`` compares the
  production composite with ``chain`` at ``L0/attn_out``.
- ``levanter_replay``: the production composite from the fp32act arm's own ``attn_norm/0`` (``levanter-dumps.pt``),
  against the arm's ``attn_out/0`` and against torch.

Recorded: the backend, device, compute capability, the implementation the module selected and its FA4 kernel
config; per path (module, gpu_fa4_cute, reference) where the 1.75 scale and 1/sqrt(d) are applied, where P is rounded
and the rope cache dtype, by source line; the reference and torch_order cores' traced dtypes; the rope cache dtype as
computed; xla_flash's block size (None means it ran the reference path); and the production composite's optimized
HLO with its custom calls. The whole set runs twice, in two processes: with XLA's defaults, and with
``--xla_allow_excess_precision=false``.

One H100 on cw-rno2a; nothing is trained or saved as a model.
"""

import argparse
import collections
import contextlib
import dataclasses
import hashlib
import importlib
import io
import json
import logging
import math
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
from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.cluster.types import Entrypoint, EnvironmentSpec
from levanter.cutlass_kernel_cache import gpu_compute_capability
from levanter.grug.attention import AttentionMask
from levanter.grug.attention._core import _rotary_cache
from levanter.grug.attention._xla_flash import _flash_block_size
from levanter.grug.sharding import compact_grug_mesh
from levanter.models import snowball
from levanter.models.snowball import GrugMoeHfConfig, SnowballAttention, SnowballConfig, long_attention_layer_mask
from marin.merging.checkpoint import CheckpointReader, CheckpointSource
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.secrets import SecretBehavior
from rigging.timing import Duration

from experiments.interp.snowball_identity.levanter_l0_ops import (
    CHECK_OPS,
    CHECK_OPS_SHA256,
    LEVANTER_DUMPS,
    LEVANTER_DUMPS_SHA256,
    RESOURCES,
    XLA_SETTINGS,
    compare,
    jaxpr_dtypes,
)
from experiments.interp.snowball_identity.logit_check import read
from experiments.interp.snowball_identity.moe_drop_diagnostic import T310
from experiments.interp.snowball_identity.precision_gate import write_json
from experiments.post_training.identity_context_distill.lr0_probe import PRIORITY_BANDS, check_pod_storage
from experiments.post_training.identity_context_distill.sft import BASE_MODEL

logger = logging.getLogger(__name__)

MODULE = "experiments.interp.snowball_identity.levanter_l0_attn"
PROJECTIONS = ("L0/q_proj", "L0/k_proj", "L0/v_proj")
NORMS = ("L0/q_norm", "L0/k_norm")
CORE_ARMS = ("selected", "reference", "xla_flash", "torch_order")
COMPOSITE_IMPLEMENTATIONS = {"selected": None, "reference": "reference", "xla_flash": "xla_flash"}
SOURCE_FACTS = {
    "module": {
        "qk_mult": "BF16 Q pre-multiply in the module, after rope, before the core (snowball.py:500)",
        "rope_cache": "cos/sin in FP32, rotation in FP32 on the upcast half, one cast to BF16 (_core.py:184-208)",
        "selection": "attention_implementation None on GPU passes gpu_fa4_cute (snowball.py:507-509)",
    },
    "gpu_fa4_cute": {
        "kernel": (
            "SegmentedFlashAttentionForwardAmpere (_fa4_cute_kernels.py:206, built at :926) unless "
            "kernel_config.sm100_forward is set (_fa4_cute_backend.py:114)"
        ),
        "accumulators": "MmaF16BF16Op(input dtype, Float32 accumulator) for QK and PV (_fa4_cute_kernels.py:348)",
        "inv_sqrt_d": (
            "inside the softmax: exp2 with 1/sqrt(d) folded into log2 (_fa4_cute.py:376; "
            "_fa4_cute_kernels.py:843-852)"
        ),
        "p_rounding": (
            "online exponential tiles exp(s - row_max), unnormalized, rounded to BF16 before PV "
            "(:743-744); row_sum accumulates the FP32 values (:848-855); deferred normalization acc_O * "
            "rcp_approx(row_sum) in FP32 (:889-890), then one cast to BF16 (:614-615)"
        ),
    },
    "reference": {
        "inv_sqrt_d": "BF16 Q pre-multiply q * (1/sqrt(d)) before the QK einsum (_core.py:240-241)",
        "scores": "BF16 QK einsum, then cast to FP32 (_core.py:241, :271)",
        "p_rounding": "fully normalized FP32 softmax cast to BF16 before PV (_core.py:272-273)",
    },
}
"""Per path, where the scales are applied, where P is rounded and the rope cache dtype, by source line at this
commit (the kernel's internals are not observable at runtime; the reference path's dtypes are also traced)."""


def load_attention(weights: str) -> SnowballAttention:
    """Block 0's attention with the checkpoint's BF16 tensors and its config, ``fp32_activations`` on as in the
    recipe and the attention implementation left to the module (production)."""
    config = json.loads(read(prefix_join(weights, "config.json")))
    cfg = dataclasses.replace(
        SnowballConfig.from_hf_config(GrugMoeHfConfig(**config)), fp32_activations=True, attention_implementation=None
    )
    if bool(long_attention_layer_mask(cfg.num_layers)[0]):
        raise ValueError("layer 0 is a long (NoPE) layer; torch's check-ops has its rope steps")
    reader = CheckpointReader(CheckpointSource(path=weights, revision=""))

    def tensor(name: str) -> jax.Array:
        value = reader.tensor(f"model.layers.0.self_attn.{name}.weight")
        if value.dtype != torch.bfloat16:
            raise ValueError(f"{name} is {value.dtype}, expected bfloat16")
        return jnp.asarray(value.float().numpy(), dtype=jnp.bfloat16).T

    return SnowballAttention(
        w_q=tensor("q_proj"),
        w_k=tensor("k_proj"),
        w_v=tensor("v_proj"),
        w_o=tensor("o_proj"),
        attn_gate=tensor("attn_gate"),
        cfg=cfg,
    )


def short_mask(cfg: SnowballConfig) -> AttentionMask:
    """Block 0's mask, as ``SnowballTransformer._forward`` builds it for short layers."""
    return AttentionMask(is_causal=True, sliding_window=cfg.sliding_window, segment_ids=None)


def torch_order_core(q: jax.Array, k: jax.Array, v: jax.Array, sliding_window: int | None) -> jax.Array:
    """torch's eager core (grug_moe.py ``_eager_attention``) in JAX: FP32 scores from upcast q and k with the
    1/sqrt(d) scale in FP32, FP32 softmax, weights rounded to BF16, then a BF16 PV einsum. Diagnostic only."""
    repeats = q.shape[2] // k.shape[2]
    k, v = jnp.repeat(k, repeats, axis=2), jnp.repeat(v, repeats, axis=2)
    scores = jnp.einsum(
        "bqhd,bkhd->bhqk",
        q.astype(jnp.float32) / math.sqrt(q.shape[-1]),
        k.astype(jnp.float32),
        precision=jax.lax.Precision.HIGHEST,
    )
    query_pos = jnp.arange(q.shape[1])[:, None]
    key_pos = jnp.arange(k.shape[1])[None, :]
    allowed = key_pos <= query_pos
    if sliding_window is not None:
        allowed = allowed & (key_pos >= query_pos - (sliding_window - 1))
    scores = jnp.where(allowed[None, None], scores, jnp.float32(-1e9))
    weights = jax.nn.softmax(scores, axis=-1).astype(v.dtype)
    return jnp.einsum("bhqk,bkhd->bqhd", weights, v)


class _Numpy:
    """``jax.numpy`` for ``levanter.models.snowball`` with ``reshape`` routed through the taps."""

    def __init__(self, taps: "Taps"):
        self._taps = taps

    def __getattr__(self, name):
        return getattr(jnp, name)

    def reshape(self, x, shape, **kwargs):
        return jnp.reshape(self._taps.before_o_proj(x), shape, **kwargs)


class Taps:
    """Wraps the helpers ``SnowballAttention.__call__`` reaches through ``levanter.models.snowball``'s globals.

    Each records its output under torch's op name in ``found``. With ``given`` (op name -> torch's value), each also
    returns torch's value, so the next step sees torch's input. ``gate_one``: the sigmoid returns 0.5, so the gate is
    exactly 1 and ``found`` holds the XSA correction alone (``L0/attn_xsa``). The attention tap also runs ``cores``:
    the core on torch's inputs per ``CORE_ARMS`` (only with ``given``).
    """

    def __init__(self, given: dict | None = None, gate_one: bool = False):
        self.given = given
        self.gate_one = gate_one
        self.found: dict[str, jax.Array] = {}
        self.cores: dict[str, jax.Array] = {}
        self.implementation = None
        self._calls: collections.Counter = collections.Counter()
        self._originals = {
            name: getattr(snowball, name)
            for name in ("rearrange", "rms_norm", "apply_rotary_embedding", "attention", "align_kv_heads")
        }
        self._originals["sigmoid_fp32"] = snowball.sigmoid_fp32

    def _take(self, name: str, ours: jax.Array) -> jax.Array:
        self.found[name] = ours
        return ours if self.given is None else self.given[name]

    def _next(self, helper: str, names: tuple[str, ...]) -> str:
        index = self._calls[helper]
        self._calls[helper] += 1
        return names[index]

    def rearrange(self, x, pattern, **kwargs):
        return self._take(self._next("rearrange", PROJECTIONS), self._originals["rearrange"](x, pattern, **kwargs))

    def rms_norm(self, x):
        return self._take(self._next("rms_norm", NORMS), self._originals["rms_norm"](x))

    def apply_rotary_embedding(self, q, k, **kwargs):
        q_rot, k_rot = self._originals["apply_rotary_embedding"](q, k, **kwargs)
        return self._take("L0/q_rope_rotated", q_rot), self._take("L0/k_rope_rotated", k_rot)

    def attention(self, q, k, v, mask, *, implementation=None):
        self.implementation = implementation
        self.found["L0/q_scaled"] = q
        core = self._originals["attention"]
        if self.given is None:
            return core(q, k, v, mask, implementation=implementation)
        q, k, v = (self.given[name] for name in ("L0/q_scaled", "L0/k_rope", "L0/v_proj"))
        self.cores = {
            "selected": core(q, k, v, mask, implementation=implementation),
            "reference": core(q, k, v, mask, implementation="reference"),
            "xla_flash": core(q, k, v, mask, implementation="xla_flash"),
            "torch_order": torch_order_core(q, k, v, mask.sliding_window),
        }
        return self.given["L0/attn_core"]

    def align_kv_heads(self, x, *, num_q_heads):
        self.found["L0/aligned_v"] = aligned = self._originals["align_kv_heads"](x, num_q_heads=num_q_heads)
        return aligned

    def sigmoid_fp32(self, x):
        gate = self._originals["sigmoid_fp32"](x)
        self.found["L0/attn_head_gate"] = 2 * gate[..., None]
        if self.gate_one:
            return jnp.full_like(gate, 0.5)
        return gate if self.given is None else self.given["L0/attn_head_gate"][..., 0] / 2

    def before_o_proj(self, x):
        if x.ndim != 4:
            raise ValueError(f"unexpected reshape of a rank-{x.ndim} array inside the attention")
        if self.gate_one:
            self.found["L0/attn_xsa"] = x
            return x if self.given is None else self.given["L0/attn_gated"]
        return self._take("L0/attn_gated", x)

    @contextlib.contextmanager
    def installed(self):
        patched = {name: getattr(self, name) for name in self._originals} | {"jnp": _Numpy(self)}
        saved = {name: getattr(snowball, name) for name in patched}
        for name, value in patched.items():
            setattr(snowball, name, value)
        try:
            yield self
        finally:
            for name, value in saved.items():
                setattr(snowball, name, value)


def tapped(attn: SnowballAttention, x: jax.Array, given: dict | None, gate_one: bool = False) -> dict:
    """The module on ``x`` under the taps in one program: every step's output, ``L0/attn_out``, the cores (with
    ``given``) under ``core/<arm>``, and the implementation the module passed."""

    def run(attn, x, given):
        taps = Taps(given, gate_one)
        with taps.installed():
            out = attn(x, short_mask(attn.cfg), disable_rope=False)
        run.implementation = taps.implementation
        return taps.found | {"L0/attn_out": out} | {f"core/{arm}": core for arm, core in taps.cores.items()}

    outputs = jax.jit(run)(attn, x, given)
    return outputs | {"implementation": run.implementation}


def composite(attn: SnowballAttention, x: jax.Array, implementation: str | None) -> jax.Array:
    """The untapped module in one program, with ``implementation`` set in its config (None: production's choice)."""
    module = dataclasses.replace(attn, cfg=dataclasses.replace(attn.cfg, attention_implementation=implementation))
    return jax.jit(lambda m, x: m(x, short_mask(m.cfg), disable_rope=False))(module, x)


def extended_reference(reference: dict, attn: SnowballAttention) -> dict:
    """torch's ops plus the views the taps read: the rotated half of q/k rope and the GQA-expanded v."""
    half = attn.cfg.inferred_head_dim // 2
    repeats = attn.cfg.num_heads // attn.cfg.num_kv_heads
    return {
        sid: (
            ops
            | {
                "L0/q_rope_rotated": ops["L0/q_rope"][..., :half],
                "L0/k_rope_rotated": ops["L0/k_rope"][..., :half],
                "L0/aligned_v": np.repeat(ops["L0/v_proj"], repeats, axis=-2),
            }
        )
        for sid, ops in reference.items()
    }


def host(ops: dict) -> dict[str, np.ndarray]:
    return {name: np.asarray(value.astype(jnp.float32))[0] for name, value in ops.items()}


def phase(root: str, setting: str) -> None:
    """One process: every mode on the check sequences, under this process's XLA flags."""
    started = time.time()
    reference = torch.load(io.BytesIO(read(CHECK_OPS)), map_location="cpu", weights_only=False)
    reference = {sid: {name: value.numpy() for name, value in ops.items()} for sid, ops in reference.items()}
    levanter_bytes = read(LEVANTER_DUMPS)
    if hashlib.sha256(levanter_bytes).hexdigest() != LEVANTER_DUMPS_SHA256:
        raise ValueError(f"{LEVANTER_DUMPS}: sha256 differs from the pinned {LEVANTER_DUMPS_SHA256}")
    levanter = torch.load(io.BytesIO(levanter_bytes), map_location="cpu", weights_only=False)
    result: dict = {"xla_flags": os.environ.get("XLA_FLAGS"), "backend": jax.default_backend()}
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        result |= measure(load_attention(BASE_MODEL), reference, levanter, root, setting)
    result["seconds"] = round(time.time() - started, 1)
    write_json(prefix_join(root, f"{setting}.json"), result)


def measure(attn: SnowballAttention, reference: dict, levanter: dict, root: str, setting: str) -> dict:
    """Every mode on the check sequences against torch's ops (``reference``: sid -> op -> array without the batch),
    the arm's own ``levanter`` dumps, and the records."""
    reference = extended_reference(reference, attn)
    found: dict[str, dict] = collections.defaultdict(dict)
    implementations = set()
    for sid, ops in reference.items():
        given = {name: jnp.asarray(value[None], dtype=jnp.bfloat16) for name, value in ops.items()}
        x = given["L0/attn_norm"]
        isolated = tapped(attn, x, given)
        implementations.add(isolated.pop("implementation"))
        for arm in CORE_ARMS:
            found[f"core/{arm}"][sid] = host({"L0/attn_core": isolated.pop(f"core/{arm}")})
        found["isolated"][sid] = host(isolated)
        xsa = tapped(attn, x, given, gate_one=True)
        found["isolated_xsa"][sid] = host({"L0/attn_xsa": xsa["L0/attn_xsa"]})
        chain = tapped(attn, x, None)
        implementations.add(chain.pop("implementation"))
        found["chain"][sid] = host(chain)
        for name, implementation in COMPOSITE_IMPLEMENTATIONS.items():
            found[f"composite/{name}"][sid] = host({"L0/attn_out": composite(attn, x, implementation)})
        own = jnp.asarray(np.asarray(levanter[sid]["attn_norm/0"])[None], dtype=jnp.bfloat16)
        found["levanter_replay"][sid] = host({"L0/attn_out": composite(attn, own, None)})
    result = {mode: compare(by_sequence, reference) for mode, by_sequence in found.items()}
    result["composite_vs_chain"] = compare(
        found["composite/selected"], {sid: {"L0/attn_out": o["L0/attn_out"]} for sid, o in found["chain"].items()}
    )
    result["levanter_replay_vs_arm"] = compare(
        found["levanter_replay"], {sid: {"L0/attn_out": np.asarray(levanter[sid]["attn_out/0"])} for sid in reference}
    )
    result["levanter_attn_norm_vs_torch"] = compare(
        {sid: {"L0/attn_norm": np.asarray(levanter[sid]["attn_norm/0"])} for sid in reference}, reference
    )
    result["records"] = records(attn, reference, implementations, root, setting)
    return result


def records(attn: SnowballAttention, reference: dict, implementations: set, root: str, setting: str) -> dict:
    """What ran: backend, device, the module's selected implementation, its kernel facts, traced dtypes, the rope
    cache dtype, the scale, xla_flash's block sizes and the production composite's HLO custom calls."""
    cfg = attn.cfg
    sid = sorted(reference)[0]
    ops = {name: jnp.asarray(value[None], dtype=jnp.bfloat16) for name, value in reference[sid].items()}
    q, k, v = ops["L0/q_scaled"], ops["L0/k_rope"], ops["L0/v_proj"]
    mask = short_mask(cfg)
    found: dict = {
        "device_kind": jax.devices()[0].device_kind,
        "selected_implementation": sorted(str(i) for i in implementations),
        "qk_mult": cfg.qk_mult,
        "source_facts": SOURCE_FACTS,
        "fp32_activations": cfg.fp32_activations,
        "sliding_window": cfg.sliding_window,
        "rope_cache_dtype": sorted(
            {str(t.dtype) for t in _rotary_cache(q.shape[1], cfg.inferred_head_dim // 2, cfg.rope)}
        ),
        "xla_flash_block_size": {
            s: _flash_block_size(len(o["L0/attn_norm"]), len(o["L0/attn_norm"]), requested=None)
            for s, o in reference.items()
        },
        "reference_core_dtypes": jaxpr_dtypes(
            lambda q, k, v: snowball.attention(q, k, v, mask, implementation="reference"), q, k, v
        ),
        "torch_order_core_dtypes": jaxpr_dtypes(lambda q, k, v: torch_order_core(q, k, v, cfg.sliding_window), q, k, v),
    }
    if jax.default_backend() == "gpu":
        from levanter.grug.attention._fa4_cute import _segmented_kernel_config  # noqa: PLC0415

        found["compute_capability"] = gpu_compute_capability()
        found["fa4_kernel_config"] = dataclasses.asdict(_segmented_kernel_config(cfg.inferred_head_dim))
    x = ops["L0/attn_norm"]
    hlo = jax.jit(lambda m, x: m(x, short_mask(m.cfg), disable_rope=False)).lower(attn, x).compile().as_text()
    StoragePath(prefix_join(root, f"{setting}-attn-composite-hlo.txt")).write_bytes(hlo.encode())
    found["attn_composite_hlo_sha256"] = hashlib.sha256(hlo.encode()).hexdigest()
    found["attn_composite_custom_calls"] = sorted(
        {
            line.split('custom_call_target="')[1].split('"')[0]
            for line in hlo.splitlines()
            if "custom_call_target" in line
        }
    )
    return found


def l0_attn_worker(root: str) -> None:
    check_pod_storage()
    started = time.time()
    check_ops = read(CHECK_OPS)
    receipt: dict = {"check_ops": CHECK_OPS, "check_ops_sha256": hashlib.sha256(check_ops).hexdigest(), "seconds": {}}
    if receipt["check_ops_sha256"] != CHECK_OPS_SHA256:
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
    return Entrypoint.from_callable(importlib.import_module(MODULE).l0_attn_worker, root)


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
            "isolated (production SnowballAttention with taps; every step fed torch's input; o_proj on torch's "
            "attn_gated)",
            "isolated_xsa (isolated with the gate held at exactly 1: the XSA correction alone)",
            "core/{selected,reference,xla_flash,torch_order} (attention core on torch's q_scaled, k_rope, v_proj)",
            "chain (taps record only, from torch's attn_norm)",
            "composite/{selected,reference,xla_flash} (untapped module, one program, from torch's attn_norm)",
            "composite_vs_chain (at L0/attn_out)",
            "levanter_replay (production composite from the fp32act arm's attn_norm/0, vs the arm's attn_out/0 "
            "and vs torch)",
            "records: backend, device, compute capability, selected implementation, FA4 kernel config and source "
            "dtype facts, reference and torch_order core dtypes, rope cache dtype, qk_mult, xla_flash block sizes, "
            "composite HLO custom calls",
        ],
        "grid": "batch-1 on the 11 check sequences",
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

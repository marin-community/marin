# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""#310 G1 failure path (jerboa's ruling 3): torch's own per-op error at block 0 when only the batch shape changes.

G1 fails at L0, which comes before any MoE: Levanter's L0 route disagreement rate is 2.77x torch's batched-vs-batch-1
rate. ``moe_drop_diagnostic``'s ``sublayers_vs_torch`` gives the Levanter-vs-torch difference per op; this job gives
torch-vs-torch for the same ops and their sub-steps, so the op whose Levanter excess exceeds torch's own reshaping
error is named by numbers.

The vendored torch GrugMoE (eager attention), embed and layer 0 only, on one H100, TF32 off. Block 0 runs step by
step as ``GrugMoeGatedNorm``, ``GrugMoeAttention`` (eager) and ``GrugMoeDecoderLayer`` order it; each composite
step is checked bit for bit against the module's own output. Two runs:

(a) ``batched``: the gate's grids (``moe_drop_diagnostic.gate_batches``), as G1's floor.
(b) ``batch1``: each sequence alone, as ``routes.pt`` was made.

Per op, over every real position of every sequence (and of the ``check`` group alone, the positions
``sublayers_vs_torch`` uses): batched vs batch-1 as ``moe_drop_diagnostic.op_diff`` measures it. The batch-1 ops of
the ``check`` sequences are saved (``<root>/check-ops.pt``) and the named ones compared with the fix-r1 rerun's
``torch-dumps.pt``.
"""

import argparse
import hashlib
import importlib
import io
import json
import logging
import os
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from fray.iris_backend import convert_constraints, convert_resources
from fray.types import ResourceConfig
from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.cluster.types import Entrypoint, EnvironmentSpec
from marin.merging.checkpoint import CheckpointReader, CheckpointSource
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.secrets import SecretBehavior
from rigging.timing import Duration

from experiments.interp.snowball_identity.harness import import_grug
from experiments.interp.snowball_identity.logit_check import read, stored_sequences
from experiments.interp.snowball_identity.moe_drop_diagnostic import (
    BATCH,
    INPUTS,
    LAYERS,
    PAD_MULTIPLE,
    T310,
    gate_batches,
)
from experiments.interp.snowball_identity.precision_gate import write_json
from experiments.interp.snowball_identity.routing import MATMUL_ENV
from experiments.interp.snowball_identity.sequences import Sequence
from experiments.post_training.identity_context_distill.lr0_probe import PRIORITY_BANDS, check_pod_storage
from experiments.post_training.identity_context_distill.sft import BASE_MODEL

logger = logging.getLogger(__name__)

MODULE = "experiments.interp.snowball_identity.torch_l0_ops"
RERUN_DUMPS = f"{T310}/diag-moe-fix-r1/torch-dumps.pt"
RERUN_NAMES = {
    "embed": "embed",
    "L0/attn_norm": "attn_norm/0",
    "L0/attn_out": "attn_out/0",
    "L0/mlp_norm": "mlp_norm/0",
    "L0/router_logits": "router_logits/0",
}
"""This job's op names that ``sublayers_vs_torch`` also measures, and their keys in the rerun's ``torch-dumps.pt``."""


def gated_norm_steps(module, x: torch.Tensor, prefix: str, ops: dict) -> torch.Tensor:
    """``GrugMoeGatedNorm.forward`` step by step."""
    down = module.down_proj(x)
    silu = F.silu(down)
    up = module.up_proj(silu)
    gate = torch.sigmoid(up)
    out = x * gate.to(x.dtype)
    if not torch.equal(out, module(x)):
        raise RuntimeError(f"{prefix}: step-by-step output differs from the module")
    ops |= {f"{prefix}/down": down, f"{prefix}/silu": silu, f"{prefix}/up": up, f"{prefix}/sigmoid": gate}
    return out


def attention_steps(attention, x: torch.Tensor, positions: torch.Tensor, is_long: bool, ops: dict) -> torch.Tensor:
    """``GrugMoeAttention.forward`` with eager attention, step by step."""
    grug = import_grug()
    cfg = attention.config
    batch, length, _ = x.shape
    q = attention.q_proj(x).view(batch, length, cfg.num_attention_heads, cfg.head_dim)
    k = attention.k_proj(x).view(batch, length, cfg.num_key_value_heads, cfg.head_dim)
    v = attention.v_proj(x).view(batch, length, cfg.num_key_value_heads, cfg.head_dim)
    ops |= {"L0/q_proj": q, "L0/k_proj": k, "L0/v_proj": v}
    q, k = grug.grug_rms_norm_no_weight(q), grug.grug_rms_norm_no_weight(k)
    ops |= {"L0/q_norm": q, "L0/k_norm": k}
    if not is_long:
        q, k = grug._apply_half_rope(q, k, positions, cfg.rope_theta)
        ops |= {"L0/q_rope": q, "L0/k_rope": k}
    q = q * (cfg.qk_mult * (cfg.qk_mult_long_scale if is_long else 1.0))
    ops["L0/q_scaled"] = q
    core, v_xsa = attention._eager_attention(q, k, v, None, is_long=is_long)
    ops["L0/attn_core"] = core
    dot = (core * v_xsa).sum(dim=-1, keepdim=True)
    xsa = core - (dot / (v_xsa.square().sum(dim=-1, keepdim=True) + grug.GRUG_XSA_EPS)) * v_xsa
    gate = grug.GRUG_ATTN_GATE_SCALE * torch.sigmoid(attention.attn_gate(x)).unsqueeze(-1)
    gated = xsa * gate.to(xsa.dtype)
    ops |= {"L0/attn_xsa": xsa, "L0/attn_head_gate": gate, "L0/attn_gated": gated}
    out = attention.o_proj(gated.reshape(batch, length, -1))
    if not torch.equal(out, attention(x, None, positions, is_long=is_long)):
        raise RuntimeError("L0 attention: step-by-step output differs from the module")
    return out


def l0_ops(inner, ids: torch.Tensor, is_long: bool) -> dict[str, torch.Tensor]:
    """Embed and block 0 up to the router logits, every step; ``ids`` [B, S]. Every value is [B, S, ...]."""
    ops: dict[str, torch.Tensor] = {}
    positions = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
    embedded = inner.embed_tokens(ids)
    normed = inner.embed_norm(embedded)
    ops |= {"embed_tokens": embedded, "embed_rmsnorm": normed}
    hidden = gated_norm_steps(inner.embed_gated_norm, normed, "embed_gated", ops)
    ops["embed"] = hidden
    block = inner.layers[0]
    attn_rms = block.input_layernorm(hidden)
    ops["L0/attn_rmsnorm"] = attn_rms
    attn_in = gated_norm_steps(block.attn_gated_norm, attn_rms, "L0/attn_gated_norm", ops)
    ops["L0/attn_norm"] = attn_in
    attn_out = attention_steps(block.self_attn, attn_in, positions, is_long, ops)
    ops["L0/attn_out"] = attn_out
    resid = hidden + attn_out
    mlp_rms = block.post_attention_layernorm(resid)
    ops |= {"L0/resid_attn": resid, "L0/mlp_rmsnorm": mlp_rms}
    router_in = gated_norm_steps(block.mlp_gated_norm, mlp_rms, "L0/mlp_gated_norm", ops)
    ops["L0/mlp_norm"] = router_in
    logits = block.mlp.router(router_in.reshape(-1, router_in.shape[-1])).router_logits
    ops["L0/router_logits"] = logits.reshape(*router_in.shape[:2], -1)
    return ops


class OpDiff:
    """``moe_drop_diagnostic.op_diff`` accumulated over sequences: max and RMS |ours - ref|, max |ref|."""

    def __init__(self):
        self.max_abs = 0.0
        self.sum_sq = 0.0
        self.count = 0
        self.ref_max_abs = 0.0

    def add(self, ours: torch.Tensor, ref: torch.Tensor) -> None:
        diff = (ours.double() - ref.double()).abs()
        self.max_abs = max(self.max_abs, float(diff.max()))
        self.sum_sq += float(diff.square().sum())
        self.count += diff.numel()
        self.ref_max_abs = max(self.ref_max_abs, float(ref.double().abs().max()))

    def report(self) -> dict:
        scale = self.ref_max_abs
        return {
            "max_abs": self.max_abs,
            "rms": (self.sum_sq / self.count) ** 0.5,
            "ref_max_abs": scale,
            "max_in_bf16_steps": self.max_abs / (scale * 2.0**-8) if scale else float("inf"),
            "elements": self.count,
        }


def load_l0(weights: str, device: torch.device):
    """The vendored GrugMoE cut to one layer, with the checkpoint's own tensors (BF16, FP32 router bias)."""
    grug = import_grug()
    source = json.loads(read(prefix_join(weights, "config.json")))
    aliases = [key for key in ("num_hidden_layers", "num_layers") if key in source]
    if not aliases or any(source[key] != LAYERS for key in aliases):
        raise ValueError(f"expected {LAYERS} layers, got {[source.get(key) for key in aliases]}")
    config = grug.GrugMoeConfig(**(source | {key: 1 for key in aliases}))
    config._attn_implementation = "eager"
    with torch.device("meta"):
        model = grug.GrugMoeForCausalLM(config).eval().requires_grad_(False)
    reader = CheckpointReader(CheckpointSource(path=weights, revision=""))
    expected = model.state_dict()
    missing = set(expected) - set(reader.weight_map)
    if missing:
        raise ValueError(f"checkpoint lacks {sorted(missing)[:5]}")
    model.load_state_dict({name: reader.tensor(name).to(device) for name in expected}, strict=True, assign=True)
    # Layer 0's attention kind in the 26-layer schedule.
    return model.model, grug.grug_long_layer_flags(LAYERS)[0]


def compare_ops(inner, is_long: bool, sequences: list[Sequence], device: torch.device) -> tuple[dict, dict]:
    """Batched vs batch-1 per op over every real position (``all``) and the ``check`` group alone; and the
    batch-1 ops of the check sequences (float32, [T, ...])."""
    order = sorted(range(len(sequences)), key=lambda i: len(sequences[i].tokens))
    stats: dict = defaultdict(lambda: defaultdict(OpDiff))
    check_ops: dict = {}
    with torch.no_grad():
        for step, grid in enumerate(gate_batches(sequences, BATCH, PAD_MULTIPLE)):
            batched = l0_ops(inner, torch.as_tensor(grid, dtype=torch.long, device=device), is_long)
            for row, index in enumerate(order[step * BATCH : (step + 1) * BATCH]):
                sequence = sequences[index]
                n = len(sequence.tokens)
                if grid[row, :n].tolist() != list(sequence.tokens):
                    raise RuntimeError(f"{sequence.id}: grid row {row} of step {step} holds other tokens")
                alone = l0_ops(inner, torch.tensor([list(sequence.tokens)], device=device), is_long)
                if set(alone) != set(batched):
                    raise RuntimeError("the two runs produced different op sets")
                groups = ("all", "check") if sequence.group == "check" else ("all",)
                for name, value in alone.items():
                    for group in groups:
                        stats[group][name].add(batched[name][row, :n], value[0])
                if sequence.group == "check":
                    check_ops[sequence.id] = {name: value[0].float().cpu() for name, value in alone.items()}
    return {group: {name: d.report() for name, d in ops.items()} for group, ops in stats.items()}, check_ops


def rerun_equality(check_ops: dict, rerun: dict) -> dict:
    """Whether this job's batch-1 ops equal the fix-r1 rerun's torch dumps of the same check sequences, bit for bit."""
    return {
        name: all(np.array_equal(check_ops[sid][name].numpy(), np.asarray(rerun[sid][key])) for sid in check_ops)
        for name, key in RERUN_NAMES.items()
    }


def l0_worker(root: str) -> None:
    check_pod_storage()
    started = time.time()
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    positions = read(INPUTS.positions)
    sequences = stored_sequences(INPUTS.positions)
    receipt: dict = {
        "weights": BASE_MODEL,
        "positions": INPUTS.positions,
        "positions_sha256": hashlib.sha256(positions).hexdigest(),
        "grids": {"batch": BATCH, "pad_multiple": PAD_MULTIPLE},
        "matmul": {
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(device),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            "allow_fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            "env": {name: os.environ.get(name) for name in MATMUL_ENV},
        },
        "seconds": {},
    }
    load_started = time.time()
    inner, is_long = load_l0(BASE_MODEL, device)
    receipt["layer0_is_long"] = bool(is_long)
    receipt["seconds"]["load"] = round(time.time() - load_started, 1)
    run_started = time.time()
    receipt["batched_vs_batch1"], check_ops = compare_ops(inner, is_long, sequences, device)
    receipt["seconds"]["ops"] = round(time.time() - run_started, 1)
    buffer = io.BytesIO()
    torch.save(check_ops, buffer)
    StoragePath(prefix_join(root, "check-ops.pt")).write_bytes(buffer.getvalue())
    rerun_bytes = read(RERUN_DUMPS)
    rerun = torch.load(io.BytesIO(rerun_bytes), map_location="cpu", weights_only=False)
    receipt["rerun_torch_dumps"] = {
        "path": RERUN_DUMPS,
        "sha256": hashlib.sha256(rerun_bytes).hexdigest(),
        "batch1_equal": rerun_equality(check_ops, rerun),
    }
    receipt["device_memory"] = {
        "bytes_in_use": torch.cuda.memory_allocated(device),
        "peak_bytes_in_use": torch.cuda.max_memory_allocated(device),
    }
    receipt["seconds"]["total"] = round(time.time() - started, 1)
    write_json(prefix_join(root, "receipt.json"), receipt)


def entrypoint_for(root: str) -> Entrypoint:
    """The worker by module reference (see ``torch_route_floor.entrypoint_for``)."""
    return Entrypoint.from_callable(importlib.import_module(MODULE).l0_worker, root)


RESOURCES = ResourceConfig.with_gpu("H100", count=1, cpu=16, ram="128GB", disk="256GB", target_cluster="cw-rno2a")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--root", required=True, help=f"fresh prefix under {T310}/diag-")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--priority", choices=("interactive", "production"), default="production")
    parser.add_argument("--expect-commit")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout
    if not args.root.startswith(f"{T310}/diag-"):
        raise ValueError(f"root must be a fresh {T310}/diag-* prefix")
    plan = {
        "job": args.job_name,
        "commit": head,
        "clean": not dirty,
        "root": args.root,
        "runs": {
            "batched": f"gate grids: batch {BATCH}, right-padded to a multiple of {PAD_MULTIPLE} (G1's floor)",
            "batch1": "each sequence alone, as routes.pt",
        },
        "model": "vendored torch GrugMoE, embed + layer 0, eager attention, checkpoint dtypes, TF32 off",
        "weights": BASE_MODEL,
        "positions": INPUTS.positions,
        "rerun_torch_dumps": RERUN_DUMPS,
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
    if StoragePath(args.root).exists():
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

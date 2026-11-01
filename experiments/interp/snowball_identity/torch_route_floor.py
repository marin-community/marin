# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Torch-vs-torch route floor for the G1 route check: how often the torch reference disagrees with itself.

The gate's route row compares Levanter routes with the torch reference ``routes.pt`` (``routing.router_rounding``:
the vendored torch GrugMoE, eager attention and experts, one sequence at a time, FP32 router dot and bias). The
torch reference itself picks different experts at near-ties when only the tensor shapes change (layer 0, position 4
flips with sequence length). This job measures that floor on the same checkpoint and the same 465 sequences, with
the gate's router arithmetic (``GrugMoeRouter.forward``'s own logits, FP32 bias, ``jax_top_k``):

(a) ``batched``: the gate's grids (``moe_drop_diagnostic.gate_batches``: sorted by length, 8 rows, short steps
    filled with copies, right-padded with ``PAD_ID`` to a multiple of 64): the shape change Levanter's gate sees.
    G1's floor.
(b) ``batch1``: each sequence alone, as ``routes.pt`` was made: the run-to-run determinism floor.

Each run saves the top-6 biased router logits and their expert ids at every real position and layer
(``<root>/<run>/top6.pt``: ``{sequence id: {"ids": int16 [T, L, 6], "biased": float32 [T, L, 6]}}``). The
receipt holds per-layer and per-group set and ordered agreement for (a) vs ``routes.pt``, (b) vs ``routes.pt`` and
(a) vs (b), over the same 805,974 position-layers the gate counts, and the distribution of the 4th/5th biased-logit
margin at the disagreeing position-layers (each side's own margin) and over all position-layers.

One H100x8 on cw-rno2a; nothing is trained or saved as a model.
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
from pathlib import Path

import numpy as np
import torch
from fray.iris_backend import convert_constraints, convert_resources
from fray.types import ResourceConfig
from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.cluster.types import Entrypoint, EnvironmentSpec
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.secrets import SecretBehavior
from rigging.timing import Duration

from experiments.interp.snowball_identity.harness import SnowballHarness, import_grug, load_harness
from experiments.interp.snowball_identity.logit_check import read, require_fresh_root, stored_sequences, write_json
from experiments.interp.snowball_identity.moe_drop_diagnostic import BATCH, INPUTS, PAD_MULTIPLE, T310, TOP, gate_batches
from experiments.interp.snowball_identity.route_proxy_check import K, agreement, margin_summary, routes_and_margins
from experiments.interp.snowball_identity.routing import MATMUL_ENV
from experiments.interp.snowball_identity.sequences import Sequence
from experiments.post_training.identity_context_distill.lr0_probe import PRIORITY_BANDS, check_pod_storage
from experiments.post_training.identity_context_distill.sft import BASE_MODEL

logger = logging.getLogger(__name__)

MODULE = "experiments.interp.snowball_identity.torch_route_floor"
RUNS = ("batched", "batch1")


class RouterTaps:
    """Top-6 biased router logits per layer from each router's own output, for the forward in progress."""

    def __init__(self, harness: SnowballHarness):
        grug = import_grug()
        self.found: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

        def tap(layer: int):
            def hook(module, args, output):
                bias = module.bias.detach().to(device=output.router_logits.device, dtype=torch.float32)
                values, ids = grug.jax_top_k(output.router_logits + bias, TOP)
                if not torch.equal(ids[:, :K], output.selected_experts):
                    raise RuntimeError(f"layer {layer}: top-{TOP} prefix differs from the router's selection")
                self.found[layer] = (ids.to(torch.int16).cpu(), values.float().cpu())

            return hook

        self.handles = [block.mlp.router.register_forward_hook(tap(layer)) for layer, block in enumerate(harness.layers)]

    def take(self, batch: int, length: int) -> tuple[np.ndarray, np.ndarray]:
        """[B, S, L, 6] ids and biased logits of the last forward."""
        layers = sorted(self.found)
        ids = torch.stack([self.found[layer][0] for layer in layers], dim=1)  # [B*S, L, 6]
        values = torch.stack([self.found[layer][1] for layer in layers], dim=1)
        self.found.clear()
        shape = (batch, length, len(layers), TOP)
        return ids.reshape(shape).numpy(), values.reshape(shape).numpy()

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()


def run_rows(harness: SnowballHarness, taps: RouterTaps, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    harness.run(grid.tolist(), assistant_start=0, logit_positions=torch.tensor([grid.shape[1] - 1]))
    return taps.take(*grid.shape)


def top6_batched(harness: SnowballHarness, taps: RouterTaps, sequences: list[Sequence]) -> dict:
    """Run (a): the gate's grids; each real sequence's rows are cut back to its own length."""
    order = sorted(range(len(sequences)), key=lambda i: len(sequences[i].tokens))
    found = {}
    for step, grid in enumerate(gate_batches(sequences, BATCH, PAD_MULTIPLE)):
        ids, values = run_rows(harness, taps, grid)
        for row, index in enumerate(order[step * BATCH : (step + 1) * BATCH]):
            sequence = sequences[index]
            if grid[row, : len(sequence.tokens)].tolist() != list(sequence.tokens):
                raise RuntimeError(f"{sequence.id}: grid row {row} of step {step} holds other tokens")
            n = len(sequence.tokens)
            found[sequence.id] = {"ids": torch.from_numpy(ids[row, :n]), "biased": torch.from_numpy(values[row, :n])}
    return found


def top6_batch1(harness: SnowballHarness, taps: RouterTaps, sequences: list[Sequence]) -> dict:
    """Run (b): each sequence alone, unpadded, as ``routes.pt`` was made."""
    found = {}
    for sequence in sequences:
        ids, values = run_rows(harness, taps, np.asarray([sequence.tokens], dtype=np.int64))
        found[sequence.id] = {"ids": torch.from_numpy(ids[0]), "biased": torch.from_numpy(values[0])}
    return found


def floor_report(top6: dict[str, dict], reference: dict[str, np.ndarray]) -> dict:
    """All three comparisons and the all-position margin distribution per run and layer."""
    routes, margins = {}, {}
    for run, found in top6.items():
        routes[run], margins[run] = routes_and_margins(found)
    report = {
        "batched_vs_routes_pt": agreement(routes["batched"], reference, {"batched": margins["batched"]}),
        "batch1_vs_routes_pt": agreement(routes["batch1"], reference, {"batch1": margins["batch1"]}),
        "batched_vs_batch1": agreement(routes["batched"], routes["batch1"], margins),
        "margins_all_positions": {},
    }
    for run, by_sequence in margins.items():
        stacked = np.concatenate([by_sequence[sid] for sid in sorted(by_sequence)])  # [positions, L]
        report["margins_all_positions"][run] = [margin_summary(stacked[:, layer]) for layer in range(stacked.shape[1])]
    return report


def device_memory() -> list[dict]:
    return [
        {
            "bytes_in_use": torch.cuda.memory_allocated(index),
            "peak_bytes_in_use": torch.cuda.max_memory_allocated(index),
        }
        for index in range(torch.cuda.device_count())
    ]


def floor_worker(root: str) -> None:
    check_pod_storage()
    started = time.time()
    positions = read(INPUTS.positions)
    routes_bytes = read(prefix_join(INPUTS.torch_routing, "routes.pt"))
    reference = {sid: routes.numpy() for sid, routes in torch.load(io.BytesIO(routes_bytes), map_location="cpu").items()}
    sequences = stored_sequences(INPUTS.positions)
    torch.backends.cuda.matmul.allow_tf32 = False  # as the torch reference job ran
    receipt: dict = {
        "weights": BASE_MODEL,
        "positions": INPUTS.positions,
        "positions_sha256": hashlib.sha256(positions).hexdigest(),
        "routes_pt": prefix_join(INPUTS.torch_routing, "routes.pt"),
        "routes_pt_sha256": hashlib.sha256(routes_bytes).hexdigest(),
        "grids": {"batch": BATCH, "pad_multiple": PAD_MULTIPLE},
        "matmul": {
            "torch": torch.__version__,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "env": {name: os.environ.get(name) for name in MATMUL_ENV},
        },
        "seconds": {},
    }
    load_started = time.time()
    harness = load_harness(BASE_MODEL)
    receipt["seconds"]["load"] = round(time.time() - load_started, 1)
    taps = RouterTaps(harness)
    top6 = {}
    for run, method in (("batched", top6_batched), ("batch1", top6_batch1)):
        run_started = time.time()
        top6[run] = method(harness, taps, sequences)
        receipt["seconds"][run] = round(time.time() - run_started, 1)
        buffer = io.BytesIO()
        torch.save(top6[run], buffer)
        StoragePath(prefix_join(root, f"{run}/top6.pt")).write_bytes(buffer.getvalue())
        receipt[f"device_memory_after_{run}"] = device_memory()
        write_json(prefix_join(root, "receipt.json"), receipt)
    taps.remove()
    receipt["floor"] = floor_report(top6, reference)
    receipt["seconds"]["total"] = round(time.time() - started, 1)
    write_json(prefix_join(root, "receipt.json"), receipt)


def entrypoint_for(root: str) -> Entrypoint:
    """The worker by module reference: run as ``python -m`` this file is ``__main__``, whose functions cloudpickle
    would copy by value, along with module objects it cannot pickle (``torch.backends.cudnn``)."""
    return Entrypoint.from_callable(importlib.import_module(MODULE).floor_worker, root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--root", required=True, help=f"fresh prefix under {T310}/")
    parser.add_argument("--timeout-seconds", type=int, default=2700)
    parser.add_argument("--priority", choices=("interactive", "production"), default="production")
    parser.add_argument("--expect-commit")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout
    if not args.root.startswith(f"{T310}/diag-"):
        raise ValueError(f"root must be a fresh {T310}/diag-* prefix")
    resources = ResourceConfig.with_gpu("H100", count=8, cpu=64, ram="768GB", disk="1TB", target_cluster="cw-rno2a")
    plan = {
        "job": args.job_name,
        "commit": head,
        "clean": not dirty,
        "root": args.root,
        "runs": {
            "batched": f"gate grids: batch {BATCH}, right-padded to a multiple of {PAD_MULTIPLE} (G1's floor)",
            "batch1": "each sequence alone, as routes.pt (determinism floor)",
        },
        "weights": BASE_MODEL,
        "positions": INPUTS.positions,
        "routes_pt": prefix_join(INPUTS.torch_routing, "routes.pt"),
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
    require_fresh_root(args.root)
    key, secret = os.environ["CW_KEY_ID"], os.environ["CW_KEY_SECRET"]
    secrets = {"CW_KEY_ID": key, "CW_KEY_SECRET": secret, "AWS_ACCESS_KEY_ID": key, "AWS_SECRET_ACCESS_KEY": secret}
    environment = EnvironmentSpec(extras=["gpu"], secrets=secrets, secret_behavior=SecretBehavior.ISOLATE)
    entrypoint = entrypoint_for(args.root)
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

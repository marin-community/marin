# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The #310 pre-update gate: score the trainer's compute view at the pinned positions and stop on any failure.

``make_pre_update_gate(inputs)`` returns the ``pre_update_gate(trainer, state)`` callback of the LR0 precision probe.

Every rank reaches the same collectives in the same order, whatever fails, so a failure on one rank cannot leave
the others waiting:

1. ``inputs``: each rank reads the stored token ids of the pinned #298 ``positions.pt`` (no collective), then an
   all-gather of every rank's status; any failure stops every rank.
2. ``scored``: every rank runs the compute-view forward and the router-dot witness (collective), then an
   all-gather of every rank's per-device memory.
3. ``gated``: process 0 alone saves ``positions.pt``, compares it with the stored vLLM reference and the torch
   reference job's routes, applies ``logit_check.parity_gate``, ``routing.route_gate`` (row-1 exceptions and
   coverage), the route proxy check named by ``inputs.route_policy`` (``G1``: ``route_proxy_check.check`` against
   the pinned torch floor; ``G1a``: ``route_proxy_check.check_vllm_floor`` against the pinned vLLM route capture;
   per-layer rate bound and near-tie margins; under ``report_g1b_l0``, G1b's direct L0 line against torch batched;
   all are computed and reported, only the named one gates) and the
   dropless check, and writes ``gate.json``, all
   inside one ``try``; then the failure count is broadcast and every rank raises
   ``PrecisionGateFailed`` unless it is zero. A process-0 exception, including a failed write, is a failure.

``gate.json`` also records the MoE settings the forward ran with (capacity factor, expert axis, FP32 combine), the
per-layer dispatch drops of valid assignments (sender, receiver, total; the gate fails on any), and each device's
memory after the forward (bytes in use, peak, limit).

Each rank writes ``ranks/<process>.json`` after every stage (best effort). A crash inside a collective cannot be
caught, so a watchdog ends the process with status 4 after ``deadline_seconds``.
"""

import io
import json
import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import haliax as hax
import jax
import numpy as np
import torch
from levanter.compute_dtype import cast_model_for_compute
from rigging.filesystem.storage_path import prefix_join

from experiments.interp.snowball_identity.levanter_positions import (
    PAD_ID,
    positions_records,
    router_dot_dtypes,
    router_dot_is_fp32,
    save_positions,
)
from experiments.interp.snowball_identity.logit_check import (
    candidate_spans,
    compare,
    parity_gate,
    read,
    reference_ids,
    stored_sequences,
    write_json,
)
from experiments.interp.snowball_identity.multihost import broadcast_host_scalar, gather_host_rows, gather_host_scalar
from experiments.interp.snowball_identity.route_proxy_check import (
    FLOOR_RECEIPT_SHA256,
    TORCH_BATCHED_TOP6_SHA256,
    TORCH_TOP6_SHA256,
    VLLM_PASS1_SHA256,
    VLLM_PASS2_SHA256,
    VLLM_RECEIPT_SHA256,
    check,
    check_vllm_floor,
    g1b_l0,
    load_pinned,
)
from experiments.interp.snowball_identity.routing import MATMUL_ENV, compare_routes_cross_stack, route_gate

logger = logging.getLogger(__name__)

DEADLINE_EXIT = 4
ROUTE_POLICIES = ("G1", "G1a", "report_g1b_l0")
"""The routes row's floor: G1 (torch floor, amendment a685d499) or G1a (served-stack floor, 27f282c2). Under
``report_g1b_l0`` both are recorded and neither gates; the routes proxy gate is G1b's direct L0 line
(``route_proxy_check.g1b_l0``)."""

GATED_REPORT = {"G1": "G1", "G1a": "G1a", "report_g1b_l0": "G1b_L0"}
"""The ``route_proxy`` report each policy gates on; its failure rows read ``routes-<report, _ as ->:``."""


class PrecisionGateFailed(RuntimeError):
    pass


@dataclass(frozen=True)
class PrecisionGateInputs:
    positions: str
    """The pinned #298 torch ``positions.pt``: its stored token ids are the inputs."""
    vllm: str
    """The stored vLLM reference JSON for the same sequences."""
    identity: str
    """identity.jsonl, for the contrastive candidate spans."""
    tokenizer_source: str
    """Checkpoint root whose tokenizer files give the spans (the regional source)."""
    torch_routing: str
    """The torch reference job's result directory, holding ``routing.json`` and ``routes.pt``."""
    floor_receipt: str
    """The torch floor job's ``receipt.json`` (G1's floor rates and torch's own margins), pinned by sha256."""
    torch_top6: str
    """The torch floor job's ``batch1/top6.pt`` (torch batch-1 top-6 ids and biased logits), pinned by sha256."""
    torch_batched_top6: str
    """The torch floor job's ``batched/top6.pt`` (torch on the gate grid; G1b's floor), pinned by sha256."""
    route_policy: str
    """``G1``, ``G1a`` or ``report_g1b_l0`` (``ROUTE_POLICIES``): which route proxy check the gate applies; G1, G1a
    and G1b's L0 line are all reported. No default."""
    vllm_capture: str
    """The vLLM route capture directory (``receipt.json``, ``pass-1/`` and ``pass-2/routes.pt``), pinned by sha256."""
    output: str
    """Where to write ``positions.pt``, ``gate.json`` and ``ranks/<process>.json``."""
    batch: int = 8
    """Real sequences per step: the mesh's batch axes."""
    pad_multiple: int = 64
    deadline_seconds: float = 1800
    """Hard limit for the whole gate on each rank."""

    def __post_init__(self):
        if self.route_policy not in ROUTE_POLICIES:
            raise ValueError(f"route_policy must be one of {ROUTE_POLICIES}, got {self.route_policy!r}")


MEMORY_FIELDS = ("bytes_in_use", "peak_bytes_in_use", "bytes_limit")


def local_device_memory() -> np.ndarray:
    """This process's devices: [local_devices, 3] of ``MEMORY_FIELDS``, -1 where the backend reports none."""
    rows = []
    for device in jax.local_devices():
        stats = device.memory_stats() or {}
        rows.append([stats.get(field, -1) for field in MEMORY_FIELDS])
    return np.asarray(rows, dtype=np.int64)


def device_memory_report(gathered: np.ndarray) -> list[dict]:
    """``gathered``: ``gather_host_rows(local_device_memory())``, [processes, local_devices, 3]."""
    return [
        {"process": process, "local_device": device, **dict(zip(MEMORY_FIELDS, map(int, values), strict=True))}
        for process, devices in enumerate(gathered)
        for device, values in enumerate(devices)
    ]


def dispatch_failures(dispatch: dict) -> list[str]:
    """The gate forward must be dropless, as vLLM is: any dropped valid assignment is a failure."""
    return [
        f"layer {layer}: {count} valid assignments dropped ({sender} sender, {receiver} receiver)"
        for layer, (count, sender, receiver) in enumerate(
            zip(dispatch["dropped"], dispatch["sender_dropped"], dispatch["receiver_dropped"], strict=True)
        )
        if count
    ]


def route_proxy(positions_path: str, inputs: PrecisionGateInputs) -> dict:
    """The route proxy checks of a written ``positions.pt``: G1 against the pinned torch floor, G1a against the pinned
    vLLM route capture, and G1b's direct L0 line against torch batched and the capture. The gate applies the one
    ``inputs.route_policy`` names (G1b's for ``report_g1b_l0``); the others are reported."""
    torch_inputs = (
        torch.load(io.BytesIO(read(positions_path)), map_location="cpu"),
        torch.load(io.BytesIO(load_pinned(inputs.torch_top6, TORCH_TOP6_SHA256)), map_location="cpu"),
        json.loads(load_pinned(inputs.floor_receipt, FLOOR_RECEIPT_SHA256)),
    )
    load_pinned(prefix_join(inputs.vllm_capture, "receipt.json"), VLLM_RECEIPT_SHA256)
    passes = [
        torch.load(
            io.BytesIO(load_pinned(prefix_join(inputs.vllm_capture, f"pass-{n}/routes.pt"), sha)), map_location="cpu"
        )
        for n, sha in ((1, VLLM_PASS1_SHA256), (2, VLLM_PASS2_SHA256))
    ]
    batched = torch.load(
        io.BytesIO(load_pinned(inputs.torch_batched_top6, TORCH_BATCHED_TOP6_SHA256)), map_location="cpu"
    )
    return {
        "policy": inputs.route_policy,
        "G1": check(*torch_inputs),
        "G1a": check_vllm_floor(*torch_inputs, *passes),
        "G1b_L0": g1b_l0(torch_inputs[0], batched, *passes),
    }


def evaluate(
    inputs: PrecisionGateInputs,
    overrides: dict | None,
    positions_path: str,
    dot_dtypes: list[dict],
    moe: dict,
    memory: list[dict],
) -> dict:
    """The gate report for a written ``positions.pt``; ``failures`` is empty only on a pass. ``moe`` holds the
    forward's MoE settings and its ``dispatch`` counts (``positions_records``)."""
    parity = compare(positions_path, inputs.vllm, None, candidate_spans(inputs.identity, inputs.tokenizer_source))
    rounding = json.loads(read(prefix_join(inputs.torch_routing, "routing.json")))
    cross = compare_routes_cross_stack(positions_path, prefix_join(inputs.torch_routing, "routes.pt"))
    proxy = route_proxy(positions_path, inputs)
    gated = GATED_REPORT[inputs.route_policy]
    failures = (
        [f"parity: {f}" for f in parity_gate(parity)]
        + [f"routes: {f}" for f in route_gate(cross, rounding, overrides, router_dot_is_fp32(dot_dtypes))]
        + [f"routes-{gated.replace('_', '-')}: {f}" for f in proxy[gated]["failures"]]
        + [f"dispatch: {f}" for f in dispatch_failures(moe["dispatch"])]
    )
    return {
        "inputs": vars(inputs),
        "compute_dtype_overrides": overrides,
        "router_dot_dtypes": dot_dtypes,
        "moe": moe,
        "device_memory": memory,
        "env": {name: os.environ.get(name) for name in MATMUL_ENV},
        "jax": jax.__version__,
        "jax_default_matmul_precision": jax.config.jax_default_matmul_precision,
        "failures": failures,
        "parity": parity,
        "routes": cross,
        "route_policy": inputs.route_policy,
        "route_proxy": proxy,
    }


class Deadline:
    """Calls ``on_expire`` (default: exit the process) unless ``cancel`` comes first."""

    def __init__(self, seconds: float, on_expire: Callable[[], None] | None = None):
        self._done = threading.Event()
        self._on_expire = on_expire or (lambda: os._exit(DEADLINE_EXIT))
        self._thread = threading.Thread(target=self._wait, args=(seconds,), daemon=True)
        self._thread.start()

    def _wait(self, seconds: float) -> None:
        if not self._done.wait(seconds):
            logger.error("precision gate passed its %.0fs deadline; exiting", seconds)
            self._on_expire()

    def cancel(self) -> None:
        self._done.set()


def any_rank_failed(failed: bool) -> bool:
    """Collective: whether any process reports a failure."""
    return bool(gather_host_scalar(np.int32(failed)).sum())


def rank_zero_count(count: int) -> int:
    """Collective: process 0's ``count`` on every process."""
    return int(broadcast_host_scalar(np.int32(count)))


def make_pre_update_gate(inputs: PrecisionGateInputs, deadline: Callable[[float], Deadline] = Deadline):
    def pre_update_gate(trainer, state) -> None:
        rank = jax.process_index()
        started = time.time()
        timer = deadline(inputs.deadline_seconds)

        def receipt(stage: str, **info) -> None:
            try:
                value = {"process": rank, "stage": stage, "seconds": round(time.time() - started, 1), **info}
                write_json(prefix_join(inputs.output, f"ranks/{rank}.json"), value)
            except Exception:
                logger.exception("rank receipt %s not written", stage)

        try:
            error: Exception | None = None
            try:
                sequences = stored_sequences(inputs.positions)
                reference = reference_ids(inputs.vllm, sequences)
            except Exception as exc:
                error = exc
            receipt("inputs", error=repr(error) if error else None)
            if any_rank_failed(error is not None):
                raise PrecisionGateFailed(
                    "a rank could not read the pinned positions or the vLLM reference ids"
                ) from error

            overrides = trainer.config.compute_dtype_overrides
            with hax.partitioning.set_mesh(trainer.device_mesh), hax.axis_mapping(trainer.compute_axis_mapping):
                view = cast_model_for_compute(state.model, trainer.mp, overrides)
                records, dispatch = positions_records(
                    view, sequences, inputs.batch, inputs.pad_multiple, PAD_ID, routes=True, reference_ids=reference
                )
                dot_dtypes = router_dot_dtypes(view, trainer.mp.compute_dtype)
            memory = device_memory_report(gather_host_rows(local_device_memory()))
            moe = {
                "capacity_factor": view.config.moe_capacity_factor,
                "expert_axis": trainer.device_mesh.shape["expert"],
                "router_combine_fp32": view.config.router_combine_fp32,
                "fp32_activations": view.config.fp32_activations,
                "dispatch": dispatch,
            }
            receipt("scored", router_dot_dtypes=dot_dtypes, moe=moe)

            failures = 0
            if rank == 0:
                report: dict = {"inputs": vars(inputs), "failures": []}
                try:
                    positions_path = prefix_join(inputs.output, "positions.pt")
                    save_positions(records, positions_path)
                    report = evaluate(inputs, overrides, positions_path, dot_dtypes, moe, memory)
                except Exception as exc:
                    error = exc
                    report["failures"].append(f"gate did not complete: {exc!r}")
                try:
                    write_json(prefix_join(inputs.output, "gate.json"), report)
                except Exception as exc:
                    error = error or exc
                    report["failures"].append(f"gate.json not written: {exc!r}")
                failures = len(report["failures"])
                for failure in report["failures"]:
                    logger.error("precision gate: %s", failure)
            failures = rank_zero_count(failures)
            receipt("gated", failures=failures)
        finally:
            timer.cancel()
        if error is not None:
            raise PrecisionGateFailed("precision gate did not complete") from error
        if failures:
            raise PrecisionGateFailed(f"{failures} precision gate failures; see {inputs.output}/gate.json")

    return pre_update_gate

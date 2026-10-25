# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The #310 pre-update gate: score the trainer's compute view at the pinned positions and stop on any failure.

``make_pre_update_gate(inputs)`` returns the ``pre_update_gate(trainer, state)`` callback of the LR0 precision probe.

Every rank reaches the same collectives in the same order, whatever fails, so a failure on one rank cannot leave
the others waiting:

1. ``inputs``: each rank reads the stored token ids of the pinned #298 ``positions.pt`` (no collective), then an
   all-gather of every rank's status; any failure stops every rank.
2. ``scored``: every rank runs the compute-view forward and the router-dot witness (collective).
3. ``gated``: process 0 alone saves ``positions.pt``, compares it with the stored vLLM reference and the torch
   reference job's routes, applies ``logit_check.parity_gate`` and ``routing.route_gate`` and writes
   ``gate.json``, all inside one ``try``; then the failure count is broadcast and every rank raises
   ``PrecisionGateFailed`` unless it is zero. A process-0 exception, including a failed write, is a failure.

Each rank writes ``ranks/<process>.json`` after every stage (best effort). A crash inside a collective cannot be
caught, so a watchdog ends the process with status 4 after ``deadline_seconds``.
"""

import json
import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils
from levanter.compute_dtype import cast_model_for_compute
from rigging.filesystem.buckets import filesystem_for
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
    stored_sequences,
)
from experiments.interp.snowball_identity.routing import MATMUL_ENV, compare_routes_cross_stack, route_gate

logger = logging.getLogger(__name__)

DEADLINE_EXIT = 4


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
    output: str
    """Where to write ``positions.pt``, ``gate.json`` and ``ranks/<process>.json``."""
    batch: int = 8
    """Real sequences per step: the mesh's batch axes."""
    pad_multiple: int = 64
    deadline_seconds: float = 1800
    """Hard limit for the whole gate on each rank."""


def write_json(path: str, value: dict) -> None:
    fs, local = filesystem_for(path)
    fs.makedirs(local.rsplit("/", 1)[0], exist_ok=True)
    fs.pipe_file(local, json.dumps(value, indent=1, default=str).encode())


def evaluate(inputs: PrecisionGateInputs, overrides: dict | None, positions_path: str, dot_dtypes: list[dict]) -> dict:
    """The gate report for a written ``positions.pt``; ``failures`` is empty only on a pass."""
    parity = compare(positions_path, inputs.vllm, None, candidate_spans(inputs.identity, inputs.tokenizer_source))
    rounding = json.loads(read(prefix_join(inputs.torch_routing, "routing.json")))
    cross = compare_routes_cross_stack(positions_path, prefix_join(inputs.torch_routing, "routes.pt"))
    failures = [f"parity: {f}" for f in parity_gate(parity)] + [
        f"routes: {f}" for f in route_gate(cross, rounding, overrides, router_dot_is_fp32(dot_dtypes))
    ]
    return {
        "inputs": vars(inputs),
        "compute_dtype_overrides": overrides,
        "router_dot_dtypes": dot_dtypes,
        "env": {name: os.environ.get(name) for name in MATMUL_ENV},
        "jax": jax.__version__,
        "jax_default_matmul_precision": jax.config.jax_default_matmul_precision,
        "failures": failures,
        "parity": parity,
        "routes": cross,
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
    return bool(np.asarray(multihost_utils.process_allgather(jnp.int32(failed))).sum())


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
            except Exception as exc:
                error = exc
            receipt("inputs", error=repr(error) if error else None)
            if any_rank_failed(error is not None):
                raise PrecisionGateFailed("a rank could not read the pinned positions") from error

            overrides = trainer.config.compute_dtype_overrides
            with hax.partitioning.set_mesh(trainer.device_mesh), hax.axis_mapping(trainer.compute_axis_mapping):
                view = cast_model_for_compute(state.model, trainer.mp, overrides)
                records = positions_records(view, sequences, inputs.batch, inputs.pad_multiple, PAD_ID, routes=True)
                dot_dtypes = router_dot_dtypes(view, trainer.mp.compute_dtype)
            receipt("scored", router_dot_dtypes=dot_dtypes)

            failures = 0
            if rank == 0:
                report: dict = {"inputs": vars(inputs), "failures": []}
                try:
                    positions_path = prefix_join(inputs.output, "positions.pt")
                    save_positions(records, positions_path)
                    report = evaluate(inputs, overrides, positions_path, dot_dtypes)
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
            failures = int(multihost_utils.broadcast_one_to_all(jnp.int32(failures)))
            receipt("gated", failures=failures)
        finally:
            timer.cancel()
        if error is not None:
            raise PrecisionGateFailed("precision gate did not complete") from error
        if failures:
            raise PrecisionGateFailed(f"{failures} precision gate failures; see {inputs.output}/gate.json")

    return pre_update_gate

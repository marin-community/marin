# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The #310 pre-update gate: score the trainer's compute view at the pinned positions and stop on any failure.

``make_pre_update_gate(inputs)`` returns the ``pre_update_gate(trainer, state)`` callback of the LR0 precision probe.
Every rank runs the forward (it is a collective); process 0 writes ``positions.pt`` (log-probs and routes, on the
stored token ids of the pinned #298 ``positions.pt``), compares it with the stored vLLM reference and the torch
reference job's routes, applies ``logit_check.parity_gate`` and ``routing.route_gate``, and writes ``gate.json``.
The failure count is broadcast, and every rank raises ``PrecisionGateFailed`` unless it is zero. An exception on
process 0 (missing input, schema or coverage error) is a failure too.
"""

import json
import logging
from dataclasses import dataclass

import haliax as hax
import jax
from jax.experimental import multihost_utils
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import prefix_join

from experiments.interp.snowball_identity.levanter_positions import PAD_ID, write_positions
from experiments.interp.snowball_identity.logit_check import (
    candidate_spans,
    compare,
    parity_gate,
    read,
    stored_sequences,
)
from experiments.interp.snowball_identity.routing import compare_routes_cross_stack, route_gate

logger = logging.getLogger(__name__)


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
    """Where to write ``positions.pt`` and ``gate.json``."""
    batch: int = 8
    """Real sequences per step: the mesh's batch axes."""
    pad_multiple: int = 64


def evaluate(inputs: PrecisionGateInputs, overrides: dict | None, positions_path: str) -> dict:
    """The gate report for a written ``positions.pt``; ``failures`` is empty only on a pass."""
    parity = compare(positions_path, inputs.vllm, None, candidate_spans(inputs.identity, inputs.tokenizer_source))
    rounding = json.loads(read(prefix_join(inputs.torch_routing, "routing.json")))
    cross = compare_routes_cross_stack(positions_path, prefix_join(inputs.torch_routing, "routes.pt"))
    failures = [f"parity: {f}" for f in parity_gate(parity)] + [
        f"routes: {f}" for f in route_gate(cross, rounding, overrides)
    ]
    return {
        "inputs": vars(inputs),
        "compute_dtype_overrides": overrides,
        "jax": jax.__version__,
        "jax_default_matmul_precision": jax.config.jax_default_matmul_precision,
        "failures": failures,
        "parity": parity,
        "routes": cross,
    }


def make_pre_update_gate(inputs: PrecisionGateInputs):
    def pre_update_gate(trainer, state) -> None:
        overrides = trainer.config.compute_dtype_overrides
        positions_path = prefix_join(inputs.output, "positions.pt")
        with hax.partitioning.set_mesh(trainer.device_mesh):
            write_positions(
                state.model,
                trainer.mp,
                stored_sequences(inputs.positions),
                positions_path,
                batch=inputs.batch,
                pad_multiple=inputs.pad_multiple,
                pad_id=PAD_ID,
                axis_mapping=trainer.compute_axis_mapping,
                routes=True,
                compute_dtype_overrides=overrides,
            )
        multihost_utils.sync_global_devices("precision_gate_positions_written")
        error = None
        failures = 0
        if jax.process_index() == 0:
            try:
                report = evaluate(inputs, overrides, positions_path)
            except Exception as exc:
                error = exc
                report = {"inputs": vars(inputs), "failures": [f"gate did not complete: {exc!r}"]}
            fs, local = filesystem_for(prefix_join(inputs.output, "gate.json"))
            fs.pipe_file(local, json.dumps(report, indent=1, default=str).encode())
            failures = len(report["failures"])
            for failure in report["failures"]:
                logger.error("precision gate: %s", failure)
        failures = int(multihost_utils.broadcast_one_to_all(failures))
        if error is not None:
            raise PrecisionGateFailed("precision gate did not complete") from error
        if failures:
            raise PrecisionGateFailed(f"{failures} precision gate failures; see {inputs.output}/gate.json")

    return pre_update_gate

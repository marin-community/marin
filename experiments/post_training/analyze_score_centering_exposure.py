# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare consumed prompt IDs saved in fully async checkpoint data trackers.

Give one or more checkpoint states per run. The script compares runs only at the
same optimizer step and writes their consumed-prompt overlap. A state covers its
current dataset epoch; do not compare states from different epochs.
"""

from __future__ import annotations

import argparse
import csv
import itertools
from dataclasses import dataclass
from pathlib import Path

import torch

from experiments.post_training.analyze_score_centering import _filesystem

FIELDS = (
    "step",
    "epoch",
    "run_a",
    "run_b",
    "consumed_a",
    "consumed_b",
    "intersection",
    "union",
    "jaccard",
)


@dataclass(frozen=True)
class ConsumedState:
    step: int
    epoch: int
    uids: set[str]


def read_state(uri: str, s3_endpoint: str) -> ConsumedState:
    """Read a trusted, agent-owned SkyRL data-consumption checkpoint."""
    fs, path = _filesystem(uri, s3_endpoint)
    with fs.open(path, "rb") as stream:
        state = torch.load(stream, map_location="cpu", weights_only=True)
    step = state["global_step"]
    epoch = state["epoch"]
    uids = set(state["consumed_uids_in_epoch"])
    if len(uids) != len(state["consumed_uids_in_epoch"]):
        raise ValueError(f"{uri}: duplicate consumed UID")
    if state["total_samples_consumed"] < len(uids):
        raise ValueError(f"{uri}: total consumed count is smaller than its UID set")
    if not uids and state["total_samples_consumed"]:
        raise ValueError(f"{uri}: UID set was cleared at an epoch boundary; use an earlier checkpoint")
    return ConsumedState(step, epoch, uids)


def compare_states(states: dict[tuple[int, str], tuple[int, set[str]]]) -> list[dict]:
    rows = []
    for step in sorted({step for step, _ in states}):
        labels = sorted(label for saved_step, label in states if saved_step == step)
        for run_a, run_b in itertools.combinations(labels, 2):
            epoch_a, uids_a = states[step, run_a]
            epoch_b, uids_b = states[step, run_b]
            if epoch_a != epoch_b:
                raise ValueError(f"cannot compare {run_a} and {run_b} at step {step}: different epochs")
            intersection = len(uids_a & uids_b)
            union = len(uids_a | uids_b)
            rows.append(
                {
                    "step": step,
                    "epoch": epoch_a,
                    "run_a": run_a,
                    "run_b": run_b,
                    "consumed_a": len(uids_a),
                    "consumed_b": len(uids_b),
                    "intersection": intersection,
                    "union": union,
                    "jaccard": intersection / union if union else 1.0,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", action="append", required=True, metavar="RUN=STATE_URI")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--s3-endpoint", default="https://cwobject.com")
    args = parser.parse_args()

    states: dict[tuple[int, str], tuple[int, set[str]]] = {}
    for item in args.state:
        label, separator, uri = item.partition("=")
        if not separator or not label or not uri:
            parser.error(f"invalid --state {item!r}; expected RUN=STATE_URI")
        state = read_state(uri, args.s3_endpoint)
        if (state.step, label) in states:
            parser.error(f"duplicate state for {label} at step {state.step}")
        states[state.step, label] = (state.epoch, state.uids)
    rows = compare_states(states)
    if not rows:
        parser.error("at least two runs must have a state at the same step")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} consumed-prompt comparisons to {args.output}")


if __name__ == "__main__":
    main()

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Summarize complete endpoint experiments without treating selected loss as validation."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def loss(record):
    values = np.asarray(record["losses"])
    counts = np.asarray(record["counts"])
    if (
        values.shape != counts.shape
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(counts))
        or np.any(counts < 0)
        or counts.sum() <= 0
    ):
        raise ValueError("Invalid monitor loss/count arrays")
    return float(np.average(values, weights=counts))


def gain(control, treatment):
    if not np.array_equal(control["counts"], treatment["counts"]):
        raise ValueError("Compared evaluations must have identical token counts")
    return loss(control) - loss(treatment)


def summarize(path, protocol_path):
    raw = json.loads(path.read_text())
    if raw["status"] != "complete":
        raise ValueError(f"Incomplete experiment: {path}")
    protocol = raw["protocol"]
    if (
        protocol != json.loads(protocol_path.read_text())
        or raw["protocol_sha256"] != hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    ):
        raise ValueError("Run differs from the frozen execution protocol")
    expected_floor = max(protocol["minimum_loss_gain"], 2 * (max(raw["repeated_loss"]) - min(raw["repeated_loss"])))
    if raw["acceptance_floor"] != expected_floor:
        raise ValueError("Acceptance floor differs from measured calibration")
    if raw["source_sha256"] != protocol["implementation_sha256"]["experiments/grug/tied_drafter_endpoints.py"]:
        raise ValueError("Driver hash differs from frozen protocol")
    if protocol["checkpoint_policy"] == "initial_and_final":
        if raw.get("copy_validation", {}).get("status") != "passed":
            raise ValueError("Missing exact state-copy validation")
        for invariant in ("inputs_exact", "source_unchanged", "retained_snapshot_unchanged"):
            if raw["copy_validation"].get(invariant) is not True:
                raise ValueError("Missing exact snapshot invariant: " + invariant)
        expected_steps = {
            "initial-full-state": protocol["warmup_updates"],
            "treatment-final-full-state": raw["final_cursor"],
            "control-final-full-state": raw["final_cursor"],
        }
        if set(raw["checkpoints"]) != set(expected_steps):
            raise ValueError("Missing complete state checkpoints")
        for name, step in expected_steps.items():
            record = raw["checkpoints"][name]
            if record["step"] != step or record["bytes"] < raw["state_bytes"]:
                raise ValueError("Invalid checkpoint metadata")
    if len(raw["episodes"]) != protocol["episodes"]:
        raise ValueError("Missing episodes")
    cursor = protocol["warmup_updates"]
    rows = []
    previous_cost = {}
    for i, e in enumerate(raw["episodes"]):
        if e["episode"] != i or e["start_cursor"] != cursor:
            raise ValueError("Broken episode/cursor sequence")
        trials = e["trials"]
        accepted = e["accepted_prefix"]
        if any(t["accepted"] for t in trials[:-1]):
            raise ValueError("Fallback continued after acceptance")
        if (
            e["check_batch"] != protocol["check_batch_start"] + i
            or e["fresh_batch_start"] != protocol["fresh_batch_start"] + i * protocol["monitor_batches"]
        ):
            raise ValueError("Unexpected monitoring data positions")
        for key, value in e["cost_seconds"].items():
            if not np.isfinite(value) or value < previous_cost.get(key, 0):
                raise ValueError("Invalid cumulative cost")
        previous_cost = e["cost_seconds"]
        if [t["prefix"] for t in trials] != [64, 32, 16, 8][: len(trials)]:
            raise ValueError("Invalid fallback order")
        for t in trials:
            expected = e["starting_check_loss"] - t["loss"] > raw["acceptance_floor"]
            if t["accepted"] != expected or not np.isclose(
                t["gain"], e["starting_check_loss"] - t["loss"], rtol=0, atol=1e-12
            ):
                raise ValueError("Recorded acceptance contradicts measured loss")
        if accepted != (trials[-1]["prefix"] if trials[-1]["accepted"] else 0):
            raise ValueError("Wrong committed prefix")
        if not accepted and len(trials) != 4:
            raise ValueError("Incomplete rejection checks")
        cursor += accepted + 1
        if e["end_cursor"] != cursor:
            raise ValueError("Incorrect consumed data cursor")
        rows.append(
            {
                "episode": i,
                "cursor": cursor,
                "accepted_prefix": accepted,
                "selected_check_gain": trials[-1]["gain"],
                "endpoint_fresh_gain": gain(e["starting_fresh"], e["endpoint_fresh"]),
                "anchor_gain_vs_native": gain(e["control_anchor"], e["treatment_anchor"]),
                "fresh_gain_vs_native": gain(e["control_fresh"], e["treatment_fresh"]),
                "treatment_loss": loss(e["treatment_anchor"]),
                "control_loss": loss(e["control_anchor"]),
                "cost_seconds": e["cost_seconds"],
            }
        )
    recovery = {}
    for r in raw["recovery"]:
        key = (r["branch"], r["native_steps"])
        if key in recovery:
            raise ValueError("Duplicate recovery point")
        if r["cursor"] != cursor + r["native_steps"]:
            raise ValueError("Recovery cursor mismatch")
        recovery[key] = r
    after = []
    for n in protocol["recovery_steps"]:
        a = recovery[("treatment", n)]
        b = recovery[("control", n)]
        after.append(
            {
                "native_steps": n,
                "anchor_gain_vs_native": gain(b["anchor"], a["anchor"]),
                "fresh_gain_vs_native": gain(b["fresh"], a["fresh"]),
            }
        )
    updates = [json.loads(s) for s in path.with_name("updates.jsonl").read_text().splitlines()]
    counts = {b: sum(r["branch"] == b for r in updates) for b in set(r["branch"] for r in updates)}
    expected_counts = {
        "warmup": protocol["warmup_updates"],
        "draft": 64 * protocol["episodes"],
        "treatment_native": protocol["episodes"],
        "control": cursor - protocol["warmup_updates"],
        "treatment_recovery": max(protocol["recovery_steps"]),
        "control_recovery": max(protocol["recovery_steps"]),
    }
    if counts != expected_counts or not all(np.isfinite(r["loss"]) and r["seconds"] > 0 for r in updates):
        raise ValueError("Invalid or missing update records")
    if raw["final_cursor"] != cursor + max(protocol["recovery_steps"]):
        raise ValueError("Incorrect final cursor")
    for branch, start, count in [
        ("warmup", 0, protocol["warmup_updates"]),
        ("control", protocol["warmup_updates"], cursor - protocol["warmup_updates"]),
        ("treatment_recovery", cursor, max(protocol["recovery_steps"])),
        ("control_recovery", cursor, max(protocol["recovery_steps"])),
    ]:
        if [u["data_step"] for u in updates if u["branch"] == branch] != list(range(start, start + count)):
            raise ValueError("Nonconsecutive data consumption: " + branch)
    expected_draft = [i for e in raw["episodes"] for i in range(e["start_cursor"], e["start_cursor"] + 64)]
    if [u["data_step"] for u in updates if u["branch"] == "draft"] != expected_draft:
        raise ValueError("Incorrect proposal data sequence")
    if [u["data_step"] for u in updates if u["branch"] == "treatment_native"] != [
        e["end_cursor"] - 1 for e in raw["episodes"]
    ]:
        raise ValueError("Incorrect native step after endpoint")
    return {
        "seed": raw["seed"],
        "source": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "protocol_sha256": raw["protocol_sha256"],
        "initial_loss": loss(raw["initial_validation"]),
        "initial_cursor": protocol["warmup_updates"],
        "accepted_blocks": sum(r["accepted_prefix"] > 0 for r in rows),
        "episodes": rows,
        "recovery": after,
        "update_counts": counts,
        "cost_seconds": raw["cost_seconds"],
        "wandb_url": raw["wandb_url"],
        "checkpoint_metadata": raw["checkpoints"],
        "scope": protocol["scope"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    runs = [summarize(p, args.protocol) for p in args.inputs]
    if len({r["seed"] for r in runs}) != len(runs):
        raise ValueError("Duplicate seed")
    if len({r["protocol_sha256"] for r in runs}) != 1:
        raise ValueError("Protocols differ")
    args.output.write_text(
        json.dumps(
            {
                "status": "verified",
                "runs": runs,
                "interpretation": (
                    "Positive gain means lower full-model cross-entropy than the matched native "
                    "control. Selected check gains are not held-out evidence."
                ),
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    for r in runs:
        rows = r["episodes"]
        label = f"Seed {r['seed']}"
        axes[0].plot([x["cursor"] for x in rows], [x["fresh_gain_vs_native"] for x in rows], "-o", label=label)
        axes[1].plot([x["episode"] + 1 for x in rows], [x["endpoint_fresh_gain"] for x in rows], "-o", label=label)
        axes[2].plot(
            [x["native_steps"] for x in r["recovery"]],
            [x["fresh_gain_vs_native"] for x in r["recovery"]],
            "-o",
            label=label,
        )
    titles = [
        "Accumulated gain vs native training",
        "Immediate endpoint gain vs block start",
        "Persistence during native-only recovery",
    ]
    xlabels = ["Consumed training updates", "Proposal block", "Subsequent native updates"]
    for ax, title, xlabel in zip(axes, titles, xlabels, strict=True):
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xlabel)
        ax.axhline(0, color="#777", ls="--", lw=1)
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("Cross-entropy gain (nats/token)")
    axes[0].legend()
    fig.savefig(args.output.with_suffix(".svg"))
    fig.savefig(args.output.with_suffix(".png"), dpi=150)
    for r in runs:
        print(
            r["seed"],
            r["accepted_blocks"],
            r["episodes"][-1]["fresh_gain_vs_native"],
            r["recovery"][-1]["fresh_gain_vs_native"],
        )


if __name__ == "__main__":
    main()

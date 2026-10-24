# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit paired rejection-frontier outputs and summarize observed acceptance rates."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def loss(values):
    a = np.asarray(values["losses"])
    w = np.asarray(values["counts"])
    if a.shape != w.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(w)) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("Invalid evaluation arrays")
    return float(np.average(a, weights=w))


def difference(before, after):
    if not np.array_equal(before["counts"], after["counts"]):
        raise ValueError("Compared token counts differ")
    return loss(before) - loss(after)


MATCHED_SETTINGS = (
    "checkpoint_step",
    "microbatches",
    "minimum_loss_gain",
    "training_batch_start",
    "training_window_stride",
    "check_batch_start",
    "fresh_batch_start",
    "monitor_batches",
    "implementation_sha256",
    "acceptance",
    "clock_policy",
    "state_policy",
    "failure_policy",
)


def matching_settings(left, right, seed):
    for key in MATCHED_SETTINGS:
        if left[key] != right[key]:
            raise ValueError(f"Incompatible experiment setting: {key}")
    if left["checkpoints"][str(seed)] != right["checkpoints"][str(seed)]:
        raise ValueError("Different starting checkpoint")


def trial_fingerprint(trial, horizon):
    return {
        key: trial[key]
        for key in (
            "training_batch_start",
            "check_batch",
            "fresh_batch_start",
            "first_training_token_sha256",
            "check_token_sha256",
            "fresh_token_sha256",
        )
    } | {"prefix_last_token_sha256": trial["prefix_last_token_sha256"][str(horizon)]}


def audit(path, protocol_path, driver_source="experiments/grug/tied_drafter_frontier.py"):
    raw = json.loads(path.read_text())
    p = json.loads(protocol_path.read_text())
    if raw["status"] != "complete" or raw["snapshot_isolation"] != "passed":
        raise ValueError("Incomplete run or copy-isolation audit")
    if raw["protocol"] != p or raw["protocol_sha256"] != hashlib.sha256(protocol_path.read_bytes()).hexdigest():
        raise ValueError("Frozen protocol mismatch")
    if raw["source_sha256"] != p["implementation_sha256"][driver_source]:
        raise ValueError("Source hash mismatch")
    if raw["seed"] not in p["seeds"]:
        raise ValueError("Unexpected initialization seed")
    if raw["checkpoint"]["restored_step"] != p["checkpoint_step"]:
        raise ValueError("Wrong optimizer start clock")
    for k, v in p["checkpoints"][str(raw["seed"])].items():
        if raw["checkpoint"][k] != v:
            raise ValueError("Wrong source checkpoint")
    trials = {r["trial"]: r for r in raw["trials"]}
    if len(trials) != len(raw["trials"]) or sorted(trials) != p["trial_ids"] or raw["completed_trials"] != len(trials):
        raise ValueError("Missing or duplicate trials")
    updates = [json.loads(line) for line in path.with_name("updates.jsonl").read_text().splitlines()]
    rows = {}
    for row in raw["endpoints"]:
        key = (row["trial"], row["active_experts"], row["horizon"])
        if key in rows:
            raise ValueError("Duplicate endpoint")
        rows[key] = row
    expected = {(t, k, h) for t in p["trial_ids"] for k in p["active_experts"] for h in p["horizons"]}
    if set(rows) != expected:
        raise ValueError("Missing or unexpected endpoints")
    for trial, t in trials.items():
        if t["training_batch_start"] != p["training_batch_start"] + trial * p["training_window_stride"]:
            raise ValueError("Training window mismatch")
        if (
            t["check_batch"] != p["check_batch_start"] + trial
            or t["fresh_batch_start"] != p["fresh_batch_start"] + trial * p["monitor_batches"]
        ):
            raise ValueError("Evaluation window mismatch")
        floor = max(p["minimum_loss_gain"], 2 * (max(t["repeated_loss"]) - min(t["repeated_loss"])))
        if t["acceptance_floor"] != floor:
            raise ValueError("Incorrect acceptance floor")
        offset = trial % len(p["active_experts"])
        order = p["active_experts"][offset:] + p["active_experts"][:offset]
        if t["variant_order"] != order:
            raise ValueError("Execution order mismatch")
        for active in p["active_experts"]:
            branch = [u for u in updates if u["trial"] == trial and u["active_experts"] == active]
            failures = [
                r["failure_step"]
                for key, r in rows.items()
                if key[:2] == (trial, active) and r["failure_step"] is not None
            ]
            limit = min(failures) if failures else max(p["horizons"])
            if [u["proposal_step"] for u in branch] != list(range(1, limit + 1)):
                raise ValueError("Missing update records")
            for u in branch:
                i = u["proposal_step"]
                if (
                    u["optimizer_step"] != p["checkpoint_step"] + i
                    or u["data_step"] != t["training_batch_start"] + i - 1
                ):
                    raise ValueError("Data offset incorrectly changes optimizer clock")
                if u["seconds"] <= 0 or not np.isfinite(u["seconds"]):
                    raise ValueError("Invalid update timing")
                if u["finite"] != (u["loss"] is not None and np.isfinite(u["loss"])):
                    raise ValueError("Invalid nonfinite-loss record")
                if not u["finite"] and (i != limit or not failures):
                    raise ValueError("Invalid divergence boundary")
            for h in p["horizons"]:
                r = rows[(trial, active, h)]
                reached = min(h, limit)
                if r["optimizer_step"] != p["checkpoint_step"] + reached:
                    raise ValueError("Endpoint clock mismatch")
                if not np.isclose(r["update_seconds"], sum(u["seconds"] for u in branch[:reached]), rtol=0, atol=1e-8):
                    raise ValueError("Endpoint cost omits generated updates")
                accepted = r["check"] is not None and difference(t["starting_check"], r["check"]) > floor
                if accepted != r["accepted"]:
                    raise ValueError("Incorrect endpoint acceptance")
                if r["check"] is not None and not np.isclose(
                    r["check_gain"], difference(t["starting_check"], r["check"]), rtol=0, atol=1e-12
                ):
                    raise ValueError("Incorrect check gain")
                if r["fresh"] is not None and not np.isclose(
                    r["fresh_gain"], difference(t["starting_fresh"], r["fresh"]), rtol=0, atol=1e-12
                ):
                    raise ValueError("Incorrect fresh gain")
                if failures and h >= limit and r["failure_step"] != limit:
                    raise ValueError("Missing unreached-horizon failure label")
    if len(updates) != sum(
        len([u for u in updates if u["trial"] == t and u["active_experts"] == k])
        for t in trials
        for k in p["active_experts"]
    ):
        raise ValueError("Unexpected update branch")
    return raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--driver-source",
        default="experiments/grug/tied_drafter_frontier.py",
        help="Protocol source-hash key for the executed experiment driver",
    )
    parser.add_argument(
        "--run-protocols", type=Path, nargs="+", help="Frozen protocol for each input, when jobs partition a phase"
    )
    parser.add_argument(
        "--native-inputs", type=Path, nargs="+", default=[], help="Previously audited native-reference runs"
    )
    parser.add_argument("--native-protocols", type=Path, nargs="+", default=[])
    args = parser.parse_args()
    protocol_paths = args.run_protocols or [args.protocol] * len(args.inputs)
    if len(protocol_paths) != len(args.inputs) or len(args.native_inputs) != len(args.native_protocols):
        raise ValueError("Each input needs its exact frozen protocol")
    runs = [
        audit(path, protocol, args.driver_source) for path, protocol in zip(args.inputs, protocol_paths, strict=True)
    ]
    references = [
        audit(path, protocol, args.driver_source)
        for path, protocol in zip(args.native_inputs, args.native_protocols, strict=True)
    ]
    p = json.loads(args.protocol.read_text())
    expected = {
        (s, t, k, h) for s in p["seeds"] for t in p["trial_ids"] for k in p["active_experts"] for h in p["horizons"]
    }
    observed = [(r["seed"], e["trial"], e["active_experts"], e["horizon"]) for r in runs for e in r["endpoints"]]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError("Incomplete or duplicate phase coverage")
    for run in runs:
        matching_settings(run["protocol"], p, run["seed"])
    native = {}
    for run in runs + references:
        trials = {t["trial"]: t for t in run["trials"]}
        for row in run["endpoints"]:
            if row["active_experts"] != 8:
                continue
            key = (run["seed"], row["trial"], row["horizon"])
            if key in native:
                raise ValueError("Ambiguous native reference")
            native[key] = (row, trials[row["trial"]], run)
    fingerprints = {}
    records = []
    for _path, run in zip(args.inputs, runs, strict=True):
        trials = {t["trial"]: t for t in run["trials"]}
        for row in run["endpoints"]:
            baseline, native_trial, native_run = native[(run["seed"], row["trial"], row["horizon"])]
            trial = trials[row["trial"]]
            matching_settings(run["protocol"], native_run["protocol"], run["seed"])
            fp = trial_fingerprint(trial, row["horizon"])
            if fp != trial_fingerprint(native_trial, row["horizon"]):
                raise ValueError("Native reference uses different data")
            fp_key = (row["trial"], row["horizon"])
            if fp_key in fingerprints and fingerprints[fp_key] != fp:
                raise ValueError("Data differs across seeds")
            fingerprints[fp_key] = fp
            if baseline["failure_step"] is not None or baseline["fresh"] is None:
                raise ValueError("Native reference failed")
            cost = (
                row["update_seconds"]
                + row["initial_copy_seconds"]
                + trial["baseline_check_seconds"]
                + row["candidate_check_seconds"]
            )
            records.append(
                {
                    "seed": run["seed"],
                    "trial": row["trial"],
                    "active_experts": row["active_experts"],
                    "horizon": row["horizon"],
                    "accepted": row["accepted"],
                    "failure_step": row["failure_step"],
                    "fresh_gain": row["fresh_gain"],
                    "fresh_gain_vs_native": (
                        None if row["fresh"] is None else difference(baseline["fresh"], row["fresh"])
                    ),
                    "check_gain": row["check_gain"],
                    "native_fresh_gain": difference(native_trial["starting_fresh"], baseline["fresh"]),
                    "update_seconds": row["update_seconds"],
                    "seconds_per_executed_update": row["update_seconds"] / (row["failure_step"] or row["horizon"]),
                    "native_update_seconds": baseline["update_seconds"],
                    "update_time_ratio": (
                        row["update_seconds"] / baseline["update_seconds"] if row["failure_step"] is None else None
                    ),
                    "single_horizon_execution_seconds": cost,
                    "single_horizon_cost_ratio": (
                        cost / baseline["update_seconds"] if row["failure_step"] is None else None
                    ),
                }
            )
    cells = []
    for active in p["active_experts"]:
        for h in p["horizons"]:
            group = [r for r in records if r["active_experts"] == active and r["horizon"] == h]
            rejected = sum(not r["accepted"] for r in group)
            per_seed = {str(seed): sum(not r["accepted"] for r in group if r["seed"] == seed) for seed in p["seeds"]}
            valid = [r for r in group if r["fresh_gain_vs_native"] is not None]
            timing = [r for r in group if r["update_time_ratio"] is not None]
            cells.append(
                {
                    "active_experts": active,
                    "horizon": h,
                    "trials": len(group),
                    "rejected": rejected,
                    "rejection_rate": rejected / len(group),
                    "passes_observed_10_percent_rule": rejected / len(group) <= 0.1,
                    "rejections_by_seed": per_seed,
                    "rejected_with_passing_shorter_prefix": sum(
                        not r["accepted"]
                        and any(
                            x["accepted"]
                            and x["seed"] == r["seed"]
                            and x["trial"] == r["trial"]
                            and x["active_experts"] == active
                            and x["horizon"] < h
                            for x in records
                        )
                        for r in group
                    ),
                    "nonfinite_fresh": len(group) - len(valid),
                    "mean_fresh_gain": None if not valid else float(np.mean([r["fresh_gain"] for r in valid])),
                    "mean_fresh_gain_vs_native": (
                        None if not valid else float(np.mean([r["fresh_gain_vs_native"] for r in valid]))
                    ),
                    "median_seconds_per_executed_update": float(
                        np.median([r["seconds_per_executed_update"] for r in group])
                    ),
                    "median_update_time_ratio": (
                        None if not timing else float(np.median([r["update_time_ratio"] for r in timing]))
                    ),
                    "median_single_horizon_cost_ratio": (
                        None if not timing else float(np.median([r["single_horizon_cost_ratio"] for r in timing]))
                    ),
                }
            )
    result = {
        "status": "audited",
        "protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        "scope": (
            "Independent checkpoint branches; observed rates on shared states, no popula"
            "tion guarantee. No shorter-prefix fallback counted as full-horizon acceptan"
            "ce."
        ),
        "sources": [
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "seed": run["seed"],
                "wandb_url": run["wandb_url"],
                "cost_seconds": run["cost_seconds"],
                "protocol_sha256": run["protocol_sha256"],
            }
            for path, run in zip(args.inputs, runs, strict=True)
        ],
        "cached_native_sources": [
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "seed": run["seed"],
                "protocol_sha256": run["protocol_sha256"],
                "wandb_url": run["wandb_url"],
            }
            for path, run in zip(args.native_inputs, references, strict=True)
        ],
        "cells": cells,
        "records": records,
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    lookup = {(c["active_experts"], c["horizon"]): c for c in cells}
    for ax, field, title in [
        (axes[0], "rejection_rate", "Observed endpoint rejection"),
        (axes[1], "mean_fresh_gain_vs_native", "Fresh loss gain vs native (nats/token)"),
    ]:
        matrix = np.array(
            [
                [lookup[(k, h)][field] if lookup[(k, h)][field] is not None else np.nan for k in p["active_experts"]]
                for h in p["horizons"]
            ]
        )
        limit = max(float(np.nanmax(np.abs(matrix))), 1e-6) if np.any(np.isfinite(matrix)) else 1.0
        picture = ax.imshow(
            matrix,
            cmap="Reds" if field == "rejection_rate" else "RdBu",
            aspect="auto",
            vmin=0 if field == "rejection_rate" else -limit,
            vmax=1 if field == "rejection_rate" else limit,
        )
        for i, h in enumerate(p["horizons"]):
            for j, k in enumerate(p["active_experts"]):
                c = lookup[(k, h)]
                label = (
                    f"{c['rejected']}/{c['trials']}"
                    if field == "rejection_rate"
                    else ("nonfinite" if c[field] is None else f"{c[field]:+.4f}")
                )
                ax.text(
                    j,
                    i,
                    label,
                    ha="center",
                    va="center",
                    fontsize=10,
                    bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
                )
        ax.set_xticks(range(len(p["active_experts"])), p["active_experts"])
        ax.set_yticks(range(len(p["horizons"])), p["horizons"])
        ax.set_xlabel("Active experts (8 = native)")
        ax.set_ylabel("Proposed update horizon")
        ax.set_title(title)
        fig.colorbar(picture, ax=ax)
    fig.savefig(args.output.with_suffix(".svg"))
    fig.savefig(args.output.with_suffix(".png"), dpi=150)
    for c in cells:
        print(c)


if __name__ == "__main__":
    main()

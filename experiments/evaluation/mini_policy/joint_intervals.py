# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit empirical joint-residual intervals without treating benchmarks as independent."""

import argparse
import io
import json
import os
from pathlib import Path

import numpy as np
from rigging.filesystem.buckets import filesystem_for

from experiments.evaluation.mini_policy.calibration import DEST
from experiments.evaluation.mini_policy.experiment import SEED, read_bytes, retained_benchmark_indices, tournament

STUDY = "s3://marin-us-east-02a/marin/experiments/mini-eval-policy/study-20261030-v6-21x21"
DRAWS = 2000


def vector_tournament(scores, sems):
    """Apply the canonical tournament to a batch of complete cohort panels."""
    difference = scores[:, :, None, :] - scores[:, None, :, :]
    cutoff = 1.96 * np.hypot(sems[:, :, None, :], sems[:, None, :, :]) / np.sqrt(0.9)
    wins = np.where(difference > cutoff, 1.0, np.where(difference < -cutoff, 0.0, 0.5))
    n, benchmarks = scores.shape[1:]
    return (wins.sum(axis=(2, 3)) - 0.5 * benchmarks) / ((n - 1) * benchmarks)


def residual_rows(train_groups, test_groups, seed):
    rng = np.random.default_rng(seed)
    families = sorted(set(train_groups))
    rows = np.empty((DRAWS, len(test_groups)), dtype=int)
    for group in sorted(set(test_groups)):
        chosen = rng.integers(len(families), size=DRAWS)
        for family_index, family in enumerate(families):
            draws = np.flatnonzero(chosen == family_index)
            available = np.flatnonzero(train_groups == family)
            for i in np.flatnonzero(test_groups == group):
                rows[draws, i] = rng.choice(available, size=len(draws))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", default=STUDY)
    parser.add_argument("--calibration", default=DEST)
    parser.add_argument("--destination", default=DEST + "/joint-report")
    args = parser.parse_args()
    settings = json.loads(read_bytes(args.study + "/settings.json"))
    source = settings["source"]
    arrays = np.load(io.BytesIO(read_bytes(source + "/matrix/responses.npz")))
    metadata = json.loads(read_bytes(source + "/matrix/tasks.json"))
    primary = retained_benchmark_indices(metadata["benchmarks"])
    if settings["target_benchmarks"] != [metadata["benchmarks"][j] for j in primary]:
        raise ValueError("Study target scope differs from the current 21-benchmark scope")
    target, sems = arrays["targets"][:, primary], arrays["sems"][:, primary]
    groups = np.asarray(settings["split_groups"])
    results = []
    for fold, split in enumerate(settings["splits"]):
        train, test = np.asarray(split["train"]), np.asarray(split["test"])
        calibration = json.loads(read_bytes(args.calibration + f"/fold-{fold}/calibration.json"))
        inner_sem = np.full((len(train), len(primary)), np.nan)
        for fit in calibration["fits"]:
            positions = [list(train).index(i) for i in fit["valid"]]
            inner_sem[positions] = np.median(sems[fit["train"]], axis=0)
        if not np.isfinite(inner_sem).all():
            raise ValueError("Incomplete inner SEM alignment")
        sem_error = sems[train] - inner_sem
        sampled_rows = residual_rows(groups[train], groups[test], SEED + fold)
        truth = tournament(np.vstack([target[train], target[test]]), np.vstack([sems[train], sems[test]]))[len(train) :]
        for record in calibration["results"]:
            method, size = record["method"], record["budget"]
            outer = json.loads(read_bytes(args.study + f"/fold-{fold}-{method}-0-{size}.json"))
            if outer["test_models"] != calibration["test_models"]:
                raise ValueError("Outer prediction order changed")
            error = target[train] - np.asarray(record["inner_predictions"])
            sampled = np.clip(np.asarray(outer["predicted"])[None, :, :] + error[sampled_rows], 0, 1)
            sampled_sem = np.clip(np.median(sems[train], axis=0)[None, None, :] + sem_error[sampled_rows], 0, 0.5)
            panels = np.concatenate([np.broadcast_to(target[train], (DRAWS, len(train), len(primary))), sampled], axis=1)
            panel_sems = np.concatenate(
                [np.broadcast_to(sems[train], (DRAWS, len(train), len(primary))), sampled_sem], axis=1
            )
            values = vector_tournament(panels, panel_sems)[:, len(train) :]
            low, high = np.quantile(values, [0.05, 0.95], axis=0)
            results.append(
                {
                    "fold": fold,
                    "method": method,
                    "budget": size,
                    "models": calibration["test_models"],
                    "lower": low.tolist(),
                    "upper": high.tolist(),
                    "actual": truth.tolist(),
                    "covered": ((truth >= low) & (truth <= high)).tolist(),
                    "width": (high - low).tolist(),
                    "calibration_families": len(set(groups[train])),
                }
            )
    summary = []
    for method in dict.fromkeys(row["method"] for row in results):
        for budget in (210, 840):
            rows = [row for row in results if row["method"] == method and row["budget"] == budget]
            covered = [value for row in rows for value in row["covered"]]
            width = [value for row in rows for value in row["width"]]
            summary.append(
                {
                    "method": method,
                    "budget": budget,
                    "models": len(covered),
                    "coverage": float(np.mean(covered)),
                    "mean_width": float(np.mean(width)),
                }
            )
    lines = [
        "# Joint-residual interval validation",
        "",
        (
            "These empirical central 90% intervals preserve each calibration model's "
            "complete vector of benchmark-score and SEM errors. Calibration uses the same"
            " nested family-held-out fits as the rectangular-bound audit. No outer "
            "held-out outcome sets an interval."
        ),
        "",
        (
            "For each of 2,000 draws, sample a calibration family uniformly for each "
            "held-out model family, then sample a residual model within that family for "
            "each held-out checkpoint. Add the whole signed residual vector to the "
            "held-out prediction, clip scores to [0, 1] and SEMs to [0, 0.5], and rerun "
            "the complete tournament against fixed training anchors. Related held-out "
            "checkpoints share the sampled family; they can draw different members within"
            " it. The interval is the 5th-95th percentile of those tournament scores."
        ),
        "",
        "| Method | Tasks | Held-out coverage | Mean tournament width (pp) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(f"| {row['method']} | {row['budget']} | {100*row['coverage']:.1f}% | {100*row['mean_width']:.1f} |")
    lines += [
        "",
        (
            "This is an empirical coverage experiment. It has no distribution-free "
            "coverage guarantee and assumes unseen-family error resembles the seven or "
            "nine calibration families. It does not validate new harnesses or fresh "
            "stochastic responses. A narrower interval than the rectangular construction "
            "reflects preservation of cross-benchmark error structure; it does not repair"
            " point-ranking failures."
        ),
        "",
    ]
    output = Path(os.environ["IRIS_OUTPUT_DIR"])
    payloads = {
        "JOINT_INTERVAL_VALIDATION.md": "\n".join(lines).encode(),
        "joint-interval-validation.json": (
            json.dumps(
                {
                    "study": args.study,
                    "calibration_source": args.calibration,
                    "matrix_source": source,
                    "draws": DRAWS,
                    "seed": SEED,
                    "summary": summary,
                    "folds": results,
                },
                indent=2,
                allow_nan=False,
            ).encode()
        ),
    }
    for name, payload in payloads.items():
        (output / name).write_bytes(payload)
        fs, key = filesystem_for(args.destination + "/" + name)
        fs.pipe_file(key, payload)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

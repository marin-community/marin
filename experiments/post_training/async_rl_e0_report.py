# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.4.0", "matplotlib==3.10.8"]
# ///
"""Report historical Qwen objective contrasts from audited, text-free receipts."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ACCEPTED_STOPS = {"complete", "end_turn", "eos", "stop"}
REPETITIONS = 100_000
BOOTSTRAP_SEED = 179


def paired_contrasts(vectors: dict) -> list[dict]:
    """Return four question-paired contrasts conditional on the observed seed 17.

    Exact two-sided McNemar tests receive Holm adjustment. Bootstrap intervals
    use Bonferroni coverage (98.75% each), a conservative simultaneous familywise
    95% convention. They are not misrepresented as Holm intervals.
    """
    contrasts = []
    for cadence in ("c1-a1", "c4-a3"):
        reference = vectors[f"qwen-factorial16-{cadence}-regular-no-tis"]["100"]
        for objective in ("regular-tis", "behavior"):
            candidate = vectors[f"qwen-factorial16-{cadence}-{objective}"]["100"]
            assert [row[:2] for row in candidate] == [row[:2] for row in reference]
            assert len(reference) == 128
            assert all(row[2] in (0, 1) for row in reference + candidate)
            ref = np.array([row[2] * (row[3] in ACCEPTED_STOPS) for row in reference])
            cand = np.array([row[2] * (row[3] in ACCEPTED_STOPS) for row in candidate])
            delta = cand - ref
            wins, losses = int((delta > 0).sum()), int((delta < 0).sum())
            discordant = wins + losses
            p = min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(wins, losses) + 1)) / 2**discordant)
            rng = np.random.default_rng(BOOTSTRAP_SEED)
            draws = np.concatenate(
                [delta[rng.integers(0, 128, (1000, 128))].mean(axis=1) for _ in range(REPETITIONS // 1000)]
            )
            contrasts.append(
                {
                    "cadence": cadence,
                    "candidate": objective,
                    "reference": "regular-no-tis",
                    "metric": "contract_completed",
                    "delta_pp": float(delta.mean() * 100),
                    "candidate_only_correct": wins,
                    "reference_only_correct": losses,
                    "mcnemar_p": p,
                    "simultaneous_interval_pp": (np.quantile(draws, [0.00625, 0.99375]) * 100).tolist(),
                }
            )
    previous = 0.0
    for rank, row in enumerate(sorted(contrasts, key=lambda row: row["mcnemar_p"])):
        previous = min(1.0, max(previous, (len(contrasts) - rank) * row["mcnemar_p"]))
        row["holm_adjusted_p"] = previous
    return contrasts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--finelog-checks", type=Path, required=True)
    parser.add_argument("--historical-final", type=Path, required=True)
    parser.add_argument("--native-history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text())
    native_checks = json.loads(args.finelog_checks.read_text())
    historical = json.loads(args.historical_final.read_text())
    native = json.loads(args.native_history.read_text())
    assert audit["clean_end_to_end"] and not audit["errors"]
    assert native_checks["status"] == historical["status"] == "PASS"
    assert len(native_checks["checks"]) == 56
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True, layout="constrained")
    arms = []
    for label, old in sorted(historical["arms"].items()):
        run = audit["runs"]["qwen-factorial16-" + label]
        assert old["attempt_id"] == run["attempt_id"]
        sums = run["history"]["sums"]
        assert math.isclose(old["core_seconds"], sums["async/performance/core_seconds"], rel_tol=1e-10)
        assert old["consumed_response_tokens"] == sums["async/performance/consumed_response_tokens"]
        rows = sorted([row for row in native["steps"] if row["run_id"] == run["run_id"]], key=lambda row: row["step"])
        assert [row["step"] for row in rows] == list(range(1, 101))
        cumulative = {
            key: np.concatenate(([0], np.cumsum([row[key] for row in rows])))
            for key in ("async/performance/consumed_response_tokens", "async/performance/core_seconds")
        }
        metrics = run["eval_dumps"][-1]["metrics"]
        count = metrics["eval/all/sequences"]
        contract = round(metrics["eval/all/completed_stop_score_contribution"] * count)
        vector = audit["vectors"]["qwen-factorial16-" + label]["100"]
        assert contract == sum(row[2] * (row[3] in ACCEPTED_STOPS) for row in vector)
        ages = old["ages"]
        age_count = sum(row["groups"] for row in ages)
        mean_age = sum(row["age"] * row["groups"] for row in ages) / age_count
        accumulated = 0
        for row in sorted(ages, key=lambda row: row["age"]):
            accumulated += row["groups"]
            if accumulated >= 0.95 * age_count:
                p95_age = row["age"]
                break
        arms.append(
            {
                "arm": label,
                "updates": 100,
                "seed": 17,
                "tokens": int(old["consumed_response_tokens"]),
                "core_seconds": old["core_seconds"],
                "task_gpu_hours": old["task_running_gpu_hours"],
                "contract_completed": contract,
                "raw_correct": round(metrics["eval/all/avg_score"] * count),
                "questions": count,
                "semantic": None,
                "truncation_percent": 100 * metrics["eval/all/length_stop_fraction"],
                "group_age_mean": mean_age,
                "group_age_p95": p95_age,
                "minimum_ess_fraction": old["minimum_ess_fraction"],
                "wandb_url": run["wandb_url"],
            }
        )
        steps = [row["step"] for row in run["eval_dumps"]]
        scores = [100 * row["metrics"]["eval/all/completed_stop_score_contribution"] for row in run["eval_dumps"]]
        xvalues = [
            steps,
            cumulative["async/performance/consumed_response_tokens"][steps] / 1e6,
            cumulative["async/performance/core_seconds"][steps],
        ]
        for ax, x in zip(axes, xvalues, strict=True):
            ax.plot(x, scores, marker=".", label=label)
    for ax, label in zip(
        axes, ["Learner updates", "Consumed response tokens (millions)", "Core RL seconds"], strict=True
    ):
        ax.set_xlabel(label)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Legacy contract_completed (%)")
    axes[2].legend(fontsize=7)
    fig.suptitle("Qwen3-0.6B: legacy 128-question development set, seed 17")
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output / "e0-quality-curves.png", dpi=150)
    fig.savefig(args.output / "e0-quality-curves.svg")
    summary = {
        "status": "PASS",
        "arms": arms,
        "contrasts": paired_contrasts(audit["vectors"]),
        "bootstrap_repetitions": REPETITIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "inference_scope": "Paired questions, fixed observed training seed; four primary final-endpoint contrasts.",
        "inputs_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (args.audit, args.finelog_checks, args.historical_final, args.native_history)
        },
    }
    (args.output / "e0-report.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print("E0_REPORT_PASS: six arms, four paired contrasts, three quality curves; familywise95% convention")
    print(json.dumps(summary["contrasts"], indent=2))


if __name__ == "__main__":
    main()

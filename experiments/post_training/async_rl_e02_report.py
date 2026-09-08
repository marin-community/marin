# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.3.5", "matplotlib==3.10.8"]
# ///
"""Report the audited, three-seed legacy Snowball cadence comparison."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

STOPS = {"complete", "end_turn", "eos", "stop"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    files = ["e02-audit-summary.json", "e02-native-history.json", "e02-task-cost-audit.json", "e02-finelog-checks.json"]
    audit, history, cost, finelog = [json.loads((args.directory / name).read_text()) for name in files]
    assert audit["clean_end_to_end"] and not audit["errors"]
    assert history["status"] == cost["status"] == finelog["status"] == "PASS"
    assert len(audit["runs"]) == 6 and len(history["steps"]) == 600 and len(finelog["checks"]) == 138
    for check in finelog["checks"]:
        expected, observed = check["expected"], check["observed"]
        pairs = (
            [(observed[k], value) for k, value in expected.items()]
            if isinstance(expected, dict)
            else [(observed, expected)]
        )
        for value, target in pairs:
            assert math.isclose(value, target, rel_tol=1e-10, abs_tol=1e-12)
    arms = []
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True, layout="constrained")
    for label, run in sorted(audit["runs"].items()):
        native = history["runs"][label]
        rows = sorted((r for r in history["steps"] if r["run_id"] == run["run_id"]), key=lambda r: r["step"])
        assert [r["step"] for r in rows] == list(range(1, 101))
        cumulative = {}
        for key in ("async/performance/consumed_response_tokens", "async/performance/core_seconds"):
            cumulative[key] = np.r_[0, np.cumsum([r[key] for r in rows])]
            assert math.isclose(cumulative[key][-1], run["history"]["sums"][key], rel_tol=1e-10)
        tasks = cost["runs"][label]["tasks"]
        assert len(tasks) == 5
        for task in tasks:
            assert task["state"] == "succeeded" and task["exit_code"] == task["attempt_number"] == 0
            assert task["execution_cluster_id"] == "cw-us-east-02a" and task["gpu_count"] == 8
        gpu_hours = sum((t["finished_at_ms"] - t["started_at_ms"]) * t["gpu_count"] / 3_600_000 for t in tasks)
        scores = []
        for step in (0, 100):
            vector = audit["vectors"][label][str(step)]
            assert len(vector) == 1191 and all(r[2] in (0, 1) for r in vector)
            completed = sum(r[2] * (r[3] in STOPS) for r in vector)
            dump = next(d for d in run["eval_dumps"] if d["step"] == step)
            assert math.isclose(completed / 1191, dump["metrics"]["eval/all/completed_stop_score_contribution"])
            scores.append(completed / 1191 * 100)
        final = audit["vectors"][label]["100"]
        arms.append(
            {
                "label": label,
                "seed": run["provenance"]["seed"],
                "tokens": int(native["tokens"]),
                "core_seconds": native["core_seconds"],
                "task_gpu_hours": gpu_hours,
                "completed": int(completed),
                "raw_correct": int(sum(r[2] for r in final)),
                "truncated": sum(r[3] == "length" for r in final),
                "questions": 1191,
                "semantic": None,
                "age_mean": native["age_mean"],
                "age_p95": native["age_p95"],
                "minimum_ess_fraction": native["ess_fraction_min"],
                "wandb_url": run["wandb_url"],
            }
        )
        xvalues = (
            [0, 100],
            cumulative["async/performance/consumed_response_tokens"][[0, 100]] / 1e6,
            cumulative["async/performance/core_seconds"][[0, 100]],
        )
        for ax, x in zip(axes, xvalues, strict=True):
            ax.plot(x, scores, marker="o", label=label.removeprefix("snowball-confirm100-"))
    study = audit["paired_studies"][0]
    deltas = []
    for seed in (17, 29, 43):
        vectors = [audit["vectors"][f"snowball-confirm100-{arm}-s{seed}"]["100"] for arm in ("c2-a1", "c4-a3")]
        assert [r[:2] for r in vectors[0]] == [r[:2] for r in vectors[1]]
        values = [np.array([r[2] * (r[3] in STOPS) for r in v]) for v in vectors]
        deltas.append(float((values[1] - values[0]).mean()))
    assert np.allclose(deltas, [s["final_reward_delta"] for s in study["seed_results"]], atol=1e-14, rtol=0)
    mean = float(np.mean(deltas))
    # Exact inverse CDF for Student t with df=2: F(t)=1/2+t/(2*sqrt(t*t+2)).
    t975 = math.sqrt(2 * 0.95**2 / (1 - 0.95**2))
    half_width = t975 * float(np.std(deltas, ddof=1)) / math.sqrt(3)
    summary = {
        "status": "PASS",
        "arms": arms,
        "seed_deltas_pp": (np.array(deltas) * 100).tolist(),
        "mean_delta_pp": mean * 100,
        "training_seed_t95_interval_pp": [(mean - half_width) * 100, (mean + half_width) * 100],
        "training_seed_interval_scope": (
            "Three independent paired training seeds on the fixed battery; Student-t df2 assumes "
            "approximately normal seed effects, uncheckable with n=3."
        ),
        "conditional_question_bootstrap95_interval_pp": (
            (np.array(study["bootstrap"]["percentile_intervals"]["final_reward_delta"]) * 100).tolist()
        ),
        "question_interval_scope": study["uncertainty_scope"],
        "paired_metric": "score_contract_completed",
        "total_historical_task_gpu_hours": sum(a["task_gpu_hours"] for a in arms),
        "new_v2_task_gpu_hours": next(a["task_gpu_hours"] for a in arms if a["label"].endswith("c4-a3-s43")),
        "inputs_sha256": {name: hashlib.sha256((args.directory / name).read_bytes()).hexdigest() for name in files},
    }
    for ax, xlabel in zip(
        axes, ["Learner updates", "Consumed response tokens (millions)", "Core RL seconds"], strict=True
    ):
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Completed correctness (%)")
    axes[2].legend(fontsize=8)
    fig.suptitle("Snowball: 1,191 locked questions, three seeds; evaluation at 0 and 100 only")
    fig.savefig(args.directory / "e02-quality-curves.png", dpi=150)
    fig.savefig(args.directory / "e02-quality-curves.svg")
    (args.directory / "e02-report.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print("E02_REPORT_PASS: six audited arms, paired completed-correctness intervals, three endpoint plots")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

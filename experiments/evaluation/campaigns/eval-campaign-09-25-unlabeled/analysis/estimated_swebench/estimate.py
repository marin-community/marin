#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Make a provisional SWE-bench tracker and rerun tournament/CD analysis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ANALYSIS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ANALYSIS / "critical_difference"))
import plot  # noqa: E402

PARTIAL_JOBS = {
    "inclusionAI/Ling-lite-1.5": ("20261003-193744-inclusionAI-Ling-lite-1.5-swebench-verified-d669", "01d95152b4ac"),
    "openai/gpt-oss-20b": ("20261003-194224-openai-gpt-oss-20b-swebench-verified-c44d", "efe1f4c21521"),
    "arcee-ai/Trinity-Mini": ("20261004-091216-arcee-ai-Trinity-Mini-swebench-verified-905e", "6553f338d457"),
    "open-athena/Snowball-67B-A2B-5.7T-Mixed-RLVR-Step38": (
        "20261003-170932-open-athena-Snowball-67B-A2B-5.7T-Mixed-RLVR-Step38-swebench-verified-8332",
        "7d02564e3954",
    ),
    "laion/snowball-67b-a2b-relay-sft-allkimi-step1203": (
        "20261004-112101-laion-snowball-67b-a2b-relay-sft-allkimi-step1203-swebench-verified-b3ab",
        "bef88ddc82c2",
    ),
    "IFM/K2-Horizon-MoVA-36B-A4B": ("20261003-174658-K2-Horizon-campaign-swebench-verified-01c6", "b16068c7482f"),
}
BUCKET = "s3://marin-us-east-02a/marin/evals"


def partial_score(model: str, run: str, job: str) -> tuple[dict[str, object], dict]:
    results = f"{BUCKET}/{run}/results"
    job_result = f"{results}/harbor_jobs/harbor_swebench-verified_{job}/result.json"
    record = plot.read_json(job_result)
    if record["n_total_trials"] != 500 or len(record["stats"]["evals"]) != 1:
        raise ValueError(f"unexpected SWE-bench task coverage for {model}")
    trial_stats = next(iter(record["stats"]["evals"].values()))
    completed = int(record["stats"]["n_completed_trials"])
    raw_mean = float(trial_stats["metrics"][0]["mean"])
    scored = int(trial_stats["n_trials"])
    successes = round(raw_mean * completed)
    if scored < 2 or scored > completed or abs(raw_mean * completed - successes) > 1e-6:
        raise ValueError(f"invalid trial counts for {model}: {scored} scored, {completed} completed")
    score = successes / scored
    return {
        "model": model,
        "score": score,
        "completed": completed,
        "scored": scored,
        "raw_completed_mean": raw_mean,
        "successes": successes,
        "total": 500,
        "updated_at": record["updated_at"],
        "results": results,
        "job_result": job_result,
    }, record


def estimated_tracker(source: Path, destination: Path, partials: dict[str, dict[str, object]]) -> None:
    lines = source.read_text().splitlines()
    header = [cell.strip() for cell in lines[0].strip("|").split("|")]
    position = header.index("swebench-verified")
    changed: set[str] = set()
    for index, line in enumerate(lines):
        if not line.startswith("|") or index < 3:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        model = cells[0]
        if model not in partials:
            continue
        if cells[position] not in {"RUNNING", "QUEUED", "CRASHED", "BLOCKED"}:
            raise ValueError(f"refusing to overwrite a scored SWE-bench cell: {model}")
        data = partials[model]
        cells[position] = f"{float(data['score']):.3f} ({data['results']})"
        lines[index] = "| " + " | ".join(cells) + " |"
        changed.add(model)
    if changed != partials.keys():
        raise ValueError(f"missing partial models: {partials.keys() - changed}")
    destination.write_text("\n".join(lines) + "\n")


def cached_statistics(path: Path) -> dict[tuple[str, str], dict[str, object]]:
    with path.open(newline="") as source:
        return {(row["model"], row["benchmark"]): row for row in csv.DictReader(source)}


def recovered_binary_statistics(cell: plot.Cell, recovered: dict[str, dict]) -> dict[str, object] | None:
    """Use an audited recovered count when an archive itself is not sealed."""
    if cell.benchmark in plot.CONTINUOUS:
        return None
    run_id = cell.results_path.removesuffix("/results").rsplit("/", 1)[-1]
    entry = recovered.get(run_id)
    if entry is None or entry.get("source") != cell.results_path:
        return None
    metrics = entry.get("metrics", {}).get(cell.benchmark, {})
    if len(metrics) != 1 or abs(float(next(iter(metrics.values()))) - cell.score) > 0.002:
        return None
    coverage = entry["coverage"]
    count = int(coverage["n_completed"])
    if count < 2 or int(coverage["n_infrastructure_errors"]) / int(coverage["n_attempted"]) > 0.1:
        return None
    sem = math.sqrt(cell.score * (1 - cell.score) / (count - 1))
    return {
        "model": cell.model,
        "benchmark": cell.benchmark,
        "score": cell.score,
        "trial_count": count,
        "raw_reward_mean": cell.score,
        "sem": sem,
        "adjusted_sem": sem / math.sqrt(plot.SURVIVAL),
        "sem_basis": "recovered_binary_coverage",
        "source": cell.results_path,
        "statistics_version": "provisional-partial-swe-v1",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracker", type=Path, required=True)
    parser.add_argument("--cached-statistics", type=Path, required=True)
    parser.add_argument("--recovered-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_tracker = args.output_dir / "TRACKER_SOURCE_SNAPSHOT.md"
    source_tracker.write_bytes(args.tracker.read_bytes())
    partials: dict[str, dict[str, object]] = {}
    source_records = args.output_dir / "source-job-results"
    source_records.mkdir(exist_ok=True)
    for model, job in PARTIAL_JOBS.items():
        data, record = partial_score(model, *job)
        snapshot = source_records / f"{model.replace('/', '--')}.json"
        snapshot.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        data["source_snapshot"] = str(snapshot.relative_to(args.output_dir))
        data["source_snapshot_sha256"] = hashlib.sha256(snapshot.read_bytes()).hexdigest()
        partials[model] = data
    tracker = args.output_dir / "TRACKER_SWEBENCH_ESTIMATED.md"
    estimated_tracker(source_tracker, tracker, partials)
    (args.output_dir / "partial-swebench-provenance.json").write_text(json.dumps(partials, indent=2) + "\n")

    tournament = args.output_dir / "tournament"
    subprocess.run(
        [
            sys.executable,
            str(ANALYSIS / "snowball_tournament" / "pairwise.py"),
            "--tracker",
            str(tracker),
            "--output-dir",
            str(tournament),
        ],
        check=True,
    )
    tournament_report = tournament / "snowball_pairwise_win_rates.md"
    tournament_report.write_text(
        "> Provisional: SWE-bench scores in this tournament include estimates from incomplete runs. "
        "See ../README.md for coverage and method.\n\n" + tournament_report.read_text()
    )
    with (tournament / "snowball_pairwise_summary.csv").open(newline="") as source:
        tournament_rows = list(csv.DictReader(source))
    best_rate = float(tournament_rows[0]["mean_pairwise_win_rate"])
    leaders = [
        row["model"]
        for row in tournament_rows
        if math.isclose(float(row["mean_pairwise_win_rate"]), best_rate, rel_tol=0, abs_tol=1e-12)
    ]
    if len(leaders) > 1:
        tournament_report.write_text(
            f"> Joint first: {', '.join(f'`{model}`' for model in leaders)}. The selection JSON uses a "
            "deterministic tie-break; plots include all tied candidates.\n\n" + tournament_report.read_text()
        )
    benchmark_order, cells = plot.tracker_cells(tracker)
    flops_path = ANALYSIS / "critical_difference" / "baseline_flops.csv"
    with flops_path.open(newline="") as source:
        baseline_rows = list(csv.DictReader(source))
    baselines = [row["model"] for row in baseline_rows]
    models = [*leaders, *baselines]
    benchmarks = [benchmark for benchmark in benchmark_order if all(benchmark in cells[model] for model in models)]
    cache = cached_statistics(args.cached_statistics)
    recovered = json.loads(args.recovered_metrics.read_text())
    stats: dict[tuple[str, str], dict[str, object]] = {}
    for model in models:
        for benchmark in benchmarks:
            cell = cells[model][benchmark]
            key = (model, benchmark)
            if benchmark == "swebench-verified" and model in partials:
                data = partials[model]
                count = int(data["scored"])
                successes = int(data["successes"])
                # Jeffreys-posterior SEM keeps sparse 0/N partials uncertain.
                sem = math.sqrt((successes + 0.5) * (count - successes + 0.5) / ((count + 1) ** 2 * (count + 2)))
                stats[key] = {
                    "model": model,
                    "benchmark": benchmark,
                    "score": cell.score,
                    "trial_count": count,
                    "raw_reward_mean": float(data["score"]),
                    "sem": sem,
                    "adjusted_sem": sem / math.sqrt(plot.SURVIVAL),
                    "sem_basis": "partial_completed_trials_jeffreys",
                    "source": cell.results_path,
                    "statistics_version": "provisional-partial-swe-v1",
                }
            elif key in cache and abs(float(cache[key]["score"]) - cell.score) < 0.001:
                stats[key] = cache[key]
            else:
                stats[key] = recovered_binary_statistics(cell, recovered) or plot.cell_statistics(cell, recovered)
    figure_dirs = {}
    for leader in leaders:
        figure_models = [leader, *baselines]
        suffix = re.sub(r"[^a-z0-9]+", "-", leader.rsplit("/", 1)[-1].lower()).strip("-")
        figures = args.output_dir / f"critical-difference-{suffix}"
        figure_dirs[leader] = figures.name
        figures.mkdir(exist_ok=True)
        statistics_rows = [stats[(m, b)] for m in figure_models for b in benchmarks]
        statistics_fields = sorted({field for row in statistics_rows for field in row})
        with (figures / "cell_statistics.csv").open("w", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=statistics_fields)
            writer.writeheader()
            writer.writerows(statistics_rows)
        flops = {row["model"]: float(row["total_flops"]) for row in baseline_rows}
        flops[leader] = plot.SNOWBALL_FLOPS
        plot.write_csv(
            figures / "model_flops.csv",
            [
                {
                    "model": model,
                    "total_flops": flops[model],
                    "basis": (
                        "September 17 Step92 proxy applied to provisional Snowball candidate"
                        if model == leader
                        else next(row["basis"] for row in baseline_rows if row["model"] == model)
                    ),
                }
                for model in figure_models
            ],
        )
        for controlled in (False, True):
            plot.critical_difference(figure_models, benchmarks, stats, flops, controlled, figures, leader)
    tracker_hash = hashlib.sha256(tracker.read_bytes()).hexdigest()
    source_tracker_hash = hashlib.sha256(source_tracker.read_bytes()).hexdigest()
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    timestamp = datetime.now(UTC).isoformat(timespec="seconds")
    summary = [
        "# Provisional SWE-bench estimate analysis",
        "",
        f"Generated {timestamp} from live tracker `{args.tracker}`; temporary tracker SHA-256 `{tracker_hash}`.",
        f"Frozen source `TRACKER_SOURCE_SNAPSHOT.md` SHA-256 `{source_tracker_hash}`; "
        f"analysis script SHA-256 `{script_hash}` at "
        "`experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/analysis/estimated_swebench/estimate.py` "
        "in the Marin campaign worktree. "
        "The exact in-flight Harbor result.json files are copied in `source-job-results/` with hashes in "
        "`partial-swebench-provenance.json`.",
        "The live tracker was not changed. These are estimates, not sealed policy scores.",
        "",
        "| Model | Estimated score | Scoreable | Completed / 500 | Last archive update (UTC) |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for data in partials.values():
        summary.append(
            f"| {data['model']} | {float(data['score']):.3f} | {data['scored']} | "
            f"{data['completed']}/500 | {data['updated_at']} |"
        )
    summary.extend(
        [
            "",
            f"Tournament: {len(tournament_rows)} Snowball/Grug models; "
            f"{'joint leaders' if len(leaders) > 1 else 'leader'}: {', '.join(f'`{model}`' for model in leaders)}. "
            "Each jointly scored benchmark has equal weight within a pair. The selection JSON uses a deterministic "
            "tie-break if needed. The allkimi SWE-bench estimate has few scoreable trials; Ling-lite has substantial "
            "unscored coverage. These two estimates are especially fragile. None of the provisional values is "
            "presented as a sealed result.",
            f"Critical-difference analysis for the leader uses {len(benchmarks)} common benchmarks and "
            "the same 10 "
            "FLOP-estimated baselines "
            "as the campaign plots. Partial SWE-bench uncertainty uses a Jeffreys-posterior "
            "SEM based on scoreable trials, expanded by 1/sqrt(0.9) for infrastructure tolerance; all other cells use "
            "the existing campaign statistics and plotting method. This prevents a sparse 0/N estimate from having zero "
            "uncertainty. The point score is observed solves / scoreable trials, matching the sealed SWE-bench "
            "metric; Harbor's in-flight mean instead divides by completed trials. FLOP estimates remain fixed. "
            "The FLOP-controlled analysis applies the September 17 Step92 FLOP proxy to the selected "
            "Snowball/Grug candidate. "
            "Lineage-specific training compute is not estimated here; that assumption limits interpretation. "
            "SOTOPIA-hard is excluded from the figures because three baselines are still pending, so these figures use "
            "SWE-bench rather than SOTOPIA-hard as their 25th benchmark. They are exploratory, not release-ready.",
            "See `tournament/` for pairwise tables, "
            + ", ".join(f"`{directory}/`" for directory in figure_dirs.values())
            + " for the controlled and uncontrolled figures and underlying CSVs, "
            "and `TRACKER_SWEBENCH_ESTIMATED.md` for the temporary score table.",
            "",
        ]
    )
    (args.output_dir / "README.md").write_text("\n".join(summary))
    print(f"estimated six SWE-bench cells; joint leaders {leaders}; plotted {len(benchmarks)} benchmarks")


if __name__ == "__main__":
    main()

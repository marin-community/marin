#!/usr/bin/env python3
"""Plot normalized critical differences for the tournament winner and baselines.

The rank simulation and FLOP control are adapted from the September 17
campaign's release-figures/generate_release_figures.py.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from finestore.reader import ReadView
from marin.evaluation.lm_eval_samples import summarize_native_eval_samples
from rigging.filesystem.buckets import filesystem_for
from scipy.stats import rankdata, studentized_range

SURVIVAL = 0.9
DRAWS = 100_000
SEED = 20260924
SCORE = re.compile(r"^([01](?:\.\d+)?)\s+\((s3://[^)]+/results)\)$")
CONTINUOUS = frozenset({"truthfulqa", "mrcr", "SOTOPIA-hard"})
SNOWBALL_FLOPS = 1.2440924703713099e23
STATISTICS_VERSION = "recovered-metrics-v3"


@dataclass(frozen=True)
class Cell:
    model: str
    benchmark: str
    score: float
    results_path: str


def tracker_cells(path: Path) -> tuple[list[str], dict[str, dict[str, Cell]]]:
    lines = [line for line in path.read_text().splitlines() if line.startswith("|")]
    header = [part.strip() for part in lines[0].strip("|").split("|")]
    benchmarks = header[1:]
    cells: dict[str, dict[str, Cell]] = {}
    for line in lines[3:]:
        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) != len(header):
            continue
        row: dict[str, Cell] = {}
        for benchmark, value in zip(benchmarks, parts[1:], strict=True):
            if match := SCORE.match(value):
                row[benchmark] = Cell(parts[0], benchmark, float(match.group(1)), match.group(2))
        cells[parts[0]] = row
    return benchmarks, cells


def read_json(url: str) -> dict:
    fs, path = filesystem_for(url)
    with fs.open(path) as source:
        return json.load(source)


def scored_count(cell: Cell) -> tuple[int, str]:
    record_url = cell.results_path.removesuffix("/results") + "/record.json"
    fs, path = filesystem_for(record_url)
    if not fs.exists(path):
        if cell.benchmark != "NUPA" or not ReadView(cell.results_path).is_sealed():
            raise ValueError(f"no canonical record or sealed Evalchemy archive for {cell.model} / {cell.benchmark}")
        summary = summarize_native_eval_samples(cell.results_path)
        if len(summary.coverage) != 1 or len(summary.canonical_metrics) != 1:
            raise ValueError(f"ambiguous native Evalchemy summary for {cell.model} / {cell.benchmark}")
        coverage = next(iter(summary.coverage.values()))
        metric = next(iter(summary.canonical_metrics.values()))["accuracy"]
        benchmark = int(coverage.n_benchmark or 0)
        infrastructure = int(coverage.errors.get("EVALCHEMY_INFRASTRUCTURE_ERROR", 0))
        if benchmark < 2 or infrastructure / benchmark > 0.1 or abs(metric - cell.score) > 0.002:
            raise ValueError(f"unreportable native Evalchemy summary for {cell.model} / {cell.benchmark}")
        return coverage.n_scored, "sealed_evalchemy_native_coverage"
    record = read_json(record_url)
    coverage = record.get("coverage") or {}
    if record.get("status") not in {"succeeded", "infra_failed"} or len(coverage) != 1:
        view = ReadView(cell.results_path)
        if not view.is_sealed():
            raise ValueError(f"unsealed results for {cell.model} / {cell.benchmark}")
        rewards, _ = continuous_rewards(cell)
        return int(rewards.size), "sealed_finestore_sample_rewards"
    task_name, task_coverage = next(iter(coverage.items()))
    count = int(task_coverage.get("n_scored") or 0)
    if count == 0:
        count = int((record.get("metrics") or {}).get(task_name, {}).get("scored_count") or 0)
    if count == 0 and task_coverage.get("errors", {}).get("ungraded") == task_coverage.get("n_attempted"):
        count = int(task_coverage["n_attempted"])
    if count < 2:
        raise ValueError(f"insufficient scored trials for {cell.model} / {cell.benchmark}: {count}")
    return count, "record_coverage"


def continuous_rewards(cell: Cell) -> tuple[np.ndarray, str]:
    """Select the scored sample metric matching the canonical tracker point estimate."""
    table = ReadView(cell.results_path).scan("samples", columns=["grading", "metrics", "filter"])
    if table is None:
        raise ValueError(f"no normalized samples for {cell.model} / {cell.benchmark}")
    candidates: dict[str, list[float]] = {}
    for row in table.to_pylist(maps_as_pydicts="strict"):
        grade = row.get("grading") or {}
        if grade.get("score") is not None:
            key = f"grading:{grade.get('metric')}:{grade.get('filter')}"
            candidates.setdefault(key, []).append(float(grade["score"]))
        for name, value in (row.get("metrics") or {}).items():
            candidates.setdefault(f"metrics:{name}:{row.get('filter')}", []).append(float(value))
    viable = {
        key: np.asarray(values, dtype=float)
        for key, values in candidates.items()
        if len(values) > 1 and np.isfinite(values).sum() > 1
    }
    if not viable:
        raise ValueError(f"no finite sample metric for {cell.model} / {cell.benchmark}")
    selector, rewards = min(
        viable.items(), key=lambda item: (abs(float(np.nanmean(item[1])) - cell.score), -item[1].size, item[0])
    )
    rewards = rewards[np.isfinite(rewards)]
    if abs(float(np.mean(rewards)) - cell.score) > 0.02:
        raise ValueError(f"sample metric differs from tracker score for {cell.model} / {cell.benchmark}: {selector}")
    return rewards, selector


def recovered_statistics(cell: Cell, recovered: dict[str, dict]) -> dict[str, object] | None:
    run_id = cell.results_path.removesuffix("/results").rsplit("/", 1)[-1]
    entry = recovered.get(run_id)
    if entry is None or entry.get("source") != cell.results_path or "reward_stderr" not in entry.get("aggregation", {}):
        return None
    if entry.get("recovered_status") not in {None, "succeeded", "infra_failed"}:
        raise ValueError(f"unreportable recovered result for {cell.model} / {cell.benchmark}")
    coverage = entry["coverage"]
    count = int(coverage["n_completed"])
    attempted = int(coverage["n_attempted"])
    infrastructure_errors = int(coverage["n_infrastructure_errors"])
    if count < 2 or attempted < count or infrastructure_errors / attempted > 0.1:
        raise ValueError(f"invalid recovered coverage for {cell.model} / {cell.benchmark}")
    if len(entry["metrics"]) != 1:
        raise ValueError(f"ambiguous recovered task for {cell.model} / {cell.benchmark}")
    metrics = next(iter(entry["metrics"].values()))
    if len(metrics) != 1:
        raise ValueError(f"ambiguous recovered metric for {cell.model} / {cell.benchmark}")
    mean = float(next(iter(metrics.values())))
    if abs(mean - cell.score) > 0.002:
        raise ValueError(f"recovered metric differs from tracker for {cell.model} / {cell.benchmark}")
    sem = float(entry["aggregation"]["reward_stderr"])
    return {
        "model": cell.model,
        "benchmark": cell.benchmark,
        "score": cell.score,
        "trial_count": count,
        "raw_reward_mean": mean,
        "sem": sem,
        "adjusted_sem": sem / math.sqrt(SURVIVAL),
        "sem_basis": "recovered_trial_reward_stderr",
        "source": cell.results_path,
        "statistics_version": STATISTICS_VERSION,
    }


def cell_statistics(cell: Cell, recovered: dict[str, dict]) -> dict[str, object]:
    if statistics := recovered_statistics(cell, recovered):
        return statistics
    count, count_basis = scored_count(cell)
    if cell.benchmark in CONTINUOUS:
        rewards, selector = continuous_rewards(cell)
        if rewards.size != count:
            raise ValueError(
                f"sample count differs from coverage for {cell.model} / {cell.benchmark}: {rewards.size} != {count}"
            )
        mean = float(np.mean(rewards))
        sem = float(np.std(rewards, ddof=1) / math.sqrt(count))
    else:
        selector = f"binary_score_from_{count_basis}"
        mean = cell.score
        sem = math.sqrt(cell.score * (1 - cell.score) / (count - 1))
    return {
        "model": cell.model,
        "benchmark": cell.benchmark,
        "score": cell.score,
        "trial_count": count,
        "raw_reward_mean": mean,
        "sem": sem,
        "adjusted_sem": sem / math.sqrt(SURVIVAL),
        "sem_basis": selector,
        "source": cell.results_path,
        "statistics_version": STATISTICS_VERSION,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect_statistics(
    models: list[str],
    benchmarks: list[str],
    cells: dict[str, dict[str, Cell]],
    output: Path,
    recovered: dict[str, dict],
    recovered_hash: str,
) -> list[dict[str, object]]:
    cache = {}
    if output.exists():
        with output.open(newline="") as source:
            for row in csv.DictReader(source):
                if (
                    row.get("statistics_version") == STATISTICS_VERSION
                    and row.get("recovered_manifest_sha256") == recovered_hash
                ):
                    cache[(row["model"], row["benchmark"], row["source"], row["score"])] = row
    rows: list[dict[str, object]] = []
    pending = []
    for model in models:
        for benchmark in benchmarks:
            cell = cells[model][benchmark]
            key = (model, benchmark, cell.results_path, str(cell.score))
            if key in cache:
                rows.append(cache[key])
            else:
                pending.append(cell)
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = {executor.submit(cell_statistics, cell, recovered): cell for cell in pending}
        failures = []
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:
                failures.append(f"{futures[future].model} / {futures[future].benchmark}: {exc}")
                continue
            result["recovered_manifest_sha256"] = recovered_hash
            rows.append(result)
            print(f"scored {result['model']} / {result['benchmark']}", flush=True)
    rows.sort(key=lambda row: (models.index(str(row["model"])), benchmarks.index(str(row["benchmark"]))))
    write_csv(output, rows)
    if failures:
        raise ValueError("Could not derive task statistics:\n" + "\n".join(failures))
    return rows


def sampled_ranks(scores: np.ndarray, sems: np.ndarray, log_flops: np.ndarray, controlled: bool) -> np.ndarray:
    rng = np.random.default_rng(SEED + int(controlled))
    models, benchmarks = scores.shape
    mean_ranks = np.empty((DRAWS, models), dtype=np.float32)
    centered = log_flops - np.mean(log_flops)
    denominator = benchmarks * np.sum(centered**2)
    for start in range(0, DRAWS, 2_000):
        stop = min(start + 2_000, DRAWS)
        draws = rng.normal(scores, sems, size=(stop - start, models, benchmarks)).clip(0, 1)
        means = np.mean(draws, axis=1, keepdims=True)
        scales = np.std(draws, axis=1, ddof=1, keepdims=True)
        normalized = np.divide(draws - means, scales, out=np.zeros_like(draws), where=scales > 0)
        values = normalized
        if controlled:
            beta = np.sum(normalized * centered[None, :, None], axis=(1, 2)) / denominator
            values = normalized - beta[:, None, None] * centered[None, :, None]
        matrix = -np.transpose(values, (0, 2, 1)).reshape(-1, models)
        ranks = rankdata(matrix, method="average", axis=1).reshape(stop - start, benchmarks, models)
        mean_ranks[start:stop] = np.mean(ranks, axis=1)
    return mean_ranks


def critical_difference(
    models: list[str],
    benchmarks: list[str],
    stats: dict[tuple[str, str], dict[str, object]],
    flops: dict[str, float],
    controlled: bool,
    output: Path,
    winner: str,
) -> None:
    scores = np.asarray([[float(stats[(model, benchmark)]["score"]) for benchmark in benchmarks] for model in models])
    sems = np.asarray(
        [[float(stats[(model, benchmark)]["adjusted_sem"]) for benchmark in benchmarks] for model in models]
    )
    draws = sampled_ranks(scores, sems, np.log10([flops[model] for model in models]), controlled)
    mean_ranks = np.mean(draws, axis=0)
    radius = float(np.quantile(np.max(np.abs(draws - mean_ranks), axis=1), 0.95))
    k, n = len(models), len(benchmarks)
    cd = float(studentized_range.ppf(0.95, k, np.inf) / math.sqrt(2) * math.sqrt(k * (k + 1) / (6 * n)))
    order = np.argsort(mean_ranks)
    stem = "critical_difference_flop_controlled" if controlled else "critical_difference_uncontrolled"
    rows = [
        {
            "model": models[index],
            "mean_rank": float(mean_ranks[index]),
            "simultaneous_95_radius": radius,
            "nemenyi_cd": cd,
            "benchmarks": n,
        }
        for index in order
    ]
    write_csv(output / f"{stem}.csv", rows)
    fig, ax = plt.subplots(figsize=(9.5, 6.3))
    for position, index in enumerate(order):
        model = models[index]
        selected = model == winner
        ax.errorbar(
            mean_ranks[index],
            position,
            xerr=radius,
            fmt="*" if selected else "o",
            markersize=12 if selected else 6,
            color="#f05a28" if selected else "#4f5963",
            ecolor="#aeb4ba",
            capsize=2.5,
            zorder=3,
        )
    labels = [models[index].split("/", 1)[-1] for index in order]
    ax.set_yticks(np.arange(k), labels)
    ax.invert_yaxis()
    ax.set_xlim(1, k)
    ax.set_xlabel("Mean rank (lower is better); bars are simultaneous 95% uncertainty bands")
    ax.set_title(
        "Normalized critical difference — FLOP-controlled"
        if controlled
        else "Normalized critical difference — uncontrolled"
    )
    ax.plot([1, 1 + cd], [-0.75, -0.75], color="#202124", linewidth=2)
    ax.text(1 + cd / 2, -1.05, f"Nemenyi CD = {cd:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(k - 0.3, -1.3)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"{stem}.{suffix}", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracker", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--recovered-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    tracker_hash = hashlib.sha256(args.tracker.read_bytes()).hexdigest()
    if tracker_hash != selection["tracker_sha256"]:
        raise ValueError("selection and plot inputs use different tracker snapshots; rerun the tournament")
    winner = selection["winner"]
    benchmark_order, cells = tracker_cells(args.tracker)
    flops_path = Path(__file__).with_name("baseline_flops.csv")
    with flops_path.open(newline="") as source:
        flops_rows = list(csv.DictReader(source))
    baselines = [row["model"] for row in flops_rows]
    models = [winner, *baselines]
    if any(model not in cells for model in models):
        raise ValueError(f"missing model rows: {set(models) - set(cells)}")
    benchmarks = [name for name in benchmark_order if all(name in cells[model] for model in models)]
    if len(benchmarks) < 2:
        raise ValueError("insufficient complete paired benchmarks")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    recovered_bytes = args.recovered_metrics.read_bytes()
    recovered = json.loads(recovered_bytes)
    recovered_hash = hashlib.sha256(recovered_bytes).hexdigest()
    stats_rows = collect_statistics(
        models, benchmarks, cells, args.output_dir / "cell_statistics.csv", recovered, recovered_hash
    )
    stats = {(str(row["model"]), str(row["benchmark"])): row for row in stats_rows}
    flops = {row["model"]: float(row["total_flops"]) for row in flops_rows}
    flops[winner] = SNOWBALL_FLOPS
    write_csv(
        args.output_dir / "model_flops.csv",
        [
            {
                "model": model,
                "total_flops": flops[model],
                "basis": "September 17 Step92 proxy"
                if model == winner
                else next(row["basis"] for row in flops_rows if row["model"] == model),
            }
            for model in models
        ],
    )
    for controlled in (False, True):
        critical_difference(models, benchmarks, stats, flops, controlled, args.output_dir, winner)
    excluded = [name for name in benchmark_order if name not in benchmarks]
    (args.output_dir / "README.md").write_text(
        f"# Critical-difference figures\n\nSelected Snowball: `{winner}`. "
        f"The cohort has {len(baselines)} non-Snowball baselines and "
        f"{len(benchmarks)} jointly scored benchmarks. Excluded pending benchmarks: {', '.join(excluded) or 'none'}. "
        f"Tracker SHA-256: `{tracker_hash}`. Recovered-metrics SHA-256: `{recovered_hash}`. "
        f"Baseline-FLOPs SHA-256: `{hashlib.sha256(flops_path.read_bytes()).hexdigest()}`.\n\n"
        "Scores are sampled 100,000 times from independent normal distributions using task-level SEM expanded by "
        "1/sqrt(0.9) for infrastructure tolerance, then clipped to [0, 1]. Binary-score SEM is the exact Bernoulli "
        "sample-variance expression from the score and scored-trial count; continuous metrics use stored trial rewards. "
        "Audited recovered cells use their independently aggregated reward SEM and scored-trial count. "
        "Within each draw, scores are normalized by benchmark. The controlled plot removes one common fitted slope "
        "against log10(total training FLOPs). FLOP estimates are fixed and their uncertainty is not propagated. "
        "Bars are simultaneous 95% rank bands; the Nemenyi CD uses alpha=0.05. "
        "Baseline FLOPs come from the September 17 campaign's NON_SNOWBALL_FLOP_ESTIMATES.csv; Step92 uses its "
        "September 17 release-figure estimate.\n"
    )
    print(f"plotted {len(models)} models over {len(benchmarks)} benchmarks; excluded {excluded}")


if __name__ == "__main__":
    main()

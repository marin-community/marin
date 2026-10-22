# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Compare the mixed-RLVR arms (marin #10214) to the October 5 Datakit SFT, paired per item.

    python -m experiments.post_training.pivotrl.mixed_rlvr_analysis collect <data-dir>
    python -m experiments.post_training.pivotrl.mixed_rlvr_analysis test <data-dir> [--series a,b,c] [--steps 4,8,12]
    python -m experiments.post_training.pivotrl.mixed_rlvr_analysis plot <data-dir> <out-prefix> [--series a,b,c]

``collect`` reads every record the arms' evaluations wrote under the CoreWeave evals prefix, keeps the fullest run per
arm label, step and benchmark (the newest among equal ones, whatever the record's status), and writes ``tasks.json``
(per-task Harbor rewards for SWE-bench Verified and OT-TBLite) and ``samples.json`` (per-question Evalchemy correctness
for AIME24, GPQA Diamond, Math500 and MMLU-Pro) beside ``records.json``. The reference model's rows come from the
October 5 cohort records in ``mixed_rlvr_evals/``.

Scoring: SWE-bench Verified counts a trial with no verifier result as unsolved and uses each task's first trial;
OT-TBLite counts a task solved when at least two of its three trials score 1; AIME24 and GPQA Diamond average their
repeated trials per question. ``test`` prints paired differences with an exact McNemar test for single-trial benchmarks
and a paired sign-flip permutation test for averaged ones. ``plot`` draws each arm's paired difference from the reference
per step with an 80% bootstrap interval over items.
"""

import argparse
import collections
import io
import json
import logging
import math
import os
import re
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from dotenv import dotenv_values
from rigging.filesystem.factory import url_to_fs
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.post_training.pivotrl.mixed_rlvr_arms import ORIGINAL_SEED
from experiments.post_training.pivotrl.mixed_rlvr_evals import EVALCHEMY, HARBOR, REFERENCE

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[3]
EVALS_ROOT = "s3://marin-us-east-02a/marin/evals"
REF = "REF"
RUN_PATTERN = re.compile(r"pivot-mixed-rlvr-([a-z0-9-]+?)-\d{8}-\d{6}-step(\d+)")
ARM_LABELS = {
    "a": ("RLVR mix", "#8C8C8C"),
    "b": ("A mix, 20% replaced by selected pivots", "#D55E00"),
    "c": ("A mix, 20% replaced by random turns", "#0072B2"),
}
PANELS = (
    ("swebench-verified", "SWE-bench Verified"),
    ("ot-tblite-recovery", "OT-TBLite"),
    ("gpqa-diamond", "GPQA-Diamond"),
    ("mmlu-pro", "MMLU-Pro"),
    ("math500", "Math500"),
    ("aime24", "AIME24"),
)
BOOTSTRAP = 10000
PERMUTATIONS = 20000
TBLITE_SOLVED_TRIALS = 2


def run_key(label: str, step: int) -> str:
    return f"{label}-step{step}"


def exact_mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for ``b`` arm-only and ``c`` reference-only successes."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n)


def sign_flip_p(diffs: Sequence[float], rng: np.random.Generator, permutations: int = PERMUTATIONS) -> float:
    """Paired sign-flip permutation p-value for the mean of per-item differences."""
    nonzero = np.array([d for d in diffs if d], dtype=float)
    if not nonzero.size:
        return 1.0
    observed = abs(nonzero.sum())
    signs = rng.choice([-1.0, 1.0], size=(permutations, nonzero.size))
    hits = int((np.abs(signs @ nonzero) >= observed - 1e-12).sum())
    return (hits + 1) / (permutations + 1)


def swebench_solved(rewards: Sequence[float | None]) -> bool:
    """Strict first-trial scoring: a missing verifier result is a failure."""
    return bool(rewards) and rewards[0] == 1.0


def tblite_solved(rewards: Sequence[float | None]) -> bool:
    return sum(r == 1.0 for r in rewards) >= TBLITE_SOLVED_TRIALS


def harbor_scores(runs: dict[str, dict[str, list[float | None]]], bench: str) -> dict[str, dict[str, float]]:
    """Per-task 0/1 score for each run of a Harbor benchmark."""
    solved = swebench_solved if bench == "swebench-verified" else tblite_solved
    return {
        key: {task: float(solved(rewards)) for task, rewards in tasks.items() if rewards} for key, tasks in runs.items()
    }


def sample_scores(runs: dict[str, dict[str, bool | None]]) -> dict[str, dict[str, float]]:
    """Per-question accuracy, averaged over repeated trials, for each run of an Evalchemy benchmark."""
    out = {}
    for key, samples in runs.items():
        per_question = collections.defaultdict(list)
        for doc_trial, correct in samples.items():
            if correct is not None:
                per_question[doc_trial.split("\t")[0]].append(float(correct))
        out[key] = {question: sum(values) / len(values) for question, values in per_question.items()}
    return out


@dataclass(frozen=True)
class PairedDelta:
    """Mean arm-minus-reference difference in points over shared items, with its 80% bootstrap interval."""

    mean: float
    low: float
    high: float
    n: int


def paired_delta(arm: dict[str, float], ref: dict[str, float], rng: np.random.Generator) -> PairedDelta:
    items = sorted(set(arm) & set(ref))
    diffs = np.array([arm[i] - ref[i] for i in items])
    boot = diffs[rng.integers(0, len(diffs), size=(BOOTSTRAP, len(diffs)))].mean(axis=1)
    low, high = np.percentile(boot, [10, 90])
    return PairedDelta(100 * diffs.mean(), 100 * low, 100 * high, len(items))


def _filesystem():
    """CoreWeave object storage through Marin's fsspec factory, with the keys from ``<repo>/.env`` when unset."""
    for key, value in dotenv_values(REPO / ".env").items():
        if value and key.startswith("CW_KEY_"):
            os.environ.setdefault(key, value)
    configure_coreweave_s3()
    fs, _ = url_to_fs(EVALS_ROOT)
    return fs


def _records(fs) -> list[dict]:
    records = []
    for directory in fs.ls(EVALS_ROOT.removeprefix("s3://")):
        if "pivot-mixed-rlvr" not in directory or not fs.exists(directory + "/record.json"):
            continue
        record = json.loads(fs.cat(directory + "/record.json"))
        match = RUN_PATTERN.search(record["model"]["name"])
        if match is None:
            continue
        records.append(
            {
                "dir": directory.rsplit("/", 1)[-1],
                "status": record["status"],
                "eval": record["eval"]["name"],
                "key": run_key(match.group(1), int(match.group(2))),
                "created": record["created_at"],
                "results": directory + "/results",
            }
        )
    return records


def _harbor_rewards(fs, base: str) -> dict[str, list[float | None]]:
    """Per-task rewards from each trial's result.json; None when an exception left no verifier result."""
    out: dict[str, list[float | None]] = {}
    jobs = base + "/harbor_jobs"
    if not fs.exists(jobs):
        return out
    depth = jobs.count("/") + 3
    files = [p for p in fs.find(jobs) if p.endswith("/result.json") and p.count("/") == depth]
    contents = fs.cat(files, on_error="return")
    unreadable = [path for path, raw in contents.items() if isinstance(raw, BaseException)]
    if unreadable:
        logger.warning(
            "%d of %d trial results under %s could not be read: %s", len(unreadable), len(files), jobs, unreadable[:3]
        )
    for raw in contents.values():
        if isinstance(raw, BaseException):
            continue
        result = json.loads(raw)
        verifier = result.get("verifier_result")
        reward = ((verifier or {}).get("rewards") or {}).get("reward")
        missing = verifier is None and result.get("exception_info") and reward is None
        out.setdefault(result["task_name"], []).append(None if missing else float(reward or 0.0))
    return out


def _sample_correctness(fs, base: str) -> dict[str, bool | None]:
    out = {}
    for path in fs.glob(base + "/data/samples/**/*.parquet"):
        table = pq.read_table(io.BytesIO(fs.cat(path)), columns=["doc_id", "trial_id", "correct"])
        columns = (table.column(name).to_pylist() for name in ("doc_id", "trial_id", "correct"))
        for doc, trial, correct in zip(*columns, strict=True):
            out[f"{doc}\t{trial}"] = correct
    return out


def _merge_fullest(parts: Iterable[tuple[str, dict]]) -> dict:
    """The fullest run wins on overlap, the newest among equal ones; partial runs only fill items it lacks.

    A resubmission supersedes an earlier run of the same checkpoint and benchmark, so among complete runs the
    latest ``created`` timestamp is the one that counts.
    """
    merged: dict = {}
    for _, part in sorted(parts, key=lambda created_part: (len(created_part[1]), created_part[0]), reverse=True):
        for item, value in part.items():
            merged.setdefault(item, value)
    return merged


def collect(data_dir: Path) -> None:
    fs = _filesystem()
    records = _records(fs)
    (data_dir / "records.json").write_text(json.dumps(records, indent=1) + "\n")
    runs: dict[tuple[str, str], list[tuple[str, str]]] = collections.defaultdict(list)
    # Every run contributes its scored items regardless of the record's status: a coordinator killed after its last
    # trial finished is recorded as infra_failed with complete results, while a resubmission supersedes it by date.
    for record in records:
        runs[(record["eval"], record["key"])].append((record["created"], record["results"]))
    for bench in (*HARBOR, *EVALCHEMY):
        runs[(bench, REF)].append(
            (REFERENCE[bench]["created_at"], REFERENCE[bench]["results_path"].removeprefix("s3://"))
        )
    tasks: dict[str, dict[str, dict]] = {bench: {} for bench in HARBOR}
    samples: dict[str, dict[str, dict]] = {bench: {} for bench in EVALCHEMY}
    with ThreadPoolExecutor(8) as pool:
        jobs = {
            (bench, key): [
                (created, pool.submit(_harbor_rewards if bench in HARBOR else _sample_correctness, fs, base))
                for created, base in bases
            ]
            for (bench, key), bases in runs.items()
        }
        for (bench, key), futures in jobs.items():
            target = tasks if bench in HARBOR else samples
            target[bench][key] = _merge_fullest((created, future.result()) for created, future in futures)
    (data_dir / "tasks.json").write_text(json.dumps(tasks) + "\n")
    (data_dir / "samples.json").write_text(json.dumps(samples) + "\n")
    for bench in (*HARBOR, *EVALCHEMY):
        source = tasks if bench in HARBOR else samples
        print(bench, {key: len(items) for key, items in sorted(source[bench].items())})


def shared_items(runs: dict[str, dict[str, float]], keys: Sequence[str]) -> dict[str, dict[str, float]]:
    """Inner join: keep only the items that every evaluated run among ``keys`` has scored.

    A run with no scored items yet is left out of the join rather than emptying it. While an agentic suite is still
    running, the join is the tasks it has finished so far, which keeps every series paired on the same instances but
    is not a random subset of the benchmark.
    """
    evaluated = [key for key in keys if runs.get(key)]
    shared = set.intersection(*(set(runs[key]) for key in evaluated)) if evaluated else set()
    return {key: {item: score for item, score in runs[key].items() if item in shared} for key in evaluated}


def load_scores(
    data_dir: Path, series: Sequence[str] = (), steps: Sequence[int] = (), shared: bool = False
) -> dict[str, dict[str, dict[str, float]]]:
    """Per-item scores for every run; with ``shared`` only the items every evaluated run in the selection scored."""
    tasks = json.loads((data_dir / "tasks.json").read_text())
    samples = json.loads((data_dir / "samples.json").read_text())
    scores = {bench: harbor_scores(tasks[bench], bench) for bench in HARBOR}
    scores |= {bench: sample_scores(samples[bench]) for bench in EVALCHEMY}
    if shared:
        keys = [REF, *(run_key(label, step) for label in series for step in steps)]
        scores = {bench: shared_items(runs, keys) for bench, runs in scores.items()}
    return scores


def report(data_dir: Path, series: Sequence[str], steps: Sequence[int], shared: bool) -> None:
    scores = load_scores(data_dir, series, steps, shared)
    rng = np.random.default_rng(0)
    samples = json.loads((data_dir / "samples.json").read_text())
    for bench in (*HARBOR, *EVALCHEMY):
        per = scores[bench]
        repeats = 1 if bench in HARBOR else max(len(v) for v in samples[bench].values()) // max(len(per[REF]), 1)
        method = "exact McNemar" if repeats == 1 else "paired sign-flip permutation on per-question accuracy"
        print(f"\n### {bench} ({method})")
        for step in steps:
            keys = [REF, *(run_key(label, step) for label in series)]
            # A run that has not scored anything yet (its record exists but no items landed) is skipped;
            # a step with no arm left, or no item shared by every present run, prints only its count.
            present = [key for key in keys if per.get(key)]
            common = set.intersection(*(set(per[key]) for key in present)) if present else set()
            n = len(common)
            if len(present) < 2 or n == 0:
                print(f"step {step:>2}: n={n}  (no comparable runs)")
                continue
            print(
                f"step {step:>2}: n={n}  "
                + "  ".join(f"{key} {sum(per[key][i] for i in common) / n:.3f}" for key in present)
            )
            arms = present[1:]
            pairs = [(x, y) for index, x in enumerate(arms) for y in arms[:index]] + [(x, REF) for x in arms]
            for x, y in pairs:
                diffs = [per[x][i] - per[y][i] for i in common]
                mean = sum(diffs) / n * 100
                if repeats == 1:
                    b = sum(d > 0 for d in diffs)
                    c = sum(d < 0 for d in diffs)
                    p = exact_mcnemar_p(b, c)
                    detail = f"{b:4d}:{c:<4d}"
                else:
                    p = sign_flip_p(diffs, rng)
                    detail = " " * 9
                print(f"   {x:>12} vs {y:<12} {detail} diff {mean:+5.1f}  p={p:.4f}" + ("  *" if p < 0.05 else ""))


def paired_points(
    scores: dict[str, dict[str, dict[str, float]]], series: Sequence[str], steps: Sequence[int]
) -> tuple[dict[tuple[str, str], list[tuple[int, PairedDelta]]], list[str]]:
    """Paired deltas per benchmark and arm label at each evaluated step, with the CSV rows that describe them.

    A label whose later checkpoints are not evaluated yet contributes only the steps it has.
    """
    rng = np.random.default_rng(0)
    points: dict[tuple[str, str], list[tuple[int, PairedDelta]]] = {}
    rows = ["benchmark,arm,step,n,reference_accuracy,arm_accuracy,delta_points,ci80_low,ci80_high"]
    for bench, _ in PANELS:
        ref = scores[bench][REF]
        for label in series:
            # A run whose samples are all unscored so far has an empty score dict and is not evaluated yet.
            evaluated = [step for step in steps if scores[bench].get(run_key(label, step))]
            deltas = [(step, paired_delta(scores[bench][run_key(label, step)], ref, rng)) for step in evaluated]
            points[(bench, label)] = deltas
            for step, delta in deltas:
                items = set(scores[bench][run_key(label, step)]) & set(ref)
                ref_acc = 100 * np.mean([ref[i] for i in items])
                rows.append(
                    f"{bench},{label},{step},{delta.n},{ref_acc:.2f},{ref_acc + delta.mean:.2f},"
                    f"{delta.mean:.2f},{delta.low:.2f},{delta.high:.2f}"
                )
    return points, rows


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Helvetica Neue",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#444444",
            "axes.labelcolor": "#222222",
            "xtick.color": "#444444",
            "ytick.color": "#444444",
        }
    )


def plot(data_dir: Path, out_prefix: Path, series: Sequence[str], steps: Sequence[int], shared: bool) -> None:
    scores = load_scores(data_dir, series, steps, shared)
    points, rows = paired_points(scores, series, steps)
    _style()
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.6), sharex=True, sharey=True)
    for ax, (bench, title) in zip(axes.flat, PANELS, strict=True):
        ref = scores[bench][REF]
        ax.axhline(0, color="#222222", lw=0.8, ls=(0, (4, 3)), zorder=1)
        for label in series:
            # A replicate label such as ``b-seed43`` shares arm B's colour and is drawn dashed with hollow markers.
            base, _, seed = label.partition("-seed")
            description, color = ARM_LABELS[base]
            deltas = points[(bench, label)]
            if not deltas:
                continue
            evaluated = [step for step, _ in deltas]
            emphasis = base != "a"
            ax.fill_between(
                evaluated,
                [d.low for _, d in deltas],
                [d.high for _, d in deltas],
                color=color,
                alpha=(0.16 if emphasis else 0.13) * (0.5 if seed else 1.0),
                lw=0,
                zorder=2,
            )
            ax.plot(
                evaluated,
                [d.mean for _, d in deltas],
                marker="o",
                linestyle="--" if seed else "-",
                color=color,
                lw=1.9 if emphasis else 1.3,
                ms=5 if emphasis else 4,
                mfc="white" if seed else color,
                mec=color if seed else "white",
                mew=1.2 if seed else 0.7,
                zorder=3,
                label=f"{base.upper()} · {description} (seed {seed or ORIGINAL_SEED})",
            )
        ax.set_title(f"{title}  ", loc="left", fontsize=10, fontweight="bold", color="#111111")
        ax.text(
            1.0,
            1.02,
            f"ref {100 * np.mean(list(ref.values())):.1f}%  ·  {'shared ' if shared else ''}n={len(ref)}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#666666",
        )
        ax.set_xticks(list(steps))
        ax.set_xlim(min(steps) - 0.6, max(steps) + 0.6)
        ax.grid(axis="y", color="#E6E6E6", lw=0.7, zorder=0)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:+g}" if v else "0")
    for ax in axes[1]:
        ax.set_xlabel("RL step")
    for ax in axes[:, 0]:
        ax.set_ylabel("Δ accuracy vs reference (pts, 80% CI)")
    # A replicate may have points on some panels only, so the legend gathers every series drawn on any panel.
    legend: dict[str, object] = {}
    for ax in axes.flat:
        for handle, label in zip(*ax.get_legend_handles_labels(), strict=True):
            legend.setdefault(label, handle)
    fig.legend(
        legend.values(), legend.keys(), loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=9
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95 if len(series) <= 3 else 0.91))
    for extension in ("png", "pdf", "svg"):
        fig.savefig(f"{out_prefix}.{extension}", dpi=220, bbox_inches="tight")
    Path(f"{out_prefix}.csv").write_text("\n".join(rows) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("collect", "test", "plot"))
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("out_prefix", type=Path, nargs="?", help="plot: output path without extension")
    parser.add_argument(
        "--series", default="a,b,c", help="comma-separated arm labels, such as a,b,c or a-seed43,b-seed43,c-seed43"
    )
    parser.add_argument("--steps", default="4,8,12")
    parser.add_argument(
        "--shared-items",
        action="store_true",
        help="inner join: score every run only on the items all evaluated runs in the selection have scored",
    )
    args = parser.parse_args()
    series = args.series.split(",")
    steps = [int(step) for step in args.steps.split(",")]
    if args.action == "collect":
        collect(args.data_dir)
    elif args.action == "test":
        report(args.data_dir, series, steps, args.shared_items)
    else:
        if args.out_prefix is None:
            raise SystemExit("plot needs an output prefix")
        plot(args.data_dir, args.out_prefix, series, steps, args.shared_items)


if __name__ == "__main__":
    main()

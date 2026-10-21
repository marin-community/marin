# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Compare the mixed-RLVR arms (marin #10214) to the October 5 Datakit SFT, paired per item.

    python -m experiments.post_training.pivotrl.mixed_rlvr_analysis collect <data-dir>
    python -m experiments.post_training.pivotrl.mixed_rlvr_analysis test <data-dir> [--series a,b,c] [--steps 4,8,12]
    python -m experiments.post_training.pivotrl.mixed_rlvr_analysis plot <data-dir> <out-prefix> [--series a,b,c]

``collect`` reads every record the arms' evaluations wrote under the CoreWeave evals prefix, keeps the fullest succeeded
run per arm label, step and benchmark, and writes ``tasks.json`` (per-task Harbor rewards for SWE-bench Verified and
OT-TBLite) and ``samples.json`` (per-question Evalchemy correctness for AIME24, GPQA Diamond, Math500 and MMLU-Pro)
beside ``records.json``. The reference model's rows come from the October 5 cohort records in ``mixed_rlvr_evals/``.

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
import math
import os
import re
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import s3fs

HERE = Path(__file__).resolve().parent
REFERENCE = json.loads((HERE / "mixed_rlvr_evals" / "reference.json").read_text())
EVALS_ROOT = "marin-us-east-02a/marin/evals"
HARBOR = ("swebench-verified", "ot-tblite-recovery")
EVALCHEMY = ("aime24", "gpqa-diamond", "math500", "mmlu-pro")
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


def paired_delta(
    arm: dict[str, float], ref: dict[str, float], rng: np.random.Generator
) -> tuple[float, float, float, int]:
    """Mean arm-minus-reference difference in points over shared items, with its 80% bootstrap interval."""
    items = sorted(set(arm) & set(ref))
    diffs = np.array([arm[i] - ref[i] for i in items])
    boot = diffs[rng.integers(0, len(diffs), size=(BOOTSTRAP, len(diffs)))].mean(axis=1)
    low, high = np.percentile(boot, [10, 90])
    return 100 * diffs.mean(), 100 * low, 100 * high, len(items)


def _filesystem() -> s3fs.S3FileSystem:
    return s3fs.S3FileSystem(
        key=os.environ["AWS_ACCESS_KEY_ID"],
        secret=os.environ["AWS_SECRET_ACCESS_KEY"],
        client_kwargs={"endpoint_url": os.environ.get("AWS_ENDPOINT_URL", "https://cwobject.com")},
        config_kwargs={"s3": {"addressing_style": "virtual"}},
    )


def _records(fs) -> list[dict]:
    records = []
    for directory in fs.ls(EVALS_ROOT):
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
    for raw in fs.cat(files, on_error="omit").values():
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


def _merge_fullest(parts: Iterable[dict]) -> dict:
    """The fullest run wins on overlap; partial runs only fill tasks it lacks."""
    merged: dict = {}
    for part in sorted(parts, key=len, reverse=True):
        for item, value in part.items():
            merged.setdefault(item, value)
    return merged


def collect(data_dir: Path) -> None:
    fs = _filesystem()
    records = _records(fs)
    (data_dir / "records.json").write_text(json.dumps(records, indent=1) + "\n")
    runs: dict[tuple[str, str], list[str]] = collections.defaultdict(list)
    for record in records:
        if record["status"] == "succeeded":
            runs[(record["eval"], record["key"])].append(record["results"])
    for bench in (*HARBOR, *EVALCHEMY):
        runs[(bench, REF)].append(REFERENCE[bench]["results_path"].removeprefix("s3://"))
    tasks: dict[str, dict[str, dict]] = {bench: {} for bench in HARBOR}
    samples: dict[str, dict[str, dict]] = {bench: {} for bench in EVALCHEMY}
    with ThreadPoolExecutor(8) as pool:
        jobs = {
            (bench, key): [
                pool.submit(_harbor_rewards if bench in HARBOR else _sample_correctness, fs, base) for base in bases
            ]
            for (bench, key), bases in runs.items()
        }
        for (bench, key), futures in jobs.items():
            target = tasks if bench in HARBOR else samples
            target[bench][key] = _merge_fullest(future.result() for future in futures)
    (data_dir / "tasks.json").write_text(json.dumps(tasks) + "\n")
    (data_dir / "samples.json").write_text(json.dumps(samples) + "\n")
    for bench in (*HARBOR, *EVALCHEMY):
        source = tasks if bench in HARBOR else samples
        print(bench, {key: len(items) for key, items in sorted(source[bench].items())})


def load_scores(data_dir: Path) -> dict[str, dict[str, dict[str, float]]]:
    tasks = json.loads((data_dir / "tasks.json").read_text())
    samples = json.loads((data_dir / "samples.json").read_text())
    scores = {bench: harbor_scores(tasks[bench], bench) for bench in HARBOR}
    scores |= {bench: sample_scores(samples[bench]) for bench in EVALCHEMY}
    return scores


def report(data_dir: Path, series: Sequence[str], steps: Sequence[int]) -> None:
    scores = load_scores(data_dir)
    rng = np.random.default_rng(0)
    samples = json.loads((data_dir / "samples.json").read_text())
    for bench in (*HARBOR, *EVALCHEMY):
        per = scores[bench]
        repeats = 1 if bench in HARBOR else max(len(v) for v in samples[bench].values()) // max(len(per[REF]), 1)
        method = "exact McNemar" if repeats == 1 else "paired sign-flip permutation on per-question accuracy"
        print(f"\n### {bench} ({method})")
        for step in steps:
            keys = [REF, *(run_key(label, step) for label in series)]
            present = [key for key in keys if key in per]
            common = set.intersection(*(set(per[key]) for key in present))
            n = len(common)
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


def plot(data_dir: Path, out_prefix: Path, series: Sequence[str], steps: Sequence[int]) -> None:
    scores = load_scores(data_dir)
    rng = np.random.default_rng(0)
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
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.6), sharex=True, sharey=True)
    rows = ["benchmark,arm,step,n,reference_accuracy,arm_accuracy,delta_points,ci80_low,ci80_high"]
    for ax, (bench, title) in zip(axes.flat, PANELS, strict=True):
        ref = scores[bench][REF]
        ax.axhline(0, color="#222222", lw=0.8, ls=(0, (4, 3)), zorder=1)
        for label in series:
            description, color = ARM_LABELS[label.split("-")[0]]
            points = [paired_delta(scores[bench][run_key(label, step)], ref, rng) for step in steps]
            for step, (delta, low, high, n) in zip(steps, points, strict=True):
                items = set(scores[bench][run_key(label, step)]) & set(ref)
                ref_acc = 100 * np.mean([ref[i] for i in items])
                rows.append(
                    f"{bench},{label},{step},{n},{ref_acc:.2f},{ref_acc + delta:.2f},{delta:.2f},{low:.2f},{high:.2f}"
                )
            emphasis = not label.startswith("a")
            ax.fill_between(
                steps,
                [p[1] for p in points],
                [p[2] for p in points],
                color=color,
                alpha=0.16 if emphasis else 0.13,
                lw=0,
                zorder=2,
            )
            ax.plot(
                steps,
                [p[0] for p in points],
                "o-",
                color=color,
                lw=1.9 if emphasis else 1.3,
                ms=5 if emphasis else 4,
                mec="white",
                mew=0.7,
                zorder=3,
                label=f"{label.upper()} · {description}",
            )
        ax.set_title(f"{title}  ", loc="left", fontsize=10, fontweight="bold", color="#111111")
        ax.text(
            1.0,
            1.02,
            f"ref {100 * np.mean(list(ref.values())):.1f}%  ·  n={len(ref)}",
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
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
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
    args = parser.parse_args()
    series = args.series.split(",")
    steps = [int(step) for step in args.steps.split(",")]
    if args.action == "collect":
        collect(args.data_dir)
    elif args.action == "test":
        report(args.data_dir, series, steps)
    else:
        if args.out_prefix is None:
            raise SystemExit("plot needs an output prefix")
        plot(args.data_dir, args.out_prefix, series, steps)


if __name__ == "__main__":
    main()

"""Score task289 runs: per-run scores, paired per-item deltas vs S with an item-clustered bootstrap, and run-to-run
disagreement counts.

Input is a local copy of each run's FineStore samples table, laid out as
    ROOT/<model>/<eval>/r<run>/samples/w=*/g=*/*.parquet
(`gcloud storage cp -r gs://.../task289/<model>-rN/<eval>/rK/data/samples ROOT/<model>/<eval>/rK/`).
"""

import argparse
import ast
import glob
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

# (per-item metric, sample filter); GSM8K's headline is flexible-extract (strict-match scores 0 for this model).
METRIC = {
    "math500": ("accuracy", "none"),
    "gsm8k-0shot": ("exact_match", "flexible-extract"),
    "humanevalplus": ("pass_rate", "none"),
    "ifbench": ("strict_prompt_accuracy", "none"),
}
BASELINE = "S"


def item_scores(run_dir: Path, metric: str, filter_name: str) -> dict[str, float]:
    """Per-doc score; the compactor generation (higher g) supersedes the writer's rows."""
    best: dict[str, tuple[int, float]] = {}
    for path in glob.glob(str(run_dir / "samples" / "*" / "*" / "*.parquet")):
        gen = int(Path(path).parent.name.removeprefix("g="))
        for row in pq.ParquetFile(path).read().to_pylist():
            if row["filter"] != filter_name:
                continue
            metrics = row["metrics"]
            if isinstance(metrics, str):
                metrics = ast.literal_eval(metrics)
            values = dict(metrics)
            if metric not in values:
                raise KeyError(f"{run_dir}: metric {metric} missing; have {sorted(values)}")
            doc = str(row["doc_id"])
            if doc not in best or gen > best[doc][0]:
                best[doc] = (gen, float(values[metric]))
    return {doc: score for doc, (_, score) in best.items()}


def bootstrap(diffs: np.ndarray, reps: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(reps, len(diffs)))
    means = diffs[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--reps", type=int, default=10000)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    report = {}
    for key, (metric, filter_name) in METRIC.items():
        models = sorted(p.name for p in args.root.iterdir() if (p / key).is_dir())
        if BASELINE not in models:
            continue
        runs = {m: {int(r.name[1:]): item_scores(r, metric, filter_name) for r in sorted((args.root / m / key).glob("r*"))} for m in models}
        docs = sorted(set.intersection(*(set(s) for m in runs for s in runs[m].values())), key=lambda d: int(d) if d.isdigit() else d)
        mean = {m: np.mean([[runs[m][r][d] for d in docs] for r in sorted(runs[m])], axis=0) for m in models}
        rows = []
        for m in models:
            per_run = {r: float(np.mean([runs[m][r][d] for d in docs])) for r in sorted(runs[m])}
            disagree = None
            if len(runs[m]) == 2:
                a, b = (runs[m][r] for r in sorted(runs[m]))
                disagree = sum(a[d] != b[d] for d in docs)
            row = dict(model=m, runs=per_run, mean=float(mean[m].mean()), disagree=disagree, n=len(docs))
            if m != BASELINE:
                diffs = mean[m] - mean[BASELINE]
                lo, hi = bootstrap(diffs, args.reps, seed=289)
                row.update(delta=float(diffs.mean()), ci=[lo, hi], better=int((diffs > 0).sum()), worse=int((diffs < 0).sum()))
            rows.append(row)
        report[key] = dict(metric=metric, rows=rows)
        print(f"\n### {key} ({metric}, n={len(docs)} items; delta = mean over items of [model 2-run mean - S 2-run mean])\n")
        print("| Model | r1 | r2 | Mean | Delta vs S | 95% CI (item bootstrap) | Items up / down vs S | r1≠r2 items |")
        print("|---|---:|---:|---:|---:|---|---:|---:|")
        for row in rows:
            r = row["runs"]
            delta = f"{row['delta']:+.4f}" if "delta" in row else "—"
            ci = f"[{row['ci'][0]:+.4f}, {row['ci'][1]:+.4f}]" if "ci" in row else "—"
            updown = f"{row['better']} / {row['worse']}" if "better" in row else "—"
            print(
                f"| {row['model']} | {r.get(1, float('nan')):.4f} | {r.get(2, float('nan')):.4f} | {row['mean']:.4f} "
                f"| {delta} | {ci} | {updown} | {row['disagree']} |"
            )
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

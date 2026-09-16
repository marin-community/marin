# /// script
# requires-python = ">=3.12"
# dependencies = ["wandb", "numpy", "matplotlib"]
# ///
"""Archive and plot the eight frozen TPP10 calibration histories."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import wandb

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
PLAN = json.loads((ROOT.parent / "calibration_plan.json").read_text())
TRACES = ROOT / "optimizer_traces"
GROUPS = {
    "train": ["train/loss", "optim/learning_rate", "optim/adam_lr"],
    "norm": ["grad/norm/total", "params/norm/total"],
}


def collect(request):
    run_name = request["run_name"]
    path = TRACES / f"{run_name}.history.json"
    if not path.exists():
        run = wandb.Api(timeout=45).run(f"marin-community/marin/{run_name}")
        assert run.state == "finished", (run_name, run.state)
        history = {
            "run_name": run_name,
            "url": run.url,
            "state": run.state,
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "groups": {
                group: list(run.scan_history(keys=["_step", *keys], page_size=1000))
                for group, keys in GROUPS.items()
            },
        }
        path.write_text(json.dumps(history, indent=2) + "\n")
    history = json.loads(path.read_text())
    assert history["run_name"] == run_name
    stats = {}
    duplicate_rows = {}
    for group, keys in GROUPS.items():
        unique = {}
        raw_rows = history["groups"][group]
        for row in raw_rows:
            step = row["_step"]
            if step in unique:
                assert row == unique[step], (run_name, group, step, "conflicting history rows")
            unique[step] = row
        rows = list(unique.values())
        duplicate_rows[group] = len(raw_rows) - len(rows)
        history["groups"][group] = rows
        assert rows, (run_name, group)
        steps = np.array([r["_step"] for r in rows])
        assert (np.diff(steps) > 0).all(), (run_name, group, "unordered steps")
        for key in keys:
            values = np.array([r[key] for r in rows], dtype=float)
            stats[key] = {
                "count": len(values),
                "nonfinite": int((~np.isfinite(values)).sum()),
                "first_step": int(steps[0]),
                "last_step": int(steps[-1]),
                "first": float(values[0]),
                "last": float(values[-1]),
                "min": float(values.min()),
                "max": float(values.max()),
                "last_10pct_mean": float(values[steps >= 0.9 * request["total_steps"]].mean()),
                "post_warmup_max": float(values[steps >= 0.02 * request["total_steps"]].max()),
            }
    return {
        "run_name": run_name,
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "identical_duplicate_rows": duplicate_rows,
        "metrics": stats,
    }, history


def main():
    TRACES.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        collected = list(pool.map(collect, PLAN["runs"]))
    summary = {
        "plan_sha256": PLAN["plan_sha256"],
        "source": "W&B scan_history; separate train and norm scans preserve their logging cadence",
        "runs": [stats for stats, _ in collected],
    }
    (ROOT / "calibration_trace_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=True, layout="constrained")
    for request, (_, history) in zip(PLAN["runs"], collected, strict=True):
        col = 0 if request["batch_size"] == 32 else 1
        color = "#0072B2" if request["arm"] == "unmatched" else "#D55E00"
        style = "-" if request["trainer_seed"] == 20260910 else "--"
        label = f"{request['arm'].capitalize()}, seed {request['trainer_seed']}"
        for row, group, key in [(0, "train", "train/loss"), (1, "norm", "grad/norm/total"),
                                (2, "train", "optim/learning_rate")]:
            rows = history["groups"][group]
            x = [r["_step"] / request["total_steps"] for r in rows]
            y = [r[key] for r in rows]
            axes[row, col].plot(x, y, color=color, linestyle=style, linewidth=1.1, alpha=0.85,
                                label=label)
    for col, batch in enumerate((32, 128)):
        axes[0, col].set_title(f"Proxy batch {batch}")
        axes[0, col].legend(fontsize=8, loc="upper right")
        axes[1, col].set_yscale("log")
        axes[2, col].set_xlabel("Fraction of training updates")
        for row, ylabel in enumerate(("Training loss", "Total gradient norm", "Muon learning rate")):
            axes[row, col].set_ylabel(ylabel)
            axes[row, col].grid(alpha=0.2)
            axes[row, col].set_xlim(0, 1)
    fig.suptitle("TPP10 calibration: all eight runs at StarCoder fraction 0.5", fontsize=13)
    fig.savefig(ROOT / "calibration_optimizer_traces.png", dpi=160)
    fig.savefig(ROOT / "calibration_optimizer_traces.pdf")
    for stats, _ in collected:
        print(json.dumps(stats))


if __name__ == "__main__":
    main()

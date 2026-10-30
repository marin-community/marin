# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export complete W&B histories and report paired losses without sampling or smoothing."""

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import wandb
from marin.execution.lazy import materialized_config

from experiments.ema_gram.plan import SELECTION_METRIC, Cell, Result, confirmations
from experiments.ema_gram.train import training_step


def summarize_attempt(
    cell: Cell,
    history: list[dict],
    *,
    expected_steps: int,
    tokens_per_step: int,
    run_state: str,
    run_url: str,
    loss_target: float | None,
) -> dict:
    """Retain an attempt's own curve; never silently join fresh process restarts."""
    observed_steps = [int(r["global_step"]) for r in history if r.get("global_step") is not None]
    completed = max(observed_steps, default=-1) + 1
    start = min(observed_steps, default=-1)
    curve = []
    for row in history:
        if row.get(SELECTION_METRIC) is None:
            continue
        step = int(row.get("global_step", row["_step"]))
        curve.append(
            {
                "step": step,
                "tokens": (step + 1) * tokens_per_step,
                "elapsed": float(row["_runtime"]),
                "loss": float(row[SELECTION_METRIC]),
            }
        )
    curve.sort(key=lambda r: r["step"])
    if len({p["step"] for p in curve}) != len(curve):
        raise ValueError("Duplicate evaluation steps; inspect history before comparing")
    final = next((p["loss"] for p in curve if p["step"] == expected_steps - 1), None)
    status = (
        "completed"
        if run_state == "finished" and completed == expected_steps and final is not None
        else ("failed" if run_state in ("failed", "crashed", "killed") else "incomplete")
    )
    crossings = [p for p in curve if loss_target is not None and p["loss"] <= loss_target]
    return {
        "cell": asdict(cell),
        "result": asdict(Result(cell.name, status, final, completed, expected_steps, run_url)),
        "curve": curve,
        "starts_at_step": start,
        "time_to_target": crossings[0]["elapsed"] if crossings and start == 0 else None,
        "tokens_to_target": crossings[0]["tokens"] if crossings and start == 0 else None,
        "loss_target": loss_target,
        "full_time_curve": start == 0,
        "nonfinite_evaluations": sum(not math.isfinite(p["loss"]) for p in curve),
    }


def paired_differences(attempts: list[dict]) -> list[dict]:
    """Pair complete runs by seed, model and horizon, at identical observed token counts."""
    complete = [r for r in attempts if r["result"]["status"] == "completed"]
    controls = {}
    for run in complete:
        cell = Cell(**run["cell"])
        key = (cell.slug, cell.horizon, cell.seed)
        if cell.alpha == 0:
            if key in controls:
                raise ValueError("Multiple alpha=0 controls for one seed; select locked settings first")
            controls[key] = run
    rows = []
    for run in complete:
        cell = Cell(**run["cell"])
        if cell.alpha == 0:
            continue
        key = (cell.slug, cell.horizon, cell.seed)
        if key not in controls:
            continue
        control = {p["tokens"]: p for p in controls[key]["curve"]}
        for point in run["curve"]:
            if point["tokens"] in control:
                rows.append(
                    {
                        "name": cell.name,
                        "seed": cell.seed,
                        "alpha": cell.alpha,
                        "tokens": point["tokens"],
                        "delta_loss": point["loss"] - control[point["tokens"]]["loss"],
                    }
                )
    return rows


def plot_curves(attempts: list[dict], paired: list[dict], output: Path):
    # Plotting is optional; install Levanter's dev extra when exporting figures.
    import matplotlib.pyplot as plt  # noqa: PLC0415

    colors = {0.0: "#5F6368", 0.125: "#9467BD", 0.25: "#1F77B4", 1 / 3: "#2CA02C", 0.5: "#FF7F0E"}
    with plt.rc_context({"font.size": 9, "svg.fonttype": "none", "pdf.fonttype": 42}):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), layout="constrained")
        for run in attempts:
            cell = Cell(**run["cell"])
            curve = run["curve"]
            label = f"alpha={cell.alpha:g}, seed={cell.seed}"
            for axis, key, divisor in ((axes[0], "tokens", 1e9), (axes[1], "elapsed", 3600)):
                if key == "elapsed" and not run["full_time_curve"]:
                    continue
                axis.plot(
                    [r[key] / divisor for r in curve],
                    [r["loss"] for r in curve],
                    color=colors[cell.alpha],
                    linewidth=1.4,
                    alpha=0.8,
                    label=label,
                )
            points = [p for p in paired if p["name"] == cell.name]
            axes[2].plot(
                [p["tokens"] / 1e9 for p in points],
                [p["delta_loss"] for p in points],
                color=colors[cell.alpha],
                linewidth=1.4,
                alpha=0.8,
            )
        for ax in axes:
            ax.grid(linestyle="--", color="#B0B0B0", alpha=0.5)
            ax.set_axisbelow(True)
        axes[0].set(xlabel="Tokens (billions)", ylabel="Held-out C4-en loss (nats)")
        axes[1].set(xlabel="Elapsed time (hours)", ylabel="Held-out C4-en loss (nats)")
        axes[2].set(xlabel="Tokens (billions)", ylabel="Paired loss difference vs alpha=0")
        axes[2].axhline(0, color="black", linestyle="--", linewidth=0.8)
        axes[0].legend(fontsize=7)
        fig.savefig(output / "curves.svg")
        fig.savefig(output / "curves.png", dpi=220)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True, help="JSON [{cell: {...}, run_path: entity/project/id}]")
    parser.add_argument("--lock", type=Path, help="Frozen pilot selection and loss target for confirmation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    entries = json.loads(args.runs.read_text())
    lock = json.loads(args.lock.read_text()) if args.lock else None
    target = lock["loss_target"] if lock else None
    args.output.mkdir(parents=True, exist_ok=True)
    api = wandb.Api()
    attempts = []
    for entry in entries:
        cell = Cell(**entry["cell"])
        if lock is not None and cell not in confirmations(lock, phase=cell.phase):
            raise ValueError("Reported run differs from locked settings")
        run = api.run(entry["run_path"])
        cfg = materialized_config(training_step(cell, "report-dev"), "/tmp/ema-gram-report").train_config
        # Verify run identity against logged settings before attaching a result to a cell.
        for field, expected in (("alpha", cell.alpha), ("ridge", cell.ridge), ("learning_rate", cell.learning_rate)):
            if run.config["optimizer"][field] != expected:
                raise ValueError(f"Run {run.url} differs from cell: {field}")
        if run.config["trainer"]["seed"] != cell.seed or run.config["data_seed"] != cell.seed:
            raise ValueError(f"Run {run.url} has unpaired seeds")
        history = list(run.scan_history())
        (args.output / f"history-{run.id}.json").write_text(json.dumps(history, indent=2) + "\n")
        attempt = summarize_attempt(
            cell,
            history,
            expected_steps=cfg.trainer.num_train_steps,
            tokens_per_step=cfg.trainer.train_batch_size * cfg.train_seq_len,
            run_state=run.state,
            run_url=run.url,
            loss_target=target,
        )
        attempt["config"] = dict(run.config)
        attempt["summary"] = dict(run.summary)
        attempts.append(attempt)
    names = [r["result"]["name"] for r in attempts]
    if len(set(names)) != len(names):
        raise ValueError("Export one attempt per cell per report; retain earlier attempts in separate reports")
    paired = paired_differences(attempts) if args.lock else []
    (args.output / "report.json").write_text(json.dumps({"attempts": attempts, "paired": paired}, indent=2) + "\n")
    (args.output / "results.json").write_text(json.dumps([r["result"] for r in attempts], indent=2) + "\n")
    if args.plot:
        plot_curves(attempts, paired, args.output)


if __name__ == "__main__":
    main()

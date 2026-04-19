# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize the standalone AdamC Qwen3 speedrun submission from finished sweep runs."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pformat

import wandb
from rigging.filesystem import open_url

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

ENTITY = "marin-community"
PROJECT = "marin"
OPTIMIZER = "adamc"
SIZES = ("130m", "300m", "520m", "1_2b")
RUN_NAME_RE = re.compile(r"qwen3_(130m|300m|520m|1_2b)_adamc_4096_lrx[0-9_]+-[a-z0-9]+$")
OUTPUT_DIR = Path(__file__).resolve().parent
SUMMARY_PATH = OUTPUT_DIR / "selection_summary.json"
SELECTED_RUNS_PATH = OUTPUT_DIR / "selected_runs.py"
PLOT_PATH = OUTPUT_DIR / "lr_vs_c4_en_bpb.png"


@dataclass(frozen=True)
class SweepRun:
    run_name: str
    size: str
    state: str
    learning_rate: float
    c4_en_bpb: float
    c4_en_loss: float | None
    run_info: dict[str, object]
    optimizer_config: dict[str, object] = field(default_factory=dict)
    raw_results: dict[str, object] | None = None
    source_results_path: str = ""
    wandb_url: str = ""
    train_batch_size: int = 0
    num_train_steps: int = 0


def select_best_runs(runs: list[SweepRun]) -> dict[str, SweepRun]:
    best_by_size: dict[str, SweepRun] = {}
    for run in runs:
        if run.state != "finished":
            continue
        current = best_by_size.get(run.size)
        if current is None or run.c4_en_bpb < current.c4_en_bpb:
            best_by_size[run.size] = run
    return best_by_size


def _fetch_results_json(results_path: str) -> dict[str, object]:
    with open_url(results_path, "r") as f:
        return json.load(f)


def _fetch_runs() -> list[SweepRun]:
    api = wandb.Api(timeout=30)
    query = {"display_name": {"$regex": RUN_NAME_RE.pattern}}
    runs = list(api.runs(f"{ENTITY}/{PROJECT}", filters=query, per_page=100, lazy=False))
    collected: list[SweepRun] = []

    for run in runs:
        run_name = run.display_name or run.name
        if run_name is None:
            continue

        match = RUN_NAME_RE.fullmatch(run_name)
        if match is None:
            continue

        size = match.group(1)
        tracker = run.config["trainer"]["tracker"]
        results_path = f"{tracker['replicate_path']}/speedrun_results.json"
        raw_results = _fetch_results_json(results_path)
        run_info = raw_results["runs"][0]["run_info"]
        optimizer_config = run.config["optimizer"]
        c4_en_bpb = float(run.summary["eval/paloma/c4_en/bpb"])
        c4_en_loss_raw = run.summary.get("eval/paloma/c4_en/loss")

        collected.append(
            SweepRun(
                run_name=run_name,
                size=size,
                state=run.state,
                learning_rate=float(optimizer_config["learning_rate"]),
                c4_en_bpb=c4_en_bpb,
                c4_en_loss=float(c4_en_loss_raw) if c4_en_loss_raw is not None else None,
                optimizer_config=optimizer_config,
                run_info=run_info,
                raw_results=raw_results,
                source_results_path=results_path,
                wandb_url=f"https://wandb.ai/{ENTITY}/{PROJECT}/runs/{run.id}",
                train_batch_size=int(run.config["trainer"]["train_batch_size"]),
                num_train_steps=int(run.config["trainer"]["num_train_steps"]),
            )
        )

    return sorted(collected, key=lambda run: (SIZES.index(run.size), run.learning_rate, run.run_name))


def _serializable_run(run: SweepRun) -> dict[str, object]:
    return {
        "run_name": run.run_name,
        "state": run.state,
        "learning_rate": run.learning_rate,
        "c4_en_bpb": run.c4_en_bpb,
        "c4_en_loss": run.c4_en_loss,
        "wandb_url": run.wandb_url,
        "source_results_path": run.source_results_path,
        "train_batch_size": run.train_batch_size,
        "num_train_steps": run.num_train_steps,
    }


def _write_selection_summary(runs: list[SweepRun], selected: dict[str, SweepRun]) -> None:
    all_runs = {
        size: [_serializable_run(run) for run in runs if run.size == size]
        for size in SIZES
    }
    selected_runs = {
        size: {
            **_serializable_run(run),
            "optimizer_config": run.optimizer_config,
            "resources": run.run_info["resources"],
            "description": run.run_info["description"],
            "author": run.run_info["author"],
            "tokenized_dataset": run.run_info["tokenized_dataset"],
        }
        for size, run in selected.items()
    }
    payload = {
        "optimizer": OPTIMIZER,
        "selection_metric": "eval/paloma/c4_en/bpb",
        "selected_runs": selected_runs,
        "all_runs": all_runs,
    }
    SUMMARY_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_selected_runs_module(selected: dict[str, SweepRun]) -> None:
    author_info = next(iter(selected.values())).run_info["author"]
    payload = {
        size: {
            "source_run_name": run.run_name,
            "learning_rate": run.learning_rate,
            "c4_en_bpb": run.c4_en_bpb,
            "c4_en_loss": run.c4_en_loss,
            "wandb_url": run.wandb_url,
            "source_results_path": run.source_results_path,
            "train_batch_size": run.train_batch_size,
            "num_train_steps": run.num_train_steps,
            "optimizer_config": run.optimizer_config,
            "resources": run.run_info["resources"],
            "description": run.run_info["description"],
            "tokenized_dataset": run.run_info["tokenized_dataset"],
        }
        for size, run in selected.items()
    }
    module = f"""# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

\"\"\"Auto-generated AdamC Qwen3 speedrun selections from finished LR sweep runs.\"\"\"

AUTHOR_INFO = {pformat(author_info, sort_dicts=True)}

SELECTED_RUNS = {pformat(payload, sort_dicts=True)}
"""
    SELECTED_RUNS_PATH.write_text(module)


def _write_results_files(selected: dict[str, SweepRun]) -> None:
    for size, run in selected.items():
        size_dir = OUTPUT_DIR / size
        size_dir.mkdir(parents=True, exist_ok=True)
        (size_dir / "speedrun_results.json").write_text(json.dumps(run.raw_results, indent=2, sort_keys=True) + "\n")


def _write_plot(runs: list[SweepRun], selected: dict[str, SweepRun]) -> None:
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=180, sharey=True)
    axes = axes.flatten()

    for axis, size in zip(axes, SIZES, strict=True):
        size_runs = sorted((run for run in runs if run.size == size), key=lambda run: run.learning_rate)
        selected_run = selected[size]
        xs = [run.learning_rate for run in size_runs]
        ys = [run.c4_en_bpb for run in size_runs]

        axis.plot(xs, ys, marker="o", linewidth=1.5, color="tab:blue")
        axis.scatter(
            [selected_run.learning_rate],
            [selected_run.c4_en_bpb],
            color="tab:red",
            s=60,
            zorder=3,
        )
        axis.annotate(
            f"best lr={selected_run.learning_rate:.4g}\nbpb={selected_run.c4_en_bpb:.4f}",
            xy=(selected_run.learning_rate, selected_run.c4_en_bpb),
            xytext=(8, -4),
            textcoords="offset points",
            fontsize=8,
            color="tab:red",
        )
        axis.set_xscale("log")
        axis.set_title(size)
        axis.set_xlabel("Learning rate")
        axis.set_ylabel("c4_en/bpb")

    fig.suptitle("AdamC Qwen3 LR Sweep: c4_en/bpb by scale", fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    runs = _fetch_runs()
    selected = select_best_runs(runs)
    missing = [size for size in SIZES if size not in selected]
    if missing:
        raise ValueError(f"Missing finished runs for sizes: {missing}")

    _write_selection_summary(runs, selected)
    _write_selected_runs_module(selected)
    _write_results_files(selected)
    _write_plot(runs, selected)

    for size in SIZES:
        run = selected[size]
        print(f"{size}: {run.run_name} lr={run.learning_rate:.6g} c4_en/bpb={run.c4_en_bpb:.6f}")


if __name__ == "__main__":
    main()

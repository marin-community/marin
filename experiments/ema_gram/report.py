# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export complete W&B histories and report paired losses without sampling or smoothing."""

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import wandb
from marin.execution.lazy import materialized_config
from wandb.util import json_friendly_val

from experiments.ema_gram.plan import (
    HORIZON,
    SCALE,
    SELECTION_METRIC,
    Cell,
    Result,
    confirmations,
    read_lock,
)
from experiments.ema_gram.train import build_training_step


@dataclass(frozen=True)
class CurvePoint:
    step: int
    tokens: int
    elapsed: float
    loss: float


@dataclass
class AttemptReport:
    cell: Cell
    result: Result
    curve: list[CurvePoint]
    starts_at_step: int
    time_to_target: float | None
    tokens_to_target: int | None
    loss_target: float | None
    full_time_curve: bool
    nonfinite_evaluations: int
    config: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PairedDifference:
    name: str
    seed: int
    alpha: float
    tokens: int
    delta_loss: float
    reference_alpha: float


def summarize_attempt(
    cell: Cell,
    history: list[dict],
    *,
    expected_steps: int,
    tokens_per_step: int,
    run_state: str,
    run_url: str,
    loss_target: float | None,
) -> AttemptReport:
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
            CurvePoint(
                step=step,
                tokens=(step + 1) * tokens_per_step,
                elapsed=float(row["_runtime"]),
                loss=float(row[SELECTION_METRIC]),
            )
        )
    curve.sort(key=lambda r: r.step)
    if len({p.step for p in curve}) != len(curve):
        raise ValueError("Duplicate evaluation steps; inspect history before comparing")
    final = next((p.loss for p in curve if p.step == expected_steps - 1), None)
    status = (
        "completed"
        if run_state == "finished" and completed == expected_steps and final is not None
        else ("failed" if run_state in ("failed", "crashed", "killed") else "incomplete")
    )
    crossings = [p for p in curve if loss_target is not None and p.loss <= loss_target]
    return AttemptReport(
        cell=cell,
        result=Result(cell.name, status, final, completed, expected_steps, run_url),
        curve=curve,
        starts_at_step=start,
        time_to_target=crossings[0].elapsed if crossings and start == 0 else None,
        tokens_to_target=crossings[0].tokens if crossings and start == 0 else None,
        loss_target=loss_target,
        full_time_curve=start == 0,
        nonfinite_evaluations=sum(not math.isfinite(p.loss) for p in curve),
    )


def reference_alpha(cell: Cell) -> float:
    return 0.25 if cell.phase in (HORIZON, SCALE) else 0.0


def validate_run_config(logged: dict, expected) -> None:
    """Require all training and evaluation controls to match before labeling run history."""
    expected = json_friendly_val(asdict(expected))
    top_fields = (
        "model",
        "optimizer",
        "data",
        "train_seq_len",
        "data_seed",
        "adapter",
        "z_loss_weight",
        "initialize_from_hf",
        "use_hf_model_config",
        "pad_tokenizer_to_match_model",
        "initialize_from_checkpoint_path",
        "initialize_model_from_checkpoint_path",
        "labeled_eval",
        "eval_harness",
        "eval_harness_steps",
    )
    trainer_fields = (
        "seed",
        "mp",
        "quantization",
        "model_averaging",
        "mesh",
        "use_explicit_mesh_axes",
        "train_batch_size",
        "per_device_parallelism",
        "allow_nondivisible_batch_size",
        "num_train_steps",
        "steps_per_eval",
        "max_eval_batches",
        "initialize_from",
        "allow_partial_checkpoint",
        "jax_config",
        "crash_on_nan",
        "crash_on_inf",
    )
    for key in top_fields:
        # W&B omits the top-level adapter mapping when it is empty.
        # Preserve strict checks for nonempty adapters and every other field.
        actual = logged.get(key, {}) if key == "adapter" else logged[key]
        target = expected[key]
        if key == "data":
            # The pod wrapper can turn cache construction off after artifacts resolve.
            actual = {k: v for k, v in actual.items() if k != "auto_build_caches"}
            target = {k: v for k, v in target.items() if k != "auto_build_caches"}
        if actual != target:
            raise ValueError(f"Run differs from protocol: {key}")
    for key in trainer_fields:
        actual, target = logged["trainer"][key], expected["trainer"][key]
        if key == "mesh":
            # The API also omits this empty mapping inside the mesh config.
            actual = {"shared_mapping": {}, **actual}
        if actual != target:
            raise ValueError(f"Run differs from protocol: trainer.{key}")


def paired_differences(attempts: list[AttemptReport]) -> list[PairedDifference]:
    """Pair complete runs against the phase reference at identical observed token counts."""
    complete = [r for r in attempts if r.result.status == "completed"]
    controls = {}
    for run in complete:
        cell = run.cell
        key = (cell.phase, cell.slug, cell.horizon, cell.seed)
        if cell.alpha == reference_alpha(cell):
            if key in controls:
                raise ValueError("Multiple reference runs for one seed; select locked settings first")
            controls[key] = run
    rows = []
    for run in complete:
        cell = run.cell
        if cell.alpha == reference_alpha(cell):
            continue
        key = (cell.phase, cell.slug, cell.horizon, cell.seed)
        if key not in controls:
            raise ValueError(f"Missing completed reference for {cell.name}")
        control = {p.tokens: p for p in controls[key].curve}
        for point in run.curve:
            if point.tokens in control:
                rows.append(
                    PairedDifference(
                        name=cell.name,
                        seed=cell.seed,
                        alpha=cell.alpha,
                        tokens=point.tokens,
                        delta_loss=point.loss - control[point.tokens].loss,
                        reference_alpha=reference_alpha(cell),
                    )
                )
    return rows


def plot_curves(attempts: list[AttemptReport], paired: list[PairedDifference], output: Path):
    # Plotting is optional; install Levanter's dev extra when exporting figures.
    import matplotlib.pyplot as plt  # noqa: PLC0415

    colors = {0.0: "#5F6368", 0.125: "#9467BD", 0.25: "#1F77B4", 1 / 3: "#2CA02C", 0.5: "#FF7F0E"}
    with plt.rc_context({"font.size": 9, "svg.fonttype": "none", "pdf.fonttype": 42}):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), layout="constrained")
        for run in attempts:
            cell = run.cell
            curve = run.curve
            label = f"alpha={cell.alpha:g}, seed={cell.seed}"
            for axis, key, divisor in ((axes[0], "tokens", 1e9), (axes[1], "elapsed", 3600)):
                if key == "elapsed" and not run.full_time_curve:
                    continue
                axis.plot(
                    [getattr(r, key) / divisor for r in curve],
                    [r.loss for r in curve],
                    color=colors[cell.alpha],
                    linewidth=1.4,
                    alpha=0.8,
                    label=label,
                )
            points = [p for p in paired if p.name == cell.name]
            axes[2].plot(
                [p.tokens / 1e9 for p in points],
                [p.delta_loss for p in points],
                color=colors[cell.alpha],
                linewidth=1.4,
                alpha=0.8,
            )
        for ax in axes:
            ax.grid(linestyle="--", color="#B0B0B0", alpha=0.5)
            ax.set_axisbelow(True)
        axes[0].set(xlabel="Tokens (billions)", ylabel="Held-out C4-en loss (nats)")
        axes[1].set(xlabel="Elapsed time (hours)", ylabel="Held-out C4-en loss (nats)")
        axes[2].set(xlabel="Tokens (billions)", ylabel="Paired loss difference vs phase reference")
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
    lock = read_lock(args.lock) if args.lock else None
    target = lock.loss_target if lock else None
    args.output.mkdir(parents=True, exist_ok=True)
    api = wandb.Api()
    attempts = []
    for entry in entries:
        cell = Cell(**entry["cell"])
        if lock is not None and cell not in confirmations(lock, phase=cell.phase):
            raise ValueError("Reported run differs from locked settings")
        run = api.run(entry["run_path"])
        cfg = materialized_config(build_training_step(cell, "report-dev"), "/tmp/ema-gram-report").train_config
        validate_run_config(dict(run.config), cfg)
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
        attempt.config = dict(run.config)
        attempt.summary = dict(run.summary)
        attempts.append(attempt)
    names = [r.result.name for r in attempts]
    if len(set(names)) != len(names):
        raise ValueError("Export one attempt per cell per report; retain earlier attempts in separate reports")
    paired = paired_differences(attempts) if args.lock else []
    (args.output / "report.json").write_text(
        json.dumps({"attempts": [asdict(r) for r in attempts], "paired": [asdict(p) for p in paired]}, indent=2) + "\n"
    )
    (args.output / "results.json").write_text(json.dumps([asdict(r.result) for r in attempts], indent=2) + "\n")
    if args.plot:
        plot_curves(attempts, paired, args.output)


if __name__ == "__main__":
    main()

"""Download and run downstream evals for selected synthetic-data 300M checkpoints.

This is a focused variant of ``eval_200m_models.py`` for the 11 runs:
- 300M regularized baseline,
- 5 shuffled WRAP runs, and
- 5 sorted WRAP runs.

Usage examples:
    uv run python experiments/data_efficiency/synth_data_eval_downstream.py --mode download
    uv run python experiments/data_efficiency/synth_data_eval_downstream.py --mode rearrange
    uv run python experiments/data_efficiency/synth_data_eval_downstream.py --mode eval
    uv run python experiments/data_efficiency/synth_data_eval_downstream.py --mode collect
"""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
import json
import shutil
import subprocess


BASE_TRAIN_STEPS = 777
DEFAULT_PREFIX = "gs://marin-us-central2"


@dataclass(frozen=True)
class DownstreamRun:
    run_name: str
    epochs: int
    teacher_ratio: float | None = None

    @property
    def total_steps(self) -> int:
        if self.teacher_ratio is None:
            return self.epochs * BASE_TRAIN_STEPS
        return int(self.epochs * BASE_TRAIN_STEPS * (1.0 / (1.0 - self.teacher_ratio)))

    @property
    def checkpoint_step(self) -> int:
        return self.total_steps - 1


TARGET_RUNS: list[DownstreamRun] = [
    # Regularized baseline (300M only)
    DownstreamRun("300m4kcda-203Mx16-dcr-cos-lr0.0030-wd1.60-bs64", epochs=16),
    # Shuffled copy-scaling best runs
    DownstreamRun("300m4kcda-203Mx4-dcr+w2s^0.75-cos-lr0.0030-wd0.80-bs64", epochs=4, teacher_ratio=0.75),
    DownstreamRun("300m4kcda-203Mx8-dcr+s4^0.75-cos-lr0.0030-wd0.80-bs64", epochs=8, teacher_ratio=0.75),
    DownstreamRun("300m4kcda-203Mx8-dcr+s8^0.75-cos-lr0.0030-wd0.80-bs64", epochs=8, teacher_ratio=0.75),
    DownstreamRun("300m4kcda-203Mx16-dcr+s16^0.75-cos-lr0.0030-wd0.40-bs64", epochs=16, teacher_ratio=0.75),
    DownstreamRun("300m4kcda-203Mx16-dcr+s32^0.75-cos-lr0.0030-wd0.40-bs64", epochs=16, teacher_ratio=0.75),
    # Sorted copy-scaling best runs
    DownstreamRun("300m4kcda-203Mx8-dcr+w2^0.75-cos-lr0.0030-wd0.80-bs64", epochs=8, teacher_ratio=0.75),
    DownstreamRun("300m4kcda-203Mx8-dcr+b4^0.75-cos-lr0.0030-wd0.40-bs64", epochs=8, teacher_ratio=0.75),
    DownstreamRun("300m4kcda-203Mx16-dcr+b8^0.75-cos-lr0.0030-wd0.40-bs64", epochs=16, teacher_ratio=0.75),
    DownstreamRun("300m4kcda-203Mx16-dcr+b16^0.75-cos-lr0.0030-wd0.40-bs64", epochs=16, teacher_ratio=0.75),
    DownstreamRun("300m4kcda-203Mx32-dcr+b32^0.9-cos-lr0.0010-wd0.40-bs64", epochs=32, teacher_ratio=0.9),
    # Latent thoughts runs
    DownstreamRun("300m4kcda-203Mx4-dcr+z2^0.75-cos-lr0.0030-wd0.80-bs64", epochs=4, teacher_ratio=0.75),
    DownstreamRun("300m4kcda-203Mx8-dcr+z4^0.75-cos-lr0.0030-wd0.80-bs64", epochs=8, teacher_ratio=0.75),
    DownstreamRun("300m4kcda-203Mx8-dcr+z8^0.75-cos-lr0.0030-wd0.40-bs64", epochs=8, teacher_ratio=0.75),
    DownstreamRun("300m4kcda-203Mx16-dcr+z16^0.75-cos-lr0.0030-wd0.40-bs64", epochs=16, teacher_ratio=0.75),
    DownstreamRun("300m4kcda-203Mx32-dcr+z32^0.9-cos-lr0.0010-wd0.40-bs64", epochs=32, teacher_ratio=0.9),
]


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _select_runs(selected_names: list[str] | None) -> list[DownstreamRun]:
    if not selected_names:
        return TARGET_RUNS
    lookup = {r.run_name: r for r in TARGET_RUNS}
    missing = [name for name in selected_names if name not in lookup]
    if missing:
        raise ValueError(f"Unknown run_name(s): {', '.join(missing)}")
    return [lookup[name] for name in selected_names]


def _download_runs(runs: list[DownstreamRun], *, prefix: str, models_dir: Path) -> None:
    for run in runs:
        target_dir = models_dir / run.run_name
        target_dir.mkdir(parents=True, exist_ok=True)
        source = f"{prefix}/checkpoints/data_efficiency/{run.run_name}/hf/step-{run.checkpoint_step}/*"
        print(f"Downloading {run.run_name} from step-{run.checkpoint_step}")
        _run(["gcloud", "storage", "cp", "-r", source, str(target_dir)])


def _rearrange_runs(runs: list[DownstreamRun], *, models_dir: Path) -> None:
    for run in runs:
        base_path = models_dir / run.run_name
        seed_path = base_path / "seed0"
        seed_path.mkdir(parents=True, exist_ok=True)
        if not base_path.exists():
            raise FileNotFoundError(f"Missing model directory: {base_path}")
        for entry in list(base_path.iterdir()):
            if entry.name == "seed0":
                continue
            shutil.move(str(entry), str(seed_path / entry.name))
        print(f"Rearranged {run.run_name} -> {seed_path}")


def _eval_runs(runs: list[DownstreamRun], *, models_dir: Path, results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    for run in runs:
        base_name = run.run_name
        model_path = models_dir / base_name
        per_model_results = results_dir / base_name
        per_model_results.mkdir(parents=True, exist_ok=True)

        print(f"Evaluating {base_name}")
        _run(["./scripts/eval_pt_ensemble.sh", str(model_path), "1"])

        root_jsons = list(results_dir.glob("*.json"))
        if not root_jsons:
            raise FileNotFoundError("No root-level results JSON produced by eval script.")
        latest_json = max(root_jsons, key=lambda p: p.stat().st_mtime)
        dest = per_model_results / "1_seeds.json"
        shutil.move(str(latest_json), str(dest))
        print(f"Saved downstream results: {dest}")


def _collect_results(runs: list[DownstreamRun], *, results_dir: Path) -> None:
    all_results: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for run in runs:
        base_name = run.run_name
        run_json = results_dir / base_name / "1_seeds.json"
        if not run_json.exists():
            raise FileNotFoundError(f"Missing eval result: {run_json}")

        with run_json.open("r") as f:
            data = json.load(f)

        metrics: dict[str, dict[str, float]] = {}
        for task, task_results in data.get("results", {}).items():
            metrics[task] = {
                "acc": float(task_results["acc,none"]),
                "acc_stderr": float(task_results["acc_stderr,none"]),
            }
        all_results[base_name] = {"1": metrics}

    output_path = results_dir / "synth_data_downstream_benchmark_results.json"
    with output_path.open("w") as f:
        json.dump(all_results, f, indent=2, sort_keys=True)
    print(f"Wrote consolidated results: {output_path}")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--mode", choices=["download", "rearrange", "eval", "collect", "all"], default="eval")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX, help="GS prefix, e.g. gs://marin-us-central2")
    parser.add_argument("--models_dir", default="models")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--run_name", action="append", default=None, help="Exact run name to include. Repeatable.")
    parser.add_argument("--list_runs", action="store_true", help="Print known run names and exit.")
    args = parser.parse_args()

    if args.list_runs:
        for run in TARGET_RUNS:
            print(run.run_name)
        return

    runs = _select_runs(args.run_name)
    models_dir = Path(args.models_dir)
    results_dir = Path(args.results_dir)

    if args.mode in {"download", "all"}:
        _download_runs(runs, prefix=args.prefix, models_dir=models_dir)
    if args.mode in {"rearrange", "all"}:
        _rearrange_runs(runs, models_dir=models_dir)
    if args.mode in {"eval", "all"}:
        _eval_runs(runs, models_dir=models_dir, results_dir=results_dir)
    if args.mode in {"collect", "all"}:
        _collect_results(runs, results_dir=results_dir)


if __name__ == "__main__":
    main()

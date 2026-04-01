# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Poll a benchmark output directory and launch validation runs once it completes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "lib" / "marin" / "src"))
sys.path.insert(0, str(REPO_ROOT / "lib" / "levanter" / "src"))
sys.path.insert(0, str(REPO_ROOT / "lib" / "iris" / "src"))
LAUNCH_SCRIPT = SCRIPT_DIR / "launch_starcoder_optima_validation.py"
REQUIRED_BENCHMARK_ARTIFACTS = (
    "selection_records.csv",
    "model_scores.csv",
    "curve_points.csv",
    "selector_summary.csv",
    "predicted_optima.csv",
    "predicted_optima.jsonl",
)
STATUS_FILENAME = "validation_monitor_status.json"


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime())


def _append_log(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as handle:
        handle.write(f"[{_now()}] {message}\n")


def _benchmark_complete(output_dir: Path) -> bool:
    return all((output_dir / name).exists() for name in REQUIRED_BENCHMARK_ARTIFACTS)


def _benchmark_process_running(output_dir: Path) -> bool:
    command = ["pgrep", "-f", f"starcoder_generic_selector_benchmark.py --output-dir {output_dir}"]
    result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    return result.returncode == 0


def _write_status(status_path: Path, payload: dict) -> None:
    status_path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _repo_relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _launch_validation_job(
    *,
    benchmark_output_dir: Path,
    dataset: str,
    region: str,
    iris_config: Path,
    data_seed: int,
) -> dict[str, str]:
    launch_script = _repo_relative_path(LAUNCH_SCRIPT)
    benchmark_dir = _repo_relative_path(benchmark_output_dir)
    command = [
        "uv",
        "run",
        "python",
        "-m",
        "marin.run.iris_run",
        "--config",
        str(iris_config.resolve()),
        "--",
        "--no-wait",
        "--region",
        region,
        "--zone",
        "us-east5-a",
        "--memory",
        "4GB",
        "--disk",
        "10GB",
        "--extra",
        "marin:tpu",
        "--extra",
        "marin:eval",
        "--",
        "python",
        launch_script,
        "--benchmark-output-dir",
        benchmark_dir,
        "--dataset",
        dataset,
        "--data-seed",
        str(data_seed),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    return {
        "dataset": dataset,
        "region": region,
        "iris_config": str(iris_config),
        "command": " ".join(command),
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor benchmark completion and launch validation runs")
    parser.add_argument("--benchmark-output-dir", type=Path, required=True)
    parser.add_argument("--region", type=str, default="us-central1")
    parser.add_argument("--iris-config", type=Path, default=Path("lib/iris/examples/marin.yaml"))
    parser.add_argument("--poll-seconds", type=int, default=600)
    parser.add_argument("--datasets", type=str, default="two_phase_starcoder,three_phase_starcoder")
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--log-path", type=Path, default=None)
    parser.add_argument("--once", action="store_true", help="Check one time and exit.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    benchmark_output_dir = args.benchmark_output_dir.resolve()
    log_path = args.log_path or (benchmark_output_dir / "validation_monitor.log")
    status_path = benchmark_output_dir / STATUS_FILENAME
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]

    _append_log(log_path, f"monitor started for {benchmark_output_dir}")

    while True:
        if status_path.exists():
            _append_log(log_path, f"status already exists at {status_path}; exiting")
            return

        complete = _benchmark_complete(benchmark_output_dir)
        running = _benchmark_process_running(benchmark_output_dir)
        _append_log(log_path, f"heartbeat complete={complete} benchmark_running={running}")

        if complete:
            launches = []
            for dataset in datasets:
                _append_log(log_path, f"launching validation for {dataset}")
                launches.append(
                    _launch_validation_job(
                        benchmark_output_dir=benchmark_output_dir,
                        dataset=dataset,
                        region=args.region,
                        iris_config=args.iris_config.resolve(),
                        data_seed=args.data_seed,
                    )
                )
            payload = {
                "status": "launched",
                "launched_at": _now(),
                "benchmark_output_dir": str(benchmark_output_dir),
                "launches": launches,
            }
            _write_status(status_path, payload)
            _append_log(log_path, f"validation launches submitted; status written to {status_path}")
            return

        if not running:
            payload = {
                "status": "benchmark_incomplete_and_not_running",
                "observed_at": _now(),
                "benchmark_output_dir": str(benchmark_output_dir),
                "missing_artifacts": [
                    name for name in REQUIRED_BENCHMARK_ARTIFACTS if not (benchmark_output_dir / name).exists()
                ],
            }
            _write_status(status_path, payload)
            _append_log(log_path, "benchmark is no longer running and required artifacts are missing; exiting")
            return

        if args.once:
            return
        time.sleep(max(args.poll_seconds, 1))


if __name__ == "__main__":
    sys.exit(main())

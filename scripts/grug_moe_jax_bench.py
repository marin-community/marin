from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import os
import statistics
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import jmp
from fray.cluster import ResourceConfig
from levanter.tracker.json_logger import JsonLoggerConfig
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import jnp_to_python
from levanter.utils.mesh import MeshConfig
from rigging.filesystem import marin_prefix

from experiments.grug.moe.launch import baseline_moe
from experiments.grug.moe.train import GrugRunConfig, GrugTrainerConfig, _run_grug_local
from marin.execution.executor import Executor

_SCRIPT_ROOT = Path(__file__).resolve().parents[1]


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_output(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], text=True, cwd=_SCRIPT_ROOT, stderr=subprocess.DEVNULL).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _extract_json_payload(line: str) -> dict[str, Any] | None:
    start = line.find("{")
    if start == -1:
        return None
    try:
        payload = json.loads(line[start:])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _read_tracker_events(log_path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in log_path.read_text().splitlines():
        payload = _extract_json_payload(line)
        if payload is None:
            continue
        if payload.get("tracker") != "json_logger":
            continue
        events.append(payload)
    return events


def _attach_tracker_log_file(log_path: Path, logger_name: str) -> logging.Handler:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(log_path, mode="a")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return handler


class _NoopCheckpointerConfig:
    def create(self, run_id: str):
        del run_id
        return None

    def expanded_path(self, run_id: str) -> str:
        raise RuntimeError(f"Checkpointing is disabled for benchmark run {run_id}")


def _summarize_metric(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    value = jnp_to_python(value)
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    return None


def _dedupe_step_records(step_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    step_to_index: dict[int, int] = {}
    for record in step_records:
        step = record.get("step")
        if not isinstance(step, int):
            deduped.append(record)
            continue

        if step in step_to_index:
            deduped[step_to_index[step]] = record
        else:
            step_to_index[step] = len(deduped)
            deduped.append(record)

    return deduped


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a short local Grug MoE throughput benchmark.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tpu-type", default="v5p-8")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--output-root", default="/tmp/grug-moe-jax-bench")
    parser.add_argument("--log-root", default="/tmp/grug-moe-jax-bench-logs")
    return parser


def _resolved_baseline_config():
    prefix = marin_prefix()
    executor = Executor(
        prefix=prefix,
        executor_info_base_path=os.path.join(prefix, "experiments"),
        description="grug-moe-jax-bench",
    )
    executor.compute_version(baseline_moe, is_pseudo_dep=False)
    return executor.configs[baseline_moe]


def main() -> None:
    args = build_parser().parse_args()
    if args.warmup_steps >= args.steps:
        raise ValueError("--warmup-steps must be smaller than --steps")

    output_root = Path(args.output_root).expanduser().resolve()
    log_root = Path(args.log_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    output_path = output_root / args.run_id
    log_path = log_root / f"{args.run_id}.log"
    if log_path.exists():
        log_path.unlink()
    tracker_logger_name = f"grug_moe_jax_bench.{args.run_id}"
    tracker_handler = _attach_tracker_log_file(log_path, tracker_logger_name)

    launch_cfg = _resolved_baseline_config()
    if args.tokenizer is not None:
        data_cfg = replace(launch_cfg.data, tokenizer=args.tokenizer)
    else:
        data_cfg = launch_cfg.data

    trainer_cfg = TrainerConfig(
        id=args.run_id,
        seed=args.seed,
        train_batch_size=args.batch_size,
        num_train_steps=args.steps,
        mp=jmp.get_policy(launch_cfg.mp),
        tracker=JsonLoggerConfig(logger_name=tracker_logger_name),
        log_dir=log_root,
        profiler=replace(TrainerConfig().profiler, enabled=False),
        use_explicit_mesh_axes=True,
        mesh=MeshConfig(axes={"expert": 1}),
        require_accelerator=True,
        allow_nondivisible_batch_size=False,
        log_jaxprs=False,
        log_xla_hlo=False,
        checkpointer=_NoopCheckpointerConfig(),
    )

    config = GrugRunConfig(
        model=launch_cfg.model,
        data=data_cfg,
        resources=ResourceConfig.with_tpu(args.tpu_type),
        optimizer=launch_cfg.optimizer,
        trainer=GrugTrainerConfig(
            trainer=trainer_cfg,
            log_every=launch_cfg.grug_trainer.log_every,
            ema_beta=launch_cfg.grug_trainer.ema_beta,
            z_loss_weight=launch_cfg.grug_trainer.z_loss_weight,
        ),
        eval=None,
    )

    try:
        _run_grug_local(config)
    finally:
        tracker_logger = logging.getLogger(tracker_logger_name)
        tracker_logger.removeHandler(tracker_handler)
        tracker_handler.close()

    if not log_path.exists():
        raise FileNotFoundError(f"Expected benchmark log at {log_path}")

    events = _read_tracker_events(log_path)
    step_records: list[dict[str, Any]] = []
    finish_summary: dict[str, Any] | None = None
    for event in events:
        if event.get("event") == "log":
            metrics = event.get("metrics", {})
            if not isinstance(metrics, dict):
                continue
            if "throughput/tokens_per_second" not in metrics:
                continue
            step_records.append(
                {
                    "step": event.get("step"),
                    "tokens_per_second": _to_float(metrics.get("throughput/tokens_per_second")),
                    "gflops_per_second": _to_float(metrics.get("throughput/gflops_per_second")),
                    "mfu": _to_float(metrics.get("throughput/mfu")),
                    "duration": _to_float(metrics.get("throughput/duration")),
                    "loading_time": _to_float(metrics.get("throughput/loading_time")),
                    "hook_time": _to_float(metrics.get("throughput/hook_time")),
                }
            )
        elif event.get("event") == "finish":
            summary = event.get("summary")
            if isinstance(summary, dict):
                finish_summary = summary

    step_records = _dedupe_step_records(step_records)

    measured_records = [
        record
        for record in step_records
        if isinstance(record.get("step"), int) and args.warmup_steps <= int(record["step"]) < args.steps
    ]

    def metric_values(name: str) -> list[float]:
        return [value for record in measured_records if (value := record.get(name)) is not None]

    report = {
        "benchmark_id": args.run_id,
        "git_head": _git_output("rev-parse", "HEAD"),
        "git_branch": _git_output("branch", "--show-current"),
        "steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "measured_steps": len(measured_records),
        "batch_size": args.batch_size,
        "tpu_type": args.tpu_type,
        "output_path": str(output_path),
        "log_path": str(log_path),
        "versions": {
            "jax": _package_version("jax"),
            "jaxlib": _package_version("jaxlib"),
            "libtpu": _package_version("libtpu"),
            "torch": _package_version("torch"),
        },
        "steady_state": {
            "tokens_per_second": _summarize_metric(metric_values("tokens_per_second")),
            "gflops_per_second": _summarize_metric(metric_values("gflops_per_second")),
            "mfu": _summarize_metric(metric_values("mfu")),
            "duration": _summarize_metric(metric_values("duration")),
            "loading_time": _summarize_metric(metric_values("loading_time")),
            "hook_time": _summarize_metric(metric_values("hook_time")),
        },
        "first_logged_step": step_records[0] if step_records else None,
        "last_logged_step": step_records[-1] if step_records else None,
        "finish_summary": finish_summary,
        "measured_records": measured_records,
        "libtpu_lockfile_exists_after_run": os.path.exists("/tmp/libtpu_lockfile"),
    }

    print(f"{args.run_id}_REPORT_START")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"{args.run_id}_REPORT_END")


if __name__ == "__main__":
    main()

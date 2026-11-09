# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reconstruct plumbing timings from preserved Iris, Finelog, W&B and Harbor evidence."""

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import median

import click


def json_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def attempt_measurements(attempts: list[dict], resources: list[dict]) -> list[dict]:
    counts = {row["job_id"]: (row["resources"]["device"] or {}).get("count", 0) for row in resources}
    result = []
    for row in attempts:
        submitted, created, started, finished = (
            int(row[k]) if row[k] else None
            for k in ["submitted_at_ms", "created_at_ms", "started_at_ms", "finished_at_ms"]
        )
        assert submitted is not None and created is not None
        result.append(
            {
                "job": row["job_id"],
                "task": row["task_id"],
                "attempt": row["attempt_id"],
                "gpus": counts[row["job_id"]],
                "submitted_ms": submitted,
                "created_ms": created,
                "started_ms": started,
                "finished_ms": finished,
                "submit_to_attempt_seconds": (created - submitted) / 1000,
                "attempt_to_running_seconds": None if started is None else (started - created) / 1000,
                "running_gpu_hours": (
                    None
                    if started is None or finished is None
                    else (finished - started) * counts[row["job_id"]] / 3600000
                ),
                "attempt_envelope_gpu_hours": (
                    None if finished is None else (finished - created) * counts[row["job_id"]] / 3600000
                ),
            }
        )
    return result


def number(value: float | None) -> str:
    return "unknown" if value is None else f"{value:.2f}"


def reconstruction(directory: Path, sft_history: Path) -> str:
    attempts = attempt_measurements(
        list(csv.DictReader((directory / "attempts.csv").open())), json.loads((directory / "resources.json").read_text())
    )
    (directory / "attempt-measurements.json").write_text(json.dumps(attempts, indent=2))
    lines = [
        "# Reconstructed plumbing component timings",
        "",
        "Back-filled from preserved native measurements and controller records. This is not a "
        "clean single-attempt loop. "
        "The BixBench continuation and archive are still pending. Raw input files in this directory are the evidence.",
        "",
        "## Queue, startup and GPU occupancy",
        "",
        "Reconstructed: submission→attempt creation is pre-attempt wait; attempt creation→RUNNING includes scheduling, "
        "pod/image/runtime startup. These are not interchangeable with exact GPU allocation boundaries. "
        "GPU-hours below use controller RUNNING→terminal windows; attempt-envelope hours include "
        "the wider creation→terminal window. "
        "Neither certifies physical release time. Missing boundaries stay unknown, never zero.",
        "",
        "| Job / task | Attempt | GPU | Pre-attempt wait s | Startup-to-running s | Running GPU-h "
        "| Attempt envelope GPU-h |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in attempts:
        lines.append(
            f"| {row['task'].removeprefix('/benfeuer/')} | {row['attempt']} | {row['gpus']} | "
            f"{number(row['submit_to_attempt_seconds'])} | {number(row['attempt_to_running_seconds'])} | "
            f"{number(row['running_gpu_hours'])} | {number(row['attempt_envelope_gpu_hours'])} |"
        )
    lines += [
        "",
        "## SFT: native W&B durations, reconstructed summary",
        "",
        "| Step (zero-based) | Train s | Input loading s | Iteration s | Checkpoint callback s | Tokens/s |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    history = sorted(json_rows(sft_history), key=lambda r: r["_step"])
    for row in history:
        lines.append(
            "| "
            + str(row["_step"])
            + " | "
            + " | ".join(
                f"{row[key]:.3f}"
                for key in [
                    "throughput/duration",
                    "throughput/loading_time",
                    "throughput/iteration_time",
                    "throughput/checkpoint_time",
                    "throughput/tokens_per_second",
                ]
            )
            + " |"
        )
    warm = [r["throughput/duration"] for r in history[1:]]
    lines += [
        "",
        f'Warm train-step median: {median(warm):.3f}s; first step: {history[0]["throughput/duration"]:.3f}s. '
        "The first-step excess is not an isolated compilation measurement. Checkpoint callback "
        "duration may only enqueue asynchronous work.",
    ]
    telemetry = json_rows(directory / "rl-telemetry.jsonl")
    phases = defaultdict(dict)
    for row in telemetry:
        attrs = json.loads(row["attributes_json"])
        if row["name"] == "phase_duration_seconds" and attrs.get("clock_domain") == "inclusive_wall":
            key = (attrs["step"], attrs["phase"])
            if attrs["phase"] in phases[attrs["step"]]:
                raise ValueError(f"Duplicate inclusive trainer phase: {key}")
            phases[attrs["step"]][attrs["phase"]] = row["value"]
    columns = [
        "step",
        "wait_for_generation_buffer",
        "assemble_generation_group_mini_batch",
        "fwd_logprobs_values_reward",
        "policy_train",
        "run_training",
        "sync_weights",
        "save_checkpoints",
    ]
    lines += [
        "",
        "## RL: native trainer phases",
        "",
        "Inclusive phases overlap and must not be summed. "
        "Waiting for generation is not total async generation work. Missing phase values are unknown, not zero.",
        "",
        "| Optimizer step | " + " | ".join(columns) + " |",
        "|---:" + "|---:" * len(columns) + "|",
    ]
    for step, values in sorted(phases.items(), key=lambda kv: int(kv[0])):
        lines.append(
            "| " + step + " | " + " | ".join(f"{values[k]:.3f}" if k in values else "—" for k in columns) + " |"
        )
    lag = Counter(row["value"] for row in telemetry if row["name"] == "rollout_staleness_steps")
    lines += [
        "",
        f"Observed consumed-group staleness histogram (lag: count): {dict(sorted(lag.items()))}.",
        "",
        "Native start/end windows are preserved in native-timers.jsonl; they include checkpoint "
        "and weight-sync boundaries.",
        "",
        "## Serving model load, compile and CUDA graphs",
        "",
        "Durations below are native log measurements, summarized by job/kind. Multiple ranks "
        "overlap; min/max are reported, not summed.",
        "",
        "| Job | Phase | Native observations | Min s | Max s |",
        "|---|---|---:|---:|---:|",
    ]
    patterns = {
        "model_load": r"Model loading took .*? and ([0-9.]+) seconds",
        "cuda_graph_capture": r"Graph capturing finished in ([0-9.]+) secs",
        "compile_and_warmup": r"torch.compile and initial profiling/warmup run together took ([0-9.]+) s",
    }
    observations = defaultdict(list)
    for row in json_rows(directory / "runtime-log-markers.jsonl"):
        for name, pattern in patterns.items():
            match = re.search(pattern, row["data"])
            if match:
                observations[(row["key"].rsplit("/", 1)[0], name)].append(float(match[1]))
    for (job, phase), values in sorted(observations.items()):
        lines.append(
            f'| {job.removeprefix("/benfeuer/")} | {phase} | {len(values)} | {min(values):.3f} | {max(values):.3f} |'
        )
    lines += [
        "",
        "## Installed-agent benchmark phases",
        "",
        "Native Harbor result timestamps; differences reconstructed in seconds.",
        "",
        "| Task | Total | Environment setup | Agent setup | Agent execution | Verifier |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for path in sorted((directory / "eval-trials").rglob("result.json")):
        row = json.loads(path.read_text())
        values = []
        for phase in [None, "environment_setup", "agent_setup", "agent_execution", "verifier"]:
            span = row if phase is None else row.get(phase)
            if not span or not span.get("started_at") or not span.get("finished_at"):
                values.append("unknown")
            else:
                values.append(
                    f"{(datetime.fromisoformat(span['finished_at'].replace('Z','+00:00'))-datetime.fromisoformat(span['started_at'].replace('Z','+00:00'))).total_seconds():.3f}"
                )
        lines.append("| " + row["task_name"] + " | " + " | ".join(values) + " |")
    lines += [
        "",
        "## Remaining gaps and interpretation",
        "",
        "Task events retain PodInitializing and terminal observations, but not every pod "
        "scheduling/image-pull boundary. "
        "Image pull and physical allocation/release cannot be inferred exactly from RUNNING timestamps. "
        "The 15 non-installed-agent benchmark and pool-wave windows were not emitted by this "
        "historical evaluation entrypoint. "
        "Do not substitute aggregate eval wall time for each benchmark. "
        "Export conversion/upload and judge startup residuals need additional native markers; "
        "measured model-load/compile windows are above. "
        "New runs must emit these windows and archive the evidence. No target-geometry run is "
        "permitted until the measurement gaps are addressed.",
    ]
    return "\n".join(lines) + "\n"


@click.command()
@click.option("--directory", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--sft-history", type=click.Path(path_type=Path, exists=True), required=True)
def main(directory: Path, sft_history: Path) -> None:
    (directory / "report.md").write_text(reconstruction(directory, sft_history))
    sources = [
        sft_history,
        directory / "attempts.csv",
        directory / "resources.json",
        directory / "rl-telemetry.jsonl",
        directory / "native-timers.jsonl",
        directory / "runtime-log-markers.jsonl",
        *sorted((directory / "eval-trials").rglob("result.json")),
    ]
    (directory / "evidence-manifest.json").write_text(
        json.dumps(
            [
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for path in sources
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

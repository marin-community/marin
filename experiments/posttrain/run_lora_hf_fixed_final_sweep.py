#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the final-step-only LoRA HF repair sweep serially on Iris."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import fsspec

from experiments.posttrain.submit_lora_hf_fixed_batch import (
    DEFAULT_CONFIG_FILE,
    DEFAULT_CPU,
    DEFAULT_DISK,
    DEFAULT_MEMORY,
    DEFAULT_OUTPUT_PREFIX,
    DEFAULT_REGION,
    DEFAULT_RUN_NAMES,
    ReexportJob,
    discover_jobs,
    latest_jobs_by_run,
)

logger = logging.getLogger(__name__)

DEFAULT_STARTUP_POLL = 120
DEFAULT_POLL = 300
DEFAULT_SUBMIT_TIMEOUT = 900


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", default=DEFAULT_CONFIG_FILE)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--cpu", type=float, default=DEFAULT_CPU)
    parser.add_argument("--memory", default=DEFAULT_MEMORY)
    parser.add_argument("--disk", default=DEFAULT_DISK)
    parser.add_argument("--run-label", default="r1")
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument(
        "--run-name", action="append", default=None, help="Restrict to specific run name(s). Repeatable."
    )
    parser.add_argument(
        "--startup-poll",
        type=int,
        default=DEFAULT_STARTUP_POLL,
        help="Seconds to sleep after each submit before the first status check.",
    )
    parser.add_argument(
        "--poll",
        type=int,
        default=DEFAULT_POLL,
        help="Seconds between steady-state controller checks.",
    )
    parser.add_argument(
        "--submit-timeout",
        type=int,
        default=DEFAULT_SUBMIT_TIMEOUT,
        help="Maximum seconds to wait for the direct submit command to return.",
    )
    parser.add_argument(
        "--state-file",
        default=None,
        help="Optional JSON state file path. Defaults to scratch/<timestamp>_lora_hf_fixed_final_sweep_state.json.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def find_json_block(text: str) -> str:
    start = text.find("[")
    if start == -1:
        start = text.find("{")
    if start == -1:
        raise ValueError(f"Could not find JSON in output:\n{text}")
    return text[start:]


def run_command(command: list[str], *, timeout: int = 600, check: bool = False) -> subprocess.CompletedProcess[str]:
    logger.debug("Running command: %s", " ".join(command))
    return subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=check)


def canonical_job_id(job_name: str) -> str:
    user = os.environ.get("USER", "ahmed")
    return f"/{user}/{job_name}"


def iris_job_list(config_file: str, job_id: str) -> list[dict[str, Any]]:
    result = run_command(
        [
            "uv",
            "run",
            "iris",
            "--config",
            config_file,
            "job",
            "list",
            "--json",
            "--prefix",
            job_id,
        ],
        timeout=180,
    )
    if result.returncode != 0:
        raise RuntimeError(f"iris job list failed for {job_id}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    payload = json.loads(find_json_block(result.stdout))
    if not isinstance(payload, list):
        raise ValueError(f"Expected JSON list from iris job list, got: {type(payload)}")
    return payload


def iris_job_logs(config_file: str, job_id: str) -> str:
    result = run_command(
        [
            "uv",
            "run",
            "iris",
            "--config",
            config_file,
            "job",
            "logs",
            "--since-seconds",
            "1800",
            "--include-children",
            job_id,
        ],
        timeout=300,
    )
    return result.stdout + result.stderr


def maybe_existing_job(config_file: str, job_name: str) -> dict[str, Any] | None:
    matches = iris_job_list(config_file, canonical_job_id(job_name))
    if not matches:
        return None
    return matches[0]


def submit_job(
    *,
    config_file: str,
    region: str,
    cpu: float,
    memory: str,
    disk: str,
    job: ReexportJob,
    submit_timeout: int,
) -> str:
    command = [
        "uv",
        "run",
        "iris",
        "--config",
        config_file,
        "job",
        "run",
        "--no-wait",
        "--job-name",
        job.job_name,
        "--cpu",
        str(int(cpu) if cpu.is_integer() else cpu),
        "--memory",
        memory,
        "--disk",
        disk,
        "--region",
        region,
        "--",
        *job.command,
    ]
    job_id = canonical_job_id(job.job_name)
    logger.info("Submitting %s", job_id)
    result = run_command(command, timeout=submit_timeout)
    if result.returncode == 0:
        if result.stdout.strip():
            logger.info("submit stdout:\n%s", result.stdout)
        if result.stderr.strip():
            logger.info("submit stderr:\n%s", result.stderr)
        return job_id

    existing = maybe_existing_job(config_file, job.job_name)
    if existing is not None:
        logger.info("Submit returned non-zero, but controller has %s", job_id)
        return job_id
    raise RuntimeError(f"submit failed for {job.job_name}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")


def path_exists(path: str) -> bool:
    fs, plain_path = fsspec.core.url_to_fs(path)
    return fs.exists(plain_path)


def read_json(path: str) -> dict[str, Any]:
    fs, plain_path = fsspec.core.url_to_fs(path)
    with fs.open(plain_path, "r") as handle:
        return json.load(handle)


def verify_reexport_output(output_path: str) -> None:
    required_files = (
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    )
    for filename in required_files:
        candidate = f"{output_path}/{filename}"
        if not path_exists(candidate):
            raise RuntimeError(f"Missing required file: {candidate}")

    index_payload = read_json(f"{output_path}/model.safetensors.index.json")
    weight_map = index_payload.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError(f"Invalid or empty weight_map in {output_path}/model.safetensors.index.json")

    for shard_name in sorted(set(weight_map.values())):
        candidate = f"{output_path}/{shard_name}"
        if not path_exists(candidate):
            raise RuntimeError(f"Missing shard referenced by index: {candidate}")

    tokenizer_config = read_json(f"{output_path}/tokenizer_config.json")
    chat_template = tokenizer_config.get("chat_template")
    if not isinstance(chat_template, str) or not chat_template.strip():
        raise RuntimeError(f"Missing embedded chat_template in {output_path}/tokenizer_config.json")


def wait_for_terminal(config_file: str, job_id: str, *, startup_poll: int, poll: int) -> dict[str, Any]:
    logger.info("Waiting %ss before first status check for %s", startup_poll, job_id)
    time.sleep(startup_poll)
    while True:
        matches = iris_job_list(config_file, job_id)
        if not matches:
            logger.warning("Job not yet visible: %s", job_id)
            time.sleep(poll)
            continue
        status = matches[0]
        state = status["state"]
        logger.info("Job %s state=%s", job_id, state)
        if state == "JOB_STATE_SUCCEEDED":
            return status
        if state in {"JOB_STATE_FAILED", "JOB_STATE_KILLED", "JOB_STATE_WORKER_FAILED", "JOB_STATE_UNSCHEDULABLE"}:
            recent_logs = iris_job_logs(config_file, job_id)
            raise RuntimeError(f"Job {job_id} failed with state={state}\nRecent logs:\n{recent_logs}")
        time.sleep(poll)


def default_state_path() -> Path:
    scratch_dir = Path("scratch")
    scratch_dir.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M")
    return scratch_dir / f"{timestamp}_lora_hf_fixed_final_sweep_state.json"


def write_state(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def build_jobs(args: argparse.Namespace) -> list[ReexportJob]:
    run_names = tuple(args.run_name) if args.run_name is not None else DEFAULT_RUN_NAMES
    jobs, warnings = discover_jobs(
        run_names=run_names,
        output_prefix=args.output_prefix,
        run_label=args.run_label,
        base_model_ref="gs://marin-us-central1/models/marin-community--marin-8b-instruct--0378f9c",
        checkpoint_subpath="model",
        lora_r=64,
        lora_alpha=64.0,
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
        verify_shapes=True,
        step_filters=None,
    )
    for warning in warnings:
        logger.warning(warning)
    return latest_jobs_by_run(jobs)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    state_path = Path(args.state_file) if args.state_file is not None else default_state_path()
    jobs = build_jobs(args)
    manifest = [
        {
            "run_name": job.run_name,
            "step_name": job.step_name,
            "job_name": job.job_name,
            "source_hf_path": job.source_hf_path,
            "raw_checkpoint_path": job.raw_checkpoint_path,
            "output_path": job.output_path,
        }
        for job in jobs
    ]
    state: dict[str, Any] = {
        "ts": int(time.time() * 1000),
        "config_file": args.config_file,
        "region": args.region,
        "run_label": args.run_label,
        "output_prefix": args.output_prefix,
        "jobs_total": len(jobs),
        "jobs_completed": 0,
        "current_job_id": None,
        "current_job_name": None,
        "manifest": manifest,
        "completed": [],
    }
    write_state(state_path, state)
    logger.info("State file: %s", state_path)

    for index, job in enumerate(jobs, start=1):
        state["ts"] = int(time.time() * 1000)
        state["current_index"] = index
        state["current_job_name"] = job.job_name
        state["current_run_name"] = job.run_name
        state["current_step_name"] = job.step_name
        state["current_output_path"] = job.output_path
        write_state(state_path, state)

        if path_exists(f"{job.output_path}/model.safetensors.index.json"):
            logger.info("Output already exists; verifying and skipping submit: %s", job.output_path)
            verify_reexport_output(job.output_path)
            state["jobs_completed"] += 1
            state["completed"].append(
                {
                    "run_name": job.run_name,
                    "step_name": job.step_name,
                    "job_name": job.job_name,
                    "job_id": canonical_job_id(job.job_name),
                    "output_path": job.output_path,
                    "status": "preexisting_verified",
                }
            )
            write_state(state_path, state)
            continue

        job_id = submit_job(
            config_file=args.config_file,
            region=args.region,
            cpu=args.cpu,
            memory=args.memory,
            disk=args.disk,
            job=job,
            submit_timeout=args.submit_timeout,
        )
        state["current_job_id"] = job_id
        write_state(state_path, state)

        terminal = wait_for_terminal(
            args.config_file,
            job_id,
            startup_poll=args.startup_poll,
            poll=args.poll,
        )
        verify_reexport_output(job.output_path)
        state["jobs_completed"] += 1
        state["completed"].append(
            {
                "run_name": job.run_name,
                "step_name": job.step_name,
                "job_name": job.job_name,
                "job_id": job_id,
                "output_path": job.output_path,
                "status": terminal["state"],
            }
        )
        write_state(state_path, state)
        logger.info("Verified repaired export for %s at %s", job.run_name, job.output_path)

    state["ts"] = int(time.time() * 1000)
    state["current_job_id"] = None
    state["current_job_name"] = None
    state["status"] = "completed"
    write_state(state_path, state)
    logger.info("Final-step LoRA HF repair sweep complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

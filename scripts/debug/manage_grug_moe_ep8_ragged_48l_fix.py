#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Manage the reconstructed 1e23 EP8 ragged resume launcher."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

# Import ray_run before Fray/Ray dashboard modules; ray_run sets Ray token-auth
# environment before Ray's auth helpers cache their mode.
from marin.run.ray_run import REMOTE_DASHBOARD_URL, _maybe_enable_ray_token_auth
from fray.v1.cluster.ray import DashboardConfig, ray_dashboard
from ray.job_submission import JobSubmissionClient

ROOT = Path(__file__).resolve().parents[2]
STATE_PATH = ROOT / "scratch" / "grug_moe_ep8_ragged_48l_fix_state.json"
CLUSTER_CONFIG = ROOT / "infra" / "marin-big-run.yaml"
LAUNCH_SCRIPT = ROOT / "scripts" / "debug" / "launch_grug_moe_ep8_ragged_48l_fix.py"
SOURCE_CHECKPOINT_BASE = (
    "gs://marin-us-central2/grug/"
    "moe_1e23_d5120_bs2048_ep8_ragged_48l_resume50654_clip15_20260502-d41be7/"
    "checkpoints"
)
DEFAULT_INITIALIZE_FROM = f"{SOURCE_CHECKPOINT_BASE}/step-51262"
COMPLETE_CHECKPOINT_MARKER = "metadata.json"
REDACTED_SECRET = "<redacted>"
SECRET_JSON_RE = re.compile(r'("(?:HF_TOKEN|WANDB_API_KEY)"\s*:\s*")[^"]+(")')
SECRET_ASSIGNMENT_RE = re.compile(r"((?:HF_TOKEN|WANDB_API_KEY)=)[^\s,'\"]+")


@dataclass
class LaunchState:
    submission_id: str
    run_id: str
    initialize_from: str
    step_name: str
    description: str
    launched_at: str
    cluster_config: str


def _parse_step_number(checkpoint_path: str) -> int:
    match = re.search(r"step-(\d+)", checkpoint_path)
    if match is None:
        raise ValueError(f"Could not parse step number from checkpoint path: {checkpoint_path}")
    return int(match.group(1))


def complete_checkpoint_paths(checkpoint_base: str) -> list[str]:
    metadata_glob = f"{checkpoint_base.rstrip('/')}/step-*/{COMPLETE_CHECKPOINT_MARKER}"
    result = subprocess.run(["gcloud", "storage", "ls", metadata_glob], cwd=ROOT, capture_output=True, text=True)
    if result.returncode == 0:
        return [
            line.strip().rsplit(f"/{COMPLETE_CHECKPOINT_MARKER}", maxsplit=1)[0]
            for line in result.stdout.splitlines()
            if line.strip()
        ]
    if "matched no objects" in result.stderr:
        return []
    raise subprocess.CalledProcessError(result.returncode, result.args, result.stdout, result.stderr)


def latest_complete_checkpoint(checkpoint_base: str = SOURCE_CHECKPOINT_BASE) -> str:
    checkpoints = complete_checkpoint_paths(checkpoint_base)
    if not checkpoints:
        raise FileNotFoundError(f"No complete checkpoints found under {checkpoint_base}")
    return max(checkpoints, key=_parse_step_number)


def _load_state() -> LaunchState:
    with STATE_PATH.open() as f:
        payload = json.load(f)
    return LaunchState(**payload)


def _save_state(state: LaunchState) -> None:
    with STATE_PATH.open("w") as f:
        json.dump(asdict(state), f, indent=2, sort_keys=True)
        f.write("\n")


def _redact_sensitive_text(text: str) -> str:
    text = SECRET_JSON_RE.sub(rf"\1{REDACTED_SECRET}\2", text)
    return SECRET_ASSIGNMENT_RE.sub(rf"\1{REDACTED_SECRET}", text)


def _with_client(cluster_config: str, fn) -> None:
    _maybe_enable_ray_token_auth(require_token=False)
    with ray_dashboard(DashboardConfig.from_cluster(cluster_config)):
        client = JobSubmissionClient(REMOTE_DASHBOARD_URL)
        fn(client)


def launch(args: argparse.Namespace) -> None:
    initialize_from = args.initialize_from or latest_complete_checkpoint()
    step = _parse_step_number(initialize_from)
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    run_id = args.run_id or f"moe_1e23_d5120_bs2048_ep8_ragged_48l_resume{step}_clip15_{timestamp}"
    submission_id = args.submission_id or f"ray-run-dlwh-moe-resume{step}-{now.strftime('%Y%m%d-%H%M%S')}"
    step_name = f"grug/{run_id}"
    description = (
        f"Resume the Grug MoE 1e23 ragged EP8 run from initialize_from={initialize_from} "
        "with max_grad_norm=1.5 and permanent checkpoint retention every 1000 steps."
    )

    cmd = [
        sys.executable,
        "-m",
        "marin.run.ray_run",
        "--cluster",
        str(CLUSTER_CONFIG),
        "--no_wait",
        "--submission-id",
        submission_id,
        "-e",
        "GRUG_RUN_ID",
        run_id,
        "-e",
        "MARIN_GRUG_INITIALIZE_FROM",
        initialize_from,
        "-e",
        "MARIN_GRUG_STEP_NAME",
        step_name,
        "-e",
        "MARIN_GRUG_DESCRIPTION",
        description,
        "--",
        "python",
        str(LAUNCH_SCRIPT.relative_to(ROOT)),
    ]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    stdout = _redact_sensitive_text(result.stdout)
    stderr = _redact_sensitive_text(result.stderr)
    if stdout:
        print(stdout, end="", flush=True)
    if stderr:
        print(stderr, end="", file=sys.stderr, flush=True)
    if result.returncode != 0:
        raise RuntimeError(f"launch failed with exit code {result.returncode}\nstdout:\n{stdout}\nstderr:\n{stderr}")
    _save_state(
        LaunchState(
            submission_id=submission_id,
            run_id=run_id,
            initialize_from=initialize_from,
            step_name=step_name,
            description=description,
            launched_at=now.isoformat(),
            cluster_config=str(CLUSTER_CONFIG),
        )
    )


def status(_: argparse.Namespace) -> None:
    state = _load_state()

    def _show(client: JobSubmissionClient) -> None:
        payload = {
            "submission_id": state.submission_id,
            "run_id": state.run_id,
            "status": client.get_job_status(state.submission_id),
            "initialize_from": state.initialize_from,
            "launched_at": state.launched_at,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))

    _with_client(state.cluster_config, _show)


def logs(args: argparse.Namespace) -> None:
    state = _load_state()

    def _show(client: JobSubmissionClient) -> None:
        lines = client.get_job_logs(state.submission_id).splitlines()
        print("\n".join(lines[-args.tail :]))

    _with_client(state.cluster_config, _show)


def stop(_: argparse.Namespace) -> None:
    state = _load_state()

    def _stop(client: JobSubmissionClient) -> None:
        client.stop_job(state.submission_id)
        print(f"Stopped {state.submission_id}")

    _with_client(state.cluster_config, _stop)


def checkpoints(_: argparse.Namespace) -> None:
    state = _load_state()
    checkpoint_base = f"gs://marin-us-central2/grug/{state.run_id}-*/checkpoints"
    checkpoints = complete_checkpoint_paths(checkpoint_base)
    if checkpoints:
        for checkpoint in checkpoints:
            print(checkpoint)
        return
    print(f"No complete checkpoints found yet for {checkpoint_base}/step-*/{COMPLETE_CHECKPOINT_MARKER}")


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    launch_parser = subparsers.add_parser("launch")
    launch_parser.add_argument(
        "--initialize-from",
        help=(
            "Checkpoint path to initialize from. Defaults to latest complete checkpoint under "
            f"{SOURCE_CHECKPOINT_BASE}."
        ),
    )
    launch_parser.add_argument("--run-id")
    launch_parser.add_argument("--submission-id")
    launch_parser.set_defaults(func=launch)

    status_parser = subparsers.add_parser("status")
    status_parser.set_defaults(func=status)

    logs_parser = subparsers.add_parser("logs")
    logs_parser.add_argument("--tail", type=int, default=200)
    logs_parser.set_defaults(func=logs)

    stop_parser = subparsers.add_parser("stop")
    stop_parser.set_defaults(func=stop)

    checkpoints_parser = subparsers.add_parser("checkpoints")
    checkpoints_parser.set_defaults(func=checkpoints)

    return parser


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

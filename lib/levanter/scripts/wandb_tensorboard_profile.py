#!/usr/bin/env python
# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Download the JAX profile directory for a W&B run and launch TensorBoard."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

import wandb
import wandb.errors as wandb_errors

from levanter.utils.fsspec_utils import join_path, mirror_directory

PROFILER_DIR_NAME = "profiler"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the JAX profile directory for a W&B run and launch TensorBoard on it."
    )
    parser.add_argument(
        "target",
        help=(
            "Run identifier. Accepts a bare run id (requires --entity and --project), an "
            "entity/project/run_id path, or a full W&B URL."
        ),
    )
    parser.add_argument("--entity", help="W&B entity (team or username) if target is a bare run id.")
    parser.add_argument("--project", help="W&B project if target is a bare run id.")
    parser.add_argument(
        "--download-root",
        type=Path,
        help="Optional directory where the profile tree will be mirrored. Defaults to a new temp directory.",
    )
    parser.add_argument(
        "--tensorboard",
        default="tensorboard",
        help="TensorBoard executable to invoke. Defaults to 'tensorboard' found on PATH.",
    )
    parser.add_argument("--port", type=int, help="Optional port to bind TensorBoard to.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mirror the profile directory and print the TensorBoard command without launching it.",
    )
    return parser.parse_args()


def normalize_run_path(target: str, entity: Optional[str], project: Optional[str]) -> Tuple[str, str, str]:
    if target.startswith(("http://", "https://")):
        parsed = urlparse(target)
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) < 3:
            raise ValueError(f"Could not parse run information from URL: {target}")
        entity = parts[0]
        project = parts[1]
        if parts[2] == "runs" and len(parts) >= 4:
            run_id = parts[3]
        else:
            run_id = parts[2]
        return entity, project, run_id

    parts = [p for p in target.split("/") if p]
    if len(parts) == 1:
        if not entity or not project:
            raise ValueError("Bare run ids require --entity and --project.")
        return entity, project, parts[0]

    if len(parts) >= 3:
        entity = parts[0]
        project = parts[1]
        if parts[2] == "runs" and len(parts) >= 4:
            run_id = parts[3]
        else:
            run_id = parts[2]
        return entity, project, run_id

    raise ValueError(f"Unrecognized run target: {target}")


def resolve_profile_dir(run: "wandb.apis.public.Run") -> str:
    run_id = resolve_profile_run_id(run)
    trainer_config = run.config.get("trainer")
    if not isinstance(trainer_config, dict):
        raise RuntimeError(f"Run {run.path} does not expose a trainer config.")

    log_dir = trainer_config.get("log_dir")
    if not log_dir:
        raise RuntimeError(f"Run {run.path} does not expose trainer.log_dir.")

    return join_path(join_path(str(log_dir), run_id), PROFILER_DIR_NAME)


def resolve_profile_run_id(run: "wandb.apis.public.Run") -> str:
    trainer_config = run.config.get("trainer")
    if not isinstance(trainer_config, dict):
        raise RuntimeError(f"Run {run.path} does not expose a trainer config.")

    run_id = trainer_config.get("id")
    if isinstance(run_id, str) and run_id:
        return run_id

    return run.path[-1]


def mirror_profile_dir(profile_dir: str, root: Path | None, *, run_id: str) -> Path:
    return mirror_directory(
        profile_dir,
        root,
        run_id=run_id,
        leaf_dirname=PROFILER_DIR_NAME,
        temp_dir_prefix="wandb-profile-",
    )


def build_tensorboard_command(executable: str, logdir: Path, port: Optional[int]) -> list[str]:
    command = [executable, f"--logdir={logdir}"]
    if port is not None:
        command.append(f"--port={port}")
    return command


def ensure_tensorboard_available(executable: str) -> None:
    if os.path.sep in executable or executable.startswith("."):
        if not Path(executable).exists():
            raise FileNotFoundError(f"TensorBoard executable '{executable}' was not found.")
        return

    if shutil.which(executable) is None:
        raise FileNotFoundError(
            f"TensorBoard executable '{executable}' not found on PATH. Use --tensorboard to point to it explicitly."
        )


def main() -> None:
    args = parse_args()

    try:
        entity, project, run_id = normalize_run_path(args.target, args.entity, args.project)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    run_path = f"{entity}/{project}/{run_id}"
    api = wandb.Api()
    try:
        run = api.run(run_path)
    except wandb_errors.CommError as exc:
        message = str(exc)
        if "not found" in message.lower() or "404" in message:
            print(f"Run '{run_path}' was not found.", file=sys.stderr)
        else:
            print(f"Failed to reach Weights & Biases: {message}", file=sys.stderr)
        sys.exit(1)
    except wandb_errors.Error as exc:
        print(f"Failed to load run '{run_path}': {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        profile_dir = resolve_profile_dir(run)
        profile_run_id = resolve_profile_run_id(run)
        download_path = mirror_profile_dir(profile_dir, args.download_root, run_id=profile_run_id)
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"Failed to resolve profile directory for '{run_path}': {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Resolved profile directory '{profile_dir}' to {download_path}")
    print("Press Ctrl+C to stop TensorBoard when finished.")

    ensure_tensorboard_available(args.tensorboard)
    command = build_tensorboard_command(args.tensorboard, download_path, args.port)
    print("TensorBoard command:", " ".join(command))

    if args.dry_run:
        return

    try:
        subprocess.run(command, check=True)
    except KeyboardInterrupt:
        print("TensorBoard interrupted by user; exiting.")
        return
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        print(f"TensorBoard exited with status {exc.returncode}.", file=sys.stderr)
        sys.exit(exc.returncode)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Download the latest JAX profile artifact for a W&B run and launch TensorBoard."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Optional, Tuple
from urllib.parse import urlparse

import wandb
import wandb.errors as wandb_errors


ARTIFACT_TYPE = "jax_profile"
DEFAULT_ALIAS = "latest"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the latest JAX profile artifact for a W&B run and launch TensorBoard on it."
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
        "--alias",
        default=DEFAULT_ALIAS,
        help="Artifact alias to use. Defaults to 'latest'. If not found, the freshest artifact is used.",
    )
    parser.add_argument(
        "--download-root",
        type=Path,
        help="Optional directory where the artifact will be downloaded. Defaults to a new temp directory.",
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
        help="Download the artifact and print the TensorBoard command without launching it.",
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


def iter_profile_artifacts(run: "wandb.apis.public.Run") -> Iterable["wandb.sdk.Artifact"]:
    for artifact in run.logged_artifacts():
        if artifact.type == ARTIFACT_TYPE:
            yield artifact


def _alias_names(artifact: "wandb.sdk.Artifact") -> set[str]:  # type: ignore[name-defined]
    names: set[str] = set()
    for alias in artifact.aliases or []:
        name = getattr(alias, "name", alias)
        if name is not None:
            names.add(str(name))
    return names


def select_profile_artifact(run: "wandb.apis.public.Run", alias: Optional[str]) -> "wandb.sdk.Artifact":
    candidates = list(iter_profile_artifacts(run))
    if not candidates:
        raise RuntimeError(f"No artifacts of type '{ARTIFACT_TYPE}' were found for run {run.path}.")

    if alias:
        for artifact in candidates:
            alias_names = _alias_names(artifact)
            if alias in alias_names:
                return artifact

    candidates.sort(key=lambda art: art.updated_at or art.created_at, reverse=True)
    return candidates[0]


def download_artifact(artifact: "wandb.sdk.Artifact", root: Optional[Path]) -> Path:
    if root is None:
        root = Path(tempfile.mkdtemp(prefix="wandb-profile-"))
    else:
        root.mkdir(parents=True, exist_ok=True)
    download_path = Path(artifact.download(root=str(root)))
    return download_path


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

    artifact = select_profile_artifact(run, args.alias)
    download_path = download_artifact(artifact, args.download_root)

    alias_list = sorted(_alias_names(artifact))
    print(f"Downloaded artifact '{artifact.name}' (aliases: {alias_list}) to {download_path}")
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

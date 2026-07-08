# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional, Protocol
from urllib.parse import urlparse

from rigging.filesystem import StoragePath, url_to_fs

PROFILER_DIR_NAME = "profiler"
WANDB_RUNS_SEGMENT = "runs"
TRAINER_CONFIG_KEY = "trainer"


@dataclass(frozen=True)
class RunTarget:
    entity: str
    project: str
    run_id: str


class WandbRunLike(Protocol):
    config: Mapping[str, Any]
    path: Sequence[str]
    id: str
    summary: Mapping[str, Any]


def normalize_run_target(target: str, entity: Optional[str], project: Optional[str]) -> RunTarget:
    if target.startswith(("http://", "https://")):
        parsed = urlparse(target)
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) < 3:
            raise ValueError(f"Could not parse run information from URL: {target}")
        entity = parts[0]
        project = parts[1]
        if parts[2] == WANDB_RUNS_SEGMENT and len(parts) >= 4:
            run_id = parts[3]
        else:
            run_id = parts[2]
        return RunTarget(entity=entity, project=project, run_id=run_id)

    parts = [p for p in target.split("/") if p]
    if len(parts) == 1:
        if not entity or not project:
            raise ValueError("Bare run ids require --entity and --project.")
        return RunTarget(entity=entity, project=project, run_id=parts[0])

    if len(parts) >= 3:
        entity = parts[0]
        project = parts[1]
        if parts[2] == WANDB_RUNS_SEGMENT and len(parts) >= 4:
            run_id = parts[3]
        else:
            run_id = parts[2]
        return RunTarget(entity=entity, project=project, run_id=run_id)

    raise ValueError(f"Unrecognized run target: {target}")


def resolve_profile_run_id(run: WandbRunLike) -> str:
    trainer_config = run.config.get(TRAINER_CONFIG_KEY)
    if not isinstance(trainer_config, dict):
        raise RuntimeError(f"Run {run.path} does not expose a trainer config.")

    run_id = trainer_config.get("id")
    if isinstance(run_id, str) and run_id:
        return run_id

    return run.path[-1]


def resolve_profile_dir(run: WandbRunLike) -> str:
    trainer_config = run.config.get(TRAINER_CONFIG_KEY)
    if not isinstance(trainer_config, dict):
        raise RuntimeError(f"Run {run.path} does not expose a trainer config.")

    log_dir = trainer_config.get("log_dir")
    if not isinstance(log_dir, str) or not log_dir:
        raise RuntimeError(f"Run {run.path} does not expose trainer.log_dir.")

    return str(StoragePath(log_dir) / resolve_profile_run_id(run) / PROFILER_DIR_NAME)


def mirror_profile_dir(profile_dir: str, root: Path | None, *, run_id: str) -> Path:
    fs, fs_path = url_to_fs(profile_dir)
    source_path = Path(fs_path)

    if root is None:
        if _is_local_fs(fs):
            if not source_path.exists():
                raise FileNotFoundError(f"Profile directory '{profile_dir}' does not exist.")
            return source_path
        root = Path(tempfile.mkdtemp(prefix="wandb-profile-"))
    else:
        root.mkdir(parents=True, exist_ok=True)

    download_path = root / run_id / PROFILER_DIR_NAME
    if _is_local_fs(fs) and download_path.resolve() == source_path.resolve():
        if not source_path.exists():
            raise FileNotFoundError(f"Profile directory '{profile_dir}' does not exist.")
        return source_path

    if download_path.exists():
        shutil.rmtree(download_path)
    download_path.parent.mkdir(parents=True, exist_ok=True)

    if _is_local_fs(fs):
        if not source_path.exists():
            raise FileNotFoundError(f"Profile directory '{profile_dir}' does not exist.")
        shutil.copytree(source_path, download_path)
    else:
        fs.get(fs_path, str(download_path), recursive=True)
    return download_path


def _is_local_fs(fs: Any) -> bool:
    protocol = fs.protocol[0] if isinstance(fs.protocol, tuple) else fs.protocol
    return protocol in (None, "", "file")

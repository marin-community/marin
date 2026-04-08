# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""YAML-based persistence for monitoring collections and run state."""

import logging
from dataclasses import asdict
from pathlib import Path

import yaml

from marin.dispatch.schema import (
    IrisRunConfig,
    MonitoringCollection,
    RayRunConfig,
    RunPointer,
    RunState,
    RunStatus,
    RunTrack,
)

logger = logging.getLogger(__name__)

COLLECTIONS_DIR = ".agents/collections"


def _collections_root(repo_root: Path) -> Path:
    d = repo_root / COLLECTIONS_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def collection_path(repo_root: Path, name: str) -> Path:
    return _collections_root(repo_root) / f"{name}.yaml"


def state_path(repo_root: Path, name: str) -> Path:
    return _collections_root(repo_root) / f"{name}.state.yaml"


def _run_pointer_to_dict(rp: RunPointer) -> dict:
    d: dict = {"track": str(rp.track)}
    if rp.ray is not None:
        d["ray"] = asdict(rp.ray)
    if rp.iris is not None:
        d["iris"] = asdict(rp.iris)
    return d


def _run_pointer_from_dict(d: dict) -> RunPointer:
    track = RunTrack(d["track"])
    ray = RayRunConfig(**d["ray"]) if "ray" in d else None
    iris = IrisRunConfig(**d["iris"]) if "iris" in d else None
    return RunPointer(track=track, ray=ray, iris=iris)


def _collection_to_dict(c: MonitoringCollection) -> dict:
    return {
        "name": c.name,
        "prompt": c.prompt,
        "logbook": c.logbook,
        "branch": c.branch,
        "issue": c.issue,
        "runs": [_run_pointer_to_dict(rp) for rp in c.runs],
        "created_at": c.created_at,
        "paused": c.paused,
    }


def _collection_from_dict(d: dict) -> MonitoringCollection:
    runs = tuple(_run_pointer_from_dict(r) for r in d.get("runs", []))
    return MonitoringCollection(
        name=d["name"],
        prompt=d["prompt"],
        logbook=d["logbook"],
        branch=d["branch"],
        issue=d["issue"],
        runs=runs,
        created_at=d.get("created_at", ""),
        paused=d.get("paused", False),
    )


def _run_state_to_dict(s: RunState) -> dict:
    return {
        "last_status": str(s.last_status),
        "last_check": s.last_check,
        "restart_count": s.restart_count,
        "last_error": s.last_error,
        "consecutive_failures": s.consecutive_failures,
    }


def _run_state_from_dict(d: dict) -> RunState:
    return RunState(
        last_status=RunStatus(d.get("last_status", "unknown")),
        last_check=d.get("last_check", ""),
        restart_count=d.get("restart_count", 0),
        last_error=d.get("last_error", ""),
        consecutive_failures=d.get("consecutive_failures", 0),
    )


def save_collection(repo_root: Path, collection: MonitoringCollection) -> None:
    path = collection_path(repo_root, collection.name)
    with open(path, "w") as f:
        yaml.dump(_collection_to_dict(collection), f, default_flow_style=False, sort_keys=False)
    logger.info("Saved collection %s to %s", collection.name, path)


def load_collection(repo_root: Path, name: str) -> MonitoringCollection:
    path = collection_path(repo_root, name)
    with open(path) as f:
        data = yaml.safe_load(f)
    return _collection_from_dict(data)


def save_state(repo_root: Path, name: str, states: list[RunState]) -> None:
    path = state_path(repo_root, name)
    data = [_run_state_to_dict(s) for s in states]
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def load_state(repo_root: Path, name: str) -> list[RunState]:
    path = state_path(repo_root, name)
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        return []
    if not data:
        return []
    return [_run_state_from_dict(d) for d in data]


def list_collections(repo_root: Path) -> list[str]:
    root = _collections_root(repo_root)
    return sorted(p.stem for p in root.glob("*.yaml") if not p.stem.endswith(".state"))


def delete_collection(repo_root: Path, name: str) -> None:
    for path in [collection_path(repo_root, name), state_path(repo_root, name)]:
        try:
            path.unlink()
            logger.info("Deleted %s", path)
        except FileNotFoundError:
            pass

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inventory local Harbor packages without building images or executing task code.

Example:
    uv run python -m experiments.evaluation.audit_harbor_tasks ./tasks \
        --source organization/dataset --revision <snapshot-commit>

Compose files describe topology; Docker command references are review candidates,
not proof that a task needs a Docker daemon. This inventory does not certify that
an image or task can run in a particular sandbox.
"""

import argparse
import hashlib
import json
import re
import tomllib
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

COMPOSE_NAMES = {"compose.yaml", "compose.yml", "docker-compose.yaml", "docker-compose.yml"}
SERVICE_KEYS = {"image", "build", "command", "entrypoint", "environment", "working_dir", "depends_on", "healthcheck"}
REVIEW_KEYS = {
    "privileged",
    "devices",
    "cap_add",
    "network_mode",
    "pid",
    "ipc",
    "volumes",
    "ports",
    "expose",
    "networks",
    "secrets",
    "configs",
    "extends",
}
MAX_SCAN_BYTES = 1_000_000
REFERENCE_PATTERNS = {
    "docker_command": re.compile(
        r"\b(?:docker(?:-compose)?|podman)\s+(?:run|exec|build|compose|info|ps|pull|start|stop)\b"
    ),
    "docker_daemon": re.compile(r"\bdockerd\b|docker\.sock|\bDOCKER_HOST\b|docker:dind"),
}


@dataclass(frozen=True)
class Reference:
    path: str
    line: int
    kind: str


@dataclass(frozen=True)
class ComposeInventory:
    path: str
    sha256: str
    services: tuple[str, ...]
    review_fields: tuple[str, ...]
    other_fields: tuple[str, ...]


@dataclass(frozen=True)
class TaskInventory:
    path: str
    manifest_sha256: str
    compose: tuple[ComposeInventory, ...]
    docker_references: tuple[Reference, ...]
    unscanned_files: tuple[str, ...]


@dataclass(frozen=True)
class InventorySummary:
    tasks: int
    tasks_with_compose: int
    tasks_with_multiple_services: int
    tasks_with_docker_references: int
    tasks_with_unscanned_files: int


@dataclass(frozen=True)
class Inventory:
    source: str
    revision: str
    root: str
    summary: InventorySummary
    tasks: tuple[TaskInventory, ...]


def compose_inventory(path: Path, task_root: Path) -> ComposeInventory:
    if path.stat().st_size > MAX_SCAN_BYTES:
        raise ValueError(f"Compose file exceeds the inventory size limit: {path.relative_to(task_root)}")
    content = path.read_bytes()
    try:
        config = yaml.safe_load(content)
    except yaml.YAMLError:
        raise ValueError(f"Invalid Compose YAML: {path.relative_to(task_root)}") from None
    if not isinstance(config, Mapping) or not isinstance(config.get("services"), Mapping):
        raise ValueError(f"Compose file requires a services mapping: {path.relative_to(task_root)}")
    services = config["services"]
    review = []
    other = [str(key) for key in config if key != "services"]
    for name, service in services.items():
        if not isinstance(name, str) or not isinstance(service, Mapping):
            raise ValueError(f"Invalid Compose service: {path.relative_to(task_root)}")
        for key in service:
            location = f"services.{name}.{key}"
            if key in REVIEW_KEYS:
                review.append(location)
            elif key not in SERVICE_KEYS:
                other.append(location)
    return ComposeInventory(
        path=path.relative_to(task_root).as_posix(),
        sha256=hashlib.sha256(content).hexdigest(),
        services=tuple(sorted(services)),
        review_fields=tuple(sorted(review)),
        other_fields=tuple(sorted(other)),
    )


def task_inventory(task_root: Path, dataset_root: Path) -> TaskInventory:
    manifest_path = task_root / "task.toml"
    if manifest_path.stat().st_size > MAX_SCAN_BYTES:
        raise ValueError(f"Task manifest exceeds the inventory size limit: {task_root.relative_to(dataset_root)}")
    manifest = manifest_path.read_bytes()
    try:
        tomllib.loads(manifest.decode())
    except (UnicodeError, tomllib.TOMLDecodeError):
        raise ValueError(f"Invalid task manifest: {task_root.relative_to(dataset_root)}") from None
    compose = []
    references = []
    unscanned = []
    for path in sorted(task_root.rglob("*")):
        if not path.is_file():
            continue
        if path.name in COMPOSE_NAMES:
            compose.append(compose_inventory(path, task_root))
        if path.suffix not in {".sh", ".py", ".md", ".toml", ".yaml", ".yml"} and path.name != "Dockerfile":
            continue
        relative = path.relative_to(task_root).as_posix()
        if path.stat().st_size > MAX_SCAN_BYTES:
            unscanned.append(relative)
            continue
        try:
            content = path.read_text()
        except UnicodeError:
            unscanned.append(relative)
            continue
        for number, line in enumerate(content.splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            for kind, pattern in REFERENCE_PATTERNS.items():
                if pattern.search(line):
                    references.append(Reference(relative, number, kind))
    return TaskInventory(
        path=task_root.relative_to(dataset_root).as_posix(),
        manifest_sha256=hashlib.sha256(manifest).hexdigest(),
        compose=tuple(compose),
        docker_references=tuple(references),
        unscanned_files=tuple(unscanned),
    )


def inventory(root: Path, *, source: str, revision: str) -> Inventory:
    """Report structural topology and reference locations for a supplied local snapshot.

    The caller supplies source and revision; this inventory does not verify either.
    Compose output contains service and field names; declaration values and
    command text are omitted.
    """
    root = root.resolve(strict=True)
    manifests = sorted(root.rglob("task.toml"))
    if not manifests:
        raise ValueError(f"No Harbor task manifests found under {root}")
    tasks = tuple(task_inventory(path.parent, root) for path in manifests)
    return Inventory(
        source=source,
        revision=revision,
        root=str(root),
        summary=InventorySummary(
            tasks=len(tasks),
            tasks_with_compose=sum(bool(task.compose) for task in tasks),
            tasks_with_multiple_services=sum(any(len(c.services) > 1 for c in task.compose) for task in tasks),
            tasks_with_docker_references=sum(bool(task.docker_references) for task in tasks),
            tasks_with_unscanned_files=sum(bool(task.unscanned_files) for task in tasks),
        ),
        tasks=tasks,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--source", required=True, help="Dataset identifier; recorded without verification")
    parser.add_argument("--revision", required=True, help="Snapshot revision; recorded without verification")
    args = parser.parse_args()
    print(json.dumps(asdict(inventory(args.root, source=args.source, revision=args.revision)), indent=2))


if __name__ == "__main__":
    main()

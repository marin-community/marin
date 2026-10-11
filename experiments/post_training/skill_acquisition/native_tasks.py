# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Adapt pinned BFCL packages to isolated verification with unchanged tests."""

import hashlib
import json
import shutil
import tomllib
from dataclasses import asdict
from pathlib import Path

from experiments.post_training.bfcl_rl.data import TaskIdentity, task_identity


def native_task_config(text: str, image: str) -> str:
    """Add runtime fields to the original BFCL package format without rewriting other fields."""
    original = tomllib.loads(text)
    if "artifacts" in original or "environment_mode" in original["verifier"]:
        raise ValueError("Expected the original shared-verifier BFCL configuration")
    if any(key in original["environment"] for key in ("docker_image", "workdir", "env")):
        raise ValueError("BFCL runtime fields are already configured")
    if "@sha256:" not in image:
        raise ValueError("BFCL native images must be digest-pinned")
    if text.count("[verifier]\n") != 1 or text.count("[environment]\n") != 1:
        raise ValueError("Unsupported BFCL configuration section layout")
    text = 'artifacts = ["/app/result.json"]\n' + text
    text = text.replace("[verifier]\n", '[verifier]\nenvironment_mode = "separate"\n')
    text = text.replace("[environment]\n", f'[environment]\ndocker_image = {json.dumps(image)}\nworkdir = "/app"\n')
    transformed = tomllib.loads(text)
    transformed.pop("artifacts")
    transformed["verifier"].pop("environment_mode")
    transformed["environment"].pop("docker_image")
    transformed["environment"].pop("workdir")
    if transformed != original:
        raise ValueError("Native transformation changed fields outside its declared runtime changes")
    return text


def materialize_native_tasks(
    source: Path, expected: tuple[TaskIdentity, ...], images: dict[str, str], output: Path
) -> dict:
    """Audit every source task, then copy tasks with explicit verifier and image settings."""
    if output.exists():
        raise FileExistsError(output)
    configs = {}
    failures = []
    for task in expected:
        directory = source / task.name
        try:
            if task_identity(directory) != task:
                raise ValueError("Source task differs from its pinned identity")
            docker_hash = hashlib.sha256((directory / "environment/Dockerfile").read_bytes()).hexdigest()
            configs[task.name] = native_task_config((directory / "task.toml").read_text(), images[docker_hash])
        except (OSError, ValueError, KeyError) as error:
            failures.append({"task": task.name, "error": str(error)})
    if failures:
        raise ValueError(json.dumps({"conversion_failures": failures}, indent=2))
    output.mkdir(parents=True)
    transformed = []
    for task in expected:
        destination = output / task.name
        shutil.copytree(source / task.name, destination)
        (destination / "task.toml").write_text(configs[task.name])
        transformed.append(asdict(task_identity(destination)))
    manifest = {
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "images": images,
        "source_tasks": [asdict(task) for task in expected],
        "transformed_tasks": transformed,
        "submission_path": "/app/result.json",
        "verifier_mode": "separate",
    }
    (output.parent / f"{output.name}-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest

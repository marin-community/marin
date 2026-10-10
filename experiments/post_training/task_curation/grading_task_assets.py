# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fingerprint the selected task-owned graders without executing archived code."""

import hashlib
import json
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from taskcompendium.convert.nemotron_ultra import blend_component
from taskcompendium.convert.tasktrove import archive_file, unpack_task_binary
from taskcompendium.pipeline.inputs import ConversionContext

from experiments.post_training.task_curation.source import GradingDatasetFile, SweGradingAssets
from infra.marina.applets.rl_data_catalog.server.grading_code import python_grading_code

VERIFIER_DIRECTORY = "tests/"
ENVIRONMENT_DOCKERFILE = "environment/Dockerfile"
TASK_CONFIG = "task.toml"


def swe_grading_assets(values: Mapping[str, Any]) -> SweGradingAssets:
    """Decode the pinned files and source selection in an Atlas snapshot."""
    return SweGradingAssets(
        blend=GradingDatasetFile(**values["blend"]),
        proxies=GradingDatasetFile(**values["proxies"]),
        membership=GradingDatasetFile(**values["membership"]),
        component=values["component"],
        partition=values["partition"],
    )


@dataclass(frozen=True)
class SweTaskKey:
    trajectory_id: str
    step: int
    turn: int
    depth: int
    instance_id: str
    agent_cls: str


def swe_task_key(values: Mapping[str, Any]) -> SweTaskKey:
    """Return a task key after validating its trajectory and pivot coordinates."""
    trajectory = values["trajectory_id"]
    if isinstance(trajectory, bool) or not isinstance(trajectory, (str, int)):
        raise ValueError("SWE trajectory identifier must be a string or integer")
    for field in ("step", "turn", "depth"):
        if type(values[field]) is not int:
            raise ValueError(f"SWE {field} must be an integer")
    for field in ("instance_id", "agent_cls"):
        if not isinstance(values[field], str) or not values[field]:
            raise ValueError(f"SWE {field} must be a nonempty string")
    return SweTaskKey(
        str(trajectory), values["step"], values["turn"], values["depth"], values["instance_id"], values["agent_cls"]
    )


def verifier_asset_revision(path: str, content: bytes) -> str:
    """Ignore presentation edits while retaining executable grader and data inputs."""
    text = content.decode()
    if path == TASK_CONFIG:
        value = json.dumps(tomllib.loads(text).get("verifier", {}), sort_keys=True)
    elif path.endswith(".json"):
        value = json.dumps(json.loads(text), sort_keys=True)
    elif path.endswith(".toml"):
        value = json.dumps(tomllib.loads(text), sort_keys=True)
    elif path.endswith(".py") or text.startswith("#!/usr/bin/env python"):
        return python_grading_code(text, ("__module_source__",)).digest
    else:
        return hashlib.sha256(content).hexdigest()
    return hashlib.sha256(value.encode()).hexdigest()


def task_verifier_assets(row: dict[str, Any]) -> tuple[SweTaskKey, dict[str, str]]:
    """Read one archived task's grading inputs, excluding its discovery metadata."""
    prepared = unpack_task_binary(row, ConversionContext(inputs={}, grader_environment=None))
    metadata = archive_file(prepared, "metadata.json")
    if metadata is None:
        raise ValueError("SWE proxy archive has no metadata")
    key = swe_task_key(json.loads(metadata))
    selected = {}
    for path in sorted(prepared["files"]):
        if not (path.startswith(VERIFIER_DIRECTORY) or path in (TASK_CONFIG, ENVIRONMENT_DOCKERFILE)):
            continue
        content = archive_file(prepared, path)
        assert content is not None
        selected[path] = verifier_asset_revision(path, content)
        if path.startswith(VERIFIER_DIRECTORY):
            selected[path + ":mode"] = prepared["file_metadata"][path]["mode"]
    if TASK_CONFIG not in selected or not any(path.startswith(VERIFIER_DIRECTORY) for path in selected):
        raise ValueError("SWE proxy lacks its native verifier configuration or files")
    return key, selected


@lru_cache(maxsize=16)
def grading_dataset_path(reference: GradingDatasetFile) -> Path:
    return Path(
        hf_hub_download(
            repo_id=reference.repository, repo_type="dataset", revision=reference.revision, filename=reference.filename
        )
    )


@lru_cache(maxsize=16)
def proxy_asset_index(reference: GradingDatasetFile) -> dict[SweTaskKey, dict[str, str]]:
    index = {}
    for batch in pq.ParquetFile(grading_dataset_path(reference)).iter_batches(
        columns=["path", "task_binary"], batch_size=64
    ):
        for row in batch.to_pylist():
            key, files = task_verifier_assets(row)
            if key in index:
                raise ValueError("Duplicate SWE proxy coordinates")
            index[key] = files
    return index


def swe_grading_asset_manifest(selection: SweGradingAssets) -> dict[str, Any]:
    """Bind only the proxy verifiers reachable from this source's selected rows."""
    members = frozenset(
        pq.read_table(grading_dataset_path(selection.membership), columns=["instance_id"])["instance_id"].to_pylist()
    )
    proxies = proxy_asset_index(selection.proxies)
    selected = {}
    with grading_dataset_path(selection.blend).open() as stream:
        for line in stream:
            row = json.loads(line)
            if blend_component(row) != selection.component:
                continue
            metadata = row["metadata"]
            if (metadata["instance_id"] in members) != (selection.partition == "swe_gym"):
                continue
            key = swe_task_key({**row["info"], **metadata, "trajectory_id": row["trajectory_id"]})
            if key not in proxies:
                continue  # The native loader drops SWE states absent from the pinned proxy index.
            identity = json.dumps([key.trajectory_id, key.step, key.turn, key.depth, key.instance_id, key.agent_cls])
            selected[identity] = proxies[key]
    if not selected:
        raise ValueError("Selected SWE source has no native proxy verifiers")
    return {"schema_version": 1, "verifiers": dict(sorted(selected.items()))}

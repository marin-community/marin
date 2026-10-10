# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve declaration-owned native grading selectors while building the Atlas."""

import ast
import hashlib
import json
from dataclasses import asdict
from typing import Any

import httpx

from experiments.post_training.task_curation.grading_task_assets import swe_grading_asset_manifest, swe_grading_assets
from infra.marina.applets.rl_data_catalog.server.grading_dependencies import (
    annotate_grading_revision,
    grading_modules,
)


def registry_entrypoints(text: str) -> dict[str, str]:
    entries = {}
    for node in ast.parse(text).body:
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if isinstance(call.func, ast.Name) and call.func.id == "register":
            fields = {keyword.arg: ast.literal_eval(keyword.value) for keyword in call.keywords}
            entries[fields["id"]] = fields["entry_point"]
    if not entries:
        raise ValueError("Native Gym registry has no declared entrypoints")
    return entries


def annotate_catalog_grading(document: dict[str, Any], client: httpx.Client) -> None:
    """Attach selected grading code and dependencies without executing source code."""
    readers = {}
    registries = {}
    for row in document["sources"]:
        selection = row.pop("grading_selection", None)
        if selection is None:
            continue
        key = (selection["marinskyrl_revision"], selection["harbor_revision"])
        if key not in readers:
            readers[key] = grading_modules(client, *key)
            repository = readers[key].packages["skyrl_gym"]
            registries[key] = registry_entrypoints(readers[key].file(repository, "skyrl-gym/skyrl_gym/envs/__init__.py"))
        environment = row["verifier_name"]
        descriptor = {
            "environment": environment,
            "gym_entrypoint": registries[key][environment],
            "verifier_mode": selection["mode"],
        }
        annotate_grading_revision(descriptor, readers[key], tuple(selection["agents"]))
        assets = selection["task_assets"]
        if selection["mode"] == "harbor":
            if assets is None:
                raise ValueError("Harbor grading declarations must identify their native task verifier assets")
            asset_selection = swe_grading_assets(assets)
            descriptor["grading_manifest"]["task_assets"] = swe_grading_asset_manifest(asset_selection)
            descriptor["grading_revision"] = hashlib.sha256(
                json.dumps(descriptor["grading_manifest"], sort_keys=True).encode()
            ).hexdigest()
            row["grading_task_assets"] = asdict(asset_selection)
        for field in ("grading_revision", "grading_manifest", "grading_scope", "grading_repositories"):
            row[field] = descriptor[field]
    document["revision"] = hashlib.sha256(
        json.dumps(document["sources"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

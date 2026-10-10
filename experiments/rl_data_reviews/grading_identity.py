# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify an Atlas grading snapshot against the actual native execution environment."""

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from infra.marina.applets.rl_data_catalog.server.grading_code import python_grading_program
from infra.marina.applets.rl_data_catalog.server.grading_routes import GradingMode, skyrl_grading_routes

PACKAGE_LOCATOR = """import importlib.util,json,sys
result={}
for name in json.loads(sys.argv[1]):
 spec=importlib.util.find_spec(name)
 if spec is None or not spec.submodule_search_locations:
  raise RuntimeError('Native grading package unavailable: '+name)
 result[name]=list(spec.submodule_search_locations)[0]
print(json.dumps(result))
"""
VERSION_LOCATOR = """import importlib.metadata,json,sys
print(json.dumps({name:importlib.metadata.version(name) for name in json.loads(sys.argv[1])}))
"""


@dataclass(frozen=True)
class ExecutionGradingIdentity:
    source_id: str
    source_revision: str
    grading_revision: str
    grading_manifest: dict
    selected_code_verified: bool
    installed_dependency_versions: dict[str, str]


class ExecutionModules:
    def __init__(self, packages: dict[str, Path]):
        self.packages = packages
        self.files: dict[str, str] = {}

    def read(self, module: str) -> str:
        if module in self.files:
            return self.files[module]
        parts = module.split(".")
        base = self.packages[parts[0]].joinpath(*parts[1:])
        path = base.with_suffix(".py") if len(parts) > 1 else base / "__init__.py"
        if not path.is_file():
            path = base / "__init__.py"
        if not path.is_file():
            raise ModuleNotFoundError(module)
        self.files[module] = path.read_text()
        return self.files[module]


def verified_execution_grading(config: dict, base: Path) -> dict | None:
    """Return the verified grading identity as JSON, or None without a snapshot."""
    snapshot_path = config["source"].get("grading_snapshot")
    if snapshot_path is None:
        return None
    snapshot = json.loads((base / snapshot_path).read_text())
    if config["source"]["source_id"] != snapshot["id"]:
        raise ValueError("The grading snapshot identifies a different source selection")
    if config["source"].get("revision") != (snapshot.get("dataset_revision") or snapshot["revision"]):
        raise ValueError("The grading snapshot identifies a different dataset revision")
    manifest = snapshot["grading_manifest"]
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    if digest != snapshot["grading_revision"]:
        raise ValueError("The grading snapshot manifest does not match its revision")
    if snapshot["verifier_mode"] == GradingMode.HARBOR:
        # Harbor asset checks need the optional catalog/TaskCompendium dependencies;
        # ordinary native Gym workers only need the grading-code reader.
        from experiments.post_training.task_curation.grading_task_assets import (  # noqa: PLC0415
            swe_grading_asset_manifest,
        )
        from experiments.post_training.task_curation.source import GradingDatasetFile, SweGradingAssets  # noqa: PLC0415

        assets = snapshot["grading_task_assets"]
        selection = SweGradingAssets(
            blend=GradingDatasetFile(**assets["blend"]),
            proxies=GradingDatasetFile(**assets["proxies"]),
            membership=GradingDatasetFile(**assets["membership"]),
            component=assets["component"],
            partition=assets["partition"],
        )
        if swe_grading_asset_manifest(selection) != manifest["task_assets"]:
            raise ValueError("Selected native task verifier assets differ from the Atlas grading snapshot")
    runtime = config["runtime"]
    if snapshot["verifier_mode"] != GradingMode.HARBOR:
        environment = snapshot["environment"]
        enabled = runtime.get("gym_config", {}).get(environment, {}).get("verifyit_enabled", False)
        if enabled != (snapshot["verifier_mode"] == GradingMode.VERIFYIT):
            raise ValueError("The effective Gym verifyit mode differs from the captured grading route")
    checkout = Path(runtime["marinskyrl_checkout"])
    packages = {
        "skyrl_gym": checkout / "skyrl-gym/skyrl_gym",
        "skyrl_train": checkout / "skyrl-train/skyrl_train",
        "harbor": Path(runtime["harbor_checkout"]) / "src/harbor",
    }
    selected_packages = {module.split(".")[0] for route in manifest["routes"].values() for module in route["modules"]}
    external = sorted(selected_packages - packages.keys())
    interpreter = runtime["harbor_python"] if snapshot["verifier_mode"] == GradingMode.HARBOR else runtime["gym_python"]
    if external:
        locations = json.loads(
            subprocess.check_output([interpreter, "-c", PACKAGE_LOCATOR, json.dumps(external)], text=True)
        )
        packages.update({name: Path(path) for name, path in locations.items()})
    source = ExecutionModules(packages)
    versions = {}
    locked = {
        name: entries for route in manifest["routes"].values() for name, entries in route["locked_packages"].items()
    }
    if locked:
        versions = json.loads(
            subprocess.check_output([interpreter, "-c", VERSION_LOCATOR, json.dumps(sorted(locked))], text=True)
        )
        for name, version in versions.items():
            allowed = {entry["version"] for entry in locked[name] if "version" in entry}
            if version not in allowed:
                raise ValueError(f"Executed grading dependency {name}=={version} differs from its captured runtime lock")
    agents = tuple(snapshot["grading_scope"]["agents"])
    for route in skyrl_grading_routes(snapshot, source, agents):
        program = python_grading_program(source, route.roots, tuple(packages), route.bindings)
        if program.digest != manifest["routes"][route.name]["program_revision"]:
            raise ValueError(f"Executed grading code differs from the Atlas snapshot for {route.name}")
        for relative, expected in manifest["routes"][route.name]["resources"].items():
            if hashlib.sha256((checkout / relative).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Executed grading resource differs at {relative}")
    return asdict(
        ExecutionGradingIdentity(
            source_id=snapshot["id"],
            source_revision=snapshot.get("dataset_revision") or snapshot["revision"],
            grading_revision=digest,
            grading_manifest=manifest,
            selected_code_verified=True,
            installed_dependency_versions=versions,
        )
    )

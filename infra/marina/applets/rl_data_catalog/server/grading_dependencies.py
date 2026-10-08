# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read immutable repositories supplying a source's selected grading program."""

import hashlib
import json
import re
import sys
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import httpx

from .grading_code import PythonGradingProgram, python_grading_program
from .grading_routes import skyrl_grading_routes

HARBOR = "marin-community/harbor"


@dataclass(frozen=True)
class GradingRepository:
    repository: str
    revision: str
    source_root: str
    project_path: str


@dataclass(frozen=True)
class GradingFile:
    repository: str
    revision: str
    path: str
    sha256: str


class RepositoryGradingModules:
    def __init__(self, client: httpx.Client, packages: Mapping[str, GradingRepository]):
        self.client = client
        self.packages = dict(packages)
        self.files: dict[tuple[str, str, str], str] = {}
        self.modules: dict[str, str] = {}

    def file(self, repository: GradingRepository, path: str) -> str:
        key = (repository.repository, repository.revision, path)
        if key not in self.files:
            response = self.client.get(
                f"https://raw.githubusercontent.com/{repository.repository}/{repository.revision}/{path}"
            )
            response.raise_for_status()
            self.files[key] = response.text
        return self.files[key]

    def read(self, module: str) -> str:
        if module in self.modules:
            return self.modules[module]
        repository = self.packages[module.split(".")[0]]
        base = repository.source_root.rstrip("/") + "/" + module.replace(".", "/")
        path = base + ".py"
        response = self.client.get(
            f"https://raw.githubusercontent.com/{repository.repository}/{repository.revision}/{path}"
        )
        if response.status_code == 404:
            path = base + "/__init__.py"
            try:
                content = self.file(repository, path)
            except httpx.HTTPStatusError as error:
                if error.response.status_code != 404:
                    raise
                raise ModuleNotFoundError(module) from error
        else:
            response.raise_for_status()
            content = response.text
            self.files[(repository.repository, repository.revision, path)] = content
        self.modules[module] = content
        return content

    def provenance(self) -> tuple[GradingFile, ...]:
        return tuple(
            GradingFile(repository, revision, path, hashlib.sha256(content.encode()).hexdigest())
            for (repository, revision, path), content in sorted(self.files.items())
        )


VERIFYIT_REQUIREMENT = re.compile(
    r"verifyit(?:\[[^]]+\])?\s*@\s*git\+https://github\.com/"
    r"(?P<repository>[\w.-]+/[\w.-]+?)(?:\.git)?@(?P<revision>[0-9a-f]{40})"
    r"(?:#subdirectory=(?P<subdirectory>[\w/-]+))?"
)


def verifyit_repository(project: str) -> GradingRepository:
    """Read the executed package location, including historical standalone releases."""
    data = tomllib.loads(project)
    requirements = [requirement for requirement in data["project"]["dependencies"] if requirement.startswith("verifyit")]
    if len(requirements) != 1:
        raise ValueError("The grading environment must declare one immutable verifyit dependency")
    match = VERIFYIT_REQUIREMENT.fullmatch(requirements[0])
    if match is None:
        raise ValueError("The verifyit dependency must identify an immutable repository revision")
    subdirectory = match["subdirectory"] or ""
    root = subdirectory + "/" if subdirectory else ""
    return GradingRepository(match["repository"], match["revision"], root + "src", root + "pyproject.toml")


def grading_requirements(
    program: PythonGradingProgram,
    projects: Mapping[str, str],
    import_distributions: Mapping[str, str],
    locked_packages: Mapping[str, list[dict[str, Any]]] | None = None,
) -> dict[str, list[str]]:
    """Keep declarations of packages imported by the selected grading closure.

    A project pin alone does not affect grading applicability. Changed relevant
    dependency declarations do. Transitive imports require a captured lock entry;
    imports with neither a declaration nor a lock entry fail the audit.
    Import/distribution names with different spellings must be mapped explicitly.
    """
    requirements: dict[str, set[str]] = {}
    for project in projects.values():
        table = tomllib.loads(project)["project"]
        declarations = [*table.get("dependencies", [])]
        declarations.extend(item for group in table.get("optional-dependencies", {}).values() for item in group)
        for declaration in declarations:
            match = re.match(r"([A-Za-z0-9_.-]+)", declaration)
            if match is None:
                raise ValueError(f"Invalid dependency declaration {declaration!r}")
            name = match[1].lower().replace("_", "-").replace(".", "-")
            requirements.setdefault(name, set()).add(declaration)
    selected = {}
    for imported in program.external_imports:
        if imported in sys.stdlib_module_names:
            continue
        name = import_distributions.get(imported, imported).lower().replace("_", "-").replace(".", "-")
        if name not in requirements and locked_packages is not None and name in locked_packages:
            selected[imported] = ["Captured transitive dependency in the runtime lock"]
            continue
        if name not in requirements:
            raise ValueError(f"Grading import {imported!r} has no captured dependency declaration")
        selected[imported] = sorted(requirements[name])
    return selected


def grading_manifest(
    program: PythonGradingProgram, requirements: Mapping[str, list[str]], resources: Mapping[str, str]
) -> dict[str, Any]:
    """Bind selected code, runtime dependency declarations and grading data."""
    return {
        "schema_version": 1,
        "program_revision": program.digest,
        "modules": dict(program.modules),
        "requirements": dict(requirements),
        "resources": {path: hashlib.sha256(content.encode()).hexdigest() for path, content in sorted(resources.items())},
    }


IMPORT_DISTRIBUTIONS = {
    "yaml": "pyyaml",
    "upath": "universal-pathlib",
    "dotenv": "python-dotenv",
    "verifiable_instructions": "verifiable-instructions",
    "func_timeout": "func-timeout",
}


def locked_grading_packages(imports: tuple[str, ...], lock: str) -> dict[str, list[dict[str, Any]]]:
    """Select imported distributions and their locked transitive dependencies."""
    table = tomllib.loads(lock)
    packages: dict[str, list[dict[str, Any]]] = {}
    for package in table["package"]:
        name = package["name"].replace("_", "-")
        packages.setdefault(name, []).append(package)
    selected = {}
    pending = [
        IMPORT_DISTRIBUTIONS.get(name, name).replace("_", "-") for name in imports if name not in sys.stdlib_module_names
    ]
    while pending:
        name = pending.pop()
        if name in selected:
            continue
        if name not in packages:
            raise ValueError(f"Grading dependency {name!r} is absent from its runtime lock")
        variants = packages[name]
        selected[name] = sorted(
            [
                {key: variant[key] for key in ("version", "source", "resolution-markers") if key in variant}
                for variant in variants
            ],
            key=lambda variant: str(variant),
        )
        pending.extend(dependency["name"] for variant in variants for dependency in variant.get("dependencies", []))
    return selected


def source_grading_manifest(
    row: Mapping[str, Any], source: RepositoryGradingModules, agents: tuple[str, ...]
) -> dict[str, Any]:
    """Resolve a source's grading routes, runtime declarations and prompt files."""
    routes = skyrl_grading_routes(row, source, agents)
    manifests = {}
    for route in routes:
        program = python_grading_program(source, route.roots, tuple(source.packages), route.bindings)
        used_packages = {module.split(".")[0] for module, _ in program.modules}
        repositories = {source.packages[package] for package in used_packages}
        projects = {
            repository.repository + ":" + repository.project_path: source.file(repository, repository.project_path)
            for repository in repositories
        }
        resources = {path: source.file(source.packages["skyrl_gym"], path) for path in route.resources}
        runtime = source.packages["harbor"] if route.name == "harbor" else source.packages["skyrl_gym"]
        locked = locked_grading_packages(program.external_imports, source.file(runtime, "uv.lock"))
        manifest = grading_manifest(
            program, grading_requirements(program, projects, IMPORT_DISTRIBUTIONS, locked), resources
        )
        manifest["locked_packages"] = locked
        manifests[route.name] = manifest
    return {"schema_version": 1, "routes": manifests}


def grading_modules(client: httpx.Client, skyrl_revision: str, harbor_revision: str) -> RepositoryGradingModules:
    """Read the captured package pins without executing their build or import code."""
    skyrl = GradingRepository("marin-community/MarinSkyRL", skyrl_revision, "skyrl-gym", "skyrl-gym/pyproject.toml")
    source = RepositoryGradingModules(client, {"skyrl_gym": skyrl})
    source.packages["skyrl_train"] = GradingRepository(skyrl.repository, skyrl.revision, "skyrl-train", "pyproject.toml")
    verifyit = verifyit_repository(source.file(skyrl, skyrl.project_path))
    source.packages["verifyit"] = verifyit
    project = tomllib.loads(source.file(verifyit, verifyit.project_path))
    config = [item for item in project["project"].get("dependencies", []) if item.startswith("harbor-config ")]
    if len(config) != 1:
        raise ValueError("The current grading package must declare its immutable Harbor config dependency")
    match = re.fullmatch(
        r"harbor-config @ git\+https://github\.com/([\w.-]+/[\w.-]+)@([0-9a-f]{40})#subdirectory=packages/harbor-config",
        config[0],
    )
    if match is None:
        raise ValueError("The Harbor config dependency does not identify an immutable supported package")
    source.packages["harbor_config"] = GradingRepository(
        match[1], match[2], "src", "packages/harbor-config/pyproject.toml"
    )
    source.packages["harbor"] = GradingRepository(HARBOR, harbor_revision, "src", "pyproject.toml")
    return source


def annotate_grading_revision(row: dict[str, Any], source: RepositoryGradingModules, agents: tuple[str, ...]) -> None:
    """Attach the implementation scope while preserving repository provenance."""
    manifest = source_grading_manifest(row, source, agents)
    row["grading_revision"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    row["grading_manifest"] = manifest
    row["grading_scope"] = {"entrypoint": row["gym_entrypoint"], "mode": row["verifier_mode"], "agents": list(agents)}
    row["grading_repositories"] = {
        name: {
            "repository": repository.repository,
            "revision": repository.revision,
            "source_root": repository.source_root,
        }
        for name, repository in source.packages.items()
    }

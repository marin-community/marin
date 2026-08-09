# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Keep native resources and controller kernels independent of RPC transport."""

import ast
from pathlib import Path

from iris.rpc import resource_pb2

_SOURCE_ROOT = Path(__file__).parents[1] / "src" / "iris"
_GENERATED_IMPORT_SUFFIXES = ("_pb2", "_connect")
_TRANSPORT_FREE_PACKAGES = (
    "resources",
    "cluster/controller/persistence",
    "cluster/controller/reconcile",
    "cluster/controller/scheduling",
)
_TRANSPORT_FREE_MODULES = ("cluster/controller/controller.py",)


def _tracked_module(module: str) -> bool:
    return (
        module == "iris.rpc"
        or module.startswith("iris.rpc.")
        or module.startswith("google.protobuf.")
        or module.endswith(_GENERATED_IMPORT_SUFFIXES)
    )


def _tracked_imports(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.Import):
        return tuple(alias.name for alias in node.names if _tracked_module(alias.name))
    if not isinstance(node, ast.ImportFrom):
        return ()

    parent = node.module or ""
    if parent == "iris.rpc":
        return tuple(f"{parent}.{alias.name}" for alias in node.names)
    if parent.startswith("iris.rpc.") or _tracked_module(parent):
        return (parent,)

    joined = tuple(f"{parent}.{alias.name}" if parent else alias.name for alias in node.names)
    return tuple(module for module in joined if _tracked_module(module))


def _rpc_imports(paths: list[Path]) -> frozenset[tuple[str, str]]:
    imports: set[tuple[str, str]] = set()
    for path in paths:
        relative_path = path.relative_to(_SOURCE_ROOT).as_posix()
        tree = ast.parse(path.read_bytes(), filename=str(path))
        for node in ast.walk(tree):
            imports.update((relative_path, module) for module in _tracked_imports(node))
    return frozenset(imports)


def test_native_resource_and_controller_kernel_packages_do_not_import_rpc_transport() -> None:
    paths = [path for package in _TRANSPORT_FREE_PACKAGES for path in (_SOURCE_ROOT / package).rglob("*.py")]
    paths.extend(_SOURCE_ROOT / module for module in _TRANSPORT_FREE_MODULES)

    violations = _rpc_imports(paths)

    assert not violations, f"RPC imports in native resource/controller packages: {sorted(violations)}"


def test_resource_service_wire_is_independent_of_the_retired_job_wire() -> None:
    assert "job.proto" not in {dependency.name for dependency in resource_pb2.DESCRIPTOR.dependencies}


def test_controller_sqlalchemy_is_confined_to_persistence() -> None:
    controller_root = _SOURCE_ROOT / "cluster" / "controller"
    violations: list[str] = []
    for path in controller_root.rglob("*.py"):
        relative_path = path.relative_to(controller_root).as_posix()
        tree = ast.parse(path.read_bytes(), filename=str(path))
        for node in ast.walk(tree):
            modules: tuple[str, ...] = ()
            if isinstance(node, ast.Import):
                modules = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                modules = (node.module,)
            if any(module == "sqlalchemy" or module.startswith("sqlalchemy.") for module in modules):
                if not relative_path.startswith("persistence/"):
                    violations.append(relative_path)
    assert not violations, f"SQLAlchemy imports outside controller/persistence: {sorted(set(violations))}"

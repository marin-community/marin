# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Keep native resources and controller kernels independent of RPC transport."""

import ast
from pathlib import Path

from iris.rpc import resource_pb2

_SOURCE_ROOT = Path(__file__).parents[1] / "src" / "iris"
_GENERATED_IMPORT_SUFFIXES = ("_pb2", "_connect")
_TRANSPORT_FREE_PACKAGES = ("resources", "cluster/federation")
_CONTROLLER_TRANSPORT_HOSTS = frozenset({"composition.py", "process.py"})
_BACKEND_TRANSPORT_HOSTS = frozenset({"k8s/logship.py"})


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
    paths.append(_SOURCE_ROOT / "cluster" / "types.py")
    paths.extend(
        path
        for path in (_SOURCE_ROOT / "cluster" / "controller").rglob("*.py")
        if path.relative_to(_SOURCE_ROOT / "cluster" / "controller").as_posix() not in _CONTROLLER_TRANSPORT_HOSTS
    )
    paths.extend(
        path
        for path in (_SOURCE_ROOT / "backends").rglob("*.py")
        if path.relative_to(_SOURCE_ROOT / "backends").as_posix() not in _BACKEND_TRANSPORT_HOSTS
    )
    violations = _rpc_imports(paths)

    assert not violations, f"RPC imports in native resource/controller packages: {sorted(violations)}"


def test_resources_and_backends_do_not_reach_through_controller_persistence() -> None:
    violations: list[tuple[str, str]] = []
    for package in ("resources", "backends"):
        for path in (_SOURCE_ROOT / package).rglob("*.py"):
            tree = ast.parse(path.read_bytes(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules = tuple(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    modules = (node.module or "",)
                else:
                    continue
                violations.extend(
                    (path.relative_to(_SOURCE_ROOT).as_posix(), module)
                    for module in modules
                    if module.startswith("iris.cluster.controller.persistence")
                )
    assert not violations, f"Controller persistence imports in native packages: {sorted(violations)}"


def test_worker_core_does_not_import_iris_rpc_transport() -> None:
    paths = [path for path in (_SOURCE_ROOT / "cluster" / "worker").rglob("*.py") if path.name != "main.py"]
    violations = {
        (path, module) for path, module in _rpc_imports(paths) if module == "iris.rpc" or module.startswith("iris.rpc.")
    }

    assert not violations, f"Iris RPC imports in worker core: {sorted(violations)}"


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


def test_rpc_adapters_do_not_import_controller_persistence() -> None:
    violations: list[tuple[str, str]] = []
    for path in (_SOURCE_ROOT / "rpc").rglob("*.py"):
        tree = ast.parse(path.read_bytes(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                modules = (node.module or "",)
            else:
                continue
            violations.extend(
                (path.name, module) for module in modules if module.startswith("iris.cluster.controller.persistence")
            )
    assert not violations, f"Controller persistence imports in RPC adapters: {sorted(violations)}"


def test_control_runtime_does_not_construct_transport_adapters() -> None:
    runtime_path = _SOURCE_ROOT / "cluster" / "controller" / "runtime.py"
    imports = {
        module
        for node in ast.walk(ast.parse(runtime_path.read_bytes(), filename=str(runtime_path)))
        for module in (
            tuple(alias.name for alias in node.names)
            if isinstance(node, ast.Import)
            else ((node.module or "",) if isinstance(node, ast.ImportFrom) else ())
        )
    }
    forbidden = {
        "iris.cluster.controller.process",
        "iris.rpc.legacy.controller_service",
        "iris.rpc.dashboard",
        "iris.rpc.endpoint_service",
        "iris.rpc.resource_service",
    }
    assert imports.isdisjoint(forbidden), f"Transport composition leaked into ControllerRuntime: {imports & forbidden}"

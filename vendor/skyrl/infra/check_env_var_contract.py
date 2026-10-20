#!/usr/bin/env python3
"""Enforce explicit ownership for production environment-variable definitions."""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from marinskyrl.environment_contract import ENV_VAR_SPECS  # noqa: E402

MANAGER_PATH = Path("marinskyrl/environment_contract.py")
EXCLUDED_PARTS = {".agents", ".git", ".venv", "__pycache__", "tests"}
ENV_MAPPING_NAMES = {"env", "env_vars", "environ", "environment", "extra_env", "runtime_env"}
UPPER_NAME = re.compile(r"^[A-Z][A-Z0-9_]+$")
SHELL_EXPORT = re.compile(r"^\s*export\s+([A-Z][A-Z0-9_]+)=", re.MULTILINE)
DOCKER_ENV = re.compile(r"^\s*ENV\s+([A-Z][A-Z0-9_]+)(?:=|\s)", re.MULTILINE | re.IGNORECASE)


def _is_excluded(path: Path) -> bool:
    return bool(EXCLUDED_PARTS.intersection(path.parts)) or path == MANAGER_PATH


def _literal_upper(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str) and UPPER_NAME.fullmatch(node.value):
        return node.value
    return None


def _attribute_path(node: ast.AST) -> tuple[str, ...]:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return tuple(reversed(parts))


class PythonDefinitions(ast.NodeVisitor):
    def __init__(self, path: Path):
        self.path = path
        self.function_stack: list[str] = []
        self.definitions: Counter[str] = Counter()

    def _record(self, name: str, kind: str) -> None:
        self.definitions[f"{self.path.as_posix()}::{kind}::{name}"] += 1

    def _record_mapping(self, node: ast.AST, kind: str) -> None:
        if not isinstance(node, ast.Dict):
            return
        for key, value in zip(node.keys, node.values, strict=True):
            if key is not None and (name := _literal_upper(key)):
                self._record(name, kind)
            self._record_mapping(value, kind)

    @staticmethod
    def _is_environment_mapping(node: ast.AST) -> bool:
        owner = _attribute_path(node)
        return bool(owner and owner[-1] in ENV_MAPPING_NAMES)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.function_stack.append(node.name)
        self.generic_visit(node)
        self.function_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._assignment_target(target)
            if self._is_environment_mapping(target):
                self._record_mapping(node.value, "python-env-mapping")
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._assignment_target(node.target)
        if node.value is not None and self._is_environment_mapping(node.target):
            self._record_mapping(node.value, "python-env-mapping")
        self.generic_visit(node)

    def _assignment_target(self, target: ast.AST) -> None:
        if not isinstance(target, ast.Subscript):
            return
        name = _literal_upper(target.slice)
        if name is None:
            return
        owner = _attribute_path(target.value)
        if owner in {("os", "environ"), ("environ",)} or (owner and owner[-1] in ENV_MAPPING_NAMES):
            self._record(name, "python-assignment")

    def visit_Call(self, node: ast.Call) -> None:
        owner = _attribute_path(node.func)
        if owner in {("os", "putenv"), ("putenv",)} and node.args:
            name = _literal_upper(node.args[0])
            if name:
                self._record(name, "python-putenv")
        if owner in {("os", "environ", "setdefault"), ("environ", "setdefault")} and node.args:
            name = _literal_upper(node.args[0])
            if name:
                self._record(name, "python-setdefault")
        if owner and owner[-1] == "update" and len(owner) > 1 and owner[-2] in ENV_MAPPING_NAMES and node.args:
            self._record_mapping(node.args[0], "python-env-update")
        for keyword in node.keywords:
            if keyword.arg in ENV_MAPPING_NAMES:
                self._record_mapping(keyword.value, "python-env-keyword")
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        function_name = self.function_stack[-1] if self.function_stack else ""
        if node.value is not None and ("env" in function_name or "environment" in function_name):
            self._record_mapping(node.value, "python-env-return")
        self.generic_visit(node)


def _python_definitions(path: Path) -> Counter[str]:
    visitor = PythonDefinitions(path)
    visitor.visit(ast.parse((REPO_ROOT / path).read_text(), filename=str(path)))
    return visitor.definitions


def _yaml_extra_env_definitions(path: Path) -> Counter[str]:
    import yaml

    document = yaml.safe_load((REPO_ROOT / path).read_text())
    definitions: Counter[str] = Counter()

    def visit(value: object) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "extra_env" and isinstance(child, dict):
                    for name in child:
                        if isinstance(name, str) and UPPER_NAME.fullmatch(name):
                            definitions[f"{path.as_posix()}::yaml-extra-env::{name}"] += 1
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(document)
    return definitions


def _repository_files() -> list[Path]:
    if not (REPO_ROOT / ".git").exists():
        return [path for path in REPO_ROOT.rglob("*") if path.is_file()]

    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        check=True,
        capture_output=True,
    )
    return [REPO_ROOT / path for path in result.stdout.decode().split("\0") if path]


def definitions() -> Counter[str]:
    found: Counter[str] = Counter()
    for absolute_path in _repository_files():
        path = absolute_path.relative_to(REPO_ROOT)
        if _is_excluded(path):
            continue
        if path.suffix == ".py":
            found.update(_python_definitions(path))
        elif path.suffix in {".yaml", ".yml"}:
            found.update(_yaml_extra_env_definitions(path))
        elif path.suffix in {".sh", ".sbatch"}:
            for name in SHELL_EXPORT.findall(absolute_path.read_text(errors="replace")):
                found[f"{path.as_posix()}::shell-export::{name}"] += 1
        elif path.name.lower().startswith("dockerfile"):
            for name in DOCKER_ENV.findall(absolute_path.read_text(errors="replace")):
                found[f"{path.as_posix()}::docker-env::{name.upper()}"] += 1
    return found


def contract_errors(current: Counter[str], specs) -> list[str]:
    specs_by_name = {spec.name: spec for spec in specs}
    errors = []
    for definition, count in sorted(current.items()):
        _, kind, name = definition.rsplit("::", 2)
        spec = specs_by_name.get(name)
        if spec is None:
            errors.append(f"{definition} ({count} occurrences): unregistered name")
            continue
        allowed = {writer.value for writer in spec.writers}
        if kind not in allowed:
            errors.append(
                f"{definition} ({count} occurrences): writer kind {kind!r} is not allowed for owner {spec.owner!r}"
            )
    return errors


def main() -> int:
    errors = contract_errors(definitions(), ENV_VAR_SPECS)
    if not errors:
        return 0
    print("Environment-variable definitions violate marinskyrl.environment_contract ownership:")
    for error in errors:
        print(f"  {error}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

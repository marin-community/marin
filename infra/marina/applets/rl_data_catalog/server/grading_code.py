# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Describe selected Python grading code without importing upstream modules."""

import ast
import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

VERIFYIT_MODES_BINDING = "__verifyit_modes__"
DEFINITION_BINDINGS = "__definition_bindings__"


@dataclass(frozen=True)
class PythonGradingCode:
    digest: str
    imports: tuple[tuple[str, str], ...]
    imported_members: tuple[tuple[str, str, tuple[str, ...]], ...]
    imported_bases: tuple[tuple[str, str], ...]


class PythonModuleSource(Protocol):
    def read(self, module: str) -> str: ...


@dataclass(frozen=True)
class PythonGradingProgram:
    digest: str
    modules: tuple[tuple[str, str], ...]
    external_imports: tuple[str, ...]


def mode_literal_bindings(mode: ast.ClassDef) -> dict[str, Any]:
    return {
        mode.name + "." + member.targets[0].id: member.value.value
        for member in mode.body
        if isinstance(member, ast.Assign)
        and isinstance(member.targets[0], ast.Name)
        and isinstance(member.value, ast.Constant)
    }


class GradingBranches(ast.NodeTransformer):
    def __init__(self, bindings: Mapping[str, Any], preserve_documentation: bool = False):
        self.bindings = bindings
        self.preserve_documentation = preserve_documentation

    def remove_docstring(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> None:
        if (
            not self.preserve_documentation
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body.pop(0)

    def value(self, node: ast.expr) -> Any:
        name = ast.unparse(node)
        if name in self.bindings:
            return self.bindings[name]
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            values = [self.value(item) for item in node.elts]
            return set(values) if isinstance(node, ast.Set) else values
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return not self.value(node.operand)
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            left, right = self.value(node.left), self.value(node.comparators[0])
            operation = node.ops[0]
            if isinstance(operation, ast.Eq):
                return left == right
            if isinstance(operation, ast.NotEq):
                return left != right
            if isinstance(operation, ast.In):
                return left in right
            if isinstance(operation, ast.NotIn):
                return left not in right
            if isinstance(operation, ast.Is):
                return left is right
        raise LookupError(name)

    def visit_If(self, node: ast.If) -> ast.AST | list[ast.stmt]:
        try:
            selected = bool(self.value(node.test))
        except LookupError:
            return self.generic_visit(node)
        branch = node.body if selected else node.orelse
        container = ast.Module(body=branch, type_ignores=[])
        self.generic_visit(container)
        return container.body

    def visit_IfExp(self, node: ast.IfExp) -> ast.AST:
        try:
            selected = bool(self.value(node.test))
        except LookupError:
            return self.generic_visit(node)
        return self.visit(node.body if selected else node.orelse)

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> ast.AST:
        self.remove_docstring(node)
        if not node.decorator_list:
            node.returns = None
            for argument in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]:
                argument.annotation = None
            if node.args.vararg:
                node.args.vararg.annotation = None
            if node.args.kwarg:
                node.args.kwarg.annotation = None
        self.generic_visit(node)
        for index, statement in enumerate(node.body):
            if isinstance(statement, (ast.Return, ast.Raise)):
                node.body = node.body[: index + 1]
                break
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        return self.visit_FunctionDef(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        self.remove_docstring(node)
        modes = self.bindings.get(VERIFYIT_MODES_BINDING)
        if modes and self.bindings.get("__name__") == "verifyit.spec" and node.name == "Mode":
            node.body = [
                item
                for item in node.body
                if not isinstance(item, ast.Assign)
                or not isinstance(item.value, ast.Constant)
                or item.value.value in modes
            ]
        return self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        modes = self.bindings.get(VERIFYIT_MODES_BINDING)
        if (
            modes
            and self.bindings.get("__name__") in {"verifyit.grade", "verifyit.spec"}
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in {"MODE_MODULES", "SPEC_TYPES"}
            and isinstance(node.value, ast.Dict)
        ):
            pairs = [
                (key, value)
                for key, value in zip(node.value.keys, node.value.values, strict=True)
                if self.value(key) in modes
            ]
            node.value.keys = [key for key, _ in pairs]
            node.value.values = [value for _, value in pairs]
        return self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST:
        if (
            node.value is not None
            and isinstance(node.target, ast.Name)
            and node.target.id in {"MODE_MODULES", "SPEC_TYPES"}
        ):
            assignment = ast.Assign(targets=[node.target], value=node.value)
            return self.visit_Assign(assignment)
        return self.generic_visit(node)


class UnusedLiteralState(ast.NodeTransformer):
    def __init__(self, reads: set[str]):
        self.reads = reads

    def visit_Assign(self, node: ast.Assign) -> ast.AST | None:
        if (
            isinstance(node.value, ast.Constant)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Attribute)
            and isinstance(node.targets[0].value, ast.Name)
            and node.targets[0].value.id == "self"
            and node.targets[0].attr not in self.reads
            and "__dict__" not in self.reads
        ):
            return None
        return node


def python_grading_code(
    source: str, roots: Sequence[str], bindings: Mapping[str, Any] | None = None
) -> PythonGradingCode:
    """Fingerprint selected definitions and their module-local dependencies.

    Known route values prune unrelated branches. Unknown conditions remain in the
    fingerprint. Imported references are returned for the dependency reader to
    resolve at the captured repository revision; upstream code never executes.
    """
    tree = ast.parse(source)
    preserve_documentation = any(
        (isinstance(item, ast.Attribute) and item.attr == "__doc__")
        or (isinstance(item, ast.Name) and item.id == "__doc__")
        for item in ast.walk(tree)
    )
    values = dict(bindings or {})
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Mode":
            values.update(mode_literal_bindings(node))
    definitions: dict[str, ast.stmt] = {}
    imports: dict[str, tuple[str, str]] = {}
    initialization = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            definitions[node.name] = node
            if isinstance(node, ast.ClassDef):
                declaration = copy.deepcopy(node)
                declaration.body = [
                    item for item in declaration.body if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]
                definitions[node.name + ".__declaration__"] = declaration
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        definitions[node.name + "." + item.name] = item
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    definitions[target.id] = node
                    try:
                        values.setdefault(target.id, ast.literal_eval(node.value))
                    except (ValueError, TypeError):
                        pass  # Nonliteral configuration stays in the dependency closure.
            calls = (
                [item for item in ast.walk(node.value) if isinstance(item, ast.Call)] if node.value is not None else []
            )
            effects = []
            for call in calls:
                function = call.func
                dictionary = (
                    definitions.get(function.value.id)
                    if isinstance(function, ast.Attribute) and isinstance(function.value, ast.Name)
                    else None
                )
                value = dictionary.value if isinstance(dictionary, (ast.Assign, ast.AnnAssign)) else None
                if (
                    isinstance(function, ast.Attribute)
                    and function.attr in {"items", "keys", "values"}
                    and isinstance(value, ast.Dict)
                ):
                    continue
                effects.append(call)
            if effects:
                initialization.append(node)
        elif isinstance(node, ast.ImportFrom):
            module = "." * node.level + (node.module or "")
            for alias in node.names:
                imports[alias.asname or alias.name] = (module, alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports[alias.asname or alias.name.split(".")[0]] = (alias.name, "*")
        elif not (
            isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
        ):
            initialization.append(node)
    if initialization:
        definitions["__module_initialization__"] = ast.If(test=ast.Constant(value=True), body=initialization, orelse=[])
    selected: dict[str, ast.AST] = {}
    dependencies: set[tuple[str, str]] = set()
    imported_members: dict[tuple[str, str], set[str]] = {}
    imported_bases: set[tuple[str, str]] = set()
    selected_roots = list(definitions) if "__module_source__" in roots else list(roots)
    pending = [*selected_roots, *(["__module_initialization__"] if initialization else [])]
    while pending:
        name = pending.pop()
        if name in selected:
            continue
        if name not in definitions:
            if name not in imports:
                raise ValueError(f"Grading root {name!r} is absent")
            dependencies.add(imports[name])
            continue
        key = values.get("__name__", "") + ":" + name
        definition_values = {**values, **values.get(DEFINITION_BINDINGS, {}).get(key, {})}
        node = GradingBranches(definition_values, preserve_documentation).visit(copy.deepcopy(definitions[name]))
        normalized = ast.Module(body=node, type_ignores=[]) if isinstance(node, list) else node
        selected[name] = normalized
        node_imports = dict(imports)
        if name.endswith(".__declaration__") and isinstance(normalized, ast.ClassDef):
            for base in normalized.bases:
                parent = base.value if isinstance(base, ast.Subscript) else base
                if isinstance(parent, ast.Name) and parent.id in imports:
                    imported_bases.add(imports[parent.id])
        for item in ast.walk(normalized):
            if isinstance(item, ast.ImportFrom):
                node_imports.update(
                    (alias.asname or alias.name, ("." * item.level + (item.module or ""), alias.name))
                    for alias in item.names
                )
            elif isinstance(item, ast.Import):
                node_imports.update(
                    (alias.asname or alias.name.split(".")[0], (alias.name, "*")) for alias in item.names
                )
        referenced = {
            item.id for item in ast.walk(normalized) if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)
        }
        pending.extend(sorted(referenced & definitions.keys() - selected.keys()))
        if "." in name:
            owner = name.split(".")[0]
            declaration = owner + ".__declaration__"
            if declaration not in selected:
                pending.append(declaration)
            pending.extend(
                owner + "." + item.attr
                for item in ast.walk(normalized)
                if isinstance(item, ast.Attribute)
                and isinstance(item.value, ast.Name)
                and item.value.id in {"self", "cls"}
                and owner + "." + item.attr in definitions
            )
        dependencies.update(imports[item] for item in referenced & imports.keys())
        for item in ast.walk(normalized):
            if isinstance(item, ast.Attribute) and isinstance(item.value, ast.Name) and item.value.id in node_imports:
                imported_members.setdefault(node_imports[item.value.id], set()).add(item.attr)
        for item in ast.walk(normalized):
            if isinstance(item, ast.ImportFrom):
                dependencies.update(("." * item.level + (item.module or ""), alias.name) for alias in item.names)
            elif isinstance(item, ast.Import):
                dependencies.update((alias.name, "*") for alias in item.names)
    reads = {
        item.attr
        for node in selected.values()
        for item in ast.walk(node)
        if isinstance(item, ast.Attribute)
        and isinstance(item.value, ast.Name)
        and item.value.id == "self"
        and isinstance(item.ctx, ast.Load)
    }
    if values.get("__retain_instance_state__"):
        reads.add("__dict__")
    for node in selected.values():
        parents = {child: parent for parent in ast.walk(node) for child in ast.iter_child_nodes(parent)}
        if any(
            isinstance(item, ast.Name)
            and item.id == "self"
            and isinstance(item.ctx, ast.Load)
            and not (isinstance(parents.get(item), ast.Attribute) and parents[item].value is item)
            for item in ast.walk(node)
        ):
            reads.add("__dict__")
    definitions_payload = {
        name: ast.dump(UnusedLiteralState(reads).visit(node), include_attributes=False)
        for name, node in selected.items()
    }
    payload = {"definitions": definitions_payload, "imports": sorted(dependencies)}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    members = tuple(
        sorted((module, symbol, tuple(sorted(names))) for (module, symbol), names in imported_members.items())
    )
    return PythonGradingCode(digest, tuple(sorted(dependencies)), members, tuple(sorted(imported_bases)))


def python_grading_program(
    source: PythonModuleSource,
    roots: Mapping[str, Sequence[str]],
    internal_packages: tuple[str, ...],
    bindings: Mapping[str, Any] | None = None,
) -> PythonGradingProgram:
    """Follow selected internal imports; preserve external requirements for auditing.

    The reader must return code at immutable captured revisions and raise when a
    required module is unavailable. Missing dependencies cannot certify an old
    review as applicable. Dynamic module dispatch and data files require explicit
    roots and resources from the selected grading route.
    """
    selected = {module: set(symbols) for module, symbols in roots.items()}
    route_bindings = dict(bindings or {})
    if route_bindings.get(VERIFYIT_MODES_BINDING):
        spec = ast.parse(source.read("verifyit.spec"))
        mode = next(node for node in spec.body if isinstance(node, ast.ClassDef) and node.name == "Mode")
        route_bindings.update(mode_literal_bindings(mode))
    pending = list(selected)
    resolved: dict[str, PythonGradingCode] = {}
    retained_state: set[str] = set()
    base_requests: set[tuple[str, str]] = set()
    external: set[str] = set()
    while pending:
        module = pending.pop()
        module_bindings = {**route_bindings, "__name__": module}
        if module in retained_state:
            module_bindings["__retain_instance_state__"] = True
        code = python_grading_code(source.read(module), sorted(selected[module]), module_bindings)
        if module in resolved and resolved[module] == code:
            continue
        resolved[module] = code
        members = {(imported, symbol): names for imported, symbol, names in code.imported_members}
        for imported, symbol in code.imports:
            is_base = (imported, symbol) in code.imported_bases or (imported, symbol) in base_requests
            attributes = members.get((imported, symbol), ())
            if imported.startswith("."):
                level = len(imported) - len(imported.lstrip("."))
                package = module.split(".")[:-level]
                imported = ".".join([*package, imported.lstrip(".")]).rstrip(".")
            if not any(imported == package or imported.startswith(package + ".") for package in internal_packages):
                external.add(imported.split(".")[0])
                continue
            if attributes and symbol != "*":
                package_source = None
                try:
                    package_source = source.read(imported)
                    python_grading_code(package_source, [symbol], {**route_bindings, "__name__": imported})
                except (ModuleNotFoundError, ValueError):
                    child = imported + "." + symbol
                    source.read(child)
                    if package_source is not None and imported not in selected:
                        selected[imported] = set()
                        pending.append(imported)
                    imported = child
                    for attribute in attributes:
                        if attribute == "__file__":
                            attribute = "__module_source__"
                        symbols = selected.setdefault(imported, set())
                        if attribute not in symbols:
                            symbols.add(attribute)
                            pending.append(imported)
                    continue
            if symbol == "*":
                raise ValueError(f"Grading module {module!r} uses an unscoped import of {imported!r}")
            symbols = selected.setdefault(imported, set())
            if is_base:
                retained_state.add(imported)
                tree = ast.parse(source.read(imported))
                base = next((node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == symbol), None)
                if base is None:
                    exported = python_grading_code(source.read(imported), [symbol], module_bindings)
                    if not exported.imports:
                        raise ValueError(f"Grading base {imported}.{symbol} has no statically resolved class")
                    base_requests.update(exported.imports)
                    if symbol not in symbols:
                        symbols.add(symbol)
                        pending.append(imported)
                    continue
                names = [symbol + ".__declaration__"]
                names.extend(
                    symbol + ".__init__"
                    for node in base.body
                    if isinstance(node, ast.FunctionDef) and node.name == "__init__"
                )
                for name in names:
                    if name not in symbols:
                        symbols.add(name)
                        pending.append(imported)
                continue
            if symbol not in symbols:
                symbols.add(symbol)
                pending.append(imported)
    modules = tuple(sorted((module, code.digest) for module, code in resolved.items()))
    payload = {"modules": modules, "external_imports": sorted(external)}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return PythonGradingProgram(digest, modules, tuple(sorted(external)))

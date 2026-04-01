# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tree-sitter parsing primitives for Python and Rust source files.

Extracts function/class definitions with signatures, source text, and
file locations. Used by packages.py for package discovery.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

import tree_sitter
import tree_sitter_python
import tree_sitter_rust

logger = logging.getLogger(__name__)

PY_LANGUAGE = tree_sitter.Language(tree_sitter_python.language())
RS_LANGUAGE = tree_sitter.Language(tree_sitter_rust.language())

# Workspace libs to scan — maps lib name to its src directory relative to repo root.
PYTHON_LIBS: dict[str, str] = {
    "marin": "lib/marin/src/marin",
    "levanter": "lib/levanter/src/levanter",
    "haliax": "lib/haliax/src/haliax",
    "fray": "lib/fray/src/fray",
    "rigging": "lib/rigging/src/rigging",
    "iris": "lib/iris/src/iris",
    "zephyr": "lib/zephyr/src/zephyr",
}

RUST_CRATES: dict[str, str] = {
    "dupekit": "rust/dupekit/src",
}

KNOWN_LIBS = set(PYTHON_LIBS.keys()) | set(RUST_CRATES.keys()) | {"draccus"}

# Path components that indicate test/cache directories to skip.
SKIP_COMPONENTS = {"__pycache__", "tests", "conftest.py"}
# File name patterns to skip (matched against the file name only).
SKIP_FILE_PREFIXES = ("test_",)
SKIP_FILE_SUFFIXES = ("_test.py",)


@dataclass
class FunctionInfo:
    name: str
    qualified_name: str
    signature: str
    source: str
    source_hash: str
    file_path: str
    line_number: int
    language: str
    calls: list[str] = field(default_factory=list)
    is_public: bool = True


@dataclass
class ClassInfo:
    name: str
    qualified_name: str
    source: str
    source_hash: str
    file_path: str
    line_number: int
    language: str
    methods: list[FunctionInfo] = field(default_factory=list)
    is_public: bool = True


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def should_skip(path: Path) -> bool:
    """Check whether a source file should be excluded from parsing."""
    for part in path.parts:
        if part in SKIP_COMPONENTS:
            return True
    name = path.name
    if any(name.startswith(p) for p in SKIP_FILE_PREFIXES):
        return True
    return any(name.endswith(s) for s in SKIP_FILE_SUFFIXES)


def node_text(node: tree_sitter.Node) -> str:
    return node.text.decode("utf-8")


def find_children(node: tree_sitter.Node, type_name: str) -> list[tree_sitter.Node]:
    return [c for c in node.children if c.type == type_name]


def find_child(node: tree_sitter.Node, type_name: str) -> tree_sitter.Node | None:
    for c in node.children:
        if c.type == type_name:
            return c
    return None


# ---------------------------------------------------------------------------
# Python parsing
# ---------------------------------------------------------------------------


def extract_python_all(root: tree_sitter.Node) -> set[str] | None:
    """Extract __all__ list if defined at module level."""
    for child in root.children:
        if child.type != "expression_statement":
            continue
        expr = child.children[0] if child.children else None
        if expr is None or expr.type != "assignment":
            continue
        left = find_child(expr, "identifier")
        if left is not None and node_text(left) == "__all__":
            right = find_child(expr, "list")
            if right is None:
                continue
            names: set[str] = set()
            for item in right.children:
                if item.type == "string":
                    text = node_text(item)
                    names.add(text.strip("\"'"))
            return names
    return None


def extract_python_imports(root: tree_sitter.Node) -> set[str]:
    """Extract top-level imported module names."""
    modules: set[str] = set()
    for child in root.children:
        if child.type == "import_statement":
            for name_node in find_children(child, "dotted_name"):
                modules.add(node_text(name_node).split(".")[0])
        elif child.type == "import_from_statement":
            dotted = find_child(child, "dotted_name")
            if dotted is not None:
                modules.add(node_text(dotted).split(".")[0])
    return modules


def extract_python_signature(func_node: tree_sitter.Node) -> str:
    """Extract function signature as a string."""
    params = find_child(func_node, "parameters")
    ret_type = find_child(func_node, "type")
    sig = node_text(params) if params else "()"
    if ret_type is not None:
        sig += f" -> {node_text(ret_type)}"
    return sig


def extract_python_calls(body_node: tree_sitter.Node) -> list[str]:
    """Best-effort extraction of called function names from a function body."""
    calls: list[str] = []

    def walk(node: tree_sitter.Node) -> None:
        if node.type == "call":
            fn = node.children[0] if node.children else None
            if fn is not None:
                calls.append(node_text(fn))
        for child in node.children:
            walk(child)

    walk(body_node)
    return calls


def is_python_public(name: str, all_names: set[str] | None) -> bool:
    if all_names is not None:
        return name in all_names
    return not name.startswith("_")


def decorated_source(decorator_parent: tree_sitter.Node, inner: tree_sitter.Node) -> str:
    """Return source text including decorators when the node is a decorated_definition."""
    if decorator_parent.type == "decorated_definition":
        return node_text(decorator_parent)
    return node_text(inner)


# ---------------------------------------------------------------------------
# Rust parsing
# ---------------------------------------------------------------------------


def extract_rust_signature(func_node: tree_sitter.Node) -> str:
    """Extract Rust function signature."""
    params = find_child(func_node, "parameters")
    sig_parts: list[str] = []
    if params:
        sig_parts.append(node_text(params))
    saw_arrow = False
    for c in func_node.children:
        if node_text(c) == "->":
            saw_arrow = True
        elif saw_arrow and c.type != "block":
            sig_parts.append(f"-> {node_text(c)}")
            break
    return " ".join(sig_parts) if sig_parts else "()"


def has_rust_attribute(node: tree_sitter.Node, attr_name: str) -> bool:
    """Check if a Rust node has a specific attribute macro (e.g. #[pyfunction])."""
    parent = node.parent
    if parent is None:
        return False
    found_attr = False
    for child in parent.children:
        if child.type == "attribute_item":
            text = node_text(child)
            if attr_name in text:
                found_attr = True
        if child.id == node.id and found_attr:
            return True
        if child.type != "attribute_item":
            found_attr = False
    return False


def is_rust_pub(node: tree_sitter.Node) -> bool:
    """Check if a Rust item has pub visibility."""
    for child in node.children:
        if child.type == "visibility_modifier":
            return True
    return False


def parse_rust_file(
    file_path: Path,
    parser: tree_sitter.Parser,
    crate_name: str,
) -> tuple[list[FunctionInfo], list[ClassInfo]]:
    """Parse a single Rust file."""
    source_bytes = file_path.read_bytes()
    tree = parser.parse(source_bytes)
    root = tree.root_node
    rel_path = str(file_path)

    functions: list[FunctionInfo] = []
    classes: list[ClassInfo] = []

    for child in root.children:
        if child.type == "function_item" and is_rust_pub(child):
            name_node = find_child(child, "identifier")
            if name_node is None:
                continue
            name = node_text(name_node)
            source = node_text(child)
            is_pyexport = has_rust_attribute(child, "pyfunction")
            functions.append(
                FunctionInfo(
                    name=name,
                    qualified_name=f"{crate_name}.py.{name}" if is_pyexport else f"{crate_name}.{name}",
                    signature=extract_rust_signature(child),
                    source=source,
                    source_hash=sha256(source),
                    file_path=rel_path,
                    line_number=child.start_point.row + 1,
                    language="rust",
                    is_public=True,
                )
            )

        elif child.type == "struct_item" and is_rust_pub(child):
            name_node = find_child(child, "type_identifier")
            if name_node is None:
                continue
            name = node_text(name_node)
            source = node_text(child)
            is_pyexport = has_rust_attribute(child, "pyclass")
            classes.append(
                ClassInfo(
                    name=name,
                    qualified_name=f"{crate_name}.py.{name}" if is_pyexport else f"{crate_name}.{name}",
                    source=source,
                    source_hash=sha256(source),
                    file_path=rel_path,
                    line_number=child.start_point.row + 1,
                    language="rust",
                    is_public=True,
                )
            )

        elif child.type == "impl_item":
            type_node = find_child(child, "type_identifier")
            if type_node is None:
                continue
            type_name = node_text(type_node)
            body = find_child(child, "declaration_list")
            if body is None:
                continue
            for item in body.children:
                if item.type == "function_item" and is_rust_pub(item):
                    mname_node = find_child(item, "identifier")
                    if mname_node is None:
                        continue
                    mname = node_text(mname_node)
                    msource = node_text(item)
                    for cls in classes:
                        if cls.name == type_name:
                            cls.methods.append(
                                FunctionInfo(
                                    name=mname,
                                    qualified_name=f"{cls.qualified_name}.{mname}",
                                    signature=extract_rust_signature(item),
                                    source=msource,
                                    source_hash=sha256(msource),
                                    file_path=rel_path,
                                    line_number=item.start_point.row + 1,
                                    language="rust",
                                    is_public=True,
                                )
                            )
                            break

    return functions, classes

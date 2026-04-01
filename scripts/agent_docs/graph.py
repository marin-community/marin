# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tree-sitter based dependency graph builder for Python and Rust source files.

Parses all library source under lib/ and rust/ to extract module structure,
function/class definitions, import edges, and a best-effort call graph.
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
# Iris has Python under lib/iris/src/iris/ like the others.
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

# Files/directories to skip
SKIP_PATTERNS = {"__pycache__", ".pyc", "test_", "_test.py", "tests/", "conftest.py"}


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


@dataclass
class ModuleInfo:
    name: str
    functions: list[FunctionInfo] = field(default_factory=list)
    classes: list[ClassInfo] = field(default_factory=list)
    imports_from: set[str] = field(default_factory=set)
    file_paths: list[str] = field(default_factory=list)
    language: str = "python"


@dataclass
class RepoGraph:
    modules: dict[str, ModuleInfo] = field(default_factory=dict)
    libs: list[str] = field(default_factory=list)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _should_skip(path: Path) -> bool:
    parts = str(path)
    return any(pat in parts for pat in SKIP_PATTERNS)


def _node_text(node: tree_sitter.Node) -> str:
    return node.text.decode("utf-8")


def _find_children(node: tree_sitter.Node, type_name: str) -> list[tree_sitter.Node]:
    return [c for c in node.children if c.type == type_name]


def _find_child(node: tree_sitter.Node, type_name: str) -> tree_sitter.Node | None:
    for c in node.children:
        if c.type == type_name:
            return c
    return None


def _module_name_for_file(file_path: Path, lib_name: str, src_dir: Path) -> str:
    """Convert a file path to a dotted module name, grouped at depth 2.

    e.g. lib/marin/src/marin/execution/executor.py -> marin.execution
    """
    rel = file_path.relative_to(src_dir)
    parts = list(rel.with_suffix("").parts)

    # Drop __init__ from the path
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]

    if not parts:
        return lib_name

    # Group at depth 2: marin.execution (not marin.execution.executor)
    # For shallow modules (e.g. rigging.filesystem), keep the full name
    if len(parts) >= 2:
        return f"{parts[0]}.{parts[1]}"
    return parts[0]


# ---------------------------------------------------------------------------
# Python parsing
# ---------------------------------------------------------------------------


def _extract_python_all(root: tree_sitter.Node) -> set[str] | None:
    """Extract __all__ list if defined at module level."""
    for child in root.children:
        if child.type != "expression_statement":
            continue
        expr = child.children[0] if child.children else None
        if expr is None or expr.type != "assignment":
            continue
        left = _find_child(expr, "identifier")
        if left is not None and _node_text(left) == "__all__":
            right = _find_child(expr, "list")
            if right is None:
                continue
            names: set[str] = set()
            for item in right.children:
                if item.type == "string":
                    # Strip quotes
                    text = _node_text(item)
                    names.add(text.strip("\"'"))
            return names
    return None


def _extract_python_imports(root: tree_sitter.Node) -> set[str]:
    """Extract top-level imported module names."""
    modules: set[str] = set()
    for child in root.children:
        if child.type == "import_statement":
            for name_node in _find_children(child, "dotted_name"):
                modules.add(_node_text(name_node).split(".")[0])
        elif child.type == "import_from_statement":
            dotted = _find_child(child, "dotted_name")
            if dotted is not None:
                modules.add(_node_text(dotted).split(".")[0])
    return modules


def _extract_python_signature(func_node: tree_sitter.Node) -> str:
    """Extract function signature as a string."""
    params = _find_child(func_node, "parameters")
    ret_type = _find_child(func_node, "type")
    sig = _node_text(params) if params else "()"
    if ret_type is not None:
        sig += f" -> {_node_text(ret_type)}"
    return sig


def _extract_python_calls(body_node: tree_sitter.Node) -> list[str]:
    """Best-effort extraction of called function names from a function body."""
    calls: list[str] = []

    def _walk(node: tree_sitter.Node) -> None:
        if node.type == "call":
            fn = node.children[0] if node.children else None
            if fn is not None:
                calls.append(_node_text(fn))
        for child in node.children:
            _walk(child)

    _walk(body_node)
    return calls


def _parse_python_file(
    file_path: Path,
    parser: tree_sitter.Parser,
    lib_name: str,
    src_dir: Path,
) -> tuple[str, list[FunctionInfo], list[ClassInfo], set[str]]:
    """Parse a single Python file and extract its contents."""
    source_bytes = file_path.read_bytes()
    tree = parser.parse(source_bytes)
    root = tree.root_node
    rel_path = str(file_path)

    module_name = _module_name_for_file(file_path, lib_name, src_dir)
    all_names = _extract_python_all(root)
    imports = _extract_python_imports(root)

    functions: list[FunctionInfo] = []
    classes: list[ClassInfo] = []

    for child in root.children:
        if child.type == "decorated_definition":
            actual = _find_child(child, "function_definition") or _find_child(child, "class_definition")
            if actual is None:
                continue
            child = actual

        if child.type == "function_definition":
            name_node = _find_child(child, "identifier")
            if name_node is None:
                continue
            name = _node_text(name_node)
            is_public = _is_python_public(name, all_names)
            source = _node_text(child)
            body = _find_child(child, "block")
            functions.append(
                FunctionInfo(
                    name=name,
                    qualified_name=f"{module_name}.{name}",
                    signature=_extract_python_signature(child),
                    source=source,
                    source_hash=_sha256(source),
                    file_path=rel_path,
                    line_number=child.start_point.row + 1,
                    language="python",
                    calls=_extract_python_calls(body) if body else [],
                    is_public=is_public,
                )
            )

        elif child.type == "class_definition":
            name_node = _find_child(child, "identifier")
            if name_node is None:
                continue
            name = _node_text(name_node)
            is_public = _is_python_public(name, all_names)
            source = _node_text(child)
            methods: list[FunctionInfo] = []
            body = _find_child(child, "block")
            if body:
                for item in body.children:
                    if item.type == "decorated_definition":
                        item = _find_child(item, "function_definition") or item
                    if item.type == "function_definition":
                        mname_node = _find_child(item, "identifier")
                        if mname_node is None:
                            continue
                        mname = _node_text(mname_node)
                        msource = _node_text(item)
                        mbody = _find_child(item, "block")
                        methods.append(
                            FunctionInfo(
                                name=mname,
                                qualified_name=f"{module_name}.{name}.{mname}",
                                signature=_extract_python_signature(item),
                                source=msource,
                                source_hash=_sha256(msource),
                                file_path=rel_path,
                                line_number=item.start_point.row + 1,
                                language="python",
                                calls=_extract_python_calls(mbody) if mbody else [],
                                is_public=not mname.startswith("_"),
                            )
                        )
            classes.append(
                ClassInfo(
                    name=name,
                    qualified_name=f"{module_name}.{name}",
                    source=source,
                    source_hash=_sha256(source),
                    file_path=rel_path,
                    line_number=child.start_point.row + 1,
                    language="python",
                    methods=methods,
                    is_public=is_public,
                )
            )

    return module_name, functions, classes, imports


def _is_python_public(name: str, all_names: set[str] | None) -> bool:
    if all_names is not None:
        return name in all_names
    return not name.startswith("_")


# ---------------------------------------------------------------------------
# Rust parsing
# ---------------------------------------------------------------------------


def _extract_rust_signature(func_node: tree_sitter.Node) -> str:
    """Extract Rust function signature."""
    params = _find_child(func_node, "parameters")
    # Try to find return type after ->
    sig_parts: list[str] = []
    if params:
        sig_parts.append(_node_text(params))
    # Walk children for return type
    saw_arrow = False
    for c in func_node.children:
        if _node_text(c) == "->":
            saw_arrow = True
        elif saw_arrow and c.type != "block":
            sig_parts.append(f"-> {_node_text(c)}")
            break
    return " ".join(sig_parts) if sig_parts else "()"


def _has_attribute(node: tree_sitter.Node, attr_name: str) -> bool:
    """Check if a node or its previous sibling has a specific Rust attribute."""
    # Attributes come as preceding siblings in tree-sitter Rust grammar
    parent = node.parent
    if parent is None:
        return False
    found_attr = False
    for child in parent.children:
        if child.type == "attribute_item":
            text = _node_text(child)
            if attr_name in text:
                found_attr = True
        if child.id == node.id and found_attr:
            return True
        if child.type != "attribute_item":
            found_attr = False
    return False


def _is_rust_pub(node: tree_sitter.Node) -> bool:
    """Check if a Rust item has pub visibility."""
    for child in node.children:
        if child.type == "visibility_modifier":
            return True
    return False


def _parse_rust_file(
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
        if child.type == "function_item" and _is_rust_pub(child):
            name_node = _find_child(child, "identifier")
            if name_node is None:
                continue
            name = _node_text(name_node)
            source = _node_text(child)
            is_pyexport = _has_attribute(child, "pyfunction")
            functions.append(
                FunctionInfo(
                    name=name,
                    qualified_name=f"{crate_name}.{name}",
                    signature=_extract_rust_signature(child),
                    source=source,
                    source_hash=_sha256(source),
                    file_path=rel_path,
                    line_number=child.start_point.row + 1,
                    language="rust",
                    is_public=True,
                )
            )
            if is_pyexport:
                # Tag for cross-language visibility
                functions[-1].qualified_name = f"{crate_name}.py.{name}"

        elif child.type == "struct_item" and _is_rust_pub(child):
            name_node = _find_child(child, "type_identifier")
            if name_node is None:
                continue
            name = _node_text(name_node)
            source = _node_text(child)
            is_pyexport = _has_attribute(child, "pyclass")
            classes.append(
                ClassInfo(
                    name=name,
                    qualified_name=f"{crate_name}.{name}" if not is_pyexport else f"{crate_name}.py.{name}",
                    source=source,
                    source_hash=_sha256(source),
                    file_path=rel_path,
                    line_number=child.start_point.row + 1,
                    language="rust",
                    is_public=True,
                )
            )

        elif child.type == "impl_item":
            # Extract public methods from impl blocks
            type_node = _find_child(child, "type_identifier")
            if type_node is None:
                continue
            type_name = _node_text(type_node)
            body = _find_child(child, "declaration_list")
            if body is None:
                continue
            for item in body.children:
                if item.type == "function_item" and _is_rust_pub(item):
                    mname_node = _find_child(item, "identifier")
                    if mname_node is None:
                        continue
                    mname = _node_text(mname_node)
                    msource = _node_text(item)
                    # Find the matching class and add method
                    for cls in classes:
                        if cls.name == type_name:
                            cls.methods.append(
                                FunctionInfo(
                                    name=mname,
                                    qualified_name=f"{cls.qualified_name}.{mname}",
                                    signature=_extract_rust_signature(item),
                                    source=msource,
                                    source_hash=_sha256(msource),
                                    file_path=rel_path,
                                    line_number=item.start_point.row + 1,
                                    language="rust",
                                    is_public=True,
                                )
                            )
                            break

    return functions, classes


# ---------------------------------------------------------------------------
# Graph building
# ---------------------------------------------------------------------------

# Known lib names for resolving import edges
_KNOWN_LIBS = set(PYTHON_LIBS.keys()) | set(RUST_CRATES.keys()) | {"draccus"}


def build_repo_graph(repo_root: Path) -> RepoGraph:
    """Build a RepoGraph from all Python and Rust source files in the repo."""
    graph = RepoGraph()
    graph.libs = sorted(set(PYTHON_LIBS.keys()) | set(RUST_CRATES.keys()))

    py_parser = tree_sitter.Parser(PY_LANGUAGE)
    rs_parser = tree_sitter.Parser(RS_LANGUAGE)

    # Parse Python libraries
    for lib_name, src_rel in PYTHON_LIBS.items():
        src_dir = repo_root / src_rel
        if not src_dir.exists():
            logger.warning("Skipping missing lib: %s (%s)", lib_name, src_dir)
            continue

        py_files = sorted(src_dir.rglob("*.py"))
        for py_file in py_files:
            if _should_skip(py_file):
                continue
            try:
                module_name, functions, classes, imports = _parse_python_file(
                    py_file, py_parser, lib_name, src_dir.parent
                )
            except Exception:
                logger.warning("Failed to parse %s", py_file, exc_info=True)
                continue

            if module_name not in graph.modules:
                graph.modules[module_name] = ModuleInfo(name=module_name, language="python")

            mod = graph.modules[module_name]
            mod.functions.extend(functions)
            mod.classes.extend(classes)
            mod.file_paths.append(str(py_file.relative_to(repo_root)))
            # Only track imports of known libraries
            mod.imports_from.update(imports & _KNOWN_LIBS)

    # Parse Rust crates
    for crate_name, src_rel in RUST_CRATES.items():
        src_dir = repo_root / src_rel
        if not src_dir.exists():
            logger.warning("Skipping missing crate: %s (%s)", crate_name, src_dir)
            continue

        rs_files = sorted(src_dir.rglob("*.rs"))
        for rs_file in rs_files:
            try:
                functions, classes = _parse_rust_file(rs_file, rs_parser, crate_name)
            except Exception:
                logger.warning("Failed to parse %s", rs_file, exc_info=True)
                continue

            if crate_name not in graph.modules:
                graph.modules[crate_name] = ModuleInfo(name=crate_name, language="rust")

            mod = graph.modules[crate_name]
            mod.functions.extend(functions)
            mod.classes.extend(classes)
            mod.file_paths.append(str(rs_file.relative_to(repo_root)))

    return graph


def print_graph_stats(graph: RepoGraph) -> None:
    """Print summary statistics for a RepoGraph."""
    total_funcs = 0
    total_classes = 0
    total_files = 0
    for mod in graph.modules.values():
        pub_funcs = sum(1 for f in mod.functions if f.is_public)
        pub_classes = sum(1 for c in mod.classes if c.is_public)
        total_funcs += pub_funcs
        total_classes += pub_classes
        total_files += len(mod.file_paths)
        imports = ", ".join(sorted(mod.imports_from)) if mod.imports_from else "(none)"
        print(
            f"  {mod.name} [{mod.language}]: {pub_funcs} funcs, "
            f"{pub_classes} classes, {len(mod.file_paths)} files | imports: {imports}"
        )

    print(
        f"\nTotal: {len(graph.modules)} modules, {total_funcs} public functions, "
        f"{total_classes} public classes, {total_files} files"
    )

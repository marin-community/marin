# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Directory-level package discovery for agent documentation.

Groups source files by filesystem directory: each directory with Python files
becomes a package, giving natural sub-package boundaries like
marin.processing.classification.deduplication.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import tree_sitter

from agent_docs.parsing import (
    KNOWN_LIBS,
    PYTHON_LIBS,
    PY_LANGUAGE,
    RS_LANGUAGE,
    RUST_CRATES,
    ClassInfo,
    FunctionInfo,
    decorated_source,
    extract_python_all,
    extract_python_calls,
    extract_python_signature,
    find_child,
    is_python_public,
    node_text,
    parse_rust_file,
    sha256,
    should_skip,
)

logger = logging.getLogger(__name__)


@dataclass
class PackageInfo:
    name: str
    description: str
    functions: list[FunctionInfo]
    classes: list[ClassInfo]
    imports_from: set[str]
    file_paths: list[str]
    language: str


def _package_name_for_dir(dir_path: Path, src_dir: Path) -> str:
    """Convert a directory path to a dotted package name.

    src_dir is the parent of the lib root — e.g. lib/marin/src for marin.
    So lib/marin/src/marin/processing/classification -> marin.processing.classification
    """
    rel = dir_path.relative_to(src_dir)
    return ".".join(rel.parts)


def _extract_full_imports(root: tree_sitter.Node) -> list[str]:
    """Extract full dotted import paths from top-level import statements."""
    modules: list[str] = []
    for child in root.children:
        if child.type == "import_statement":
            for name_node in [c for c in child.children if c.type == "dotted_name"]:
                modules.append(node_text(name_node))
        elif child.type == "import_from_statement":
            dotted = find_child(child, "dotted_name")
            if dotted is not None:
                modules.append(node_text(dotted))
    return modules


def _parse_python_file_for_package(
    file_path: Path,
    parser: tree_sitter.Parser,
    package_name: str,
) -> tuple[list[FunctionInfo], list[ClassInfo], list[str]]:
    """Parse a Python file, returning functions, classes, and full import paths."""
    source_bytes = file_path.read_bytes()
    tree = parser.parse(source_bytes)
    root = tree.root_node
    rel_path = str(file_path)

    all_names = extract_python_all(root)
    imports = _extract_full_imports(root)

    functions: list[FunctionInfo] = []
    classes: list[ClassInfo] = []

    for child in root.children:
        # Track the outer node for decorator source
        outer = child
        if child.type == "decorated_definition":
            actual = find_child(child, "function_definition") or find_child(child, "class_definition")
            if actual is None:
                continue
            child = actual

        if child.type == "function_definition":
            name_node = find_child(child, "identifier")
            if name_node is None:
                continue
            name = node_text(name_node)
            source = decorated_source(outer, child)
            body = find_child(child, "block")
            functions.append(
                FunctionInfo(
                    name=name,
                    qualified_name=f"{package_name}.{name}",
                    signature=extract_python_signature(child),
                    source=source,
                    source_hash=sha256(source),
                    file_path=rel_path,
                    line_number=outer.start_point.row + 1,
                    language="python",
                    calls=extract_python_calls(body) if body else [],
                    is_public=is_python_public(name, all_names),
                )
            )

        elif child.type == "class_definition":
            name_node = find_child(child, "identifier")
            if name_node is None:
                continue
            name = node_text(name_node)
            source = decorated_source(outer, child)
            methods: list[FunctionInfo] = []
            body = find_child(child, "block")
            if body:
                for item in body.children:
                    method_outer = item
                    if item.type == "decorated_definition":
                        item = find_child(item, "function_definition") or item
                    if item.type == "function_definition":
                        mname_node = find_child(item, "identifier")
                        if mname_node is None:
                            continue
                        mname = node_text(mname_node)
                        msource = decorated_source(method_outer, item)
                        mbody = find_child(item, "block")
                        methods.append(
                            FunctionInfo(
                                name=mname,
                                qualified_name=f"{package_name}.{name}.{mname}",
                                signature=extract_python_signature(item),
                                source=msource,
                                source_hash=sha256(msource),
                                file_path=rel_path,
                                line_number=method_outer.start_point.row + 1,
                                language="python",
                                calls=extract_python_calls(mbody) if mbody else [],
                                is_public=not mname.startswith("_"),
                            )
                        )
            classes.append(
                ClassInfo(
                    name=name,
                    qualified_name=f"{package_name}.{name}",
                    source=source,
                    source_hash=sha256(source),
                    file_path=rel_path,
                    line_number=outer.start_point.row + 1,
                    language="python",
                    methods=methods,
                    is_public=is_python_public(name, all_names),
                )
            )

    return functions, classes, imports


def _collect_python_directories(src_dir: Path) -> dict[Path, list[Path]]:
    """Map each directory containing .py files to its list of Python files."""
    dir_files: dict[Path, list[Path]] = defaultdict(list)
    for py_file in sorted(src_dir.rglob("*.py")):
        if should_skip(py_file):
            continue
        dir_files[py_file.parent].append(py_file)
    return dict(dir_files)


def _should_merge_into_parent(dir_path: Path, py_files: list[Path]) -> bool:
    """A leaf directory with only __init__.py should merge into its parent."""
    if len(py_files) != 1:
        return False
    return py_files[0].name == "__init__.py"


def _resolve_imports_to_packages(
    raw_imports: list[str],
    known_packages: set[str],
) -> set[str]:
    """Resolve full dotted import paths to known package names.

    For an import like "marin.processing.classification.classifier", we find
    the longest matching package prefix.
    """
    resolved: set[str] = set()
    for imp in raw_imports:
        top_level = imp.split(".")[0]
        if top_level not in KNOWN_LIBS:
            continue
        best = ""
        for pkg_name in known_packages:
            if imp == pkg_name or imp.startswith(pkg_name + "."):
                if len(pkg_name) > len(best):
                    best = pkg_name
        if best:
            resolved.add(best)
        elif top_level in known_packages:
            resolved.add(top_level)
    return resolved


def discover_packages(repo_root: Path) -> dict[str, PackageInfo]:
    """Walk all library source trees and group by filesystem directory."""
    packages: dict[str, PackageInfo] = {}
    py_parser = tree_sitter.Parser(PY_LANGUAGE)
    rs_parser = tree_sitter.Parser(RS_LANGUAGE)

    # --- Python libraries ---
    for lib_name, src_rel in PYTHON_LIBS.items():
        src_dir = repo_root / src_rel
        if not src_dir.exists():
            logger.warning("Skipping missing lib: %s (%s)", lib_name, src_dir)
            continue

        parent_dir = src_dir.parent
        dir_files = _collect_python_directories(src_dir)

        # Identify leaf directories with only __init__.py that should merge into parent
        merge_targets: dict[Path, Path] = {}
        for dir_path, py_files in dir_files.items():
            if dir_path == src_dir:
                continue
            if _should_merge_into_parent(dir_path, py_files):
                has_children = any(
                    other_dir != dir_path and str(other_dir).startswith(str(dir_path) + "/") for other_dir in dir_files
                )
                if not has_children:
                    merge_targets[dir_path] = dir_path.parent

        effective_files: dict[Path, list[Path]] = defaultdict(list)
        for dir_path, py_files in dir_files.items():
            target = merge_targets.get(dir_path, dir_path)
            effective_files[target].extend(py_files)

        raw_imports_by_pkg: dict[str, list[str]] = {}

        for dir_path, py_files in sorted(effective_files.items()):
            pkg_name = _package_name_for_dir(dir_path, parent_dir)

            all_functions: list[FunctionInfo] = []
            all_classes: list[ClassInfo] = []
            all_imports: list[str] = []
            all_file_paths: list[str] = []

            for py_file in sorted(py_files):
                try:
                    functions, classes, imports = _parse_python_file_for_package(py_file, py_parser, pkg_name)
                except Exception:
                    logger.warning("Failed to parse %s", py_file, exc_info=True)
                    continue

                all_functions.extend(functions)
                all_classes.extend(classes)
                all_imports.extend(imports)
                all_file_paths.append(str(py_file.relative_to(repo_root)))

            packages[pkg_name] = PackageInfo(
                name=pkg_name,
                description="",
                functions=all_functions,
                classes=all_classes,
                imports_from=set(),
                file_paths=all_file_paths,
                language="python",
            )
            raw_imports_by_pkg[pkg_name] = all_imports

        all_pkg_names = set(packages.keys())
        for pkg_name, raw_imports in raw_imports_by_pkg.items():
            resolved = _resolve_imports_to_packages(raw_imports, all_pkg_names)
            resolved.discard(pkg_name)
            packages[pkg_name].imports_from = resolved

    # --- Rust crates ---
    for crate_name, src_rel in RUST_CRATES.items():
        src_dir = repo_root / src_rel
        if not src_dir.exists():
            logger.warning("Skipping missing crate: %s (%s)", crate_name, src_dir)
            continue

        all_functions: list[FunctionInfo] = []
        all_classes: list[ClassInfo] = []
        all_file_paths: list[str] = []

        for rs_file in sorted(src_dir.rglob("*.rs")):
            try:
                functions, classes = parse_rust_file(rs_file, rs_parser, crate_name)
            except Exception:
                logger.warning("Failed to parse %s", rs_file, exc_info=True)
                continue

            all_functions.extend(functions)
            all_classes.extend(classes)
            all_file_paths.append(str(rs_file.relative_to(repo_root)))

        packages[crate_name] = PackageInfo(
            name=crate_name,
            description="",
            functions=all_functions,
            classes=all_classes,
            imports_from=set(),
            file_paths=all_file_paths,
            language="rust",
        )

    return packages

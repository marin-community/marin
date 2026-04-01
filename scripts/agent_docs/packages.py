# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Directory-level package discovery for agent documentation.

The module grouping in graph.py groups at depth 2 (e.g. marin.processing),
which lumps 40+ functions across dedup, classification, and tokenization into
one doc. This module groups by filesystem directory instead: each directory
with Python files becomes a package, giving natural sub-package boundaries
like marin.processing.classification.deduplication (~5 public functions,
perfect for a 2-4K reference card).
"""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import tree_sitter

from agent_docs.cache import DocCache
from agent_docs.graph import (
    PYTHON_LIBS,
    PY_LANGUAGE,
    RS_LANGUAGE,
    RUST_CRATES,
    ClassInfo,
    FunctionInfo,
    _extract_python_all,
    _extract_python_calls,
    _extract_python_signature,
    _find_child,
    _is_python_public,
    _node_text,
    _parse_rust_file,
    _sha256,
    _should_skip,
)

logger = logging.getLogger(__name__)

# Known lib names for resolving import edges to packages
_KNOWN_LIBS = set(PYTHON_LIBS.keys()) | set(RUST_CRATES.keys()) | {"draccus"}


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
    """Extract full dotted import paths from top-level import statements.

    Unlike _extract_python_imports (which returns only the top-level module
    name), this returns the full dotted path so we can resolve imports to
    specific packages.
    """
    modules: list[str] = []
    for child in root.children:
        if child.type == "import_statement":
            for name_node in [c for c in child.children if c.type == "dotted_name"]:
                modules.append(_node_text(name_node))
        elif child.type == "import_from_statement":
            dotted = _find_child(child, "dotted_name")
            if dotted is not None:
                modules.append(_node_text(dotted))
    return modules


def _parse_python_file_for_package(
    file_path: Path,
    parser: tree_sitter.Parser,
    package_name: str,
) -> tuple[list[FunctionInfo], list[ClassInfo], list[str]]:
    """Parse a Python file, returning functions, classes, and full import paths.

    Similar to _parse_python_file in graph.py but uses the package name
    directly (no depth-2 grouping) and returns full dotted imports.
    """
    source_bytes = file_path.read_bytes()
    tree = parser.parse(source_bytes)
    root = tree.root_node
    rel_path = str(file_path)

    all_names = _extract_python_all(root)
    imports = _extract_full_imports(root)

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
                    qualified_name=f"{package_name}.{name}",
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
                                qualified_name=f"{package_name}.{name}.{mname}",
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
                    qualified_name=f"{package_name}.{name}",
                    source=source,
                    source_hash=_sha256(source),
                    file_path=rel_path,
                    line_number=child.start_point.row + 1,
                    language="python",
                    methods=methods,
                    is_public=is_public,
                )
            )

    return functions, classes, imports


def _collect_python_directories(src_dir: Path) -> dict[Path, list[Path]]:
    """Map each directory containing .py files to its list of Python files.

    Skips __pycache__ and test files.
    """
    dir_files: dict[Path, list[Path]] = defaultdict(list)
    for py_file in sorted(src_dir.rglob("*.py")):
        if _should_skip(py_file):
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
    the longest matching package prefix. If no package matches but the
    top-level name is a known lib, we resolve to that lib's root package.
    """
    resolved: set[str] = set()
    for imp in raw_imports:
        top_level = imp.split(".")[0]
        if top_level not in _KNOWN_LIBS:
            continue
        # Find the longest package name that is a prefix of the import
        best = ""
        for pkg_name in known_packages:
            if imp == pkg_name or imp.startswith(pkg_name + "."):
                if len(pkg_name) > len(best):
                    best = pkg_name
        if best:
            resolved.add(best)
        else:
            # Fall back to the top-level lib name if it's a known package
            if top_level in known_packages:
                resolved.add(top_level)
    return resolved


def discover_packages(repo_root: Path) -> dict[str, PackageInfo]:
    """Walk all library source trees and group by filesystem directory.

    Each directory with Python files becomes a package. Leaf directories
    containing only __init__.py merge into their parent. Rust crates become
    a single package each (flat structure).
    """
    packages: dict[str, PackageInfo] = {}
    py_parser = tree_sitter.Parser(PY_LANGUAGE)
    rs_parser = tree_sitter.Parser(RS_LANGUAGE)

    # --- Python libraries ---
    for lib_name, src_rel in PYTHON_LIBS.items():
        src_dir = repo_root / src_rel
        if not src_dir.exists():
            logger.warning("Skipping missing lib: %s (%s)", lib_name, src_dir)
            continue

        # src_dir is e.g. lib/marin/src/marin; parent is lib/marin/src
        parent_dir = src_dir.parent
        dir_files = _collect_python_directories(src_dir)

        # First pass: identify directories that should merge into parent
        merge_targets: dict[Path, Path] = {}
        for dir_path, py_files in dir_files.items():
            if dir_path == src_dir:
                continue
            if _should_merge_into_parent(dir_path, py_files):
                # Only merge if this is a true leaf (no child directories with .py files)
                has_children = any(
                    other_dir != dir_path and str(other_dir).startswith(str(dir_path) + "/") for other_dir in dir_files
                )
                if not has_children:
                    merge_targets[dir_path] = dir_path.parent

        # Second pass: build packages, merging as needed
        # Accumulate: effective_dir -> list of (file, original_dir)
        effective_files: dict[Path, list[Path]] = defaultdict(list)
        for dir_path, py_files in dir_files.items():
            target = merge_targets.get(dir_path, dir_path)
            effective_files[target].extend(py_files)

        # Parse files and build PackageInfo for each effective directory
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
                imports_from=set(),  # resolved after all packages are known
                file_paths=all_file_paths,
                language="python",
            )
            raw_imports_by_pkg[pkg_name] = all_imports

        # Resolve imports now that all package names are known
        all_pkg_names = set(packages.keys())
        for pkg_name, raw_imports in raw_imports_by_pkg.items():
            resolved = _resolve_imports_to_packages(raw_imports, all_pkg_names)
            resolved.discard(pkg_name)  # no self-imports
            packages[pkg_name].imports_from = resolved

    # --- Rust crates (each crate = one package) ---
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
                functions, classes = _parse_rust_file(rs_file, rs_parser, crate_name)
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


def _package_source_hash(pkg: PackageInfo) -> str:
    """Hash the combined source of all public items in a package."""
    hashes = sorted(item.source_hash for item in pkg.functions + pkg.classes if item.is_public)
    if not hashes:
        return hashlib.sha256(b"empty").hexdigest()[:16]
    combined = "\n".join(hashes)
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


def compute_stale_packages(
    packages: dict[str, PackageInfo],
    cache: DocCache,
) -> set[str]:
    """Determine which packages need doc regeneration.

    A package is stale if its source hash changed or it is missing from cache.
    Staleness propagates through imports: if B is stale and A imports from B,
    A is also stale.
    """
    stale: set[str] = set()

    for pkg_name, pkg in packages.items():
        public_items = [f for f in pkg.functions if f.is_public] + [c for c in pkg.classes if c.is_public]
        if not public_items:
            continue
        source_hash = _package_source_hash(pkg)
        if cache.is_stale(pkg_name, source_hash):
            stale.add(pkg_name)

    # Propagate: if package A imports from B and B is stale, A is also stale
    changed = True
    while changed:
        changed = False
        for pkg_name, pkg in packages.items():
            if pkg_name in stale:
                continue
            if pkg.imports_from & stale:
                stale.add(pkg_name)
                changed = True

    return stale

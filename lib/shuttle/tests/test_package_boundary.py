# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path


def test_shuttle_sources_do_not_import_tile_lifetime() -> None:
    source_root = Path(__file__).parents[1] / "src" / "shuttle"
    imported_modules = {
        node.module
        for source in source_root.rglob("*.py")
        for node in ast.walk(ast.parse(source.read_text()))
        if isinstance(node, ast.ImportFrom)
    }
    imported_modules.update(
        alias.name
        for source in source_root.rglob("*.py")
        for node in ast.walk(ast.parse(source.read_text()))
        if isinstance(node, ast.Import)
        for alias in node.names
    )

    assert all(module is None or not module.startswith("tile_lifetime") for module in imported_modules)


def test_current_shuttle_surface_does_not_import_experimental_modules() -> None:
    source_root = Path(__file__).parents[1] / "src" / "shuttle"
    current_sources = tuple(
        source for source in source_root.rglob("*.py") if "experimental" not in source.relative_to(source_root).parts
    )
    imported_modules = {
        node.module
        for source in current_sources
        for node in ast.walk(ast.parse(source.read_text()))
        if isinstance(node, ast.ImportFrom)
    }
    imported_modules.update(
        alias.name
        for source in current_sources
        for node in ast.walk(ast.parse(source.read_text()))
        if isinstance(node, ast.Import)
        for alias in node.names
    )

    assert all(module is None or not module.startswith("shuttle.experimental") for module in imported_modules)

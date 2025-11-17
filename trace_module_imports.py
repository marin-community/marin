#!/usr/bin/env python3
"""
Script to identify unused marin modules.

This script:
1. Finds all Python files in experiments/ (excluding experiments/crawl)
2. Traces all transitively imported modules from lib/marin/src/marin
3. Reports modules in src/marin that are NOT referenced by any experiment
"""

import ast
import sys
from pathlib import Path
from typing import Set, List, Dict
import importlib.util
import os


def find_experiment_files(experiments_dir: Path) -> List[Path]:
    """Find all Python files in experiments/, excluding crawl subdirectory."""
    all_py_files = list(experiments_dir.rglob("*.py"))
    # Exclude files in experiments/crawl
    filtered_files = [
        f for f in all_py_files
        if not any(part == "crawl" for part in f.relative_to(experiments_dir).parts)
    ]
    return filtered_files


def find_all_marin_modules(marin_src_dir: Path) -> Set[str]:
    """Find all Python modules in lib/marin/src/marin."""
    marin_modules = set()

    for py_file in marin_src_dir.rglob("*.py"):
        # Get relative path from marin package root
        rel_path = py_file.relative_to(marin_src_dir)

        # Convert file path to module name
        if rel_path.name == "__init__.py":
            # For __init__.py, use the parent directory as module name
            if rel_path.parent == Path("."):
                module_name = "marin"
            else:
                module_name = "marin." + str(rel_path.parent).replace(os.sep, ".")
        else:
            # For regular files, include the filename without .py
            module_path = rel_path.with_suffix("")
            module_name = "marin." + str(module_path).replace(os.sep, ".")

        marin_modules.add(module_name)

    return marin_modules


def extract_imports_from_ast(file_path: Path) -> Set[str]:
    """Extract direct imports from a Python file using AST parsing."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=str(file_path))
    except Exception as e:
        print(f"Warning: Could not parse {file_path}: {e}", file=sys.stderr)
        return set()

    imports = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)
                # Also add submodule imports
                for alias in node.names:
                    full_name = f"{node.module}.{alias.name}"
                    imports.add(full_name)

    return imports


def get_transitive_imports(
    file_path: Path,
    base_dir: Path,
    visited: Set[Path] = None
) -> Set[str]:
    """
    Recursively get all imports from a file and its dependencies.
    Only follows imports within the repository.
    """
    if visited is None:
        visited = set()

    if file_path in visited:
        return set()

    visited.add(file_path)

    all_imports = extract_imports_from_ast(file_path)

    # Try to resolve and follow local imports
    for imported_module in list(all_imports):
        # Try to resolve as experiments.* or marin.* module
        module_parts = imported_module.split('.')

        # Check if it's an experiments module
        if module_parts[0] == "experiments":
            subpath = "/".join(module_parts[1:])
            possible_paths = [
                base_dir / "experiments" / subpath / "__init__.py",
                base_dir / "experiments" / (subpath + ".py"),
            ]

            for possible_path in possible_paths:
                if possible_path.exists() and possible_path not in visited:
                    transitive = get_transitive_imports(possible_path, base_dir, visited)
                    all_imports.update(transitive)
                    break

    return all_imports


def find_marin_imports_in_experiments(
    experiment_files: List[Path],
    base_dir: Path,
    all_existing_modules: Set[str]
) -> Set[str]:
    """Find all marin modules imported (transitively) by experiment files."""
    all_marin_imports = set()

    for exp_file in experiment_files:
        print(f"Processing {exp_file.relative_to(base_dir)}...", file=sys.stderr)
        imports = get_transitive_imports(exp_file, base_dir)

        # Filter for marin imports
        marin_imports = {imp for imp in imports if imp.startswith("marin.")}
        all_marin_imports.update(marin_imports)

    # Also include parent modules that actually exist
    # If marin.foo.bar.baz is imported, also consider marin.foo and marin.foo.bar as imported
    # but only if they exist as actual modules
    expanded_imports = set(all_marin_imports)
    for imp in all_marin_imports:
        parts = imp.split('.')
        for i in range(1, len(parts)):
            parent = '.'.join(parts[:i+1])
            if parent in all_existing_modules:
                expanded_imports.add(parent)

    return expanded_imports


def main():
    base_dir = Path(__file__).parent.resolve()
    experiments_dir = base_dir / "experiments"
    marin_src_dir = base_dir / "lib" / "marin" / "src" / "marin"

    print("=" * 80)
    print("Marin Module Import Analysis")
    print("=" * 80)
    print()

    # Step 1: Find experiment files
    print("Step 1: Finding experiment files (excluding experiments/crawl)...")
    experiment_files = find_experiment_files(experiments_dir)
    print(f"Found {len(experiment_files)} experiment files")
    print()

    # Step 2: Find all marin modules
    print("Step 2: Finding all marin modules...")
    all_marin_modules = find_all_marin_modules(marin_src_dir)
    print(f"Found {len(all_marin_modules)} marin modules")
    print()

    # Step 3: Find marin imports in experiments
    print("Step 3: Tracing imports from experiments...")
    imported_marin_modules = find_marin_imports_in_experiments(
        experiment_files, base_dir, all_marin_modules
    )
    print(f"Found {len(imported_marin_modules)} unique marin imports")
    print()

    # Step 4: Compute unreferenced modules
    print("Step 4: Computing unreferenced modules...")
    unreferenced_modules = all_marin_modules - imported_marin_modules

    print("=" * 80)
    print(f"RESULTS: {len(unreferenced_modules)} modules NOT referenced by experiments")
    print("=" * 80)
    print()

    if unreferenced_modules:
        # Sort for better readability
        for module in sorted(unreferenced_modules):
            print(f"  {module}")
    else:
        print("  All modules are referenced!")

    # Calculate actual referenced modules (only those that exist)
    referenced_existing_modules = all_marin_modules & imported_marin_modules

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total marin modules:              {len(all_marin_modules)}")
    print(f"Modules referenced by experiments: {len(referenced_existing_modules)}")
    print(f"Unreferenced modules:             {len(unreferenced_modules)}")
    print(f"Coverage:                         {len(referenced_existing_modules) / len(all_marin_modules) * 100:.1f}%")
    print()
    print(f"Note: Found {len(imported_marin_modules)} total import statements,")
    print(f"      including {len(imported_marin_modules) - len(referenced_existing_modules)} imports of functions/classes/variables")

    # Optionally write results to a file
    output_file = base_dir / "unreferenced_modules.txt"
    with open(output_file, 'w') as f:
        f.write("Unreferenced marin modules:\n")
        f.write("=" * 80 + "\n\n")
        for module in sorted(unreferenced_modules):
            f.write(f"{module}\n")
    print()
    print(f"Results written to: {output_file}")


if __name__ == "__main__":
    main()

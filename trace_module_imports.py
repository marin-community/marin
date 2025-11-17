#!/usr/bin/env python3
"""
Script to identify unused marin modules by tracing imports.

This script:
1. Finds all Python files in experiments/ (excluding experiments/crawl)
2. Imports each experiment file and traces which marin modules get imported
3. Identifies all modules from lib/marin/src/marin that are imported
4. Reports modules in src/marin that are NOT referenced by any experiment
"""

import importlib
import logging
import os
import sys
from pathlib import Path
from typing import Set, List

# Suppress most logging to reduce noise
logging.basicConfig(level=logging.ERROR)


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


def import_experiment_and_trace(
    script_path: Path,
    base_dir: Path
) -> Set[str]:
    """
    Import an experiment file and capture imported marin modules.
    Returns the set of marin.* modules that were imported.
    """
    # Clear marin modules from sys.modules before importing
    marin_modules_before = {k for k in list(sys.modules.keys()) if k.startswith("marin.")}
    for mod in marin_modules_before:
        if mod in sys.modules:
            del sys.modules[mod]

    # Convert file path to module name
    rel_path = script_path.relative_to(base_dir)
    module_name = str(rel_path.with_suffix("")).replace(os.sep, ".")

    try:
        # Import the module
        importlib.import_module(module_name)
    except Exception:
        # Ignore import errors - we just want to see what got imported before the error
        pass

    # Capture all marin modules that are now loaded
    marin_modules = {m for m in sys.modules.keys() if m.startswith("marin.")}

    return marin_modules


def trace_imports_from_experiments(
    experiment_files: List[Path],
    base_dir: Path
) -> Set[str]:
    """
    Import all experiment files and collect marin imports.
    """
    all_marin_imports = set()
    successful = 0
    failed = 0

    for i, exp_file in enumerate(experiment_files, 1):
        rel_path = exp_file.relative_to(base_dir)
        print(f"[{i}/{len(experiment_files)}] Importing {rel_path}...", end=" ")

        try:
            marin_imports = import_experiment_and_trace(exp_file, base_dir)
            all_marin_imports.update(marin_imports)
            print(f"OK ({len(marin_imports)} marin modules)")
            successful += 1

        except Exception as e:
            print(f"FAIL ({type(e).__name__}: {str(e)[:50]})")
            failed += 1

    print()
    print(f"Summary: {successful} successful, {failed} failed")
    print()

    return all_marin_imports


def main():
    base_dir = Path(__file__).parent.resolve()
    experiments_dir = base_dir / "experiments"
    marin_src_dir = base_dir / "lib" / "marin" / "src" / "marin"

    # Add library paths to sys.path
    lib_paths = [
        base_dir / "lib" / "marin" / "src",
        base_dir / "lib" / "levanter" / "src",
        base_dir / "lib" / "haliax" / "src",
        base_dir / "lib" / "zephyr" / "src",
        base_dir / "lib" / "fray" / "src",
    ]
    for lib_path in lib_paths:
        if lib_path.exists() and str(lib_path) not in sys.path:
            sys.path.insert(0, str(lib_path))

    # Also add base_dir to sys.path so we can import experiments
    if str(base_dir) not in sys.path:
        sys.path.insert(0, str(base_dir))

    print("=" * 80)
    print("Marin Module Import Analysis (Import Tracing)")
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

    # Step 3: Import experiments and trace imports
    print("Step 3: Importing experiments and tracing marin modules...")
    print()
    imported_marin_modules = trace_imports_from_experiments(experiment_files, base_dir)
    print(f"Found {len(imported_marin_modules)} unique marin module imports")
    print()

    # Step 4: Compute unreferenced modules
    print("Step 4: Computing unreferenced modules...")
    # Only consider modules that actually exist
    referenced_existing_modules = all_marin_modules & imported_marin_modules
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

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total marin modules:               {len(all_marin_modules)}")
    print(f"Modules referenced by experiments: {len(referenced_existing_modules)}")
    print(f"Unreferenced modules:              {len(unreferenced_modules)}")
    print(f"Coverage:                          {len(referenced_existing_modules) / len(all_marin_modules) * 100:.1f}%")

    # Write results to a file
    output_file = base_dir / "unreferenced_modules.txt"
    with open(output_file, 'w') as f:
        f.write("Unreferenced marin modules:\n")
        f.write("=" * 80 + "\n\n")
        for module in sorted(unreferenced_modules):
            f.write(f"{module}\n")
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total marin modules:               {len(all_marin_modules)}\n")
        f.write(f"Modules referenced by experiments: {len(referenced_existing_modules)}\n")
        f.write(f"Unreferenced modules:              {len(unreferenced_modules)}\n")
        f.write(f"Coverage:                          {len(referenced_existing_modules) / len(all_marin_modules) * 100:.1f}%\n")

    print()
    print(f"Results written to: {output_file}")


if __name__ == "__main__":
    main()

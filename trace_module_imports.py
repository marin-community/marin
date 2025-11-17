#!/usr/bin/env python3
"""
Script to identify unused marin modules by tracing imports.

This script:
1. Finds all Python files in experiments/ (excluding experiments/crawl)
2. Runs each experiment with import tracing enabled
3. Identifies all modules from lib/marin/src/marin that are imported
4. Reports modules in src/marin that are NOT referenced by any experiment
"""

import json
import logging
import os
import subprocess
import sys
import tempfile
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


def trace_imports_for_script(script_path: Path, base_dir: Path) -> Set[str]:
    """
    Run a script with import tracing and extract marin imports.
    Returns the set of marin.* modules that were imported.
    """
    # Create a trace script that will capture imports
    trace_script = """
import sys
import json

try:
    # Run the script
    with open(sys.argv[1], 'r') as f:
        code = compile(f.read(), sys.argv[1], 'exec')
        exec(code, {'__name__': '__main__', '__file__': sys.argv[1]})
except SystemExit:
    pass
except Exception:
    pass
finally:
    # Capture all marin modules in sys.modules after execution
    imported_modules = set()
    for mod in list(sys.modules.keys()):
        if mod.startswith('marin.') or mod == 'marin':
            imported_modules.add(mod)

    # Write results to a temp file
    output_file = sys.argv[2]
    with open(output_file, 'w') as f:
        json.dump(list(imported_modules), f)
"""

    with tempfile.TemporaryDirectory(prefix="trace-") as temp_dir:
        trace_file = Path(temp_dir) / "trace.py"
        output_file = Path(temp_dir) / "imports.json"

        with open(trace_file, 'w') as f:
            f.write(trace_script)

        # Check if script has executor_main
        with open(script_path, 'r') as f:
            content = f.read()
            if "executor_main(" not in content:
                return set()
            if "nodryrun" in content:
                return set()

        # Run with uv
        try:
            subprocess.run(
                [
                    "uv", "run", "python", str(trace_file), str(script_path), str(output_file),
                    "--dry_run", "True",
                    "--executor_info_base_path", temp_dir,
                    "--prefix", temp_dir
                ],
                cwd=base_dir,
                capture_output=True,
                timeout=60,
                check=False
            )
        except subprocess.TimeoutExpired:
            pass
        except Exception:
            pass

        # Read the imports
        if output_file.exists():
            with open(output_file, 'r') as f:
                try:
                    imports = json.load(f)
                    return set(imports)
                except Exception:
                    pass

    return set()


def trace_imports_from_experiments(
    experiment_files: List[Path],
    base_dir: Path
) -> Set[str]:
    """
    Run all experiment files with tracing and collect marin imports.
    """
    all_marin_imports = set()
    successful = 0
    skipped = 0
    failed = 0

    for i, exp_file in enumerate(experiment_files, 1):
        rel_path = exp_file.relative_to(base_dir)
        print(f"[{i}/{len(experiment_files)}] Tracing {rel_path}...", end=" ", flush=True)

        # Check if file has executor_main (skip if not)
        try:
            with open(exp_file, 'r') as f:
                content = f.read()

            if "executor_main(" not in content:
                print("SKIP (no executor_main)")
                skipped += 1
                continue

            if "nodryrun" in content:
                print("SKIP (nodryrun marker)")
                skipped += 1
                continue

        except Exception as e:
            print(f"SKIP (read error)")
            skipped += 1
            continue

        # Try to trace the experiment
        try:
            marin_imports = trace_imports_for_script(exp_file, base_dir)
            if marin_imports:
                all_marin_imports.update(marin_imports)
                print(f"OK ({len(marin_imports)} marin imports)")
                successful += 1
            else:
                print("SKIP (no imports or failed)")
                skipped += 1

        except Exception as e:
            print(f"FAIL ({type(e).__name__})")
            failed += 1

    print()
    print(f"Summary: {successful} successful, {skipped} skipped, {failed} failed")
    print()

    return all_marin_imports


def main():
    base_dir = Path(__file__).parent.resolve()
    experiments_dir = base_dir / "experiments"
    marin_src_dir = base_dir / "lib" / "marin" / "src" / "marin"

    print("=" * 80)
    print("Marin Module Import Analysis (uv run with tracing)")
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

    # Step 3: Run experiments with tracing
    print("Step 3: Running experiments with import tracing...")
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

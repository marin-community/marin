#!/usr/bin/env python3
"""
Script to identify unused marin modules using conservative static analysis.

This script:
1. Finds all Python files in experiments/ and infra/ (excluding experiments/crawl)
2. Uses AST to extract all import statements
3. Also scans for string literals that look like module references (e.g., "marin.download.huggingface")
4. Builds a conservative set of all potentially referenced modules
5. Reports modules in src/marin that are NOT referenced by any seed file
"""

import ast
import os
import re
from pathlib import Path
from typing import Set, List


def find_seed_files(base_dir: Path) -> List[Path]:
    """Find all Python files in experiments/ and infra/, excluding experiments/crawl."""
    all_files = []

    # Add files from experiments/ (excluding experiments/crawl)
    experiments_dir = base_dir / "experiments"
    if experiments_dir.exists():
        exp_files = list(experiments_dir.rglob("*.py"))
        # Exclude files in experiments/crawl
        exp_files = [
            f for f in exp_files
            if not any(part == "crawl" for part in f.relative_to(experiments_dir).parts)
        ]
        all_files.extend(exp_files)

    # Add files from infra/
    infra_dir = base_dir / "infra"
    if infra_dir.exists():
        infra_files = list(infra_dir.rglob("*.py"))
        all_files.extend(infra_files)

    return all_files


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


def extract_marin_references_from_file(file_path: Path) -> Set[str]:
    """
    Extract all marin module references from a file using AST and string scanning.
    Returns a set of module names that might be referenced.
    """
    references = set()

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Warning: Could not read {file_path}: {e}")
        return references

    # Parse with AST to get imports
    try:
        tree = ast.parse(content, filename=str(file_path))

        for node in ast.walk(tree):
            # Handle "import marin.x.y"
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith('marin'):
                        references.add(alias.name)
            # Handle "from marin.x.y import z"
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith('marin'):
                    references.add(node.module)
                    # Also add the full path for imported names
                    for alias in node.names:
                        if node.module:
                            full_name = f"{node.module}.{alias.name}"
                            references.add(full_name)
            # Also look for string constants that look like marin modules
            elif isinstance(node, ast.Constant):
                if isinstance(node.value, str):
                    if node.value.startswith('marin.') or node.value == 'marin':
                        # Only add if it looks like a module path (no spaces, special chars, etc)
                        if re.match(r'^marin(\.[a-zA-Z_][a-zA-Z0-9_]*)*$', node.value):
                            references.add(node.value)

    except SyntaxError as e:
        print(f"Warning: Could not parse {file_path}: {e}")

    # Also do a regex scan for marin module references in strings
    # This catches cases where modules are referenced dynamically
    # Pattern: "marin.something.something" or 'marin.something.something'
    pattern = r'["\']marin(\.[a-zA-Z_][a-zA-Z0-9_]*)+["\']'
    for match in re.finditer(pattern, content):
        # Extract the module name without quotes
        module_ref = match.group(0).strip('"\'')
        references.add(module_ref)

    return references


def analyze_seed_files(
    seed_files: List[Path],
    base_dir: Path
) -> Set[str]:
    """
    Analyze all seed files and collect marin module references.
    """
    all_references = set()
    successful = 0
    failed = 0

    for i, seed_file in enumerate(seed_files, 1):
        rel_path = seed_file.relative_to(base_dir)
        print(f"[{i}/{len(seed_files)}] Analyzing {rel_path}...", end=" ", flush=True)

        try:
            references = extract_marin_references_from_file(seed_file)
            all_references.update(references)
            print(f"OK ({len(references)} references)")
            successful += 1
        except Exception as e:
            print(f"FAIL ({type(e).__name__})")
            failed += 1

    print()
    print(f"Summary: {successful} successful, {failed} failed")
    print()

    return all_references


def expand_module_references(references: Set[str], all_modules: Set[str]) -> Set[str]:
    """
    Expand references to include parent modules and resolve wildcards.

    For example:
    - If "marin.download.huggingface.download_hf" is referenced, also mark
      "marin.download", "marin.download.huggingface" as referenced
    - Only include modules that actually exist
    """
    expanded = set()

    for ref in references:
        # Add the reference itself if it's a real module
        if ref in all_modules:
            expanded.add(ref)

        # Add all parent modules that exist
        parts = ref.split('.')
        for i in range(1, len(parts) + 1):
            parent = '.'.join(parts[:i])
            if parent in all_modules:
                expanded.add(parent)

    return expanded


def main():
    base_dir = Path(__file__).parent.resolve()
    marin_src_dir = base_dir / "lib" / "marin" / "src" / "marin"

    print("=" * 80)
    print("Marin Module Import Analysis (Conservative AST + String Scanning)")
    print("=" * 80)
    print()

    # Step 1: Find seed files
    print("Step 1: Finding seed files from experiments/ and infra/ (excluding experiments/crawl)...")
    seed_files = find_seed_files(base_dir)
    print(f"Found {len(seed_files)} seed files")
    print()

    # Step 2: Find all marin modules
    print("Step 2: Finding all marin modules...")
    all_marin_modules = find_all_marin_modules(marin_src_dir)
    print(f"Found {len(all_marin_modules)} marin modules")
    print()

    # Step 3: Analyze seed files for references
    print("Step 3: Analyzing seed files for marin module references...")
    print()
    raw_references = analyze_seed_files(seed_files, base_dir)
    print(f"Found {len(raw_references)} raw marin module references")
    print()

    # Step 4: Expand references to include parent modules
    print("Step 4: Expanding references to include parent modules...")
    referenced_modules = expand_module_references(raw_references, all_marin_modules)
    print(f"Expanded to {len(referenced_modules)} referenced modules")
    print()

    # Step 5: Compute unreferenced modules
    print("Step 5: Computing unreferenced modules...")
    unreferenced_modules = all_marin_modules - referenced_modules

    print("=" * 80)
    print(f"RESULTS: {len(unreferenced_modules)} modules NOT referenced by seed files")
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
    print(f"Modules referenced by seed files:  {len(referenced_modules)}")
    print(f"Unreferenced modules:              {len(unreferenced_modules)}")
    print(f"Coverage:                          {len(referenced_modules) / len(all_marin_modules) * 100:.1f}%")

    # Write results to a file
    output_file = base_dir / "unreferenced_modules.txt"
    with open(output_file, 'w') as f:
        f.write("Unreferenced marin modules:\n")
        f.write("(Not referenced by experiments/ or infra/)\n")
        f.write("=" * 80 + "\n\n")
        for module in sorted(unreferenced_modules):
            f.write(f"{module}\n")
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total marin modules:               {len(all_marin_modules)}\n")
        f.write(f"Modules referenced by seed files:  {len(referenced_modules)}\n")
        f.write(f"Unreferenced modules:              {len(unreferenced_modules)}\n")
        f.write(f"Coverage:                          {len(referenced_modules) / len(all_marin_modules) * 100:.1f}%\n")

    print()
    print(f"Results written to: {output_file}")


if __name__ == "__main__":
    main()

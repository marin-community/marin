#!/usr/bin/env python3
"""
Script to iteratively prune files listed in prune.txt.

For each file:
1. Remove the file
2. Run dry_run tests
3. If no new tests fail, commit the removal
4. Otherwise, restore the file
"""

import subprocess
import sys
import os
import shutil
from pathlib import Path


def run_command(cmd, cwd=None, capture_output=True):
    """Run a shell command and return the result."""
    print(f"Running: {cmd}")
    result = subprocess.run(
        cmd,
        shell=True,
        cwd=cwd,
        capture_output=capture_output,
        text=True
    )
    return result


def get_test_results():
    """Run the dry_run tests and return the result."""
    result = run_command("CI=true uv run pytest tests/test_dry_run.py -v", capture_output=True)
    return result


def parse_test_output(output):
    """Parse pytest output to extract failed test names."""
    failed_tests = set()
    lines = output.split('\n')

    for line in lines:
        # Look for FAILED lines
        if 'FAILED' in line:
            # Extract test name from lines like "FAILED tests/test_dry_run.py::test_name - ..."
            parts = line.split('::')
            if len(parts) >= 2:
                test_name = parts[1].split(' ')[0].split('-')[0].strip()
                failed_tests.add(test_name)

    return failed_tests


def main():
    # Read the prune.txt file
    prune_file = Path("prune.txt")
    if not prune_file.exists():
        print("Error: prune.txt not found!")
        sys.exit(1)

    with open(prune_file) as f:
        files_to_prune = [line.strip() for line in f if line.strip()]

    print(f"Found {len(files_to_prune)} files to potentially prune")

    # Get baseline test results
    print("\n" + "="*80)
    print("Running baseline tests...")
    print("="*80)
    baseline_result = get_test_results()
    baseline_failed = parse_test_output(baseline_result.stdout + baseline_result.stderr)

    print(f"\nBaseline: {len(baseline_failed)} tests failing")
    if baseline_failed:
        print(f"Baseline failures: {sorted(baseline_failed)}")

    # Track statistics
    pruned_count = 0
    skipped_count = 0
    nonexistent_count = 0

    # Process each file
    for i, file_path in enumerate(files_to_prune, 1):
        print("\n" + "="*80)
        print(f"Processing {i}/{len(files_to_prune)}: {file_path}")
        print("="*80)

        # Check if file exists
        if not Path(file_path).exists():
            print(f"  ⚠️  File does not exist, skipping: {file_path}")
            nonexistent_count += 1
            continue

        # Backup the file
        backup_path = f"{file_path}.backup"
        print(f"  📋 Creating backup: {backup_path}")
        shutil.copy2(file_path, backup_path)

        # Remove the file
        print(f"  🗑️  Removing: {file_path}")
        os.remove(file_path)

        # Run tests
        print(f"  🧪 Running tests...")
        test_result = get_test_results()
        current_failed = parse_test_output(test_result.stdout + test_result.stderr)

        # Check if new tests failed
        new_failures = current_failed - baseline_failed

        if new_failures:
            print(f"  ❌ New test failures detected: {sorted(new_failures)}")
            print(f"  ↩️  Restoring file...")
            shutil.move(backup_path, file_path)
            skipped_count += 1
        else:
            print(f"  ✅ No new test failures!")
            print(f"  💾 Committing removal...")

            # Remove backup
            if Path(backup_path).exists():
                os.remove(backup_path)

            # Commit the removal
            commit_msg = f"Remove unused file: {file_path}"
            run_command(f'git add "{file_path}"')
            run_command(f'git commit -m "{commit_msg}"')

            pruned_count += 1

    # Summary
    print("\n" + "="*80)
    print("PRUNING COMPLETE")
    print("="*80)
    print(f"✅ Successfully pruned: {pruned_count}")
    print(f"❌ Skipped (caused test failures): {skipped_count}")
    print(f"⚠️  Skipped (file not found): {nonexistent_count}")
    print(f"📊 Total processed: {len(files_to_prune)}")


if __name__ == "__main__":
    main()

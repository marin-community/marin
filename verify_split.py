
import subprocess
import sys
from collections import defaultdict

BRANCHES = [
    "kevin/ray-run-auth",
    "kevin/gsm8k",
    "kevin/inflight-weights",
    "kevin/math500",
    "kevin/rl-loss-improvements",
    "kevin/classification",
    "kevin/inference-ctx",
    "kevin/upgrade-deps-misc",
    "kevin/remove-old-scripts",
    "kevin/mock-env"
]

SOURCE_BRANCH = "chris/exp-rl"
BASE_BRANCH = "main"

def run_cmd(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"Command failed: {' '.join(cmd)}\nStderr: {result.stderr}")
    return result.stdout.strip()

def get_changed_files(branch):
    # relative to main
    output = run_cmd(["git", "diff", "--name-only", f"{BASE_BRANCH}...{branch}"])
    lines = output.split('\n')
    return set(line for line in lines if line.strip())

def main():
    print(f"Verifying split against {SOURCE_BRANCH}...\n")

    # 1. Get changes in source
    source_files = get_changed_files(SOURCE_BRANCH)
    print(f"Total changed files in {SOURCE_BRANCH}: {len(source_files)}")

    # 2. Get changes in each sub-branch
    branch_files = {}
    all_sub_files = set()
    file_to_branch = defaultdict(list)

    for branch in BRANCHES:
        files = get_changed_files(branch)
        branch_files[branch] = files
        for f in files:
            all_sub_files.add(f)
            file_to_branch[f].append(branch)
        print(f"  {branch}: {len(files)} files")

    print("\n---------------------------------------------------")
    print("CHECK 1: DISJOINTNESS")
    # Check if any file is in multiple branches
    overlaps = {f: branches for f, branches in file_to_branch.items() if len(branches) > 1}
    
    if not overlaps:
        print("✅ PASS: All PRs are disjoint (no file overlaps).")
    else:
        print("❌ FAIL: The following files verify appear in multiple PRs:")
        for f, branches in overlaps.items():
            print(f"  - {f}: {', '.join(branches)}")

    print("\n---------------------------------------------------")
    print("CHECK 2: COMPLETENESS (File Coverage)")
    # Check if all source files are covered
    missing = source_files - all_sub_files
    extras = all_sub_files - source_files

    if not missing and not extras:
        print("✅ PASS: The union of sub-PRs exactly matches the file list of the original PR.")
    else:
        if missing:
            print(f"❌ FAIL: The following files from {SOURCE_BRANCH} are MISSING in sub-PRs:")
            for f in sorted(missing):
                print(f"  - {f}")
        if extras:
            print(f"⚠️  NOTE: The following files exist in sub-PRs but NOT in {SOURCE_BRANCH} (Extra files):")
            for f in sorted(extras):
                print(f"  - {f}")
        if not missing:
            print("✅ PASS: All original files are covered.")

    print("\n---------------------------------------------------")
    print("CHECK 3: CONTENT FIDELITY (Exact Match)")
    # For every file in the sub-PRs, diff it against the source branch
    diffs_found = []
    
    # We need to fetch/ensure we can read from branches.
    # We will use git diff branch1:file branch2:file
    
    for f in sorted(all_sub_files):
        # Find which branch has it
        owners = file_to_branch[f]
        if not owners: continue 
        # checking the first owner is enough if disjoint checked passed, but let's check all
        for owner in owners:
            # Check if file exists in source
            if f not in source_files:
                # It's an extra file, skip diff against source (or diff against /dev/null if we cared)
                continue
            
            # Diff content
            # git diff b1:path b2:path
            proc = subprocess.run(
                ["git", "diff", f"{SOURCE_BRANCH}:{f}", f"{owner}:{f}"],
                capture_output=True, text=True
            )
            
            if proc.stdout:
                diffs_found.append((f, owner))
                # print(f"  Diff found in {f} ({owner})")

    if not diffs_found:
         print(f"✅ PASS: Content of all covered files exactly matches {SOURCE_BRANCH}.")
    else:
         print(f"⚠️  WARNING: Content deviations found in {len(diffs_found)} files (likely due to fixes applied):")
         for f, owner in diffs_found:
             print(f"  - {f} (in {owner}) differs from original.")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Analyze stale branches using GitHub's exact merge detection logic.
Detects squash merges using git cherry with temporary squashed commits.
"""

import subprocess
import re
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple


def run_git(cmd: str) -> Tuple[str, int]:
    """Run a git command and return (output, return_code)."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60
        )
        return result.stdout.strip(), result.returncode
    except Exception as e:
        print(f"Error running: {cmd}")
        print(f"Error: {e}")
        return "", 1


def get_remote_branches() -> List[str]:
    """Get list of all remote branches except HEAD and main."""
    output, _ = run_git("git for-each-ref --format='%(refname:short)' refs/remotes/origin")
    branches = []
    for line in output.split('\n'):
        line = line.strip()
        if line and line != 'origin/HEAD' and line != 'origin/main':
            branches.append(line)
    return branches


def check_if_squash_merged(branch: str) -> bool:
    """
    Check if a branch was squash-merged into main.
    Uses git cherry to detect if the branch's changes (as a squashed commit) exist in main.

    Algorithm:
    1. Find merge-base between branch and main
    2. Create a temporary squashed commit representing all branch changes
    3. Use git cherry to check if this patch exists in main
    """
    # Get merge-base
    merge_base, code = run_git(f"git merge-base {branch} origin/main")
    if code != 0 or not merge_base:
        return False

    # Create a tree object representing the branch state
    tree, code = run_git(f"git rev-parse {branch}^{{tree}}")
    if code != 0:
        return False

    # Create a temporary squashed commit
    # This commit has the merge-base as parent and the branch tree
    commit_msg = f"Temporary squashed commit for {branch}"
    temp_commit, code = run_git(f'git commit-tree {tree} -p {merge_base} -m "{commit_msg}"')
    if code != 0:
        return False

    # Use git cherry to check if this commit's patch-id exists in main
    # git cherry returns commits that are NOT in the upstream
    # If our temp commit is NOT in the list, it means it's already in main (squash-merged)
    cherry_output, code = run_git(f"git cherry {branch} origin/main {temp_commit}")

    # If cherry returns empty or doesn't list our commit, it's merged
    # If it returns "+ {temp_commit}", it's not merged
    if cherry_output.strip():
        # Commit is listed = not in main
        return False
    else:
        # Commit not listed = already in main (squash-merged!)
        return True


def check_merge_status(branch: str) -> Tuple[bool, str]:
    """
    Check if branch is merged into main.
    Returns: (is_merged, merge_type)
    """
    # Method 1: Traditional merge (ancestor check)
    _, code = run_git(f"git merge-base --is-ancestor {branch} origin/main")
    if code == 0:
        return True, 'traditional_merge'

    # Method 2: No commits ahead
    ahead_output, _ = run_git(f"git log origin/main..{branch} --oneline")
    ahead_count = len([l for l in ahead_output.split('\n') if l.strip()])
    if ahead_count == 0:
        return True, 'no_diff'

    # Method 3: Squash merge detection
    if check_if_squash_merged(branch):
        return True, 'squash_merge'

    return False, 'not_merged'


def get_branch_metadata(branch: str) -> Optional[Dict]:
    """Get metadata for a branch."""
    clean_name = branch.replace('origin/', '')

    # Get date and author
    date_str, _ = run_git(f"git log -1 --format='%ci' {branch}")
    author, _ = run_git(f"git log -1 --format='%an' {branch}")

    if not date_str:
        return None

    # Parse date
    try:
        date_clean = re.sub(r'\s+[+-]\d{4}$', '', date_str)
        date = datetime.strptime(date_clean, '%Y-%m-%d %H:%M:%S')
        age_days = (datetime.now() - date).days
    except:
        return None

    # Check merge status
    is_merged, merge_type = check_merge_status(branch)

    # Get ahead/behind counts
    ahead_output, _ = run_git(f"git log origin/main..{branch} --oneline")
    ahead = len([l for l in ahead_output.split('\n') if l.strip()])

    behind_output, _ = run_git(f"git log {branch}..origin/main --oneline")
    behind = len([l for l in behind_output.split('\n') if l.strip()])

    return {
        'branch': clean_name,
        'ahead': ahead,
        'behind': behind,
        'is_merged': is_merged,
        'merge_type': merge_type,
        'date': date_str,
        'age_days': age_days,
        'author': author,
    }


def main():
    """Main entry point."""
    print("Fetching latest from origin...")
    run_git("git fetch origin --prune")

    print("Getting branch list...")
    branches = get_remote_branches()
    print(f"Found {len(branches)} branches to analyze\n")

    # Test with specific branch first
    test_branch = 'origin/ksalahi/openqa'
    if test_branch in branches:
        print(f"Testing squash detection on {test_branch}...")
        metadata = get_branch_metadata(test_branch)
        if metadata:
            print(f"  Result: is_merged={metadata['is_merged']}, type={metadata['merge_type']}")
            print(f"  Ahead: {metadata['ahead']}, Behind: {metadata['behind']}\n")

    results = []
    for i, branch in enumerate(branches):
        if i % 25 == 0:
            print(f"Progress: {i}/{len(branches)}")

        metadata = get_branch_metadata(branch)
        if metadata:
            results.append(metadata)

    print(f"\nAnalyzed {len(results)} branches successfully\n")

    # Sort: merged first, then by ahead count, then by age
    results.sort(key=lambda x: (not x['is_merged'], x['ahead'], -x['age_days']))

    # Save JSON
    with open('/tmp/github_logic_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to /tmp/github_logic_analysis.json\n")

    # Generate report
    print("=" * 80)
    print("STALE BRANCH ANALYSIS - GITHUB LOGIC")
    print("=" * 80)
    print()

    # Merged branches
    merged = [b for b in results if b['is_merged']]
    print(f"## MERGED BRANCHES ({len(merged)} branches)")
    print("These are fully merged into main and safe to delete:\n")

    # Group by merge type
    trad_merged = [b for b in merged if b['merge_type'] == 'traditional_merge']
    squash_merged = [b for b in merged if b['merge_type'] == 'squash_merge']
    no_diff = [b for b in merged if b['merge_type'] == 'no_diff']

    if trad_merged:
        print(f"### Traditional merges ({len(trad_merged)} branches):")
        for b in trad_merged[:20]:
            print(f"- `{b['branch']}` - {b['behind']} behind - {b['date'][:10]} - {b['author']}")
        if len(trad_merged) > 20:
            print(f"  ... and {len(trad_merged) - 20} more")
        print()

    if squash_merged:
        print(f"### Squash merges ({len(squash_merged)} branches):")
        for b in squash_merged[:20]:
            print(f"- `{b['branch']}` - {b['ahead']} ahead (squashed), {b['behind']} behind - {b['date'][:10]} - {b['author']}")
        if len(squash_merged) > 20:
            print(f"  ... and {len(squash_merged) - 20} more")
        print()

    if no_diff:
        print(f"### No differences ({len(no_diff)} branches):")
        for b in no_diff[:20]:
            print(f"- `{b['branch']}` - {b['date'][:10]} - {b['author']}")
        print()

    # Not merged
    not_merged = [b for b in results if not b['is_merged']]
    print(f"## NOT MERGED ({len(not_merged)} branches)")
    print(f"Branches with unmerged changes\n")

    # Show old branches with minimal changes
    old_minimal = [b for b in not_merged if b['age_days'] > 365 and b['ahead'] <= 100]
    if old_minimal:
        print(f"### Old branches with minimal changes ({len(old_minimal)} branches):")
        for b in old_minimal[:20]:
            print(f"- `{b['branch']}` - {b['ahead']} ahead, {b['age_days']} days old - {b['author']}")
        print()

    # Statistics
    print("\n## STATISTICS\n")
    print(f"Total branches: {len(results)}")
    print(f"- Merged (safe to delete): {len(merged)}")
    print(f"  - Traditional merges: {len(trad_merged)}")
    print(f"  - Squash merges: {len(squash_merged)}")
    print(f"  - No differences: {len(no_diff)}")
    print(f"- Not merged: {len(not_merged)}")


if __name__ == '__main__':
    main()

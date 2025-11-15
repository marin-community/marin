#!/usr/bin/env python3
"""
Analyze stale branches - FINAL VERSION.
Uses origin/rw/main (old main) for merge-base checks since main's history was rewritten.
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
    """Get list of all remote branches except HEAD and main variants."""
    output, _ = run_git("git for-each-ref --format='%(refname:short)' refs/remotes/origin")
    branches = []
    for line in output.split('\n'):
        line = line.strip()
        if line and line not in ('origin/HEAD', 'origin/main', 'origin/rw/main'):
            branches.append(line)
    return branches


def check_merge_status(branch: str) -> Tuple[bool, str]:
    """
    Check if branch is merged using BOTH current main and old main (rw/main).
    """
    # Method 1: Check against current main (for recent merges)
    _, code = run_git(f"git merge-base --is-ancestor {branch} origin/main")
    if code == 0:
        return True, 'merged_into_current_main'

    # Method 2: Check if no commits ahead of current main
    ahead_output, _ = run_git(f"git log origin/main..{branch} --oneline")
    ahead_count = len([l for l in ahead_output.split('\n') if l.strip()])
    if ahead_count == 0:
        return True, 'no_commits_ahead_of_main'

    # Method 3: Check against old main (rw/main) and see if branch merged there
    old_main_base, code = run_git(f"git merge-base {branch} origin/rw/main")
    if code == 0 and old_main_base:
        # Has common ancestor with old main
        # Check if branch is ancestor of old main
        _, code = run_git(f"git merge-base --is-ancestor {branch} origin/rw/main")
        if code == 0:
            return True, 'merged_into_old_main'

        # Check commits ahead of old main
        old_ahead_output, _ = run_git(f"git log origin/rw/main..{branch} --oneline")
        old_ahead = len([l for l in old_ahead_output.split('\n') if l.strip()])

        if old_ahead == 0:
            return True, 'no_commits_ahead_of_old_main'

    # Method 4: Content comparison with current main
    diff_output, _ = run_git(f"git diff --stat origin/main {branch}")
    if not diff_output.strip():
        return True, 'identical_content_to_main'

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

    # Get ahead/behind counts relative to CURRENT main
    ahead_output, _ = run_git(f"git log origin/main..{branch} --oneline")
    ahead = len([l for l in ahead_output.split('\n') if l.strip()])

    behind_output, _ = run_git(f"git log {branch}..origin/main --oneline")
    behind = len([l for l in behind_output.split('\n') if l.strip()])

    # Check merge-base status with both mains
    _, mb_current = run_git(f"git merge-base {branch} origin/main")
    _, mb_old = run_git(f"git merge-base {branch} origin/rw/main")

    return {
        'branch': clean_name,
        'ahead': ahead,
        'behind': behind,
        'is_merged': is_merged,
        'merge_type': merge_type,
        'has_mb_current': mb_current == 0,
        'has_mb_old': mb_old == 0,
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

    # Test with ksalahi/openqa first
    test_branches = ['origin/ksalahi/openqa', 'origin/og_alpaca', 'origin/chris/fix-alpaca']
    print("Testing with example branches:")
    for test_branch in test_branches:
        if test_branch in branches:
            metadata = get_branch_metadata(test_branch)
            if metadata:
                print(f"\n{test_branch}:")
                print(f"  is_merged: {metadata['is_merged']}")
                print(f"  merge_type: {metadata['merge_type']}")
                print(f"  ahead: {metadata['ahead']}, behind: {metadata['behind']}")
                print(f"  has_mb_current: {metadata['has_mb_current']}, has_mb_old: {metadata['has_mb_old']}")
    print()

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
    with open('/tmp/final_branch_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to /tmp/final_branch_analysis.json\n")

    # Generate report
    print("=" * 80)
    print("STALE BRANCH ANALYSIS - FINAL")
    print("=" * 80)
    print()

    # Merged branches
    merged = [b for b in results if b['is_merged']]
    print(f"## MERGED BRANCHES ({len(merged)} branches)")
    print("These are fully merged and safe to delete:\n")

    # Group by merge type
    for merge_type in ['merged_into_current_main', 'no_commits_ahead_of_main', 'merged_into_old_main',
                       'no_commits_ahead_of_old_main', 'identical_content_to_main']:
        type_branches = [b for b in merged if b['merge_type'] == merge_type]
        if type_branches:
            display_name = merge_type.replace('_', ' ').title()
            print(f"### {display_name} ({len(type_branches)} branches):")
            for b in type_branches[:20]:
                mb_status = f"MB:{'✓C' if b['has_mb_current'] else '✗C'}/{'✓O' if b['has_mb_old'] else '✗O'}"
                print(f"- `{b['branch']}` [{mb_status}] - {b['ahead']} ahead, {b['behind']} behind - {b['date'][:10]} - {b['author']}")
            if len(type_branches) > 20:
                print(f"  ... and {len(type_branches) - 20} more")
            print()

    # Not merged
    not_merged = [b for b in results if not b['is_merged']]
    print(f"## NOT MERGED ({len(not_merged)} branches)")
    print("Branches with unmerged changes\n")

    # Show old branches that might be stale
    old_branches = [b for b in not_merged if b['age_days'] > 180]
    if old_branches:
        print(f"### Old branches (180+ days): {len(old_branches)} branches")
        # Group by commits ahead
        for threshold in [(1, 50), (51, 100), (101, 999999)]:
            start, end = threshold
            in_range = [b for b in old_branches if start <= b['ahead'] <= end]
            if in_range:
                print(f"\n  {start}-{end if end < 999999 else '+'} commits ahead:")
                for b in in_range[:10]:
                    print(f"    - `{b['branch']}` - {b['ahead']} ahead, {b['age_days']} days - {b['author']}")
                if len(in_range) > 10:
                    print(f"      ... and {len(in_range) - 10} more")
        print()

    # Statistics
    print("\n## STATISTICS\n")
    print(f"Total branches: {len(results)}")
    print(f"- Merged (safe to delete): {len(merged)}")
    print(f"- Not merged: {len(not_merged)}")
    print()
    print(f"Merge-base status:")
    with_mb_current = len([b for b in results if b['has_mb_current']])
    with_mb_old = len([b for b in results if b['has_mb_old']])
    print(f"- With merge-base to current main: {with_mb_current}")
    print(f"- With merge-base to old main (rw/main): {with_mb_old}")
    print()

    # Generate deletion script
    if merged:
        print(f"\n## DELETION SCRIPT ({len(merged)} branches)\n")
        print("# Review before running!")
        print("# These branches are fully merged:\n")
        for b in merged[:50]:
            print(f"git push origin --delete {b['branch']}")
        if len(merged) > 50:
            print(f"\n# ... and {len(merged) - 50} more (see JSON file)")


if __name__ == '__main__':
    main()

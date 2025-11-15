#!/usr/bin/env python3
"""
Analyze stale branches using content-based comparison.
Works even when git history has been rewritten and merge-base doesn't exist.
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
            timeout=120
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


def check_if_content_merged(branch: str) -> Tuple[bool, str]:
    """
    Check if branch content is merged into main using tree comparison.
    This works even without a common merge-base.

    Returns: (is_merged, method)
    """
    # Method 1: Try traditional ancestor check first
    _, code = run_git(f"git merge-base --is-ancestor {branch} origin/main")
    if code == 0:
        return True, 'ancestor'

    # Method 2: Check if branch has any commits not in main (by commit hash)
    ahead_output, _ = run_git(f"git log origin/main..{branch} --oneline")
    ahead_commits = [l for l in ahead_output.split('\n') if l.strip()]

    if not ahead_commits:
        # No commits ahead = fully merged
        return True, 'no_unique_commits'

    # Method 3: Content-based comparison
    # Compare the actual file trees directly (two-dot diff: FROM main TO branch)
    # If there are no differences, the branch content is in main
    diff_output, code = run_git(f"git diff --stat origin/main {branch}")

    if not diff_output.strip():
        # No differences in content = merged (squash or otherwise)
        return True, 'identical_content'

    # Method 4: Check if branch would merge cleanly with no changes
    # Use diff-tree to compare tree objects directly
    main_tree, _ = run_git("git rev-parse origin/main^{tree}")
    branch_tree, _ = run_git(f"git rev-parse {branch}^{{tree}}")

    if main_tree == branch_tree:
        # Exact same tree = definitely merged
        return True, 'same_tree'

    # Method 5: Check number of differing files
    # If very few files differ and they're minor, might be effectively merged
    files_output, _ = run_git(f"git diff --name-only origin/main {branch}")
    files_changed = [f for f in files_output.split('\n') if f.strip()]

    if len(files_changed) == 0:
        return True, 'no_file_changes'

    # Not merged
    return False, f'{len(files_changed)}_files_differ'


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

    # Check if merge-base exists
    _, mb_code = run_git(f"git merge-base {branch} origin/main")
    has_merge_base = (mb_code == 0)

    # Check merge status using content comparison
    is_merged, method = check_if_content_merged(branch)

    # Get ahead/behind counts (may not be accurate without merge-base)
    ahead_output, _ = run_git(f"git log origin/main..{branch} --oneline")
    ahead = len([l for l in ahead_output.split('\n') if l.strip()])

    behind_output, _ = run_git(f"git log {branch}..origin/main --oneline")
    behind = len([l for l in behind_output.split('\n') if l.strip()])

    return {
        'branch': clean_name,
        'ahead': ahead,
        'behind': behind,
        'has_merge_base': has_merge_base,
        'is_merged': is_merged,
        'merge_method': method,
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
    test_branch = 'origin/ksalahi/openqa'
    if test_branch in branches:
        print(f"Testing content-based detection on {test_branch}...")
        metadata = get_branch_metadata(test_branch)
        if metadata:
            print(f"  has_merge_base: {metadata['has_merge_base']}")
            print(f"  is_merged: {metadata['is_merged']}")
            print(f"  method: {metadata['merge_method']}")
            print(f"  ahead: {metadata['ahead']}, behind: {metadata['behind']}\n")

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
    with open('/tmp/content_based_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to /tmp/content_based_analysis.json\n")

    # Generate report
    print("=" * 80)
    print("STALE BRANCH ANALYSIS - CONTENT-BASED DETECTION")
    print("=" * 80)
    print()

    # Merged branches
    merged = [b for b in results if b['is_merged']]
    print(f"## MERGED BRANCHES ({len(merged)} branches)")
    print("These have identical or merged content into main:\n")

    # Group by method
    for method_name in ['ancestor', 'no_unique_commits', 'identical_content', 'same_tree', 'no_file_changes']:
        method_branches = [b for b in merged if b['merge_method'] == method_name]
        if method_branches:
            print(f"### {method_name.replace('_', ' ').title()} ({len(method_branches)} branches):")
            for b in method_branches[:15]:
                mb = "✓" if b['has_merge_base'] else "✗"
                print(f"- `{b['branch']}` [MB:{mb}] - {b['ahead']} ahead, {b['behind']} behind - {b['date'][:10]} - {b['author']}")
            if len(method_branches) > 15:
                print(f"  ... and {len(method_branches) - 15} more")
            print()

    # Not merged
    not_merged = [b for b in results if not b['is_merged']]
    print(f"## NOT MERGED ({len(not_merged)} branches)")
    print(f"Branches with unique content not in main\n")

    # Show old branches with minimal differences
    minimal_changes = [b for b in not_merged if b['age_days'] > 180 and 'files_differ' in b['merge_method']]
    if minimal_changes:
        print(f"### Old branches (180+ days) with file differences:")
        for b in minimal_changes[:20]:
            mb = "✓" if b['has_merge_base'] else "✗"
            print(f"- `{b['branch']}` [MB:{mb}] - {b['merge_method']} - {b['age_days']} days - {b['author']}")
        if len(minimal_changes) > 20:
            print(f"  ... and {len(minimal_changes) - 20} more")
        print()

    # Statistics
    print("\n## STATISTICS\n")
    print(f"Total branches: {len(results)}")
    print(f"- Merged (safe to delete): {len(merged)}")
    print(f"- Not merged: {len(not_merged)}")
    print()
    print(f"Merge-base status:")
    with_mb = len([b for b in results if b['has_merge_base']])
    without_mb = len([b for b in results if not b['has_merge_base']])
    print(f"- With merge-base: {with_mb}")
    print(f"- Without merge-base (orphaned): {without_mb}")
    print()

    # Deletion candidates
    safe_to_delete = [b for b in merged if b['age_days'] > 30]
    if safe_to_delete:
        print(f"\n## DELETION CANDIDATES ({len(safe_to_delete)} branches)\n")
        print("Merged branches older than 30 days (review before deleting):\n")
        for b in safe_to_delete[:30]:
            print(f"git push origin --delete {b['branch']}")
        if len(safe_to_delete) > 30:
            print(f"\n# ... and {len(safe_to_delete) - 30} more (see JSON file)")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Analyze stale branches in the repository.
Identifies branches that can be safely deleted based on merge status and age.
"""

import subprocess
import re
import json
from datetime import datetime
from collections import defaultdict
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


def check_if_merged_into_main(branch: str) -> Tuple[bool, str]:
    """
    Check if a branch is fully merged into main.
    Uses multiple methods to detect squash merges and regular merges.

    Returns: (is_merged, merge_type)
    merge_type: 'ancestor' | 'squash' | 'no_diff' | 'not_merged'
    """
    # Method 1: Check if branch tip is ancestor of main (regular merge/rebase)
    _, code = run_git(f"git merge-base --is-ancestor {branch} origin/main")
    if code == 0:
        return True, 'ancestor'

    # Method 2: Check for commits on branch not in main
    output, _ = run_git(f"git log origin/main..{branch} --oneline")
    commits_not_in_main = [l for l in output.split('\n') if l.strip()]

    if not commits_not_in_main:
        return True, 'no_diff'

    # Method 3: Detect squash merges by comparing file trees
    # If the trees are identical, the content was squash-merged
    diff_output, _ = run_git(f"git diff origin/main..{branch} --stat")

    if not diff_output.strip():
        # No file differences = squash merged or identical content
        return True, 'squash'

    # Method 4: Check if diff is minimal (just whitespace/formatting)
    diff_name_only, _ = run_git(f"git diff origin/main..{branch} --name-only")
    files_changed = [f for f in diff_name_only.split('\n') if f.strip()]

    if not files_changed:
        return True, 'squash'

    # Has meaningful differences - not merged
    return False, 'not_merged'


def count_commits_ahead_behind(branch: str) -> Tuple[int, int]:
    """
    Count commits ahead and behind main.
    Returns: (ahead_count, behind_count)
    """
    # Commits on branch not in main (ahead)
    ahead_output, _ = run_git(f"git log origin/main..{branch} --oneline")
    ahead = len([l for l in ahead_output.split('\n') if l.strip()])

    # Commits on main not in branch (behind)
    behind_output, _ = run_git(f"git log {branch}..origin/main --oneline")
    behind = len([l for l in behind_output.split('\n') if l.strip()])

    return ahead, behind


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
    except:
        return None

    # Determine age category
    one_year_ago = datetime(2024, 5, 15)
    six_months_ago = datetime(2024, 11, 15)

    if date < one_year_ago:
        age = 'VERY_OLD'
    elif date < six_months_ago:
        age = 'OLD'
    else:
        age = 'RECENT'

    # Check merge status and counts
    is_merged, merge_type = check_if_merged_into_main(branch)
    ahead, behind = count_commits_ahead_behind(branch)

    return {
        'branch': clean_name,
        'ahead': ahead,
        'behind': behind,
        'is_fully_merged': is_merged,
        'merge_type': merge_type,
        'date': date_str,
        'date_obj': date,
        'author': author,
        'age': age,
    }


def analyze_all_branches() -> List[Dict]:
    """Analyze all branches and return sorted results."""
    print("Fetching latest from origin...")
    run_git("git fetch origin --prune")

    print("Getting branch list...")
    branches = get_remote_branches()
    print(f"Found {len(branches)} branches to analyze\n")

    results = []
    for i, branch in enumerate(branches):
        if i % 50 == 0:
            print(f"Progress: {i}/{len(branches)}")

        metadata = get_branch_metadata(branch)
        if metadata:
            results.append(metadata)

    print(f"\nAnalyzed {len(results)} branches successfully\n")

    # Sort: fully merged first, then by ahead count, then by date
    results.sort(key=lambda x: (not x['is_fully_merged'], x['ahead'], x['date_obj']))

    return results


def generate_report(results: List[Dict]) -> str:
    """Generate a detailed report."""
    lines = []

    lines.append("# STALE BRANCH ANALYSIS REPORT")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Total Branches:** {len(results)}")
    lines.append("")

    # Fully merged branches
    fully_merged = [b for b in results if b['is_fully_merged']]
    lines.append(f"## PRIORITY 1: Fully Merged Branches ({len(fully_merged)} branches)")
    lines.append("")
    lines.append("These branches are completely merged into main and can be safely deleted:")
    lines.append("")

    if fully_merged:
        # Group by merge type first
        squash_merged = [b for b in fully_merged if b.get('merge_type') in ['squash', 'no_diff']]
        ancestor_merged = [b for b in fully_merged if b.get('merge_type') == 'ancestor']

        if squash_merged:
            lines.append(f"### Squash-merged branches ({len(squash_merged)} branches)")
            lines.append("")
            lines.append("These were squash-merged (content is in main, commits are not):")
            lines.append("")
            for b in squash_merged:
                lines.append(f"- `{b['branch']}` - {b['behind']} behind, {b['ahead']} commits (squashed) - {b['date'][:10]} - by {b['author']}")
            lines.append("")

        if ancestor_merged:
            lines.append(f"### Traditionally merged branches ({len(ancestor_merged)} branches)")
            lines.append("")
            for b in ancestor_merged:
                lines.append(f"- `{b['branch']}` - {b['behind']} behind - {b['date'][:10]} - by {b['author']}")
            lines.append("")
    else:
        lines.append("**No fully merged branches found.**")
        lines.append("")

    # Not merged with minimal changes
    not_merged_minimal = [b for b in results if not b['is_fully_merged'] and b['ahead'] <= 50]
    lines.append(f"## PRIORITY 2: Branches with Minimal Changes ({len(not_merged_minimal)} branches)")
    lines.append("")
    lines.append("Branches with 1-50 commits ahead (may be abandoned experiments):")
    lines.append("")

    for age, age_label in [('VERY_OLD', '>1 year'), ('OLD', '6-12 months'), ('RECENT', '<6 months')]:
        age_branches = [b for b in not_merged_minimal if b['age'] == age]
        if age_branches:
            lines.append(f"### {age_label} ({len(age_branches)} branches)")
            lines.append("")
            for b in age_branches[:30]:
                lines.append(f"- **{b['ahead']} ahead, {b['behind']} behind** | `{b['branch']}` | {b['date'][:10]} | by {b['author']}")
            if len(age_branches) > 30:
                lines.append(f"\n  ... and {len(age_branches) - 30} more")
            lines.append("")

    # Statistics
    lines.append("## STATISTICS")
    lines.append("")
    lines.append("### By Merge Status:")
    lines.append(f"- Fully merged: {len(fully_merged)} branches")
    lines.append(f"- Not merged: {len(results) - len(fully_merged)} branches")
    lines.append("")

    lines.append("### By Age:")
    for age, label in [('VERY_OLD', '>1 year'), ('OLD', '6-12 months'), ('RECENT', '<6 months')]:
        count = sum(1 for b in results if b['age'] == age)
        lines.append(f"- {label}: {count} branches")
    lines.append("")

    lines.append("### By Commits Ahead (not merged only):")
    not_merged = [b for b in results if not b['is_fully_merged']]
    for start, end in [(1, 10), (11, 50), (51, 100), (101, 500), (501, 1000), (1001, 5000), (5001, 999999)]:
        count = sum(1 for b in not_merged if start <= b['ahead'] <= end)
        if count > 0:
            lines.append(f"- {start}-{end if end < 999999 else '+'} commits: {count} branches")

    return '\n'.join(lines)


def main():
    """Main entry point."""
    # Analyze branches
    results = analyze_all_branches()

    # Save JSON results
    json_results = []
    for r in results:
        r_copy = r.copy()
        r_copy['date_obj'] = r_copy['date_obj'].isoformat()
        json_results.append(r_copy)

    with open('/tmp/branch_analysis_results.json', 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"Saved detailed results to /tmp/branch_analysis_results.json\n")

    # Generate and print report
    report = generate_report(results)
    print(report)

    # Save report
    with open('STALE_BRANCHES_REPORT.md', 'w') as f:
        f.write(report)
    print(f"\n✓ Report saved to STALE_BRANCHES_REPORT.md")


if __name__ == '__main__':
    main()

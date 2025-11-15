#!/usr/bin/env python3
"""
Analyze stale branches - practical approach.
Focus on identifying branches safe to delete based on:
1. Traditional merge detection (commit ancestry)
2. Age and abandonment indicators
3. Pattern matching (bot/temp branches)
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


def check_merge_status(branch: str) -> Tuple[str, str]:
    """
    Check merge status of a branch.
    Returns: (status, details)
    status: 'merged' | 'probably_merged' | 'active' | 'stale'
    """
    # Check 1: Is it an ancestor of main? (traditional merge)
    _, code = run_git(f"git merge-base --is-ancestor {branch} origin/main")
    if code == 0:
        return 'merged', 'traditionally merged (ancestor)'

    # Check 2: Are there commits not in main?
    ahead_output, _ = run_git(f"git log origin/main..{branch} --oneline")
    ahead_commits = [l for l in ahead_output.split('\n') if l.strip()]
    ahead_count = len(ahead_commits)

    if ahead_count == 0:
        return 'merged', 'no unique commits'

    # Check 3: Behind count
    behind_output, _ = run_git(f"git log {branch}..origin/main --oneline")
    behind_commits = [l for l in behind_output.split('\n') if l.strip()]
    behind_count = len(behind_commits)

    # Check 4: Date
    date_str, _ = run_git(f"git log -1 --format='%ci' {branch}")
    try:
        date_clean = re.sub(r'\s+[+-]\d{4}$', '', date_str)
        date = datetime.strptime(date_clean, '%Y-%m-%d %H:%M:%S')
        age_days = (datetime.now() - date).days
    except:
        age_days = 0

    # Classification logic (adjusted for this repo's patterns)
    if ahead_count == 0:
        return 'merged', 'no differences'
    elif ahead_count <= 10 and age_days > 180:
        return 'stale', f'{ahead_count} commits, {age_days} days old'
    elif ahead_count <= 50 and age_days > 365:
        return 'stale', f'{ahead_count} commits, {age_days} days old'
    elif ahead_count <= 100 and age_days > 365:
        return 'probably_stale', f'{ahead_count} commits, {age_days} days old'
    elif age_days > 365:
        return 'old', f'{ahead_count} commits, {age_days} days old'
    else:
        return 'active', f'{ahead_count} ahead, {behind_count} behind'


def is_bot_or_automated(branch: str) -> bool:
    """Check if branch is from a bot or automated system."""
    bot_prefixes = ['claude/', 'codex/', 'gh/', 'actions/', 'dependabot/', 'renovate/']
    return any(branch.replace('origin/', '').startswith(prefix) for prefix in bot_prefixes)


def is_temp_or_test(branch: str) -> bool:
    """Check if branch appears to be temporary or test."""
    patterns = ['wip', 'temp', 'test', 'tmp', 'experiment', 'exp_', 'fix_', 'bug_']
    clean_name = branch.replace('origin/', '').lower()
    return any(pattern in clean_name for pattern in patterns)


def analyze_branch(branch: str) -> Optional[Dict]:
    """Analyze a single branch."""
    clean_name = branch.replace('origin/', '')

    # Get metadata
    date_str, _ = run_git(f"git log -1 --format='%ci' {branch}")
    author, _ = run_git(f"git log -1 --format='%an' {branch}")

    if not date_str:
        return None

    try:
        date_clean = re.sub(r'\s+[+-]\d{4}$', '', date_str)
        date = datetime.strptime(date_clean, '%Y-%m-%d %H:%M:%S')
        age_days = (datetime.now() - date).days
    except:
        return None

    # Get ahead/behind counts
    ahead_output, _ = run_git(f"git log origin/main..{branch} --oneline")
    ahead = len([l for l in ahead_output.split('\n') if l.strip()])

    behind_output, _ = run_git(f"git log {branch}..origin/main --oneline")
    behind = len([l for l in behind_output.split('\n') if l.strip()])

    # Check merge status
    status, details = check_merge_status(branch)

    # Categorization
    is_bot = is_bot_or_automated(branch)
    is_temp = is_temp_or_test(branch)

    # Deletion recommendation
    if status == 'merged':
        recommendation = 'DELETE'
        priority = 1
    elif status == 'stale' and (is_bot or is_temp):
        recommendation = 'DELETE'
        priority = 2
    elif status == 'stale':
        recommendation = 'REVIEW'
        priority = 3
    elif status == 'probably_stale' and is_temp:
        recommendation = 'REVIEW'
        priority = 3
    elif status == 'old' and is_temp and age_days > 365:
        recommendation = 'REVIEW'
        priority = 4
    elif status == 'old' and is_bot and age_days > 180:
        recommendation = 'REVIEW'
        priority = 4
    else:
        recommendation = 'KEEP'
        priority = 5

    return {
        'branch': clean_name,
        'ahead': ahead,
        'behind': behind,
        'status': status,
        'details': details,
        'date': date_str,
        'age_days': age_days,
        'author': author,
        'is_bot': is_bot,
        'is_temp': is_temp,
        'recommendation': recommendation,
        'priority': priority,
    }


def main():
    """Main entry point."""
    print("Fetching latest from origin...")
    run_git("git fetch origin --prune")

    print("Getting branch list...")
    branches = get_remote_branches()
    print(f"Found {len(branches)} branches to analyze\n")

    results = []
    for i, branch in enumerate(branches):
        if i % 50 == 0:
            print(f"Progress: {i}/{len(branches)}")

        data = analyze_branch(branch)
        if data:
            results.append(data)

    print(f"\nAnalyzed {len(results)} branches successfully\n")

    # Sort by priority, then age
    results.sort(key=lambda x: (x['priority'], -x['age_days']))

    # Save JSON
    json_results = [r for r in results]
    with open('/tmp/stale_branches_v2.json', 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"Saved results to /tmp/stale_branches_v2.json\n")

    # Generate report
    print("=" * 80)
    print("STALE BRANCH ANALYSIS - PRACTICAL APPROACH")
    print("=" * 80)
    print()

    # Priority 1: Merged branches (safe to delete)
    merged = [b for b in results if b['recommendation'] == 'DELETE' and b['status'] == 'merged']
    print(f"## PRIORITY 1: MERGED BRANCHES ({len(merged)} branches)")
    print("These are fully merged into main and safe to delete:\n")
    for b in merged[:30]:
        print(f"- `{b['branch']}` - {b['details']} - {b['date'][:10]} - {b['author']}")
    if len(merged) > 30:
        print(f"\n... and {len(merged) - 30} more")
    print()

    # Priority 2: Stale bot/temp branches
    stale_bot = [b for b in results if b['recommendation'] == 'DELETE' and b['status'] == 'stale']
    print(f"## PRIORITY 2: STALE BOT/TEMP BRANCHES ({len(stale_bot)} branches)")
    print("Automated or temporary branches that appear abandoned:\n")
    for b in stale_bot[:30]:
        print(f"- `{b['branch']}` - {b['details']} - {b['date'][:10]}")
    if len(stale_bot) > 30:
        print(f"\n... and {len(stale_bot) - 30} more")
    print()

    # Priority 3: Stale user branches (need review)
    stale_review = [b for b in results if b['recommendation'] == 'REVIEW']
    print(f"## PRIORITY 3: STALE BRANCHES NEEDING REVIEW ({len(stale_review)} branches)")
    print("Old branches that might be abandoned but need owner confirmation:\n")
    for b in stale_review[:30]:
        print(f"- `{b['branch']}` - {b['ahead']} ahead, {b['age_days']} days old - by {b['author']}")
    if len(stale_review) > 30:
        print(f"\n... and {len(stale_review) - 30} more")
    print()

    # Statistics
    print("## STATISTICS\n")
    print(f"Total branches: {len(results)}")
    print(f"- Safe to DELETE: {len([b for b in results if b['recommendation'] == 'DELETE'])}")
    print(f"- Need REVIEW: {len([b for b in results if b['recommendation'] == 'REVIEW'])}")
    print(f"- Should KEEP: {len([b for b in results if b['recommendation'] == 'KEEP'])}")
    print()
    print(f"By status:")
    print(f"- Merged: {len([b for b in results if b['status'] == 'merged'])}")
    print(f"- Stale: {len([b for b in results if b['status'] == 'stale'])}")
    print(f"- Active: {len([b for b in results if b['status'] == 'active'])}")
    print()

    # Generate deletion script
    delete_branches = [b for b in results if b['recommendation'] == 'DELETE']
    if delete_branches:
        print("\n## DELETION SCRIPT\n")
        print("# Review this list before running!")
        print("# To delete these branches, run:\n")
        for b in delete_branches[:50]:
            print(f"git push origin --delete {b['branch']}")
        if len(delete_branches) > 50:
            print(f"\n# ... and {len(delete_branches) - 50} more (see JSON file)")


if __name__ == '__main__':
    main()

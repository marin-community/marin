#!/usr/bin/env python3
# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Check which users with write access have submitted PRs recently."""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone


def get_collaborators_with_write_access(repo: str) -> set[str]:
    """Get all collaborators with push (write) permissions."""
    result = subprocess.run(
        ["gh", "api", f"repos/{repo}/collaborators", "--paginate"],
        capture_output=True,
        text=True,
        check=True,
    )
    collaborators = json.loads(result.stdout)
    return {c["login"] for c in collaborators if c["permissions"]["push"]}


def get_recent_pr_authors(repo: str, days: int) -> dict[str, int]:
    """Get authors who created PRs in the past N days and their PR counts."""
    cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    result = subprocess.run(
        [
            "gh",
            "pr",
            "list",
            "--repo",
            repo,
            "--limit",
            "1000",
            "--state",
            "all",
            "--json",
            "author,createdAt",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    prs = json.loads(result.stdout)

    author_counts: dict[str, int] = {}
    for pr in prs:
        if pr["createdAt"] >= cutoff_date:
            author = pr["author"]["login"]
            author_counts[author] = author_counts.get(author, 0) + 1

    return author_counts


def remove_collaborator(repo: str, username: str, dry_run: bool) -> bool:
    """Remove a collaborator from the repository."""
    if dry_run:
        print(f"  [DRY RUN] Would remove: {username}")
        return True

    try:
        subprocess.run(
            ["gh", "api", "-X", "DELETE", f"repos/{repo}/collaborators/{username}"],
            capture_output=True,
            text=True,
            check=True,
        )
        print(f"  ✓ Removed: {username}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Failed to remove {username}: {e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Check which users with write access have submitted PRs recently")
    parser.add_argument(
        "--repo",
        default="marin-community/marin",
        help="Repository in owner/name format (default: marin-community/marin)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Number of days to look back (default: 90)",
    )
    parser.add_argument(
        "--remove-inactive",
        action="store_true",
        help="Remove inactive users from the repository",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without actually removing",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Exclude specific users from removal (can be specified multiple times)",
    )
    args = parser.parse_args()

    print(f"Fetching collaborators with write access to {args.repo}...")
    write_users = get_collaborators_with_write_access(args.repo)

    print(f"Fetching PRs from the past {args.days} days...")
    pr_authors = get_recent_pr_authors(args.repo, args.days)

    # Find active users (write access + recent PRs)
    active_users = {user: pr_authors[user] for user in write_users if user in pr_authors}

    # Find inactive users (write access but no recent PRs)
    inactive_users = write_users - set(pr_authors.keys())

    print(f"\n{'='*60}")
    print(f"Results for {args.repo} (past {args.days} days)")
    print(f"{'='*60}\n")

    print(f"Active users with write access ({len(active_users)}):")
    for user in sorted(active_users.keys(), key=lambda u: active_users[u], reverse=True):
        print(f"  {user}: {active_users[user]} PRs")

    print(f"\nInactive users with write access ({len(inactive_users)}):")
    for user in sorted(inactive_users):
        print(f"  {user}")

    print("\nSummary:")
    print(f"  Total users with write access: {len(write_users)}")
    print(f"  Active (submitted PRs): {len(active_users)}")
    print(f"  Inactive (no PRs): {len(inactive_users)}")

    # Remove inactive users if requested
    if args.remove_inactive:
        # Apply exclusions
        users_to_remove = inactive_users - set(args.exclude)
        excluded = inactive_users & set(args.exclude)

        if excluded:
            print(f"\nExcluding {len(excluded)} user(s) from removal:")
            for user in sorted(excluded):
                print(f"  {user}")

        if not users_to_remove:
            print("\nNo users to remove after applying exclusions.")
            return

        print(f"\n{'='*60}")
        if args.dry_run:
            print("DRY RUN: Removing inactive users")
        else:
            print("Removing inactive users")
        print(f"{'='*60}\n")

        # Confirm if not dry run
        if not args.dry_run:
            print(f"About to remove {len(users_to_remove)} user(s):")
            for user in sorted(users_to_remove):
                print(f"  - {user}")
            response = input("\nAre you sure you want to proceed? (yes/no): ")
            if response.lower() != "yes":
                print("Cancelled.")
                sys.exit(0)

        # Remove users
        success_count = 0
        fail_count = 0
        for user in sorted(users_to_remove):
            if remove_collaborator(args.repo, user, args.dry_run):
                success_count += 1
            else:
                fail_count += 1

        print(f"\n{'='*60}")
        if args.dry_run:
            print(f"DRY RUN: Would remove {success_count} user(s)")
        else:
            print(f"Successfully removed: {success_count}")
            if fail_count > 0:
                print(f"Failed to remove: {fail_count}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()

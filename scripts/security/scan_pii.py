# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Lightweight PII scanner for the Marin codebase.

Scans git-tracked files for patterns that may indicate PII exposure:
email addresses, hardcoded API keys/tokens, private key material, and SSN patterns.

Uses a YAML allowlist to suppress known-acceptable findings (e.g., contributor
attribution in CONTRIBUTORS.md, academic emails in test fixtures from public
arxiv papers).

Usage:
    uv run scripts/security/scan_pii.py
    uv run scripts/security/scan_pii.py --allowlist scripts/security/pii_allowlist.yaml
"""

import argparse
import fnmatch
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "api_key": re.compile(
        r"(?:"
        r"sk-[a-zA-Z0-9]{20,}"  # OpenAI-style
        r"|ghp_[a-zA-Z0-9]{36,}"  # GitHub PAT
        r"|glpat-[a-zA-Z0-9\-]{20,}"  # GitLab PAT
        r"|AKIA[0-9A-Z]{16}"  # AWS access key
        r"|xox[bprs]-[a-zA-Z0-9\-]+"  # Slack tokens
        r")"
    ),
    "private_key": re.compile(r"-----BEGIN\s+(?:RSA\s+|EC\s+|DSA\s+|OPENSSH\s+)?PRIVATE\s+KEY-----"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
}

SKIP_EXTENSIONS = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".ico",
        ".svg",
        ".webp",
        ".woff",
        ".woff2",
        ".ttf",
        ".eot",
        ".zip",
        ".tar",
        ".gz",
        ".bz2",
        ".xz",
        ".pyc",
        ".pyo",
        ".so",
        ".dylib",
        ".dll",
        ".pdf",
        ".parquet",
        ".arrow",
        ".npy",
        ".npz",
        ".mov",
        ".mp4",
        ".avi",
        ".mkv",
        ".lock",
    }
)


@dataclass(frozen=True)
class Finding:
    file: str
    line_number: int
    pattern_name: str
    matched_text: str

    @property
    def key(self) -> str:
        """Stable key used for allowlist matching: file::pattern_name::matched_text."""
        return f"{self.file}::{self.pattern_name}::{self.matched_text}"


@dataclass(frozen=True)
class AllowlistEntry:
    file_glob: str
    pattern_name: str
    reason: str


def load_allowlist(path: Path) -> list[AllowlistEntry]:
    if not path.exists():
        return []
    with open(path) as f:
        data = yaml.safe_load(f)
    if not data or "allowlist" not in data:
        return []
    return [
        AllowlistEntry(
            file_glob=entry["file_glob"],
            pattern_name=entry["pattern_name"],
            reason=entry.get("reason", ""),
        )
        for entry in data["allowlist"]
    ]


def is_allowed(finding: Finding, allowlist: list[AllowlistEntry]) -> bool:
    for entry in allowlist:
        if entry.pattern_name != finding.pattern_name:
            continue
        if fnmatch.fnmatch(finding.file, entry.file_glob):
            return True
    return False


def git_tracked_files(repo_root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return [f for f in result.stdout.strip().split("\n") if f]


def scan_file(filepath: Path, relative: str) -> list[Finding]:
    if filepath.suffix.lower() in SKIP_EXTENSIONS:
        return []
    try:
        content = filepath.read_text(errors="ignore")
    except (OSError, UnicodeDecodeError):
        return []

    findings: list[Finding] = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        for pattern_name, pattern in PII_PATTERNS.items():
            for match in pattern.finditer(line):
                findings.append(
                    Finding(
                        file=relative,
                        line_number=line_number,
                        pattern_name=pattern_name,
                        matched_text=match.group(0),
                    )
                )
    return findings


def scan_repo(
    repo_root: Path,
    allowlist_path: Path | None = None,
) -> tuple[list[Finding], list[Finding]]:
    """Scan the repo and return (unallowed_findings, allowed_findings)."""
    if allowlist_path is None:
        allowlist_path = repo_root / "scripts" / "security" / "pii_allowlist.yaml"
    allowlist = load_allowlist(allowlist_path)
    files = git_tracked_files(repo_root)

    all_findings: list[Finding] = []
    for relative in files:
        filepath = repo_root / relative
        all_findings.extend(scan_file(filepath, relative))

    unallowed = [f for f in all_findings if not is_allowed(f, allowlist)]
    allowed = [f for f in all_findings if is_allowed(f, allowlist)]
    return unallowed, allowed


def main() -> int:

    parser = argparse.ArgumentParser(description="Scan for PII in tracked files")
    parser.add_argument(
        "--allowlist",
        type=Path,
        default=REPO_ROOT / "scripts" / "security" / "pii_allowlist.yaml",
        help="Path to YAML allowlist file",
    )
    parser.add_argument("--show-allowed", action="store_true", help="Also print allowed findings")
    args = parser.parse_args()

    unallowed, allowed = scan_repo(REPO_ROOT, args.allowlist)

    if args.show_allowed and allowed:
        print(f"\n--- {len(allowed)} allowed findings (suppressed by allowlist) ---")
        for f in allowed:
            print(f"  {f.file}:{f.line_number} [{f.pattern_name}] {f.matched_text}")

    if unallowed:
        print(f"\n!!! {len(unallowed)} PII finding(s) NOT in allowlist !!!\n")
        for f in unallowed:
            print(f"  {f.file}:{f.line_number} [{f.pattern_name}] {f.matched_text}")
        print("\nTo suppress a finding, add it to the allowlist YAML file.")
        return 1

    print(f"PII scan passed. {len(allowed)} findings suppressed by allowlist, 0 new findings.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

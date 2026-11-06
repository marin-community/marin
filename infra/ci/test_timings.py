# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sum per-file test time from the unit job's JUnit reports, for balancing CI shards.

Each unit leg of a full CI run writes ``<package>.xml`` with ``junit_family=xunit1``,
whose test cases carry a ``file`` path relative to that leg's pytest rootdir
(``tests/...`` for Levanter, which has its own pytest configuration). This script maps
each path to the repo-relative test file the selector uses, sums the case times per file,
and overlays the sums on the previous timings so files missing from a partial run keep
their last value.

Usage:
    python3 -m infra.ci.test_timings --reports DIR --output test_seconds.json [--previous OLD.json]
"""

import argparse
import json
import logging
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from infra.ci.select_tests import SCOPES, UV_PACKAGE, all_test_files, load_test_seconds

logger = logging.getLogger(__name__)

SCOPE_BY_PACKAGE = {UV_PACKAGE[scope]: scope for scope in SCOPES}


def report_seconds(report: Path, scope_files: list[str]) -> dict[str, float]:
    """Summed test-case time per repo-relative test file in one JUnit report."""
    seconds: dict[str, float] = defaultdict(float)
    unmatched: set[str] = set()
    for case in ET.parse(report).iter("testcase"):
        rootdir_path = case.get("file")
        if rootdir_path is None:
            continue
        matches = [path for path in scope_files if path == rootdir_path or path.endswith(f"/{rootdir_path}")]
        if len(matches) != 1:
            unmatched.add(rootdir_path)
            continue
        seconds[matches[0]] += float(case.get("time", 0.0))
    if unmatched:
        logger.warning("%s: no unique test file for %s", report, sorted(unmatched))
    return dict(seconds)


def merged_seconds(reports: Path, previous: dict[str, float], repo_root: Path) -> dict[str, float]:
    """Previous timings for files that still exist, overlaid with the reports' sums."""
    files_by_scope = {scope: all_test_files(scope, repo_root) for scope in SCOPES}
    existing = {path for files in files_by_scope.values() for path in files}
    merged = {path: seconds for path, seconds in previous.items() if path in existing}
    for report in sorted(reports.rglob("*.xml")):
        scope = SCOPE_BY_PACKAGE.get(report.stem)
        if scope is None:
            raise ValueError(f"{report}: report name is not a known package")
        merged.update(report_seconds(report, files_by_scope[scope]))
    return {path: round(merged[path], 1) for path in sorted(merged)}


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reports", type=Path, required=True, help="Directory holding <package>.xml JUnit reports")
    parser.add_argument("--output", type=Path, required=True, help="Where to write the per-file timings JSON")
    parser.add_argument("--previous", type=Path, help="Earlier timings JSON to carry forward")
    args = parser.parse_args()

    previous = load_test_seconds(args.previous) if args.previous and args.previous.exists() else {}
    repo_root = Path(__file__).parent.parent.parent
    seconds = merged_seconds(args.reports, previous, repo_root)
    args.output.write_text(json.dumps(seconds, indent=1) + "\n")
    logger.info("Wrote %d file timings to %s", len(seconds), args.output)


if __name__ == "__main__":
    main()

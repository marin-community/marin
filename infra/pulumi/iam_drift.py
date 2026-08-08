#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Scan live GCP IAM for drift from iam_data.py and file it as a GitHub issue for triage.

    uv run --package marin-iac --extra deploy python infra/pulumi/iam_drift.py --no-github
    uv run --package marin-iac --extra deploy python infra/pulumi/iam_drift.py --run-url "$RUN_URL"

Every grant iam_data.py declares is non-authoritative, so `pulumi preview` only detects drift on
bindings already in Pulumi state. This re-enumerates the live IAM surface iam_data.py declares and
reports what diverges: undeclared bindings (including new service agents and hand-run grants),
broad roles bound outside config, and service agents whose backing API looks disabled. Findings
land on one sticky GitHub issue that updates in place and closes when the drift clears.

The scanned project is iam_data.py's own (`iac.gcp.iam_kms.PROJECT`); this scan is not
parameterizable to another project, since the KMS key and declared grants it diffs against are
that project's. See `iac.gcp.iam_scan` for what is and isn't in scope (human `user:` grants are
not diffed).
"""

import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path

# Match __main__.py: Pulumi's src/ layout means `iac` needs src/ on the path when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from iac.gcp import iam_data
from iac.gcp.iam_kms import PROJECT, crypto_key_id
from iac.gcp.iam_live import GcpIamReader, IamReader, iter_resources
from iac.gcp.iam_scan import (
    MARKER,
    Finding,
    classify,
    declared_bindings,
    declared_resources,
    fingerprint,
    render_markdown,
)
from iac.github.tracking_issue import sync_tracking_issue

logger = logging.getLogger("iam_drift")

DEFAULT_REPO = "marin-community/marin"
LABEL = "iam-drift"
TITLE = "[iam-drift] GCP IAM drift outside Pulumi"


def scan(reader: IamReader) -> list[Finding]:
    """Enumerate the live IAM surface, diff it against iam_data.py, and return the findings."""
    resources = declared_resources(
        PROJECT,
        crypto_key_id(),
        project_grants=iam_data.PROJECT_GRANTS,
        kms_grants=iam_data.KMS_GRANTS,
        secrets=iam_data.SECRETS,
        buckets=iam_data.BUCKETS,
        artifact_repositories=iam_data.ARTIFACT_REPOSITORIES,
        service_accounts=iam_data.SERVICE_ACCOUNTS,
    )
    live = [binding for target in iter_resources(resources) for binding in reader.bindings(target, PROJECT)]
    declared = declared_bindings(resources)
    enabled_services = reader.enabled_services(PROJECT)
    return classify(live, declared, project=PROJECT, enabled_services=enabled_services)


def _resolved_comment(run_url: str | None) -> str:
    where = run_url or "the latest scheduled scan"
    return (
        f"🤖 GCP IAM drift has cleared as of {where}; closing. The scheduled scan "
        "(`infra/pulumi/iam_drift.py`) reopens a fresh issue if drift returns."
    )


def _changed_comment(findings: list[Finding], run_url: str | None) -> str:
    where = run_url or "the latest scheduled scan"
    return (
        f"🤖 The GCP IAM drift set changed: {len(findings)} finding(s) as of {where}. See the "
        "issue body for the current list."
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=DEFAULT_REPO, help="owner/name the tracking issue lives in.")
    parser.add_argument("--run-url", default=None, help="Link to this scan run, embedded in the issue.")
    parser.add_argument("--json", action="store_true", help="Print findings as JSON and exit without touching GitHub.")
    parser.add_argument("--no-github", action="store_true", help="Print the rendered report instead of filing it.")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parser().parse_args()

    findings = scan(GcpIamReader())
    logger.info("found %d IAM drift finding(s) on %s", len(findings), PROJECT)

    if args.json:
        print(json.dumps([dataclasses.asdict(f) for f in findings], indent=2, default=str))
        return

    body = render_markdown(findings, project=PROJECT, run_url=args.run_url) if findings else None

    if args.no_github:
        print(body if body is not None else "No IAM drift detected.")
        return

    action = sync_tracking_issue(
        repo=args.repo,
        label=LABEL,
        marker=MARKER,
        title=TITLE,
        body=body,
        fingerprint=fingerprint(findings),
        changed_comment=_changed_comment(findings, args.run_url),
        resolved_comment=_resolved_comment(args.run_url),
    )
    logger.info("tracking issue %s", action.value)


if __name__ == "__main__":
    main()

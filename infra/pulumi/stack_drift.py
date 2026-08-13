#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Report Pulumi stack drift from refreshed-preview artifacts to a sticky GitHub issue.

    uv run --package marin-iac python infra/pulumi/stack_drift.py --previews-dir iac-previews --no-github

Consumes the `meta.json`/`diff.txt` pairs `.github/actions/pulumi-preview` uploads when run with
`refresh: true`. Any stack whose refreshed preview plans an operation has drifted away from
Pulumi out of band; the report lands on one sticky issue that updates in place and closes once
every stack is clean again.

Exits non-zero only on a failed preview, so an infrastructure fault is a red job while ordinary
drift is a triage issue.
"""

import argparse
import logging
import sys
from pathlib import Path

# Match __main__.py: Pulumi's src/ layout means `iac` needs src/ on the path when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from iac.github.tracking_issue import sync_tracking_issue
from iac.stack_drift import MARKER, fingerprint, load_stack_drifts, render_markdown

logger = logging.getLogger("stack_drift")

DEFAULT_REPO = "marin-community/marin"
LABEL = "stack-drift"
TITLE = "[stack-drift] Pulumi stacks diverge from live infrastructure"


def _resolved_comment(run_url: str | None) -> str:
    where = run_url or "the latest refresh"
    return (
        f"🤖 Every Pulumi stack matches its live infrastructure again as of {where}; closing. "
        "The scheduled refresh (`infra/pulumi/stack_drift.py`) opens a fresh issue if drift returns."
    )


def _changed_comment(drifted: int, run_url: str | None) -> str:
    where = run_url or "the latest refresh"
    return f"🤖 Pulumi stack drift changed: {drifted} stack(s) diverge as of {where}. See the issue body."


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--previews-dir", type=Path, required=True, help="Directory of downloaded preview artifacts.")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="owner/name the tracking issue lives in.")
    parser.add_argument("--run-url", default=None, help="Link to this refresh run, embedded in the issue.")
    parser.add_argument("--no-github", action="store_true", help="Print the rendered report instead of filing it.")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parser().parse_args()

    drifts = load_stack_drifts(args.previews_dir)
    if not drifts:
        raise SystemExit(f"no preview artifacts under {args.previews_dir}")

    drifted = [d for d in drifts if not d.clean]
    logger.info("%d of %d stack(s) diverge from live infrastructure", len(drifted), len(drifts))

    body = render_markdown(drifts, run_url=args.run_url) if drifted else None

    if args.no_github:
        print(body if body is not None else "No stack drift detected.")
    else:
        action = sync_tracking_issue(
            repo=args.repo,
            label=LABEL,
            marker=MARKER,
            title=TITLE,
            body=body,
            fingerprint=fingerprint(drifts),
            changed_comment=_changed_comment(len(drifted), args.run_url),
            resolved_comment=_resolved_comment(args.run_url),
        )
        logger.info("tracking issue %s", action.value)

    failed = [d.stack for d in drifts if d.severity == "error"]
    if failed:
        raise SystemExit(f"refreshed preview failed for: {', '.join(sorted(failed))}")


if __name__ == "__main__":
    main()

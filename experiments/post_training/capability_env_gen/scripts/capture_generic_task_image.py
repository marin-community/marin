#!/usr/bin/env python3
"""Trusted worker entry point for one externally approved rootfs capture.

The last stdout line is always one JSON object the image controller parses:

* a receipt was written: ``{"state": <receipt state>, "role", "receipt",
  "failure_class", "error_type", "failed_step"}``;
* the capture failed before any sandbox existed (so no receipt):
  ``{"state": "failed", "stage", "failure_class", "error_type", ...}`` where
  ``failure_class`` is transient / content / harness (see generic_image_capture).

Only our own static messages, exception type names and HTTP statuses are
printed -- never provider or secret text.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from capability_pipeline import sandbox_provider
from capability_pipeline.generic_image_capture import (
    HARNESS,
    capture_role,
    error_summary,
    validate_plan,
)
from capability_pipeline.generic_image_publication import validate_review

# Which pre-capture step is running, for the failure line.  Failures in these
# steps mean the job or the staged inputs are wrong, not the provider.
_STAGE = {"name": "arguments"}


def main(argv: list[str] | None = None) -> int:
    _STAGE["name"] = "arguments"
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--capture-tools", type=Path, required=True)
    parser.add_argument("--role", choices=("candidate", "private_verifier"), required=True)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--builder-session-id", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    # Capture sandboxes of earlier attempts whose deletion was never verified.
    parser.add_argument("--prior-sandbox-id", action="append", default=[])
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    _STAGE["name"] = "validate_plan"
    plan = json.loads(args.plan.read_text())
    validate_plan(plan, args.workspace, args.capture_tools)
    if not args.execute:
        print(json.dumps({"state": "review_required", "roles": [row["role"] for row in plan["images"]]}, sort_keys=True))
        return 0
    _STAGE["name"] = "validate_review"
    validate_review(args.approval, args.plan, builder_session_ids=set(args.builder_session_id))
    # Harness-side credentials only: the sandbox provider's, plus the object
    # store the captured archive lands in.  None of them enters a sandbox, and
    # no registry credential is involved in capture at all.
    _STAGE["name"] = "credentials"
    provider = sandbox_provider.provider()
    if not sandbox_provider.credentials_present() or not all(os.environ.get(name) for name in ("CW_KEY_ID", "CW_KEY_SECRET")):
        raise ValueError("trusted capture credentials are unavailable")
    _STAGE["name"] = "capture_tools_import"
    sys.path.insert(0, str(args.capture_tools.resolve()))
    import dtx  # type: ignore[import-not-found]

    # dtx prepends its parent and lazily imports the parent dt.py; validate_plan
    # pinned those exact bytes before the first provider call.
    client_factory = s3_factory = None
    if provider == sandbox_provider.SILO:
        # dtx.client() is Daytona-only.  The staged parent dt.py is silo's
        # drop-in (hash-pinned by the plan); its client() speaks to the broker.
        import cw_presign  # type: ignore[import-not-found]
        import dt  # type: ignore[import-not-found]

        client_factory = dt.client
        def s3_factory():
            return cw_presign.presigner().c

    _STAGE["name"] = "capture"
    receipt = capture_role(
        args.plan, args.workspace, args.role, args.output,
        approved_plan_sha256=json.loads(args.approval.read_text())["plan_sha256"], dtx=dtx,
        capture_tools=args.capture_tools, provider=provider,
        client_factory=client_factory, s3_factory=s3_factory,
        prior_sandbox_ids=list(args.prior_sandbox_id),
    )
    print(json.dumps({
        "state": receipt["state"], "role": args.role, "receipt": str(args.output),
        "failure_class": receipt.get("failure_class"), "error_type": receipt.get("error_type"),
        "failed_step": receipt.get("failed_step"),
    }, sort_keys=True))
    return 0 if receipt["state"] == "captured_pending_privacy_and_publication" else 1


def failure_line(error: BaseException) -> dict:
    """The structured, secret-free failure record for an exception out of main()."""
    summary = error_summary(error)
    stage = summary.pop("stage", None) or _STAGE["name"]
    if stage != "capture" and summary.get("failure_class") != "content" and "error_message" not in summary:
        # Anything that fails before the provider is involved is our own
        # staging, configuration or code: a harness failure, never "try again".
        summary["failure_class"] = HARNESS
    if isinstance(error, ValueError) and stage == "credentials":
        summary["error_message"] = "trusted capture credentials are unavailable"
    return {"state": "failed", "stage": stage, **summary}


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit as exit_:
        if exit_.code in (0, 1, None):
            raise
        # dt.py / dtx.py / cw_presign.py exit with a message when a credential is absent.
        print(json.dumps(failure_line(exit_), sort_keys=True))
        raise SystemExit(1) from None
    except BaseException as error:  # noqa: BLE001 - provider and secret text must not enter logs.
        print(json.dumps(failure_line(error), sort_keys=True))
        raise SystemExit(1) from None

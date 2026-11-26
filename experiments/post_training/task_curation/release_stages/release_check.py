# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Release check orchestration: preflight, hosted and postpublish phases.

One maintained module replaces the per-release gate and recheck drivers (see
INVENTORY.md): ``gates-browser-benchmark-v1.py`` and its siblings, plus
``recheck-v18-arc-evidence-v1.py``. The checks stay in the maintained
``infra.marina.apps.rl_data_catalog.release_check`` module; this stage is the
typed orchestration around it. It verifies the sealed preview-package receipt
and the phase-appropriate prior checker receipts (hosted/postpublish require a
passing preflight baseline; postpublish also requires the blocking hosted
receipt), computes the planned checker command, and delegates the write run to
an injected checker callable. The default checker runs the maintained module in
a subprocess: its ``main()`` is argparse-driven with no argument-taking entry
point, and it drives heavy gates of its own — the module invocation in the
release checkout is the pattern the maintained postpublish launcher itself uses.
The stage seals its receipt only after the produced checker receipt exists,
parses, and reports this phase as passed at the expected revision.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from rigging.filesystem.storage_path import StoragePath

import experiments.post_training.task_curation.release_stages.alerts as alerts
from experiments.post_training.task_curation.release_stages.guard import attested_revision
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef, read_ref
from experiments.post_training.task_curation.release_stages.package_preview import receipt_kind as package_receipt_kind
from infra.marina.apps.rl_data_catalog.release_receipts import PreflightBaseline

CHECKER_MODULE = "infra.marina.apps.rl_data_catalog.release_check"
RELEASE_CHECK_PHASES = ("preflight", "hosted", "postpublish")
CHECKER_RECEIPT_PATTERN = "checker/{phase}.json"
STAGE_RECEIPT_PATTERN = "release-check-{phase}.json"


@dataclass(frozen=True)
class ReleaseCheckInputs:
    """Typed inputs for one release-check run; every value is a graph output."""

    release_id: str
    source_revision: str
    phase: str
    output_root: str
    service_url: str
    applet_id: str
    expected_source_count: int
    production_version: int
    package: ArtifactRef | None = None
    oracle_inventory: ArtifactRef | None = None
    source_counts: ArtifactRef | None = None
    preflight_receipt: ArtifactRef | None = None
    hosted_receipt: ArtifactRef | None = None
    published_revision: int | None = None

    def validate(self) -> None:
        """Refuse malformed inputs before any verification runs."""
        for name in ("release_id", "source_revision", "output_root", "service_url", "applet_id"):
            if not getattr(self, name):
                raise ValueError(f"{name} is required")
        if self.phase not in RELEASE_CHECK_PHASES:
            raise ValueError(f"phase must be one of {RELEASE_CHECK_PHASES}, got {self.phase!r}")
        if self.expected_source_count <= 0:
            raise ValueError(f"expected_source_count must be positive, got {self.expected_source_count}")
        if self.production_version <= 0:
            raise ValueError("production_version must be the explicit expected production version")
        if self.phase == "preflight":
            if self.package is None or self.oracle_inventory is None or self.source_counts is None:
                raise ValueError("the preflight check requires the package, oracle-inventory and source-counts refs")
            if self.published_revision is not None:
                raise ValueError("the preflight check stages the package and must not name a published revision")
            return
        if self.package is not None or self.oracle_inventory is not None or self.source_counts is not None:
            raise ValueError(f"the {self.phase} check runs the published revision with its preflight baseline inputs")
        if self.preflight_receipt is None:
            raise ValueError(f"the {self.phase} check requires the sealed preflight receipt ref")
        if self.published_revision is None:
            raise ValueError(f"the {self.phase} check requires the published revision to verify")
        if self.phase == "postpublish" and self.hosted_receipt is None:
            raise ValueError("the postpublish audit requires the blocking hosted receipt ref")


def seal(path: str, value: dict) -> ArtifactRef:
    """Write-once seal with byte-identical readback; returns the artifact reference."""
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    target = StoragePath(path)
    if target.exists():
        if target.read_bytes() != raw:
            raise ValueError(f"sealed artifact {path} exists with different content; use a new output root")
    else:
        target.write_bytes(raw, auto_mkdir=True)
    if target.read_bytes() != raw:
        raise ValueError(f"seal readback mismatch at {path}")
    return ArtifactRef(uri=str(target), sha256=hashlib.sha256(raw).hexdigest())


def verify_phase_receipt(ref: ArtifactRef, expected_phase: str, revision: str) -> dict:
    """Digest-verify a prior checker receipt; require its phase, pass state and revision."""
    payload = StoragePath(ref.uri).read_bytes()
    if hashlib.sha256(payload).hexdigest() != ref.sha256:
        raise ValueError(f"{expected_phase} receipt digest mismatch: {ref.uri}")
    document = json.loads(payload)
    if document.get("phase") != expected_phase:
        raise ValueError(f"prior receipt phase is {document.get('phase')!r}, expected {expected_phase!r}")
    if document.get("passed") is not True:
        raise ValueError(f"the prior {expected_phase} receipt did not pass")
    if document.get("source_revision") != revision:
        raise ValueError(f"prior receipt pins revision {document.get('source_revision')!r}, expected {revision!r}")
    return document


def verify_package_receipt(inputs: ReleaseCheckInputs) -> dict:
    """Digest-verify the sealed package receipt and its release bindings."""
    document = read_ref(inputs.package)
    expected_kind = package_receipt_kind(inputs.release_id)
    if document.get("kind") != expected_kind:
        raise ValueError(f"package receipt kind is {document.get('kind')!r}, expected {expected_kind!r}")
    if document.get("release_id") != inputs.release_id or document.get("source_revision") != inputs.source_revision:
        raise ValueError("the package receipt binds a different release or revision")
    packaged = document.get("packaged") or {}
    if packaged.get("source_revision") != inputs.source_revision:
        raise ValueError(
            f"packaged provenance pins revision {packaged.get('source_revision')!r}, expected {inputs.source_revision!r}"
        )
    for name, ref in (("oracle_inventory", inputs.oracle_inventory), ("source_counts", inputs.source_counts)):
        if ref is not None and document.get(name) != {"uri": ref.uri, "sha256": ref.sha256}:
            raise ValueError(f"the package receipt's {name} ref differs from the typed input")
    # The package derives its counts from the release census; sources without a
    # measured population carry no count row, so the census covers the plan's sources.
    if document.get("census_sources") != inputs.expected_source_count:
        raise ValueError(
            f"the package derives from a census of {document.get('census_sources')} sources,"
            f" expected {inputs.expected_source_count}"
        )
    counted = document.get("counted_sources")
    counts = (packaged.get("source_counts") or {}).get("sources")
    if counts != counted:
        raise ValueError(f"the package carries {counts} count rows, the census measured {counted} sources")
    inventory = (packaged.get("oracle_inventory") or {}).get("sources")
    if inventory is not None and inventory != counted:
        raise ValueError(f"the packaged oracle inventory covers {inventory} sources, the census measured {counted}")
    if not StoragePath(document["package"]["uri"]).exists():
        raise ValueError(f"the staged package directory is missing: {document['package']['uri']}")
    return document


def planned_checker_arguments(
    inputs: ReleaseCheckInputs, package_document: dict | None, receipt_path: str, attested: str | None
) -> list[str]:
    """The exact maintained-checker arguments for the phase, computed as data.

    ``attested`` is the in-region bundle's attested revision; the bundle has no git
    checkout for the checker's own landed-cut checks.
    """
    argv = [
        "--receipt",
        receipt_path,
        *(("--attested-revision", attested) if attested is not None else ()),
        "--service-url",
        inputs.service_url,
        "--production-version",
        str(inputs.production_version),
    ]
    if inputs.phase == "preflight":
        if package_document is None:
            raise ValueError("the preflight release check needs the sealed package receipt")
        argv += [
            "--dist",
            package_document["package"]["uri"],
            "--source-counts",
            inputs.source_counts.uri,
            "--oracle-inventory",
            inputs.oracle_inventory.uri,
        ]
        return argv
    argv += ["--revision", str(inputs.published_revision), "--baseline", inputs.preflight_receipt.uri]
    if inputs.phase == "postpublish":
        argv.append("--full-browser-sweep")
    return argv


def subprocess_checker(repo_root: Path, argv: list[str], receipt_path: str) -> None:
    """Run the maintained checker module in a subprocess and require its receipt."""
    result = subprocess.run(
        [sys.executable, "-m", CHECKER_MODULE, *argv],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not StoragePath(receipt_path).exists():
        raise ValueError(
            f"the maintained release check exited {result.returncode} without a receipt at {receipt_path}:"
            f"\n{result.stderr[-2000:]}"
        )


def verify_checker_receipt(receipt_path: str, inputs: ReleaseCheckInputs) -> dict:
    """The produced checker receipt must exist, parse, and pass this phase."""
    path = StoragePath(receipt_path)
    if not path.exists():
        raise ValueError(f"the maintained checker left no receipt at {receipt_path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("phase") != inputs.phase:
        raise ValueError(f"checker receipt phase is {document.get('phase')!r}, expected {inputs.phase!r}")
    if document.get("passed") is not True:
        raise ValueError(f"the maintained {inputs.phase} check failed; see {receipt_path}")
    if document.get("source_revision") != inputs.source_revision:
        raise ValueError(
            f"checker receipt pins revision {document.get('source_revision')!r}, expected {inputs.source_revision!r}"
        )
    return document


def _ref_pointer(ref: ArtifactRef | None) -> dict | None:
    return None if ref is None else {"uri": ref.uri, "sha256": ref.sha256}


def run_release_check(
    inputs: ReleaseCheckInputs,
    *,
    dry_run: bool,
    repo_root: Path,
    feeds: tuple | None = None,
    checker: Callable[[Path, list[str], str], None] | None = None,
) -> dict:
    """Verify the release gates; ``dry_run`` plans the command with zero execution."""
    alert_feeds = feeds if feeds is not None else (alerts.WATCHDOG_FEED,)
    mode = "dry-run" if dry_run else "write"
    alerts.stage_started(inputs.release_id, "release_check", {"mode": mode, "phase": inputs.phase}, alert_feeds)
    try:
        inputs.validate()
        package_document = verify_package_receipt(inputs) if inputs.package is not None else None
        if inputs.preflight_receipt is not None:
            preflight_document = verify_phase_receipt(inputs.preflight_receipt, "preflight", inputs.source_revision)
            if inputs.phase in ("hosted", "postpublish"):
                baseline = PreflightBaseline.model_validate(preflight_document)
                if baseline.production_version != inputs.production_version:
                    raise ValueError("the preflight baseline pins a different production version")
        if inputs.hosted_receipt is not None:
            verify_phase_receipt(inputs.hosted_receipt, "hosted", inputs.source_revision)
        receipt_path = f"{inputs.output_root}/{CHECKER_RECEIPT_PATTERN.format(phase=inputs.phase)}"
        argv = planned_checker_arguments(inputs, package_document, receipt_path, attested_revision())
        proof: dict = {
            "kind": f"atlas-{inputs.release_id}-release-check",
            "release_id": inputs.release_id,
            "source_revision": inputs.source_revision,
            "phase": inputs.phase,
            "service_url": inputs.service_url,
            "expected_source_count": inputs.expected_source_count,
            "package": _ref_pointer(inputs.package),
            "preflight_receipt": _ref_pointer(inputs.preflight_receipt),
            "hosted_receipt": _ref_pointer(inputs.hosted_receipt),
            "published_revision": inputs.published_revision,
            "planned_checker": {"module": CHECKER_MODULE, "arguments": argv, "receipt": receipt_path},
            "dry_run": dry_run,
            "sql_writes": 0,
        }
        if dry_run:
            alerts.stage_finished(inputs.release_id, "release_check", {"mode": mode, "phase": inputs.phase}, alert_feeds)
            return proof
        (checker if checker is not None else subprocess_checker)(repo_root, argv, receipt_path)
        checker_document = verify_checker_receipt(receipt_path, inputs)
        proof["passed"] = checker_document.get("passed") is True
        # The marina-release contract reads the maintained receipt vocabulary: the
        # boolean outcome field and the applet the check targets.
        proof["target"] = {"id": inputs.applet_id, "service_url": inputs.service_url}
        proof["completed_at"] = datetime.now(tz=UTC).isoformat(timespec="seconds")
        receipt = seal(f"{inputs.output_root}/{STAGE_RECEIPT_PATTERN.format(phase=inputs.phase)}", proof)
        proof["receipt"] = {"uri": receipt.uri, "sha256": receipt.sha256}
        alerts.stage_finished(
            inputs.release_id,
            "release_check",
            {"mode": mode, "phase": inputs.phase, **proof["receipt"]},
            alert_feeds,
        )
        return proof
    except Exception as error:
        alerts.stage_failed(
            inputs.release_id,
            "release_check",
            {"mode": mode, "phase": inputs.phase, "error": str(error)[:500]},
            alert_feeds,
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True, help="Path to a JSON ReleaseCheckInputs document")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(), help="Checkout the checker runs from")
    parser.add_argument("--mode", choices=("dry-run", "write"), default="dry-run")
    args = parser.parse_args()
    document = json.loads(args.inputs.read_text(encoding="utf-8"))
    for name in ("package", "oracle_inventory", "source_counts", "preflight_receipt", "hosted_receipt"):
        if document.get(name) is not None:
            document[name] = ArtifactRef(**document[name])
    inputs = ReleaseCheckInputs(**document)
    proof = run_release_check(inputs, dry_run=args.mode == "dry-run", repo_root=args.repo_root)
    label = "RELEASE-CHECK-DRY-RUN" if args.mode == "dry-run" else "RELEASE-CHECK-SEALED"
    print(f"{label} {json.dumps(proof)}", flush=True)


if __name__ == "__main__":
    main()

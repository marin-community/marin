# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Marina release: publish the preview applet and seal the deployment manifest.

One maintained module replaces the per-release publish-and-seal scripts (see
INVENTORY.md). The stage verifies the release-check receipt chain and the clean
landed checkout, publishes the packaged preview through Marina's applet API, and
seals two write-once documents under the release prefix: the publish record and the
deployment manifest that the hosted stages verify against.

Receipt contracts (decided here): the preflight and hosted receipts are the
maintained release-check receipts (``ReleaseReceipt`` in
``infra.marina.apps.rl_data_catalog.release_receipts``) and must record ``passed``
true (the receipt's boolean outcome field), the matching ``phase``, the typed
``source_revision``, and the typed applet id in ``target.id``. The package receipt
is the sealed package-preview receipt: its ``package.uri`` is the staged applet
directory and its ``packaged`` provenance pins the release revision and the packaged
module digests, which the repackaged payload must match before publishing.

The deployment manifest carries only fields the typed inputs provide: revision
continuity against the prior manifest through ``base_version``, the applet id, mode
and publish URL, the source revision, and the receipt chain. Browser-derived
continuity fields the per-release manifests carried (source-table headers, status
rollups) return as typed inputs land.
"""

import argparse
import hashlib
import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from marina.applets import package_applet, read_applet_package
from marina.client import bearer_token, marina_token_provider, publish_applet
from rigging.filesystem.storage_path import StoragePath

import experiments.post_training.task_curation.release_stages.alerts as alerts
from experiments.post_training.task_curation.release_stages import live_applet as live_applet_module
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef, read_ref

ACCESS_MODES = ("private", "public")
PACKAGE_RECEIPT_KIND_SUFFIX = "preview-package"
RELEASE_RECORD_SUFFIX = "marina-release"
DEPLOYMENT_MANIFEST_SUFFIX = "deployment-manifest"
RELEASE_DOCUMENT = "release-v1.json"
MANIFEST_DOCUMENT = "deployment-manifest.json"

Publish = Callable[[bytes], dict]


@dataclass(frozen=True)
class MarinaReleaseInputs:
    """Typed inputs for one applet release; every value is a graph output."""

    release_id: str
    source_revision: str
    applet_id: str
    service_url: str
    package: ArtifactRef
    preflight_receipt: ArtifactRef
    hosted_receipt: ArtifactRef
    prior_manifest: ArtifactRef | None
    base_version: int
    access_mode: str
    output_root: str

    def validate(self) -> None:
        """Refuse malformed inputs before any read runs."""
        if not self.release_id or not self.source_revision or not self.applet_id:
            raise ValueError("release_id, source_revision and applet_id are required")
        if not self.service_url or not self.output_root:
            raise ValueError("service_url and output_root are required")
        if self.base_version <= 0:
            raise ValueError(
                f"base_version must be the positive live revision this release updates, got {self.base_version}"
            )
        if self.access_mode not in ACCESS_MODES:
            raise ValueError(f"access_mode must be one of {ACCESS_MODES}, got {self.access_mode!r}")


def verify_checkout(repo_root: Path, source_revision: str) -> str:
    """The releasing checkout must be clean and contain the plan's pinned revision.

    In the in-region bundle mode the submitter attests the workspace (the guard's
    landing-log check carries the landed attestation) and the git checks cannot run.
    """
    from experiments.post_training.task_curation.release_stages.guard import (  # noqa: PLC0415 — avoids the guard/marina cycle at module load
        attested_revision,
    )

    attested = attested_revision()
    if attested is not None:
        return attested
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo_root, capture_output=True, text=True, check=True
    ).stdout.strip()
    if dirty:
        raise ValueError(f"releasing checkout is dirty:\n{dirty[:400]}")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=True
    ).stdout.strip()
    pinned = subprocess.run(
        ["git", "merge-base", "--is-ancestor", source_revision, revision],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if pinned.returncode != 0:
        raise ValueError(
            f"checkout HEAD {revision} does not contain the plan's pinned revision {source_revision}; "
            "cut from a checkout that contains it"
        )
    return revision


def verify_release_check_receipt(receipt: ArtifactRef, phase: str, inputs: MarinaReleaseInputs) -> dict:
    """Digest-verify one release-check receipt against the decided contract fields."""
    document = read_ref(receipt)
    if document.get("passed") is not True:
        raise ValueError(f"the {phase} release-check receipt did not pass: {receipt.uri}")
    if document.get("phase") != phase:
        raise ValueError(f"receipt at {receipt.uri} records phase {document.get('phase')!r}, expected {phase!r}")
    if document.get("source_revision") != inputs.source_revision:
        raise ValueError(
            f"the {phase} receipt pins source revision {document.get('source_revision')!r},"
            f" expected {inputs.source_revision!r}"
        )
    target = document.get("target")
    if not isinstance(target, dict) or target.get("id") != inputs.applet_id:
        raise ValueError(f"the {phase} receipt targets a different applet than {inputs.applet_id!r}")
    return document


def verify_package_receipt(package: ArtifactRef, inputs: MarinaReleaseInputs) -> dict:
    """Digest-verify the sealed package receipt and its staged directory."""
    document = read_ref(package)
    expected_kind = f"atlas-{inputs.release_id}-{PACKAGE_RECEIPT_KIND_SUFFIX}"
    if document.get("kind") != expected_kind:
        raise ValueError(f"package receipt kind is {document.get('kind')!r}, expected {expected_kind!r}")
    if document.get("release_id") != inputs.release_id or document.get("source_revision") != inputs.source_revision:
        raise ValueError("the package receipt binds a different release or revision")
    packaged = document.get("packaged") or {}
    if packaged.get("source_revision") != document.get("code_revision"):
        raise ValueError(
            f"packaged provenance pins revision {packaged.get('source_revision')!r},"
            f" expected the packaging checkout {document.get('code_revision')!r}"
        )
    root = (document.get("package") or {}).get("uri")
    if not isinstance(root, str) or not root:
        raise ValueError(f"package receipt records no package directory: {package.uri}")
    if not StoragePath(root).exists():
        raise ValueError(f"the staged package directory is missing: {root}")
    return document


def verify_prior_manifest(prior: ArtifactRef, inputs: MarinaReleaseInputs) -> dict:
    """The prior manifest must be the live revision this release continues."""
    document = read_ref(prior)
    if document.get("revision") != inputs.base_version:
        raise ValueError(
            f"prior manifest revision {document.get('revision')!r} is not the typed base_version {inputs.base_version}"
        )
    if document.get("app_id") != inputs.applet_id:
        raise ValueError("the prior manifest names a different applet than this release")
    if document.get("mode") != inputs.access_mode:
        raise ValueError(f"prior manifest mode is {document.get('mode')!r}, expected {inputs.access_mode!r}")
    return document


def verify_packaged_modules(payload: bytes, provenance: dict) -> None:
    """The repackaged applet must carry the provenance-recorded module bytes."""
    files = read_applet_package(payload).files
    for module in provenance.get("modules", []):
        packaged = files.get(module["packaged"])
        digest = hashlib.sha256(packaged).hexdigest() if packaged is not None else None
        if digest != module["packaged_sha256"]:
            raise ValueError(f"packaged module {module['packaged']!r} differs from the package provenance")


def publish_with_iap_token(inputs: MarinaReleaseInputs, payload: bytes) -> dict:
    """The real publish: resolve the IAP token eagerly, then POST the gzip payload."""
    bearer_token(marina_token_provider())
    return publish_applet(
        inputs.service_url,
        payload,
        applet_id=inputs.applet_id,
        base_version=inputs.base_version,
        mode=inputs.access_mode,
    )


def verify_publish_result(result: dict, inputs: MarinaReleaseInputs) -> int:
    """Marina must have released this applet, in the live access mode, at exactly the next revision."""
    if result.get("id") != inputs.applet_id:
        raise ValueError(f"Marina released applet {result.get('id')!r}, expected {inputs.applet_id!r}")
    if result.get("mode") != inputs.access_mode:
        raise ValueError(f"Marina released in mode {result.get('mode')!r}, expected {inputs.access_mode!r}")
    version = result.get("version")
    if not isinstance(version, int) or version != inputs.base_version + 1:
        raise ValueError(f"Marina released version {version!r}, expected {inputs.base_version + 1}")
    if not isinstance(result.get("url"), str) or not result["url"]:
        raise ValueError("Marina returned no release URL")
    return version


def ref_fields(ref: ArtifactRef) -> dict:
    return {"uri": ref.uri, "sha256": ref.sha256}


def write_once(path: StoragePath, value: dict, label: str) -> ArtifactRef:
    """Write one release document once, with byte-identical readback."""
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    if path.exists():
        raise ValueError(f"{label} already exists at {path}; bump the release prefix instead of republishing")
    path.write_bytes(raw, auto_mkdir=True)
    if path.read_bytes() != raw:
        raise ValueError(f"{label} readback mismatch at {path}")
    return ArtifactRef(uri=str(path), sha256=hashlib.sha256(raw).hexdigest())


def run_marina_release(
    inputs: MarinaReleaseInputs,
    *,
    dry_run: bool,
    repo_root: Path,
    publish: Publish | None = None,
    feeds: tuple | None = None,
) -> dict:
    """Publish the preview applet and seal the release documents; dry run writes nothing."""
    alert_feeds = feeds if feeds is not None else (alerts.WATCHDOG_FEED,)
    mode = "dry-run" if dry_run else "write"
    alerts.stage_started(inputs.release_id, "marina_release", {"mode": mode}, alert_feeds)
    try:
        inputs.validate()
        verify_release_check_receipt(inputs.preflight_receipt, "preflight", inputs)
        verify_release_check_receipt(inputs.hosted_receipt, "hosted", inputs)
        package = verify_package_receipt(inputs.package, inputs)
        prior = verify_prior_manifest(inputs.prior_manifest, inputs) if inputs.prior_manifest is not None else None
        verify_checkout(repo_root, inputs.source_revision)
        receipts = {
            "preflight": ref_fields(inputs.preflight_receipt),
            "hosted": ref_fields(inputs.hosted_receipt),
            "package": ref_fields(inputs.package),
        }
        proof: dict = {
            "kind": f"atlas-{inputs.release_id}-{RELEASE_RECORD_SUFFIX}",
            "release_id": inputs.release_id,
            "source_revision": inputs.source_revision,
            "applet_id": inputs.applet_id,
            "service_url": inputs.service_url,
            "base_version": inputs.base_version,
            "access_mode": inputs.access_mode,
            "receipts": receipts,
            "prior_manifest": ref_fields(inputs.prior_manifest) if prior is not None else None,
            "dry_run": dry_run,
            "sql_writes": 0,
        }
        if dry_run:
            alerts.stage_finished(inputs.release_id, "marina_release", {"mode": mode}, alert_feeds)
            return proof
        # Refuse a rerun before publishing: a second POST would create a duplicate version.
        release_path = StoragePath(f"{inputs.output_root}/{RELEASE_DOCUMENT}")
        manifest_path = StoragePath(f"{inputs.output_root}/{MANIFEST_DOCUMENT}")
        for path, label in ((release_path, "release document"), (manifest_path, "deployment manifest")):
            if path.exists():
                raise ValueError(f"{label} already exists at {path}; bump the release prefix instead of republishing")
        # The hosted check recorded the live revision and mode; the publish continues
        # exactly that state (the compare-and-set also refuses a moved base).
        live = live_applet_module.live_applet(inputs.service_url, inputs.applet_id)
        if (live.version, live.mode) != (inputs.base_version, inputs.access_mode):
            raise ValueError(
                f"the preview moved to v{live.version} ({live.mode}) since the hosted check recorded"
                f" v{inputs.base_version} ({inputs.access_mode})"
            )
        payload = package_applet(Path(package["package"]["uri"]))
        verify_packaged_modules(payload, package["packaged"])
        publish_fn = (
            publish if publish is not None else (lambda payload_bytes: publish_with_iap_token(inputs, payload_bytes))
        )
        result = publish_fn(payload)
        version = verify_publish_result(result, inputs)
        sealed_at = datetime.now(tz=UTC).isoformat(timespec="seconds")
        release_document = {
            "kind": f"atlas-{inputs.release_id}-{RELEASE_RECORD_SUFFIX}",
            "release_id": inputs.release_id,
            "source_revision": inputs.source_revision,
            "publish": result,
            "receipts": receipts,
            "dry_run": False,
            "sealed_at": sealed_at,
        }
        manifest = {
            "kind": f"atlas-{inputs.release_id}-{DEPLOYMENT_MANIFEST_SUFFIX}",
            "release_id": inputs.release_id,
            "revision": version,
            "base_version": inputs.base_version,
            "url": result["url"],
            "app_id": result["id"],
            "mode": result["mode"],
            "source_sha": inputs.source_revision,
            "code_revision": package["code_revision"],
            "service_url": inputs.service_url,
            "package": {"root": package["package"]["uri"]},
            "receipts": receipts,
            **({"prior_manifest": ref_fields(inputs.prior_manifest)} if prior is not None else {}),
            "sealed_at": sealed_at,
        }
        release_ref = write_once(release_path, release_document, "release document")
        manifest_ref = write_once(manifest_path, manifest, "deployment manifest")
        proof["publish"] = {key: result[key] for key in ("id", "version", "url", "mode")}
        proof["release"] = ref_fields(release_ref)
        proof["manifest"] = ref_fields(manifest_ref)
        alerts.stage_finished(
            inputs.release_id,
            "marina_release",
            {"mode": mode, "revision": version, **proof["manifest"]},
            alert_feeds,
        )
        return proof
    except Exception as error:
        alerts.stage_failed(inputs.release_id, "marina_release", {"mode": mode, "error": str(error)[:500]}, alert_feeds)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True, help="Path to a JSON MarinaReleaseInputs document")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(), help="Checkout the release publishes from")
    parser.add_argument("--mode", choices=("dry-run", "write"), default="dry-run")
    args = parser.parse_args()
    document = json.loads(args.inputs.read_text(encoding="utf-8"))
    for name in ("package", "preflight_receipt", "hosted_receipt"):
        document[name] = ArtifactRef(**document[name])
    if document.get("prior_manifest") is not None:
        document["prior_manifest"] = ArtifactRef(**document["prior_manifest"])
    inputs = MarinaReleaseInputs(**document)
    proof = run_marina_release(inputs, dry_run=args.mode == "dry-run", repo_root=args.repo_root)
    label = "MARINA-RELEASE-DRY-RUN" if args.mode == "dry-run" else "MARINA-RELEASE-SEALED"
    print(f"{label} {json.dumps(proof)}", flush=True)


if __name__ == "__main__":
    main()

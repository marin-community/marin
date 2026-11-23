# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The release artifact graph (DESIGN.md sections 1-3): one entry point builds a release.

Every stage is a lazy ``ArtifactStep`` whose build runs the stage module's
``run_*`` and publishes a typed receipt artifact; downstream stages bind their
typed inputs from ``ctx.resolved`` of their dependencies, so hashes and URIs flow
through the graph and are never copied by hand. Independent stages are sibling
dependencies of their join step and run concurrently under the runner.

The release plan (``ReleasePlan``) is the typed configuration resolved once at the
CLI boundary — the only place a release id enters the system.
"""

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.task_curation.release_stages.manifest import PriorPin

PLAN_KIND = "atlas-release-plan"


class ReleasePlan(BaseModel):
    """Typed release configuration resolved once at the CLI boundary.

    Every field is data the graph passes as typed inputs; no stage source carries
    these values as literals. The plan document is sealed under the release prefix
    so the guard's no-literal rule and the preflight receipts bind it.
    """

    kind: str = PLAN_KIND
    release_id: str
    source_revision: str
    release_root: str
    catalog_root: str
    prior: PriorPin
    targets: tuple[str, ...]
    expected_source_count: int
    expected_browse_source_count: int
    expected_contract_count: int
    expected_contract_tasks: int
    expected_selected_panels: int
    region_prefix: str
    qc_exclusions: tuple[str, ...] = ()
    migrations: tuple[str, ...] = ()
    slice_output_root: str = ""

    def validate_shape(self) -> None:
        """Refuse a plan that cannot drive the graph."""
        if self.kind != PLAN_KIND:
            raise ValueError(f"release plan kind is {self.kind!r}, expected {PLAN_KIND!r}")
        if not self.release_id or not self.source_revision:
            raise ValueError("release_id and source_revision are required")
        if not self.release_root or not self.catalog_root:
            raise ValueError("release_root and catalog_root are required")
        if len(self.prior.load_id) != 32 or any(c not in "0123456789abcdef" for c in self.prior.load_id):
            raise ValueError(f"prior load id must be 32-hex, got {self.prior.load_id!r}")
        if not self.targets or self.expected_source_count <= 0:
            raise ValueError("targets and a positive expected_source_count are required")
        if not self.region_prefix.endswith("/"):
            raise ValueError("region_prefix must end with '/' so region pins are unambiguous")


def load_plan(path: Path) -> ReleasePlan:
    """Load and validate the sealed release plan document."""
    plan = ReleasePlan.model_validate_json(path.read_text(encoding="utf-8"))
    plan.validate_shape()
    return plan


def plan_ref(release_root: str, relative: str, sha256: str | None = None) -> dict:
    """A {uri, sha256} reference into the release prefix (sha filled by the stage)."""
    uri = str(StoragePath(f"{release_root}/{relative}"))
    return {"uri": uri, "sha256": sha256} if sha256 else {"uri": uri, "sha256": ""}


def stage_output_root(plan: ReleasePlan, stage: str) -> str:
    """The deterministic output prefix of one stage's artifacts."""
    return f"{plan.release_root}/{stage}"


def write_plan(plan: ReleasePlan, release_root: str) -> dict:
    """Seal the release plan document write-once; the graph's first artifact."""
    raw = (json.dumps(plan.model_dump(mode="json"), indent=2, sort_keys=True) + "\n").encode()
    path = StoragePath(f"{release_root}/release-plan.json")
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError(f"release plan already exists with different content at {path}")
    else:
        path.write_bytes(raw, auto_mkdir=True)
    if path.read_bytes() != raw:
        raise ValueError("release plan readback mismatch")
    return {"uri": str(path), "sha256": hashlib.sha256(raw).hexdigest()}

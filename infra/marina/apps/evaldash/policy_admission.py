# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read released eval policy manifests, and the compliance files of their versions' other runs.

``policy release`` writes one manifest per released version under ``RELEASES_PREFIX``, and ``policy verify`` writes
a compliance file beside each run it checks (``evalrecords.policy_release``). EvalDash cannot see the git tag that
makes a manifest a release, so it treats every manifest as one.
"""

import logging
from collections.abc import Callable, Collection, Iterable

from evalrecords.policy_release import (
    PolicyRelease,
    ReleaseAdmission,
    RunCompliance,
    compliance_path,
)
from evalrecords.records import EvalRunRecord
from rigging.filesystem.factory import open_url
from rigging.filesystem.storage_path import StoragePath, prefix_join
from sqlalchemy.engine import Engine

from .results_db import catalog_records_of_versions, store_admission

logger = logging.getLogger(__name__)


def read_releases(prefix: str) -> dict[str, PolicyRelease]:
    """Every manifest under ``prefix``, by version."""
    releases = {}
    for path in StoragePath(prefix_join(prefix, "*.json")).glob():
        with open_url(str(path), "rb") as handle:
            release = PolicyRelease.model_validate_json(handle.read())
        releases[release.version] = release
    return releases


def _compliance(record: EvalRunRecord) -> RunCompliance | None:
    """The compliance file beside a policy run's record, or None if the run has none."""
    if record.results_path is None:
        return None
    run_dir = StoragePath(record.results_path).parent
    if run_dir.name != record.run_id:
        return None
    try:
        with open_url(compliance_path(str(run_dir.parent), record.run_id), "rb") as handle:
            return RunCompliance.model_validate_json(handle.read())
    except FileNotFoundError:
        return None


def read_admission(
    releases: dict[str, PolicyRelease], records_of: Callable[[Collection[str]], Iterable[EvalRunRecord]]
) -> ReleaseAdmission:
    """The admission for ``releases``: the compliance file of each of their versions' runs the manifests don't pin.

    ``records_of`` gives the records labelled with the given versions.
    """
    pinned = {run.run_id for release in releases.values() for run in release.runs}
    compliance = {
        record.run_id: found
        for record in records_of(releases.keys())
        if record.run_id not in pinned and (found := _compliance(record)) is not None
    }
    return ReleaseAdmission(releases=releases, compliance=compliance)


def ingest_policy_admission(engine: Engine, releases_prefix: str) -> bool:
    """Store the current admission, publishing a generation if it changed; False when reading it failed."""
    try:
        admission = read_admission(
            read_releases(releases_prefix), lambda versions: catalog_records_of_versions(engine, versions)
        )
    except Exception:
        logger.exception("policy admission: reading releases or compliance files failed; keeping the stored one")
        return False
    if store_admission(engine, admission):
        logger.info(
            "policy admission: %d releases, %d compliance files", len(admission.releases), len(admission.compliance)
        )
    return True

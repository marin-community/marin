# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Capture small authoritative metadata before claiming or staging a source."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from taskcompendium.identity import canonical_sha256

from experiments.post_training.task_curation.backend_census_manifest import seal_manifest
from experiments.post_training.task_curation.ingest_stages.eligibility import require_starting_state
from experiments.post_training.task_curation.ingest_stages.ingest_config import IngestConfig, IngestMode
from experiments.post_training.task_curation.ingest_stages.licensing import capture_license_context
from experiments.post_training.task_curation.ingest_stages.queue import (
    QueueClaim,
    QueueSelection,
    claimed_row,
    select_queue,
)
from experiments.post_training.task_curation.ingest_stages.registry_update import pointers_at_revision
from experiments.post_training.task_curation.license_binding import LICENSE_RECEIPT_INDEX
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef

EXPORT_STATE = """import json
from experiments.post_training.task_curation.sources import all_sources, pending_measurement_sources
from experiments.post_training.task_curation.exclusion_registry import manual_exclusion_claims
from experiments.post_training.task_curation.metadata_exclusions import METADATA_EXCLUSIONS
sources = list(all_sources().values())
catalog = {s.info.id: s.declaration.model_dump(mode='json') for s in sources}
starting = {s.info.id: {**catalog[s.info.id], 'dataset_url': s.dataset.url if s.dataset else None} for s in sources}
for source in pending_measurement_sources():
    source_id = source.info.id
    if source_id in starting:
        continue
    dataset = source.dataset
    claims = manual_exclusion_claims(source_id)
    starting[source_id] = dict(id=source_id, status='Excluded' if claims else 'Pending',
        dataset_id=dataset.name if dataset else None, dataset_revision=dataset.revision if dataset else None,
        dataset_url=dataset.url if dataset else None,
        exclusion_claims=[c.model_dump(mode='json') for c in claims],
        upstream_name=source.info.upstream_name, build_path=source.info.build_path,
        family=source.info.family, tags=list(source.info.tags), notes=source.info.notes,
        data_restriction=source.info.data_restriction.value)
for row in METADATA_EXCLUSIONS:
    if row.id in starting:
        continue
    starting[row.id] = dict(id=row.id, status='Excluded', dataset_id=row.dataset_id,
        dataset_revision=row.dataset_revision, dataset_url=row.upstream_url,
        exclusion_claims=[c.model_dump(mode='json') for c in row.claims],
        upstream_name=row.upstream_dataset, build_path='metadata_exclusions', family=row.origin,
        tags=[], notes=row.notes)
print(json.dumps(dict(catalog=catalog, starting=starting), sort_keys=True))
"""


def catalog_snapshot(repo_root: Path) -> dict:
    paths = [repo_root, *sorted(repo_root.glob("lib/*/src")), repo_root / "infra/marina/src"]
    result = subprocess.check_output(
        [sys.executable, "-c", EXPORT_STATE],
        cwd=repo_root,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(map(str, paths))},
        text=True,
    )
    return json.loads(result)


def metadata_inputs(policy: IngestConfig, repo_root: Path, queue: Path, output: Path) -> dict[str, ArtifactRef]:
    """Check eligibility before queue mutation, issue comments or any payload transfer."""
    state = catalog_snapshot(repo_root)
    require_starting_state(policy, state["starting"])
    if policy.mode is IngestMode.INGEST:
        selection = select_queue(queue.read_bytes(), policy)
    else:
        declaration = state["starting"][policy.source_id]
        url = declaration["dataset_url"]
        row = f"| {policy.source_id} | {url} | Excluded repair |"
        selection = QueueSelection(
            source_id=policy.source_id,
            policy_sha256=policy.digest(),
            queue_sha256=canonical_sha256(declaration),
            row=row,
            requester_issue=None,
        )
    queue_claim = QueueClaim(
        selection=selection,
        claimed_row_sha256=hashlib.sha256(claimed_row(selection).encode()).hexdigest(),
    )
    documents = {
        "catalog-declarations": state["catalog"],
        "starting-declarations": state["starting"],
        "queue-claim": queue_claim.model_dump(mode="json"),
        "license-context": (
            capture_license_context(policy, repo_root / LICENSE_RECEIPT_INDEX.relative_to(repo_root)).model_dump(
                mode="json"
            )
        ),
        "registry-pointers": pointers_at_revision(repo_root, "HEAD").model_dump(mode="json"),
    }
    return {name: seal_manifest(document, str(output / f"{name}.json")) for name, document in documents.items()}

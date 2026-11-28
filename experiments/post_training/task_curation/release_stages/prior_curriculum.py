# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve a frozen curriculum load to its digest-bound release and coverage.

The confidence sidecar pins both documents. Read only that small reference from
the recorded load, rather than the mutable current pointer or the large embedded
SQL manifest. Re-cuts must retain the same served curriculum they targeted.
"""

from experiments.post_training.task_curation.release_stages.history_staging import Query
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef, read_ref


def prior_curriculum_pins(query: Query, load_id: str) -> dict[str, ArtifactRef]:
    """Verify the exact recorded load and return its immutable composition pins."""
    rows = query(
        "SELECT load_id,release_id,release_uri,manifest->'confidence'->'reference' AS confidence"
        " FROM curriculum_loads WHERE load_id=:load",
        {"load": load_id},
    )["rows"]
    if len(rows) != 1 or rows[0]["load_id"] != load_id or not rows[0].get("confidence"):
        raise ValueError(f"prior curriculum load {load_id} has no digest-bound confidence reference")
    row = rows[0]
    confidence = read_ref(ArtifactRef.model_validate(row["confidence"]))
    release_ref = ArtifactRef.model_validate(confidence["release"])
    coverage_ref = ArtifactRef.model_validate(confidence["coverage"])
    release = read_ref(release_ref)
    coverage = read_ref(coverage_ref)
    if (
        row["release_uri"] != release_ref.uri
        or row["release_id"] != coverage["coverage_id"]
        or coverage["alignment_release_id"] != release["release_id"]
        or coverage["task_count"] != release["task_count"]
        or coverage["curriculum_key"] != release["curriculum_key"]
    ):
        raise ValueError(f"prior curriculum load {load_id} does not bind its release and coverage")
    return {"composed_release": release_ref, "composed_coverage": coverage_ref}

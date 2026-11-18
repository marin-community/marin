# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Keep archived assessment evidence across catalog refreshes and source retirement."""

from collections.abc import Mapping

from marina.client import AppletQuery

# Copy evidence before its assessment header; each query selects missing headers.
ASSESSMENT_TABLES = ("populations", "items", "packets", "assessments")
# The per-task tables copy in task-identity hash buckets: their rows dominate the
# retained volume and the gateway's per-statement timeout bounds the copy size.
# The bucket count stays a power of two so the mask avoids a literal modulo, which
# the SQL gateway's parameter parser would read as a placeholder.
RETAIN_BUCKETED_TABLES = ("items", "packets")
RETAIN_COPY_BUCKETS = 16


def retain_prior_assessments(
    query: AppletQuery, columns: Mapping[str, str], prior_load_id: str, load_id: str
) -> dict[str, int]:
    """Copy complete prior assessment units absent from a fully staged catalog load.

    Each missing pipeline copies in bounded statements: one INSERT..SELECT over every
    missing pipeline exceeded the SQL gateway's per-statement timeout once the retained
    packet history grew past the r22 volume, and per-pipeline bounds overtook the biggest
    retained populations as well. The per-task tables (items, packets) copy in
    deterministic hash buckets of the task identity; the per-pipeline header tables
    (populations, assessments) are small and copy whole. The missing-pipeline probe
    deduplicates before its anti-join for the same reason.
    """
    copied = {}
    for table in ASSESSMENT_TABLES:
        names = [column.strip().split()[0] for column in columns[table].split(",")]
        missing = query(
            f"SELECT pipeline_key FROM ("
            f"SELECT DISTINCT previous.pipeline_key FROM atlas_{table} previous "
            f"WHERE previous.load_id = :prior_load_id"
            f") candidates WHERE NOT EXISTS (SELECT 1 FROM atlas_assessments current_assessment "
            "WHERE current_assessment.load_id = :load_id "
            "AND current_assessment.pipeline_key = candidates.pipeline_key)",
            {"load_id": load_id, "prior_load_id": prior_load_id},
        )["rows"]
        total = 0
        for row in missing:
            parameters = {
                "load_id": load_id,
                "prior_load_id": prior_load_id,
                "pipeline_key": row["pipeline_key"],
            }
            if table in RETAIN_BUCKETED_TABLES:
                for bucket in range(RETAIN_COPY_BUCKETS):
                    result = query(
                        f"INSERT INTO atlas_{table} (load_id, {', '.join(names)}) "
                        f"SELECT :load_id, {', '.join('previous.' + name for name in names)} "
                        f"FROM atlas_{table} previous WHERE previous.load_id = :prior_load_id "
                        "AND previous.pipeline_key = :pipeline_key "
                        f"AND hashtext(previous.task_id) & {RETAIN_COPY_BUCKETS - 1} = :bucket "
                        "AND NOT EXISTS (SELECT 1 FROM atlas_assessments current_assessment "
                        "WHERE current_assessment.load_id = :load_id "
                        "AND current_assessment.pipeline_key = previous.pipeline_key)",
                        {**parameters, "bucket": bucket},
                    )
                    total += result["row_count"]
            else:
                result = query(
                    f"INSERT INTO atlas_{table} (load_id, {', '.join(names)}) "
                    f"SELECT :load_id, {', '.join('previous.' + name for name in names)} "
                    f"FROM atlas_{table} previous WHERE previous.load_id = :prior_load_id "
                    "AND previous.pipeline_key = :pipeline_key "
                    "AND NOT EXISTS (SELECT 1 FROM atlas_assessments current_assessment "
                    "WHERE current_assessment.load_id = :load_id "
                    "AND current_assessment.pipeline_key = previous.pipeline_key)",
                    parameters,
                )
                total += result["row_count"]
        copied[table] = total
    return copied

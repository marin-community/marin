# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Keep archived assessment evidence across catalog refreshes and source retirement."""

from collections.abc import Mapping

from marina.client import AppletQuery

# Copy evidence before its assessment header; each query selects missing headers.
ASSESSMENT_TABLES = ("populations", "items", "packets", "assessments")


def retain_prior_assessments(
    query: AppletQuery, columns: Mapping[str, str], prior_load_id: str, load_id: str
) -> dict[str, int]:
    """Copy complete prior assessment units absent from a fully staged catalog load.

    Each missing pipeline copies in its own bounded statement: one INSERT..SELECT
    over every missing pipeline exceeded the SQL gateway's per-statement timeout
    once the retained packet history grew past the r22 volume.
    """
    copied = {}
    for table in ASSESSMENT_TABLES:
        names = [column.strip().split()[0] for column in columns[table].split(",")]
        missing = query(
            f"SELECT DISTINCT previous.pipeline_key FROM atlas_{table} previous "
            "WHERE previous.load_id = :prior_load_id "
            "AND NOT EXISTS (SELECT 1 FROM atlas_assessments current_assessment "
            "WHERE current_assessment.load_id = :load_id "
            "AND current_assessment.pipeline_key = previous.pipeline_key)",
            {"load_id": load_id, "prior_load_id": prior_load_id},
        )["rows"]
        total = 0
        for row in missing:
            result = query(
                f"INSERT INTO atlas_{table} (load_id, {', '.join(names)}) "
                f"SELECT :load_id, {', '.join('previous.' + name for name in names)} "
                f"FROM atlas_{table} previous WHERE previous.load_id = :prior_load_id "
                "AND previous.pipeline_key = :pipeline_key "
                "AND NOT EXISTS (SELECT 1 FROM atlas_assessments current_assessment "
                "WHERE current_assessment.load_id = :load_id "
                "AND current_assessment.pipeline_key = previous.pipeline_key)",
                {"load_id": load_id, "prior_load_id": prior_load_id, "pipeline_key": row["pipeline_key"]},
            )
            total += result["row_count"]
        copied[table] = total
    return copied

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Store top curriculum subjects once per immutable curriculum load."""

from .cohort_summaries import SummaryQuery

SOURCE_AREA_VERSION = 1


def materialize_source_areas(query: SummaryQuery, load_id: str) -> None:
    """Fill missing or obsolete source summaries before selecting a release."""
    parameters = {"load_id": load_id, "version": SOURCE_AREA_VERSION}
    missing = query(
        "SELECT s.source_id FROM curriculum_sources s LEFT JOIN curriculum_source_areas a "
        "ON a.load_id=s.load_id AND a.source_id=s.source_id AND a.version=:version "
        "WHERE s.load_id=:load_id AND a.source_id IS NULL LIMIT 1",
        parameters,
    )
    if not missing:
        return
    query(
        """
        INSERT INTO curriculum_source_areas (load_id,source_id,version,area)
        SELECT s.load_id,s.source_id,:version,
            CASE WHEN a.node_id IS NULL THEN NULL ELSE jsonb_build_object(
                'subject_id',a.subject_id,'node_id',a.node_id,'name',a.name,'instances',a.instances
            ) END
        FROM curriculum_sources s LEFT JOIN (
            SELECT DISTINCT ON (statistics.source_id)
                statistics.source_id,n.subject_id,n.node_id,n.name,statistics.instances
            FROM curriculum_statistics statistics JOIN curriculum_nodes n USING (load_id,subject_id,node_id)
            WHERE statistics.load_id=:load_id AND statistics.metric='placement'
                AND n.kind='subject' AND statistics.instances>0
            ORDER BY statistics.source_id,statistics.instances DESC,n.subject_id
        ) a ON a.source_id=s.source_id
        WHERE s.load_id=:load_id
        ON CONFLICT (load_id,source_id) DO UPDATE SET version=EXCLUDED.version,area=EXCLUDED.area
        WHERE curriculum_source_areas.version<>EXCLUDED.version
        """,
        parameters,
    )

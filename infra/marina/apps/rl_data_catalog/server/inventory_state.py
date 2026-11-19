# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Track the committed inputs of source inventory payloads."""

from dataclasses import dataclass
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Connection


@dataclass(frozen=True)
class CachedInventory:
    input_key: str
    payload: dict[str, Any]


# This is a disposable invalidation key, not an evidence digest. Primary keys
# detect additions/deletions; xmin detects updates, and the transaction epoch
# prevents reuse across XID wraparound. Read only the tables the inventory uses:
# top areas come from their materialized rows, never raw curriculum statistics.
INVENTORY_KEY_SQL = """
SELECT md5(jsonb_build_array(
    (txid_snapshot_xmin(txid_current_snapshot()) >> 32)::text,
    (SELECT jsonb_build_array(l.load_id,l.xmin::text)
        FROM atlas_loads l WHERE l.load_id=:load_id),
    (SELECT md5(string_agg(jsonb_build_array(s.id,s.xmin::text)::text,','
        ORDER BY s.id))
        FROM atlas_sources s WHERE s.load_id=:load_id),
    (SELECT md5(string_agg(jsonb_build_array(b.source_id,b.xmin::text)::text,','
        ORDER BY b.source_id))
        FROM atlas_backend_sources b WHERE b.load_id=:load_id),
    (SELECT md5(string_agg(jsonb_build_array(a.pipeline_key,a.xmin::text)::text,','
        ORDER BY a.pipeline_key))
        FROM atlas_assessments a WHERE a.load_id=:load_id),
    (SELECT md5(string_agg(jsonb_build_array(s.pipeline_key,s.cohort,s.xmin::text)::text,','
        ORDER BY s.pipeline_key,s.cohort,s.version))
        FROM atlas_cohort_summaries s WHERE s.load_id=:load_id),
    (SELECT md5(string_agg(jsonb_build_array(q.load_id,q.source_id,q.xmin::text,
        a.xmin::text,l.xmin::text,s.xmin::text)::text,',' ORDER BY q.load_id,q.source_id))
        FROM atlas_review_current q JOIN atlas_assessments a USING (load_id,pipeline_key)
        JOIN atlas_loads l USING (load_id)
        LEFT JOIN atlas_cohort_summaries s ON s.load_id=q.load_id AND s.pipeline_key=q.pipeline_key
            AND s.cohort='admitted'
        WHERE a.scope='full'),
    (SELECT md5(string_agg(jsonb_build_array(measurement_id,xmin::text)::text,','
        ORDER BY measurement_id))
        FROM atlas_difficulty),
    (SELECT md5(string_agg(jsonb_build_array(source_id,receipt_id,xmin::text)::text,','
        ORDER BY source_id,receipt_id))
        FROM atlas_derived_contracts),
    (SELECT md5(string_agg(jsonb_build_array(left_source,right_source,xmin::text)::text,','
        ORDER BY left_source,right_source))
        FROM atlas_duplicate_reviews WHERE load_id=:load_id),
    (SELECT jsonb_build_array(c.measurement_id,c.xmin::text)
        FROM atlas_identity_measurement_current c WHERE c.load_id=:load_id),
    (SELECT md5(string_agg(jsonb_build_array(r.left_source,r.right_source,r.xmin::text)::text,','
        ORDER BY r.left_source,r.right_source))
        FROM atlas_identity_measurement_reviews r
        JOIN atlas_identity_measurement_current c USING (load_id,measurement_id)
        WHERE r.load_id=:load_id),
    (SELECT md5(string_agg(jsonb_build_array(c.name,c.xmin::text)::text,','
        ORDER BY c.name)) FROM curriculum_current c),
    (SELECT md5(string_agg(jsonb_build_array(s.source_id,s.xmin::text)::text,','
        ORDER BY s.source_id))
        FROM curriculum_sources s JOIN curriculum_current c USING (load_id) WHERE c.name='latest'),
    (SELECT md5(string_agg(jsonb_build_array(s.source_id,s.xmin::text)::text,','
        ORDER BY s.source_id))
        FROM curriculum_source_areas s JOIN curriculum_current c USING (load_id) WHERE c.name='latest')
)::text)
"""


def inventory_input_key(connection: Connection, load_id: str) -> str:
    """Return the committed row-version key for the selected inventory's inputs."""
    return connection.execute(text(INVENTORY_KEY_SQL), {"load_id": load_id}).scalar_one()

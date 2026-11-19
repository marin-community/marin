# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical cohort counters and grade projection for views and bounded staging."""

COHORT_COUNTS_SQL = """
SELECT p.load_id, p.pipeline_key, p.cohort, p.population, p.sample_count,
    COUNT(i.task_id) FILTER (WHERE i.status = 'reviewed') AS reviewed,
    COUNT(i.task_id) FILTER (WHERE NOT i.has_task) AS missing_evidence,
    COUNT(i.task_id) FILTER (WHERE jsonb_array_length(i.omissions) > 0) AS packets_with_omissions,
    jsonb_build_object(
        'good', COUNT(i.task_id) FILTER (WHERE i.verdict = 'good'),
        'some_issues', COUNT(i.task_id) FILTER (WHERE i.verdict = 'some_issues'),
        'bad', COUNT(i.task_id) FILTER (WHERE i.verdict = 'bad'),
        'unknown', COUNT(i.task_id) FILTER (WHERE i.verdict = 'unknown')
    ) AS verdict_counts,
    jsonb_build_object(
        'missing_context', COUNT(i.task_id) FILTER (WHERE i.defects @> '["missing_context"]'::jsonb),
        'ambiguity', COUNT(i.task_id) FILTER (WHERE i.defects @> '["ambiguity"]'::jsonb),
        'wrong_reference', COUNT(i.task_id) FILTER (WHERE i.defects @> '["wrong_reference"]'::jsonb),
        'answer_leakage', COUNT(i.task_id) FILTER (WHERE i.defects @> '["answer_leakage"]'::jsonb),
        'malformed', COUNT(i.task_id) FILTER (WHERE i.defects @> '["malformed"]'::jsonb),
        'rubric_mismatch', COUNT(i.task_id) FILTER (WHERE i.defects @> '["rubric_mismatch"]'::jsonb)
    ) AS defect_counts,
    jsonb_build_object(
        'high', COUNT(i.task_id) FILTER (WHERE i.confidence = 'high'),
        'medium', COUNT(i.task_id) FILTER (WHERE i.confidence = 'medium'),
        'low', COUNT(i.task_id) FILTER (WHERE i.confidence = 'low'),
        'unrecorded', COUNT(i.task_id) FILTER (WHERE i.confidence IS NULL)
    ) AS confidence_counts,
    COUNT(i.task_id) AS item_count,
    COUNT(i.task_id) FILTER (
        WHERE i.status = 'reviewed'
        AND (
            (i.has_task AND i.verdict = 'good')
            OR (
                a.scope = 'full' AND NOT i.has_task AND i.confidence IS DISTINCT FROM 'low'
                AND i.verdict = 'good' AND COALESCE(jsonb_array_length(i.defects), 0) = 0
                AND i.reference_status IS DISTINCT FROM 'conflict'
            )
        )
    ) AS good_count,
    COUNT(i.task_id) FILTER (
        WHERE i.status = 'reviewed'
        AND (
            (i.has_task AND i.verdict <> 'unknown')
            OR (
                a.scope = 'full' AND NOT i.has_task AND i.confidence IS DISTINCT FROM 'low'
                AND (
                    i.verdict <> 'unknown'
                    OR COALESCE(jsonb_array_length(i.defects), 0) > 0
                    OR i.reference_status = 'conflict'
                )
            )
        )
    ) AS conclusive_count
FROM atlas_populations p JOIN atlas_assessments a USING (load_id, pipeline_key)
LEFT JOIN atlas_items i
    ON i.load_id = p.load_id AND i.pipeline_key = p.pipeline_key AND i.cohort = p.cohort
    /* selected_items */
/* selected_population */
GROUP BY p.load_id, p.pipeline_key, p.cohort, p.population, p.sample_count
"""

COHORT_PROJECTION_SQL = """
SELECT load_id, pipeline_key, cohort, population, sample_count, reviewed,
    missing_evidence, packets_with_omissions, verdict_counts,
    CASE
        WHEN sample_count = 0 OR item_count <> sample_count OR conclusive_count = 0 THEN 'unknown'
        WHEN 2 * (conclusive_count - good_count) >= conclusive_count THEN 'bad'
        WHEN 10 * (conclusive_count - good_count) <= conclusive_count THEN 'good'
        ELSE 'some_issues'
    END AS grade,
    good_count::double precision / NULLIF(conclusive_count, 0) AS good_fraction,
    CASE
        WHEN sample_count = 0 THEN 'No sample available'
        WHEN item_count <> sample_count THEN 'Sample coverage is incomplete'
        WHEN conclusive_count = 0 THEN 'No conclusive reviewed tasks'
        WHEN 2 * (conclusive_count - good_count) >= conclusive_count THEN 'At least 50% of reviewed tasks have issues'
        WHEN 10 * (conclusive_count - good_count) <= conclusive_count THEN 'At least 90% of reviewed tasks are clean'
        ELSE 'More than 10% and fewer than 50% of reviewed tasks have issues'
    END AS grade_reason,
    conclusive_count AS result_reviewed,
    conclusive_count - good_count AS issue_count,
    (conclusive_count - good_count)::double precision / NULLIF(conclusive_count, 0) AS issue_rate,
    reviewed - conclusive_count AS uncertain_count,
    sample_count - reviewed AS unavailable_count,
    -- Appended after the retained column prefix: Postgres refuses CREATE OR REPLACE VIEW
    -- when an existing column changes name or position.
    defect_counts,
    confidence_counts
FROM counts;

"""

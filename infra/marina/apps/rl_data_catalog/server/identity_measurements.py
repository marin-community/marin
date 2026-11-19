# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select current-input overlap evidence without rewriting retained catalog archives."""

IDENTITY_REVIEWS_SQL = """
SELECT r.review FROM atlas_identity_measurement_reviews r
JOIN atlas_identity_measurement_current c USING (load_id,measurement_id)
WHERE r.load_id=:load_id
UNION ALL
SELECT r.review FROM atlas_duplicate_reviews r WHERE r.load_id=:load_id
AND NOT EXISTS (SELECT 1 FROM atlas_identity_measurement_current c WHERE c.load_id=:load_id)
"""
IDENTITY_SURVEY_SQL = """
SELECT m.measurement_id AS survey_id,m.metadata FROM atlas_identity_measurements m
JOIN atlas_identity_measurement_current c USING (load_id,measurement_id) WHERE m.load_id=:load_id
UNION ALL
SELECT survey_id,metadata FROM atlas_identity_surveys WHERE load_id=:load_id
AND NOT EXISTS (SELECT 1 FROM atlas_identity_measurement_current c WHERE c.load_id=:load_id)
"""

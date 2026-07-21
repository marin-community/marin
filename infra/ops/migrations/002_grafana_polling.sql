-- Copyright The Marin Authors
-- SPDX-License-Identifier: Apache-2.0

CREATE TABLE grafana_polls (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    poll_slot timestamptz NOT NULL UNIQUE,
    observed_at timestamptz NOT NULL,
    alert_count integer NOT NULL CHECK (alert_count >= 0),
    created_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE signals
    ADD COLUMN missing_successful_polls integer NOT NULL DEFAULT 0
    CHECK (missing_successful_polls >= 0);

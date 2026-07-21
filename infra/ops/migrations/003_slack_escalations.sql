-- Copyright The Marin Authors
-- SPDX-License-Identifier: Apache-2.0

CREATE TABLE slack_escalations (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    incident_key text NOT NULL UNIQUE,
    case_id uuid NOT NULL REFERENCES cases(id),
    turn_id uuid NOT NULL UNIQUE REFERENCES agent_turns(id),
    severity text NOT NULL CHECK (severity IN ('error', 'critical')),
    reason text NOT NULL,
    message text NOT NULL,
    state text NOT NULL CHECK (state IN ('pending', 'sending', 'sent', 'abandoned')),
    attempts integer NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    available_at timestamptz NOT NULL DEFAULT now(),
    lease_expires_at timestamptz,
    last_error text,
    created_at timestamptz NOT NULL DEFAULT now(),
    sent_at timestamptz
);

CREATE INDEX slack_escalations_delivery_queue
    ON slack_escalations (available_at, created_at)
    WHERE state = 'pending';

CREATE INDEX slack_escalations_case
    ON slack_escalations (case_id, created_at DESC);

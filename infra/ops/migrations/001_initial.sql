-- Copyright The Marin Authors
-- SPDX-License-Identifier: Apache-2.0

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE source_deliveries (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source text NOT NULL,
    delivery_key text NOT NULL,
    key_id text NOT NULL,
    source_timestamp timestamptz NOT NULL,
    received_at timestamptz NOT NULL DEFAULT now(),
    body_sha256 text NOT NULL,
    normalized_payload jsonb NOT NULL,
    result jsonb,
    UNIQUE (source, delivery_key)
);

CREATE TABLE signals (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source text NOT NULL,
    fingerprint text NOT NULL,
    generation integer NOT NULL DEFAULT 1,
    state text NOT NULL CHECK (state IN ('firing', 'resolved')),
    receiver text NOT NULL,
    group_key text NOT NULL,
    alert_name text NOT NULL,
    severity text NOT NULL,
    cluster text,
    namespace text,
    object_kind text,
    object_name text,
    summary text NOT NULL,
    labels jsonb NOT NULL,
    annotations jsonb NOT NULL,
    values jsonb NOT NULL,
    generator_url text,
    silence_url text,
    dashboard_url text,
    panel_url text,
    source_version text NOT NULL,
    first_seen_at timestamptz NOT NULL,
    last_seen_at timestamptz NOT NULL,
    latest_source_timestamp timestamptz NOT NULL,
    resolved_at timestamptz,
    latest_delivery_id uuid NOT NULL REFERENCES source_deliveries(id),
    missing_successful_polls integer NOT NULL DEFAULT 0 CHECK (missing_successful_polls >= 0),
    UNIQUE (source, fingerprint)
);

CREATE TABLE grafana_polls (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    poll_slot timestamptz NOT NULL UNIQUE,
    observed_at timestamptz NOT NULL,
    alert_count integer NOT NULL CHECK (alert_count >= 0),
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE delivery_signals (
    delivery_id uuid NOT NULL REFERENCES source_deliveries(id) ON DELETE CASCADE,
    signal_id uuid NOT NULL REFERENCES signals(id),
    disposition text NOT NULL CHECK (disposition IN (
        'created', 'updated', 'resolved', 'reopened', 'stale'
    )),
    PRIMARY KEY (delivery_id, signal_id)
);

CREATE TABLE cases (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    trigger text NOT NULL CHECK (trigger IN ('automatic', 'manual')),
    receiver text NOT NULL,
    group_key text NOT NULL,
    grouping_rule text NOT NULL,
    state text NOT NULL CHECK (state IN (
        'pending', 'investigating', 'waiting_human',
        'investigated', 'failed', 'archived'
    )),
    priority integer NOT NULL,
    title text NOT NULL,
    question text,
    opened_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    next_eligible_at timestamptz NOT NULL DEFAULT now(),
    investigated_at timestamptz,
    archived_at timestamptz,
    outcome text CHECK (outcome IN (
        'no_action', 'action_recommended', 'blocked', 'unknown'
    )),
    summary text,
    CHECK (
        (trigger = 'automatic' AND question IS NULL)
        OR
        (trigger = 'manual' AND question IS NOT NULL)
    )
);

CREATE UNIQUE INDEX cases_one_open_group
    ON cases (receiver, group_key)
    WHERE state <> 'archived';

CREATE TABLE case_signals (
    case_id uuid NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
    signal_id uuid NOT NULL REFERENCES signals(id),
    signal_generation integer NOT NULL,
    attached_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (case_id, signal_id, signal_generation)
);

CREATE UNIQUE INDEX case_signals_one_case_per_generation
    ON case_signals (signal_id, signal_generation);

CREATE TABLE agent_sessions (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    case_id uuid NOT NULL REFERENCES cases(id),
    deterministic_name text NOT NULL UNIQUE,
    loom_session_id text UNIQUE,
    loom_session_url text,
    state text NOT NULL CHECK (state IN (
        'new', 'ready', 'active', 'idle', 'lost', 'archived'
    )),
    repo_revision text NOT NULL,
    skill_revision text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    archived_at timestamptz
);

CREATE UNIQUE INDEX agent_sessions_one_current_per_case
    ON agent_sessions (case_id)
    WHERE state <> 'archived';

CREATE TABLE agent_turns (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id uuid NOT NULL REFERENCES agent_sessions(id),
    kind text NOT NULL CHECK (kind IN ('automatic', 'question', 'follow_up')),
    state text NOT NULL CHECK (state IN (
        'queued', 'launching', 'running', 'succeeded',
        'failed', 'interrupted', 'cancelled'
    )),
    priority integer NOT NULL,
    requested_by text NOT NULL,
    client_request_id text NOT NULL,
    prompt text NOT NULL,
    available_at timestamptz NOT NULL DEFAULT now(),
    lease_owner text,
    lease_expires_at timestamptz,
    attempt integer NOT NULL DEFAULT 1,
    retry_of uuid REFERENCES agent_turns(id),
    loom_turn_number bigint,
    result_artifact_revision integer,
    result jsonb,
    error text,
    created_at timestamptz NOT NULL DEFAULT now(),
    started_at timestamptz,
    deadline_at timestamptz,
    completed_at timestamptz,
    UNIQUE (requested_by, client_request_id)
);

CREATE UNIQUE INDEX agent_turns_one_queued_per_session
    ON agent_turns (session_id)
    WHERE state = 'queued';

CREATE UNIQUE INDEX agent_turns_one_active_per_session
    ON agent_turns (session_id)
    WHERE state IN ('launching', 'running');

CREATE UNIQUE INDEX agent_turns_one_active_global
    ON agent_turns ((true))
    WHERE state IN ('launching', 'running');

CREATE TABLE case_events (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    case_id uuid NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
    turn_id uuid REFERENCES agent_turns(id) ON DELETE SET NULL,
    event_type text NOT NULL,
    actor text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    data jsonb NOT NULL DEFAULT '{}'::jsonb
);

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

CREATE INDEX cases_queue_order
    ON cases (priority DESC, next_eligible_at, opened_at)
    WHERE state = 'pending';

CREATE INDEX signals_group_state
    ON signals (receiver, group_key, state, last_seen_at DESC);

CREATE INDEX case_signals_case
    ON case_signals (case_id, attached_at);

CREATE INDEX agent_turns_queue_order
    ON agent_turns (priority DESC, available_at, created_at)
    WHERE state = 'queued';

CREATE INDEX agent_turns_expired_leases
    ON agent_turns (lease_expires_at)
    WHERE state IN ('launching', 'running');

CREATE INDEX case_events_timeline
    ON case_events (case_id, created_at, id);

CREATE INDEX slack_escalations_delivery_queue
    ON slack_escalations (available_at, created_at)
    WHERE state = 'pending';

CREATE INDEX slack_escalations_case
    ON slack_escalations (case_id, created_at DESC);

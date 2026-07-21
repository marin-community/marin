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

CREATE TABLE source_streams (
    source text NOT NULL,
    cluster text NOT NULL,
    latest_observed_at timestamptz NOT NULL,
    latest_body_sha256 text NOT NULL,
    latest_delivery_id uuid NOT NULL REFERENCES source_deliveries(id),
    PRIMARY KEY (source, cluster)
);

CREATE TABLE signals (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source text NOT NULL,
    fingerprint text NOT NULL,
    generation integer NOT NULL DEFAULT 1,
    state text NOT NULL CHECK (state IN ('firing', 'resolved')),
    cluster text NOT NULL,
    namespace text,
    object_kind text,
    object_name text,
    reason text NOT NULL,
    summary text NOT NULL,
    source_version text NOT NULL,
    first_seen_at timestamptz NOT NULL,
    last_seen_at timestamptz NOT NULL,
    last_snapshot_at timestamptz NOT NULL,
    resolved_at timestamptz,
    occurrence_count bigint NOT NULL,
    latest_delivery_id uuid NOT NULL REFERENCES source_deliveries(id),
    UNIQUE (source, fingerprint)
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
    signal_id uuid REFERENCES signals(id),
    signal_generation integer,
    trigger text NOT NULL CHECK (trigger IN ('automatic', 'manual')),
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
        (trigger = 'automatic' AND signal_id IS NOT NULL
            AND signal_generation IS NOT NULL AND question IS NULL)
        OR
        (trigger = 'manual' AND signal_id IS NULL
            AND signal_generation IS NULL AND question IS NOT NULL)
    ),
    UNIQUE (signal_id, signal_generation)
);

CREATE TABLE agent_sessions (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    case_id uuid NOT NULL REFERENCES cases(id),
    deterministic_name text NOT NULL UNIQUE,
    loom_session_id text UNIQUE,
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

CREATE UNIQUE INDEX agent_turns_one_pending_per_session
    ON agent_turns (session_id)
    WHERE state IN ('queued', 'launching', 'running');

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

CREATE TABLE operation_requests (
    actor text NOT NULL,
    idempotency_key text NOT NULL,
    operation text NOT NULL,
    request_sha256 text NOT NULL,
    response_status integer NOT NULL,
    response_body jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (actor, idempotency_key)
);

CREATE INDEX cases_queue_order
    ON cases (priority DESC, next_eligible_at, opened_at)
    WHERE state = 'pending';

CREATE INDEX agent_turns_queue_order
    ON agent_turns (priority DESC, available_at, created_at)
    WHERE state = 'queued';

CREATE INDEX agent_turns_expired_leases
    ON agent_turns (lease_expires_at)
    WHERE state IN ('launching', 'running');

CREATE INDEX case_events_timeline
    ON case_events (case_id, created_at, id);

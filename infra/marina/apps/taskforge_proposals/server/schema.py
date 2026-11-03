# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tables of the proposal viewer; ``ingest.py`` fills them and ``app.py`` reads them."""

SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS capabilities (
        id TEXT PRIMARY KEY,
        name TEXT,
        outcome TEXT,
        subject_id TEXT,
        domain TEXT,
        area TEXT,
        section_path TEXT,
        status TEXT NOT NULL,
        slots_planned INTEGER,
        proposed INTEGER,
        accepted INTEGER,
        rejected INTEGER,
        cut_off INTEGER,
        requests INTEGER,
        prompt_tokens BIGINT,
        completion_tokens BIGINT,
        reasoning_tokens BIGINT,
        cached_tokens BIGINT,
        wall_time DOUBLE PRECISION,
        plan_prompt_tokens INTEGER,
        plan_completion_tokens INTEGER,
        plan_reasoning_tokens INTEGER,
        coverage_rationale TEXT,
        git_sha TEXT,
        plan JSONB,
        result JSONB,
        failure JSONB,
        catalog_entry JSONB,
        plan_request_path TEXT,
        plan_completions_path TEXT,
        source_key TEXT NOT NULL,
        loaded_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS proposals (
        capability_id TEXT NOT NULL,
        slot INTEGER NOT NULL,
        subject_id TEXT,
        domain TEXT,
        area TEXT,
        status TEXT NOT NULL,
        title TEXT,
        workflow TEXT,
        challenge TEXT,
        summary TEXT,
        plan_status TEXT,
        planned_verification TEXT,
        planned_environment TEXT,
        verification TEXT,
        environment TEXT,
        solver_machine TEXT,
        grounding TEXT,
        research_need TEXT,
        verdict TEXT,
        reason TEXT,
        cut_off BOOLEAN NOT NULL,
        failure TEXT,
        completions INTEGER,
        prompt_tokens INTEGER,
        completion_tokens INTEGER,
        reasoning_tokens INTEGER,
        cached_tokens INTEGER,
        finish_reasons JSONB,
        wall_time DOUBLE PRECISION,
        body_chars INTEGER,
        request_path TEXT,
        completions_path TEXT,
        completions_bytes BIGINT,
        proposal JSONB,
        verdict_record JSONB,
        source_key TEXT NOT NULL,
        loaded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        search TSVECTOR GENERATED ALWAYS AS (
            setweight(to_tsvector('english'::regconfig, coalesce(title, '')), 'A')
            || setweight(to_tsvector('english'::regconfig,
                coalesce(summary, '') || ' ' || coalesce(workflow, '') || ' ' || coalesce(challenge, '')), 'B')
            || setweight(to_tsvector('english'::regconfig, coalesce(proposal ->> 'body', '')), 'C')
        ) STORED,
        PRIMARY KEY (capability_id, slot)
    )
    """,
    "CREATE INDEX IF NOT EXISTS proposals_search ON proposals USING GIN (search)",
    "CREATE INDEX IF NOT EXISTS proposals_domain ON proposals (domain)",
    "CREATE INDEX IF NOT EXISTS proposals_verification ON proposals (verification)",
    "CREATE INDEX IF NOT EXISTS proposals_environment ON proposals (environment)",
    "CREATE INDEX IF NOT EXISTS proposals_verdict ON proposals (verdict)",
    "CREATE INDEX IF NOT EXISTS proposals_completion_tokens ON proposals (completion_tokens)",
    "CREATE INDEX IF NOT EXISTS capabilities_domain ON capabilities (domain)",
    """
    CREATE TABLE IF NOT EXISTS run_stats (
        key TEXT PRIMARY KEY,
        value JSONB NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
)

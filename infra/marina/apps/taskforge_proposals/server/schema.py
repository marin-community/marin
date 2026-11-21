# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tables of the proposal viewer; ``ingest.py`` fills them and ``app.py`` reads them.

``CAPABILITY_COLUMNS`` and ``PROPOSAL_COLUMNS`` are the columns ingest loads, in the column-definition
syntax both ``CREATE TABLE`` and ``jsonb_to_recordset`` accept, so the table and the loader share one list.
"""

TS_CONFIG = "english"
"""The text-search configuration of the stored ``search`` vector; queries must parse with the same one."""

CAPABILITY_COLUMNS = (
    "id TEXT, name TEXT, outcome TEXT, subject_id TEXT, domain TEXT, area TEXT, section_path TEXT, status TEXT, "
    "slots_planned INTEGER, proposed INTEGER, accepted INTEGER, rejected INTEGER, cut_off INTEGER, "
    "requests INTEGER, prompt_tokens BIGINT, completion_tokens BIGINT, reasoning_tokens BIGINT, "
    "cached_tokens BIGINT, wall_time DOUBLE PRECISION, plan_prompt_tokens INTEGER, plan_completion_tokens INTEGER, "
    "plan_reasoning_tokens INTEGER, coverage_rationale TEXT, git_sha TEXT, plan JSONB, result JSONB, "
    "failure JSONB, catalog_entry JSONB, plan_request_path TEXT, plan_completions_path TEXT, source_key TEXT"
)
PROPOSAL_COLUMNS = (
    "capability_id TEXT, slot INTEGER, subject_id TEXT, domain TEXT, area TEXT, status TEXT, title TEXT, "
    "workflow TEXT, challenge TEXT, summary TEXT, plan_status TEXT, planned_verification TEXT, "
    "planned_environment TEXT, verification TEXT, environment TEXT, solver_machine TEXT, grounding TEXT, "
    "research_need TEXT, verdict TEXT, reason TEXT, cut_off BOOLEAN, failure TEXT, completions INTEGER, "
    "prompt_tokens INTEGER, completion_tokens INTEGER, reasoning_tokens INTEGER, cached_tokens INTEGER, "
    "finish_reasons JSONB, wall_time DOUBLE PRECISION, body_chars INTEGER, request_path TEXT, "
    "completions_path TEXT, completions_bytes BIGINT, proposal JSONB, verdict_record JSONB, superseded BOOLEAN, "
    "source_key TEXT"
)
SEARCH_VECTOR = f"""
    setweight(to_tsvector('{TS_CONFIG}'::regconfig, coalesce(title, '')), 'A')
    || setweight(to_tsvector('{TS_CONFIG}'::regconfig,
        coalesce(summary, '') || ' ' || coalesce(workflow, '') || ' ' || coalesce(challenge, '')), 'B')
    || setweight(to_tsvector('{TS_CONFIG}'::regconfig, coalesce(proposal ->> 'body', '')), 'C')
"""

SCHEMA_STATEMENTS = (
    f"""
    CREATE TABLE IF NOT EXISTS capabilities (
        {CAPABILITY_COLUMNS},
        loaded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        PRIMARY KEY (id)
    )
    """,
    f"""
    CREATE TABLE IF NOT EXISTS proposals (
        {PROPOSAL_COLUMNS},
        loaded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        search TSVECTOR GENERATED ALWAYS AS ({SEARCH_VECTOR}) STORED,
        PRIMARY KEY (capability_id, slot)
    )
    """,
    # CREATE TABLE IF NOT EXISTS leaves an existing table as it is, so a column added after a table
    # was first created needs its own idempotent ALTER.
    "ALTER TABLE proposals ADD COLUMN IF NOT EXISTS superseded BOOLEAN",
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

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Restricted database operations for a driver assigned to one lab run."""

from uuid import UUID

from sqlalchemy import text
from sqlalchemy.engine import Connection

WORKER_FUNCTION_SIGNATURES = (
    "worker_run(text)",
    "worker_cases(text,text,integer)",
    "worker_case(text,text)",
    "worker_save_run(jsonb)",
    "worker_save_cases(jsonb)",
    "worker_save_traces(jsonb)",
    "worker_claim(text,text)",
)


def grant_worker_functions(connection: Connection, applet: UUID, role: str) -> None:
    """Grant a database login only the lab operations that check its run assignment."""
    privileges = connection.execute(
        text(
            "SELECT rolcanlogin AND NOT (rolsuper OR rolcreaterole OR rolcreatedb OR rolbypassrls) "
            "FROM pg_roles WHERE rolname=:role"
        ),
        {"role": role},
    ).scalar()
    if privileges is not True:
        raise ValueError("Worker must be an existing restricted database login")
    unsafe_membership = connection.execute(
        text(
            "SELECT EXISTS (SELECT 1 FROM pg_roles parent WHERE "
            "(parent.rolsuper OR parent.rolcreaterole OR parent.rolcreatedb OR parent.rolbypassrls "
            "OR parent.rolname='marina_reader' OR starts_with(parent.rolname,'applet_')) "
            "AND pg_has_role(:role,parent.oid,'MEMBER'))"
        ),
        {"role": role},
    ).scalar_one()
    if unsafe_membership:
        raise ValueError("Worker login inherits permissions beyond restricted lab operations")
    namespace = connection.dialect.identifier_preparer.quote_schema("applet_" + applet.hex)
    worker = connection.dialect.identifier_preparer.quote(role)
    connection.execute(text(f"GRANT USAGE ON SCHEMA {namespace} TO {worker}"))
    functions = ",".join(f"{namespace}.{signature}" for signature in WORKER_FUNCTION_SIGNATURES)
    connection.execute(text(f"GRANT EXECUTE ON FUNCTION {functions} TO {worker}"))


def install_worker_functions(connection: Connection) -> None:
    schema = connection.execute(text("SELECT current_schema()")).scalar_one()
    namespace = connection.dialect.identifier_preparer.quote_schema(schema)
    bodies = {
        "worker_require_run(run_name text) RETURNS void": (
            f"""
            BEGIN
                IF NOT EXISTS (SELECT 1 FROM {namespace}.runs
                    WHERE run_id=run_name AND database_worker=session_user) THEN
                    RAISE EXCEPTION 'Database login is not assigned to this run' USING ERRCODE='42501';
                END IF;
            END
        """
        ),
        "worker_run(run_name text) RETURNS jsonb": (
            f"""
            BEGIN
                PERFORM {namespace}.worker_require_run(run_name);
                RETURN (SELECT to_jsonb(r) FROM {namespace}.runs r WHERE run_id=run_name);
            END
        """
        ),
        "worker_cases(run_name text, case_state text, page_limit integer) RETURNS jsonb": (
            f"""
            BEGIN
                PERFORM {namespace}.worker_require_run(run_name);
                IF page_limit < 1 OR page_limit > 200 THEN
                    RAISE EXCEPTION 'Case page limit must be between 1 and 200' USING ERRCODE='22023';
                END IF;
                RETURN jsonb_build_object(
                    'total', (SELECT count(*) FROM {namespace}.cases
                        WHERE run_id=run_name AND (case_state='' OR state=case_state)),
                    'cases', COALESCE((SELECT jsonb_agg(value) FROM (
                        SELECT to_jsonb(c) || jsonb_build_object('metadata_payload', encode(m.payload,'base64')) AS value
                        FROM {namespace}.cases c LEFT JOIN {namespace}.case_metadata m USING (case_id)
                        WHERE run_id=run_name AND (case_state='' OR state=case_state)
                        ORDER BY updated_at DESC LIMIT page_limit
                    ) selected), '[]'::jsonb));
            END
        """
        ),
        "worker_case(case_name text, trace_state text) RETURNS jsonb": (
            f"""
            DECLARE result jsonb; run_name text;
            BEGIN
                SELECT run_id INTO run_name FROM {namespace}.cases WHERE case_id=case_name;
                PERFORM {namespace}.worker_require_run(run_name);
                SELECT to_jsonb(c) || jsonb_build_object('metadata_payload', encode(m.payload,'base64'))
                    INTO result FROM {namespace}.cases c LEFT JOIN {namespace}.case_metadata m USING (case_id)
                    WHERE case_id=case_name;
                RETURN result || jsonb_build_object('traces', COALESCE((SELECT jsonb_agg(value) FROM (
                    SELECT to_jsonb(t) || jsonb_build_object('payload',encode(t.payload,'base64')) AS value
                    FROM {namespace}.traces t WHERE case_id=case_name AND (trace_state='' OR state=trace_state)
                    ORDER BY sequence,model,message_id
                ) selected), '[]'::jsonb));
            END
        """
        ),
        "worker_save_run(record jsonb) RETURNS jsonb": (
            f"""
            DECLARE previous_config jsonb; next_config jsonb;
            BEGIN
                PERFORM {namespace}.worker_require_run(record->>'run_id');
                SELECT config INTO previous_config FROM {namespace}.runs WHERE run_id=record->>'run_id' FOR UPDATE;
                next_config := record->'config';
                IF (previous_config - 'coordinator' - 'driver') || jsonb_build_object(
                        'driver',COALESCE(previous_config->'driver','{{}}'::jsonb) - 'gateway')
                    IS DISTINCT FROM (next_config - 'coordinator' - 'driver') || jsonb_build_object(
                        'driver',COALESCE(next_config->'driver','{{}}'::jsonb) - 'gateway') THEN
                    RAISE EXCEPTION 'Run checkpoint/config identity is immutable' USING ERRCODE='23514';
                END IF;
                IF (record->>'generated_tokens')::bigint < 0 OR (record->>'reserved_tokens')::bigint < 0
                    OR (record->>'active_chats')::integer < 0 THEN
                    RAISE EXCEPTION 'Run accounting must be nonnegative' USING ERRCODE='23514';
                END IF;
                UPDATE {namespace}.runs SET config=next_config,
                    generated_tokens=(record->>'generated_tokens')::bigint,
                    reserved_tokens=(record->>'reserved_tokens')::bigint,
                    state=record->>'state', active_chats=(record->>'active_chats')::integer,
                    error=record->>'error', updated_at=now() WHERE run_id=record->>'run_id';
                RETURN (SELECT jsonb_build_object('desired_state',desired_state)
                    FROM {namespace}.runs WHERE run_id=record->>'run_id');
            END
        """
        ),
        "worker_save_cases(records jsonb) RETURNS jsonb": (
            f"""
            DECLARE record jsonb; saved_name text;
            BEGIN
                FOR record IN SELECT value FROM jsonb_array_elements(records) LOOP
                    PERFORM {namespace}.worker_require_run(record->>'run_id');
                    INSERT INTO {namespace}.cases AS c
                        (case_id,run_id,title,category,mode,state,slot,round,seed,review)
                    VALUES (record->>'case_id',record->>'run_id',record->>'title',record->>'category',
                        record->>'mode',record->>'state',(record->>'slot')::integer,(record->>'round')::integer,
                        record->'seed',record->'review')
                    ON CONFLICT (case_id) DO UPDATE SET title=EXCLUDED.title, category=EXCLUDED.category,
                        mode=EXCLUDED.mode, state=EXCLUDED.state, round=EXCLUDED.round,
                        seed=EXCLUDED.seed, review=EXCLUDED.review, updated_at=now()
                    WHERE c.run_id=EXCLUDED.run_id RETURNING case_id INTO saved_name;
                    IF saved_name IS NULL THEN
                        RAISE EXCEPTION 'Case belongs to another run' USING ERRCODE='42501';
                    END IF;
                    INSERT INTO {namespace}.case_metadata (case_id,payload)
                    VALUES (saved_name,decode(record->>'metadata_payload','base64'))
                    ON CONFLICT (case_id) DO UPDATE SET payload=EXCLUDED.payload;
                END LOOP;
                RETURN jsonb_build_object('saved',jsonb_array_length(records));
            END
        """
        ),
        "worker_save_traces(records jsonb) RETURNS jsonb": (
            f"""
            DECLARE record jsonb; run_name text; saved_name text; encoded_payload bytea;
            BEGIN
                FOR record IN SELECT value FROM jsonb_array_elements(records) LOOP
                    SELECT run_id INTO run_name FROM {namespace}.cases WHERE case_id=record->>'case_id';
                    PERFORM {namespace}.worker_require_run(run_name);
                    encoded_payload := decode(record->>'payload','base64');
                    IF octet_length(encoded_payload) > 4194304 THEN
                        RAISE EXCEPTION 'Trace exceeds 4 MiB' USING ERRCODE='22023';
                    END IF;
                    INSERT INTO {namespace}.traces AS t
                        (message_id,case_id,model,role,round,sequence,state,payload)
                    VALUES (record->>'message_id',record->>'case_id',record->>'model',record->>'role',
                        (record->>'round')::integer,(record->>'sequence')::integer,record->>'state',encoded_payload)
                    ON CONFLICT (message_id) DO UPDATE SET state=EXCLUDED.state,payload=EXCLUDED.payload,updated_at=now()
                    WHERE t.case_id=EXCLUDED.case_id RETURNING message_id INTO saved_name;
                    IF saved_name IS NULL THEN
                        RAISE EXCEPTION 'Trace belongs to another case' USING ERRCODE='42501';
                    END IF;
                END LOOP;
                RETURN jsonb_build_object('saved',jsonb_array_length(records));
            END
        """
        ),
        "worker_claim(run_name text, case_name text) RETURNS jsonb": (
            f"""
            DECLARE idea {namespace}.ideas%ROWTYPE;
            BEGIN
                PERFORM {namespace}.worker_require_run(run_name);
                PERFORM run_id FROM {namespace}.runs WHERE run_id=run_name FOR UPDATE;
                SELECT i.* INTO idea FROM {namespace}.ideas i JOIN {namespace}.idea_claims a USING (idea_id)
                    WHERE a.case_id=case_name;
                IF FOUND THEN
                    IF idea.run_id <> run_name THEN
                        RAISE EXCEPTION 'Idea claim belongs to another run' USING ERRCODE='42501';
                    END IF;
                    RETURN jsonb_build_object('idea',to_jsonb(idea));
                END IF;
                IF (SELECT desired_state FROM {namespace}.runs WHERE run_id=run_name) <> 'running' THEN
                    RETURN jsonb_build_object('idea',NULL);
                END IF;
                SELECT * INTO idea FROM {namespace}.ideas WHERE run_id=run_name AND NOT cancelled AND assigned<chats
                    ORDER BY created_at,idea_id LIMIT 1 FOR UPDATE;
                IF NOT FOUND THEN RETURN jsonb_build_object('idea',NULL); END IF;
                INSERT INTO {namespace}.idea_claims(case_id,idea_id) VALUES(case_name,idea.idea_id);
                UPDATE {namespace}.ideas SET assigned=assigned+1 WHERE idea_id=idea.idea_id;
                RETURN jsonb_build_object('idea',to_jsonb(idea));
            END
        """
        ),
    }
    for signature, body in bodies.items():
        name_and_arguments = signature.split(" RETURNS ")[0]
        connection.execute(
            text(
                f"""CREATE OR REPLACE FUNCTION {namespace}.{signature}
            LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $worker$ {body} $worker$"""
            )
        )
        connection.execute(text(f"REVOKE ALL ON FUNCTION {namespace}.{name_and_arguments} FROM PUBLIC"))

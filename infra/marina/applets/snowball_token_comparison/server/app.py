# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Private, durable trace storage and comparison queries for the conversation lab."""

import json
import secrets
import uuid
import zlib
from enum import StrEnum
from threading import Lock
from time import monotonic
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from marina.applets import AppletServices
from pydantic import BaseModel, Field
from rigging.server_auth import get_verified_identity
from sqlalchemy import inspect, text
from sqlalchemy.engine import Connection
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse

from .records import case_record, searchable_metadata
from .sessions import SESSION_HEADER, key_hash, session_run, session_run_id

SUMMARY_TTL = 8.0
SUMMARY_CACHE_SIZE = 128


class RunState(StrEnum):
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"


class RunUpdate(BaseModel):
    run_id: str
    config: dict
    token_cap: int = Field(gt=0)
    concurrency: int = Field(gt=0)
    generated_tokens: int = Field(ge=0)
    reserved_tokens: int = Field(ge=0)
    state: str
    active_chats: int = Field(ge=0)
    error: str | None = None


class CaseUpdate(BaseModel):
    case_id: str
    run_id: str
    title: str
    category: str
    mode: str
    state: str
    slot: int
    round: int
    seed: dict
    review: dict = Field(default_factory=dict)


class TraceUpdate(BaseModel):
    message_id: str
    case_id: str
    model: str
    role: str
    round: int
    sequence: int
    state: str
    payload: dict


class CaseBatch(BaseModel):
    cases: list[CaseUpdate] = Field(min_length=1, max_length=128)


class TraceBatch(BaseModel):
    traces: list[TraceUpdate] = Field(min_length=1, max_length=128)


class Control(BaseModel):
    state: RunState


class IdeaCreate(BaseModel):
    text: str = Field(min_length=1, max_length=6000)
    kind: Literal["topic", "prompt"]
    chats: int = Field(ge=1, le=100)
    mode: Literal["auto", "chat", "agentic"]


class IdeaClaim(BaseModel):
    case_id: str


class WorkerGrant(BaseModel):
    identity: str | None = Field(pattern=r"^[a-zA-Z0-9._-]+@[a-zA-Z0-9.-]+\.gserviceaccount\.com$")


def migrate(connection: Connection) -> None:
    catalog = inspect(connection)
    schema = connection.execute(text("SELECT current_schema()")).scalar_one()
    tables = set(catalog.get_table_names(schema=schema))
    for name, definition in (
        (
            "runs",
            """(
            run_id TEXT PRIMARY KEY, owner TEXT NOT NULL, config JSONB NOT NULL,
            token_cap BIGINT NOT NULL, concurrency INTEGER NOT NULL, generated_tokens BIGINT NOT NULL DEFAULT 0,
            reserved_tokens BIGINT NOT NULL DEFAULT 0, state TEXT NOT NULL,
            desired_state TEXT NOT NULL DEFAULT 'running',
            active_chats INTEGER NOT NULL DEFAULT 0, error TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(), updated_at TIMESTAMPTZ NOT NULL DEFAULT now())""",
        ),
        (
            "cases",
            """(
            case_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id),
            title TEXT NOT NULL, category TEXT NOT NULL, mode TEXT NOT NULL, state TEXT NOT NULL,
            slot INTEGER NOT NULL, round INTEGER NOT NULL, seed JSONB NOT NULL, review JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(), updated_at TIMESTAMPTZ NOT NULL DEFAULT now())""",
        ),
        (
            "case_metadata",
            """(
            case_id TEXT PRIMARY KEY REFERENCES cases(case_id), payload BYTEA NOT NULL)""",
        ),
        (
            "traces",
            """(
            message_id TEXT PRIMARY KEY, case_id TEXT NOT NULL REFERENCES cases(case_id), model TEXT NOT NULL,
            role TEXT NOT NULL, round INTEGER NOT NULL, sequence INTEGER NOT NULL, state TEXT NOT NULL,
            payload BYTEA NOT NULL, updated_at TIMESTAMPTZ NOT NULL DEFAULT now())""",
        ),
        (
            "ideas",
            """(
            idea_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id),
            text TEXT NOT NULL, kind TEXT NOT NULL, chats INTEGER NOT NULL, mode TEXT NOT NULL,
            assigned INTEGER NOT NULL DEFAULT 0, cancelled BOOLEAN NOT NULL DEFAULT false,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now())""",
        ),
        (
            "idea_claims",
            """(
            case_id TEXT PRIMARY KEY, idea_id TEXT NOT NULL REFERENCES ideas(idea_id),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now())""",
        ),
    ):
        if name not in tables:
            connection.execute(text(f"CREATE TABLE {name} {definition}"))
    # Avoid taking index/column DDL locks on every publish while workers upload.
    indexes = {
        index["name"]
        for table in ("cases", "traces", "ideas", "idea_claims")
        for index in catalog.get_indexes(table, schema=schema)
    }
    for name, definition in (
        ("cases_run_updated", "ON cases(run_id, updated_at DESC)"),
        ("cases_review", "ON cases USING GIN(review)"),
        ("cases_seed", "ON cases USING GIN(seed)"),
        ("traces_case", "ON traces(case_id, sequence)"),
        ("ideas_run_created", "ON ideas(run_id, created_at)"),
        ("idea_claims_idea", "ON idea_claims(idea_id)"),
    ):
        if name not in indexes:
            connection.execute(text(f"CREATE INDEX {name} {definition}"))
    columns = {column["name"] for column in catalog.get_columns("runs", schema=schema)}
    for name in ("worker_identity", "worker_session_hash"):
        if name not in columns:
            connection.execute(text(f"ALTER TABLE runs ADD COLUMN {name} TEXT"))
    namespace = connection.dialect.identifier_preparer.quote_schema(schema)
    for signature in (
        "worker_run(text)",
        "worker_cases(text,text,integer)",
        "worker_case(text,text)",
        "worker_save_run(jsonb)",
        "worker_save_cases(jsonb)",
        "worker_save_traces(jsonb)",
        "worker_claim(text,text)",
        "worker_require_run(text)",
    ):
        connection.execute(text(f"DROP FUNCTION IF EXISTS {namespace}.{signature}"))


def identity() -> str:
    caller = get_verified_identity()
    return caller.user_id if caller is not None else "anonymous"


def owned_run(connection: Connection, run_id: str) -> None:
    if session_run.get() is not None:
        raise HTTPException(403, "Session keys cannot change owner controls")
    owner = connection.execute(text("SELECT owner FROM runs WHERE run_id=:run_id"), {"run_id": run_id}).scalar()
    if owner is None:
        raise HTTPException(404, "Run not found")
    if owner != identity():
        raise HTTPException(403, "Only this run's owner may write or control it")


def writable_run(connection: Connection, run_id: str) -> None:
    delegated_run = session_run.get()
    if delegated_run is not None:
        if delegated_run != run_id:
            raise HTTPException(403, "Session key is assigned to another run")
        return
    run = (
        connection.execute(text("SELECT owner,worker_identity FROM runs WHERE run_id=:run_id"), {"run_id": run_id})
        .mappings()
        .first()
    )
    if run is None:
        raise HTTPException(404, "Run not found")
    if identity() not in {run["owner"], run["worker_identity"]}:
        raise HTTPException(403, "Only this run's owner or assigned worker may save results")


def readable_run(connection: Connection, run_id: str) -> None:
    if session_run.get() is not None or identity().endswith(".gserviceaccount.com"):
        writable_run(connection, run_id)


def run_record(row: dict) -> dict:
    return {key: value for key, value in row.items() if key != "worker_session_hash"}


def comparison_identity(config: dict) -> dict:
    """Pin evaluation semantics while allowing serving jobs to move during recovery."""
    return {
        **{key: value for key, value in config.items() if key != "coordinator"},
        "driver": {key: value for key, value in config.get("driver", {}).items() if key != "gateway"},
    }


def run_summary(connection: Connection, run_id: str, mode: str | None) -> dict:
    parameters = {"run_id": run_id, "mode": mode or ""}
    categories = [
        dict(row)
        for row in connection.execute(
            text(
                """SELECT category,mode,count(*) AS cases,
        count(*) FILTER(WHERE state='complete') AS completed
        FROM cases WHERE run_id=:run_id AND (:mode='' OR mode=:mode) GROUP BY category,mode"""
            ),
            parameters,
        ).mappings()
    ]
    breakages = [
        dict(row)
        for row in connection.execute(
            text(
                """SELECT item->>'category' AS category,
        item->>'model' AS model,item->>'severity' AS severity,count(*) AS count
        FROM cases, jsonb_array_elements(review->'breakages') AS item
        WHERE run_id=:run_id AND (:mode='' OR mode=:mode) GROUP BY 1,2,3 ORDER BY count(*) DESC"""
            ),
            parameters,
        ).mappings()
    ]
    scores = [
        dict(row)
        for row in connection.execute(
            text(
                """SELECT item->>'model' AS model,category,
        avg((item->>'score')::float) AS score,count(*) AS n
        FROM cases,jsonb_array_elements(review->'assessments') AS item
        WHERE run_id=:run_id AND (:mode='' OR mode=:mode)
        AND item->>'score' IS NOT NULL GROUP BY 1,2"""
            ),
            parameters,
        ).mappings()
    ]
    verified_scores = [
        dict(row)
        for row in connection.execute(
            text(
                """SELECT item->>'model' AS model,category,
                avg((item->>'score')::float) AS score,count(*) AS n
                FROM cases,jsonb_array_elements(review->'verification') AS item
                WHERE run_id=:run_id AND (:mode='' OR mode=:mode)
                AND review->>'verification_eligible'='true'
                AND item->>'score' IS NOT NULL GROUP BY 1,2"""
            ),
            parameters,
        ).mappings()
    ]
    counts = dict(
        connection.execute(
            text(
                """SELECT count(*) AS cases,
        count(*) FILTER(WHERE state='complete') AS completed,
        count(*) FILTER(WHERE state IN ('generating','reviewing')) AS active,
        count(*) FILTER(WHERE state IN ('error','review_error','interrupted')) AS errors
        FROM cases WHERE run_id=:run_id AND (:mode='' OR mode=:mode)"""
            ),
            parameters,
        )
        .mappings()
        .one()
    )
    return {
        "categories": categories,
        "breakages": breakages,
        "scores": scores,
        "verified_scores": verified_scores,
        **counts,
    }


def create_api(services: AppletServices) -> FastAPI:
    api = FastAPI()
    engine = services.engine()
    summary_lock = Lock()
    summaries: dict[tuple[str, str | None], tuple[float, dict]] = {}

    @api.middleware("http")
    async def authenticate_session(request: Request, call_next):
        key = request.headers.get(SESSION_HEADER)
        if key is None:
            return await call_next(request)
        run_id = await run_in_threadpool(session_run_id, engine, key)
        if run_id is None:
            return JSONResponse({"detail": "Invalid or revoked session key"}, status_code=401)
        token = session_run.set(run_id)
        try:
            return await call_next(request)
        finally:
            session_run.reset(token)

    @api.post("/runs")
    def update_run(update: RunUpdate) -> dict:
        values = update.model_dump()
        values["config"] = json.dumps(update.config)
        values["owner"] = identity()
        with engine.begin() as connection:
            existing = (
                connection.execute(text("SELECT owner, config FROM runs WHERE run_id=:run_id"), values)
                .mappings()
                .first()
            )
            if existing is not None:
                writable_run(connection, update.run_id)
                if comparison_identity(existing["config"]) != comparison_identity(update.config):
                    raise HTTPException(409, "Run checkpoint/config identity is immutable; create a new run")
            elif session_run.get() is not None or values["owner"].endswith(".gserviceaccount.com"):
                raise HTTPException(403, "A user must create the run and assign its worker")
            connection.execute(
                text(
                    """INSERT INTO runs
                (run_id,owner,config,token_cap,concurrency,generated_tokens,reserved_tokens,state,active_chats,error)
                VALUES (:run_id,:owner,CAST(:config AS JSONB),:token_cap,:concurrency,:generated_tokens,
                :reserved_tokens,:state,:active_chats,:error)
                ON CONFLICT (run_id) DO UPDATE SET generated_tokens=EXCLUDED.generated_tokens,
                reserved_tokens=EXCLUDED.reserved_tokens, state=EXCLUDED.state,
                config=EXCLUDED.config, active_chats=EXCLUDED.active_chats,
                error=EXCLUDED.error, updated_at=now()"""
                ),
                values,
            )
            desired = connection.execute(
                text("SELECT desired_state FROM runs WHERE run_id=:run_id"), values
            ).scalar_one()
        return {"desired_state": desired}

    @api.get("/runs")
    def runs() -> list[dict]:
        delegated_run = session_run.get()
        caller = identity()
        values = {"run_id": delegated_run, "caller": caller}
        where = ""
        if delegated_run is not None:
            where = "WHERE run_id=:run_id"
        elif caller.endswith(".gserviceaccount.com"):
            where = "WHERE worker_identity=:caller"
        with engine.connect() as connection:
            return [
                run_record(dict(row))
                for row in connection.execute(
                    text(f"SELECT * FROM runs {where} ORDER BY created_at DESC LIMIT 100"), values
                ).mappings()
            ]

    @api.get("/runs/{run_id}")
    def run(run_id: str) -> dict:
        with engine.connect() as connection:
            readable_run(connection, run_id)
            row = (
                connection.execute(text("SELECT * FROM runs WHERE run_id=:run_id"), {"run_id": run_id})
                .mappings()
                .first()
            )
            if row is None:
                raise HTTPException(404, "Run not found")
            return run_record(dict(row))

    @api.post("/runs/{run_id}/create-session-key")
    def create_session_key(run_id: str) -> dict:
        with engine.begin() as connection:
            owned_run(connection, run_id)
            key = secrets.token_urlsafe(32)
            connection.execute(
                text("UPDATE runs SET worker_session_hash=:hash WHERE run_id=:run_id"),
                {"run_id": run_id, "hash": key_hash(key)},
            )
        return {"session_key": key}

    @api.post("/runs/{run_id}/revoke-session-key")
    def revoke_session_key(run_id: str) -> dict:
        with engine.begin() as connection:
            owned_run(connection, run_id)
            connection.execute(text("UPDATE runs SET worker_session_hash=NULL WHERE run_id=:run_id"), {"run_id": run_id})
        return {"revoked": True}

    @api.post("/runs/{run_id}/worker")
    def assign_worker(run_id: str, grant: WorkerGrant) -> dict:
        with engine.begin() as connection:
            owned_run(connection, run_id)
            connection.execute(
                text("UPDATE runs SET worker_identity=:identity WHERE run_id=:run_id"),
                {"run_id": run_id, "identity": grant.identity},
            )
        return {"assigned": grant.identity is not None}

    @api.post("/runs/{run_id}/control")
    def control(run_id: str, update: Control) -> dict:
        with engine.begin() as connection:
            owned_run(connection, run_id)
            connection.execute(
                text("UPDATE runs SET desired_state=:state WHERE run_id=:run_id"),
                {"run_id": run_id, "state": update.state},
            )
        return {"desired_state": update.state}

    @api.post("/cases")
    def update_case(update: CaseUpdate) -> dict:
        values = update.model_dump()
        values["seed"] = json.dumps(searchable_metadata(update.seed))
        values["review"] = json.dumps(searchable_metadata(update.review))
        values["metadata_payload"] = zlib.compress(json.dumps({"seed": update.seed, "review": update.review}).encode())
        with engine.begin() as connection:
            writable_run(connection, update.run_id)
            existing = connection.execute(text("SELECT run_id FROM cases WHERE case_id=:case_id"), values).scalar()
            if existing is not None and existing != update.run_id:
                raise HTTPException(409, "Case belongs to a different run")
            connection.execute(
                text(
                    """INSERT INTO cases
                (case_id,run_id,title,category,mode,state,slot,round,seed,review)
                VALUES (:case_id,:run_id,:title,:category,:mode,:state,:slot,:round,
                CAST(:seed AS JSONB),CAST(:review AS JSONB))
                ON CONFLICT (case_id) DO UPDATE SET title=EXCLUDED.title, category=EXCLUDED.category,
                mode=EXCLUDED.mode, seed=EXCLUDED.seed, state=EXCLUDED.state,
                round=EXCLUDED.round, review=EXCLUDED.review, updated_at=now()"""
                ),
                values,
            )
            connection.execute(
                text(
                    """INSERT INTO case_metadata (case_id,payload) VALUES (:case_id,:metadata_payload)
                    ON CONFLICT (case_id) DO UPDATE SET payload=EXCLUDED.payload"""
                ),
                values,
            )
        return {"saved": update.case_id}

    @api.post("/cases/batch")
    def update_cases(batch: CaseBatch) -> dict:
        for case in batch.cases:
            update_case(case)
        return {"saved": len(batch.cases)}

    @api.post("/runs/{run_id}/ideas")
    def add_idea(run_id: str, idea: IdeaCreate) -> dict:
        values = {**idea.model_dump(), "run_id": run_id, "idea_id": uuid.uuid4().hex}
        if not idea.text.strip():
            raise HTTPException(422, "Enter a topic or prompt")
        with engine.begin() as connection:
            owned_run(connection, run_id)
            state = connection.execute(text("SELECT desired_state FROM runs WHERE run_id=:run_id"), values).scalar_one()
            if state == "stopped":
                raise HTTPException(409, "This run is stopped; choose an active run")
            connection.execute(
                text(
                    """INSERT INTO ideas (idea_id,run_id,text,kind,chats,mode)
                    VALUES (:idea_id,:run_id,:text,:kind,:chats,:mode)"""
                ),
                values,
            )
        return {"idea_id": values["idea_id"]}

    @api.get("/runs/{run_id}/ideas")
    def ideas(run_id: str) -> list[dict]:
        with engine.connect() as connection:
            readable_run(connection, run_id)
            rows = connection.execute(
                text(
                    """SELECT i.*, count(c.case_id) FILTER (WHERE c.state='complete') AS completed,
                    count(c.case_id) FILTER (WHERE c.state NOT IN ('generating','reviewing','complete')) AS errors,
                    count(c.case_id) FILTER (WHERE c.state NOT IN ('generating','reviewing')) AS finished
                    FROM (SELECT * FROM ideas WHERE run_id=:run_id ORDER BY created_at DESC LIMIT 100) i
                    LEFT JOIN idea_claims a ON a.idea_id=i.idea_id
                    LEFT JOIN cases c ON c.case_id=a.case_id
                    GROUP BY i.idea_id,i.run_id,i.text,i.kind,i.chats,i.mode,i.assigned,i.cancelled,i.created_at
                    ORDER BY i.created_at DESC"""
                ),
                {"run_id": run_id},
            ).mappings()
            return [dict(row) for row in rows]

    @api.post("/runs/{run_id}/ideas/claim")
    def claim_idea(run_id: str, claim: IdeaClaim) -> dict:
        values = {"run_id": run_id, "case_id": claim.case_id}
        with engine.begin() as connection:
            writable_run(connection, run_id)
            # Serialize admission and retries across slots; a lost HTTP response must not consume another chat.
            connection.execute(text("SELECT run_id FROM runs WHERE run_id=:run_id FOR UPDATE"), values)
            old = (
                connection.execute(
                    text(
                        """SELECT i.* FROM ideas i JOIN idea_claims a ON a.idea_id=i.idea_id
                    WHERE a.case_id=:case_id"""
                    ),
                    values,
                )
                .mappings()
                .first()
            )
            if old is not None:
                if old["run_id"] != run_id:
                    raise HTTPException(409, "Idea claim belongs to a different run")
                return {"idea": dict(old)}
            desired = connection.execute(
                text("SELECT desired_state FROM runs WHERE run_id=:run_id"), values
            ).scalar_one()
            if desired != "running":
                return {"idea": None}
            row = (
                connection.execute(
                    text(
                        """SELECT * FROM ideas WHERE run_id=:run_id AND NOT cancelled AND assigned<chats
                    ORDER BY created_at,idea_id LIMIT 1 FOR UPDATE"""
                    ),
                    values,
                )
                .mappings()
                .first()
            )
            if row is None:
                return {"idea": None}
            values["idea_id"] = row["idea_id"]
            connection.execute(text("INSERT INTO idea_claims (case_id,idea_id) VALUES (:case_id,:idea_id)"), values)
            connection.execute(text("UPDATE ideas SET assigned=assigned+1 WHERE idea_id=:idea_id"), values)
            return {"idea": dict(row)}

    @api.post("/runs/{run_id}/ideas/{idea_id}/cancel")
    def cancel_idea(run_id: str, idea_id: str) -> dict:
        with engine.begin() as connection:
            owned_run(connection, run_id)
            result = connection.execute(
                text("UPDATE ideas SET cancelled=true WHERE run_id=:run_id AND idea_id=:idea_id"),
                {"run_id": run_id, "idea_id": idea_id},
            )
            if result.rowcount != 1:
                raise HTTPException(404, "Queued idea not found")
        return {"cancelled": idea_id}

    @api.post("/traces")
    def update_trace(update: TraceUpdate) -> dict:
        payload = json.dumps(update.payload, ensure_ascii=False).encode()
        if len(payload) > 4 * 1024 * 1024:
            raise HTTPException(413, "One trace message exceeds 4 MiB")
        values = {**update.model_dump(), "payload": zlib.compress(payload)}
        with engine.begin() as connection:
            run_id = connection.execute(text("SELECT run_id FROM cases WHERE case_id=:case_id"), values).scalar()
            if run_id is None:
                raise HTTPException(404, "Case not found")
            writable_run(connection, run_id)
            old_case = connection.execute(
                text("SELECT case_id FROM traces WHERE message_id=:message_id"), values
            ).scalar()
            if old_case is not None and old_case != update.case_id:
                raise HTTPException(409, "Trace belongs to a different case")
            connection.execute(
                text(
                    """INSERT INTO traces (message_id,case_id,model,role,round,sequence,state,payload)
                VALUES (:message_id,:case_id,:model,:role,:round,:sequence,:state,:payload)
                ON CONFLICT(message_id) DO UPDATE SET state=EXCLUDED.state,payload=EXCLUDED.payload,updated_at=now()"""
                ),
                values,
            )
        return {"saved": update.message_id}

    @api.post("/traces/batch")
    def update_traces(batch: TraceBatch) -> dict:
        for trace in batch.traces:
            update_trace(trace)
        return {"saved": len(batch.traces)}

    @api.get("/cases")
    def cases(
        run_id: str,
        category: str = "",
        mode: str = "",
        state: str = "",
        breakage: str = "",
        model: str = "",
        idea_id: str = "",
        limit: int = Query(60, ge=1, le=200),
        offset: int = Query(0, ge=0),
    ) -> dict:
        filters = ["run_id=:run_id"]
        values: dict = {"run_id": run_id, "limit": limit, "offset": offset}
        if idea_id:
            filters.append("seed @> CAST(:idea AS JSONB)")
            values["idea"] = json.dumps({"queued_idea": {"idea_id": idea_id}})
        for column, value in (("category", category), ("mode", mode), ("state", state)):
            if value:
                filters.append(f"{column}=:{column}")
                values[column] = value
        if breakage or model:
            item = {}
            if breakage:
                item["category"] = breakage
            if model:
                item["model"] = model
            filters.append("review @> CAST(:breakages AS JSONB)")
            values["breakages"] = json.dumps({"breakages": [item]})
        where = " AND ".join(filters)
        with engine.connect() as connection:
            readable_run(connection, run_id)
            total = connection.execute(text(f"SELECT count(*) FROM cases WHERE {where}"), values).scalar_one()
            rows = connection.execute(
                text(
                    f"""SELECT cases.*,case_metadata.payload AS metadata_payload FROM cases
                    LEFT JOIN case_metadata USING (case_id) WHERE {where}
                    ORDER BY updated_at DESC LIMIT :limit OFFSET :offset"""
                ),
                values,
            ).mappings()
            return {"total": total, "cases": [case_record(dict(row)) for row in rows]}

    @api.get("/cases/{case_id}")
    def case(case_id: str, trace_state: str | None = None) -> dict:
        with engine.connect() as connection:
            row = (
                connection.execute(
                    text(
                        """SELECT cases.*,case_metadata.payload AS metadata_payload FROM cases
                        LEFT JOIN case_metadata USING (case_id) WHERE case_id=:case_id"""
                    ),
                    {"case_id": case_id},
                )
                .mappings()
                .first()
            )
            if row is None:
                raise HTTPException(404, "Case not found")
            readable_run(connection, row["run_id"])
            statement = "SELECT * FROM traces WHERE case_id=:case_id"
            if trace_state is not None:
                statement += " AND state=:trace_state"
            statement += " ORDER BY sequence,model,message_id"
            traces = connection.execute(text(statement), {"case_id": case_id, "trace_state": trace_state}).mappings()
            messages = [
                {
                    **{key: value for key, value in trace.items() if key != "payload"},
                    "payload": json.loads(zlib.decompress(bytes(trace["payload"]))),
                }
                for trace in traces
            ]
            return {**case_record(dict(row)), "traces": messages}

    @api.get("/summary")
    def summary(run_id: str, mode: Literal["chat", "agentic"] | None = None) -> dict:
        # Authorization must run even when another caller already populated the cache.
        with engine.connect() as connection:
            readable_run(connection, run_id)
        key = (run_id, mode)
        with summary_lock:
            now = monotonic()
            for expired in [key for key, (deadline, _) in summaries.items() if deadline <= now]:
                del summaries[expired]
            cached = summaries.get(key)
            if cached is not None:
                return cached[1]
            # Serialize refreshes so simultaneous viewers share one aggregation.
            with engine.connect() as connection:
                result = run_summary(connection, run_id, mode)
            if len(summaries) >= SUMMARY_CACHE_SIZE:
                del summaries[next(iter(summaries))]
            summaries[key] = (monotonic() + SUMMARY_TTL, result)
            return result

    return api

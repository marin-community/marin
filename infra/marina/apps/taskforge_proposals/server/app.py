# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Serve the n=25 Taskforge proposal run from the tables ``ingest.py`` loads.

Every statement is a fixed SQL string: filters only add clauses from this module with bound values,
and sort keys are looked up in a fixed map.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import Response
from marina.applets import AppletServices
from sqlalchemy import bindparam, text
from sqlalchemy.engine import Connection

from .proposal import body_sections
from .schema import SCHEMA_STATEMENTS, TS_CONFIG

MAX_PAGE_SIZE = 200
HISTOGRAM_BINS = 30
TOKEN_FIELDS = ("completion_tokens", "reasoning_tokens", "prompt_tokens")
LIST_COLUMNS = """capability_id, slot, domain, subject_id, area, status, title, summary, verification,
    environment, solver_machine, verdict, reason, cut_off, completions, prompt_tokens, completion_tokens,
    reasoning_tokens, cached_tokens"""
PROPOSAL_SORTS = {
    "capability": "capability_id {dir}, slot {dir}",
    "domain": "subject_id {dir}, capability_id, slot",
    "slot": "slot {dir}, capability_id",
    "kind": "verification {dir} NULLS LAST, capability_id, slot",
    "environment": "environment {dir} NULLS LAST, capability_id, slot",
    "verdict": "verdict {dir} NULLS LAST, capability_id, slot",
    "title": "title {dir} NULLS LAST, capability_id, slot",
    "completion_tokens": "completion_tokens {dir} NULLS LAST, capability_id, slot",
    "reasoning_tokens": "reasoning_tokens {dir} NULLS LAST, capability_id, slot",
    "prompt_tokens": "prompt_tokens {dir} NULLS LAST, capability_id, slot",
    "rank": f"ts_rank(search, websearch_to_tsquery('{TS_CONFIG}', :q)) {{dir}}, capability_id, slot",
}
CAPABILITY_SORTS = {
    "id": "id {dir}",
    "domain": "subject_id {dir}, id",
    "accepted": "accepted {dir} NULLS LAST, id",
    "rejected": "rejected {dir} NULLS LAST, id",
    "cut_off": "cut_off {dir} NULLS LAST, id",
    "completion_tokens": "completion_tokens {dir} NULLS LAST, id",
    "wall_time": "wall_time {dir} NULLS LAST, id",
    "status": "status {dir}, id",
}
NONE = "none"
CAPABILITY_LIKE = re.compile(r"[\w-]+(\.[\w-]+)+")


def migrate(connection: Connection) -> None:
    for statement in SCHEMA_STATEMENTS:
        connection.execute(text(statement))


@dataclass
class Where:
    """A conjunction of fixed clauses and the values bound into them."""

    clauses: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    lists: list[str] = field(default_factory=list)

    def add(self, clause: str, **params: Any) -> None:
        self.clauses.append(clause)
        self.params.update(params)

    def any_of(self, column: str, name: str, values: list[str]) -> None:
        """``column`` is one of ``values``; the value ``none`` also matches NULL."""
        values = [v for v in values if v]
        if not values:
            return
        named = [v for v in values if v != NONE]
        parts = []
        if named:
            parts.append(f"{column} IN :{name}")
            self.params[name] = named
            self.lists.append(name)
        if NONE in values:
            parts.append(f"{column} IS NULL")
        self.clauses.append("(" + " OR ".join(parts) + ")")

    def sql(self) -> str:
        return " AND ".join(self.clauses) if self.clauses else "TRUE"

    def statement(self, sql: str):
        statement = text(sql)
        if self.lists:
            statement = statement.bindparams(*(bindparam(name, expanding=True) for name in self.lists))
        return statement


def proposal_filter(
    q: str | None = None,
    capability: str | None = None,
    domain: list[str] = Query(default=[]),
    kind: list[str] = Query(default=[]),
    environment: list[str] = Query(default=[]),
    verdict: list[str] = Query(default=[]),
    status: list[str] = Query(default=[]),
    cut_off: bool | None = None,
    completion_tokens_min: int | None = None,
    completion_tokens_max: int | None = None,
    reasoning_tokens_min: int | None = None,
    reasoning_tokens_max: int | None = None,
    prompt_tokens_min: int | None = None,
    prompt_tokens_max: int | None = None,
) -> Where:
    """The proposal filters of the query string, shared by the table and the statistics."""
    tokens = {
        "completion_tokens": (completion_tokens_min, completion_tokens_max),
        "reasoning_tokens": (reasoning_tokens_min, reasoning_tokens_max),
        "prompt_tokens": (prompt_tokens_min, prompt_tokens_max),
    }
    where = Where()
    if q and q.strip():
        q = q.strip()
        if CAPABILITY_LIKE.fullmatch(q):
            # Something shaped like a capability id also matches ids containing it; plain text uses the index.
            where.add(
                f"(search @@ websearch_to_tsquery('{TS_CONFIG}', :q) OR capability_id ILIKE :q_like ESCAPE '\\')",
                q=q,
                q_like="%" + q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%",
            )
        else:
            where.add(f"search @@ websearch_to_tsquery('{TS_CONFIG}', :q)", q=q)
    if capability:
        where.add("capability_id = :capability", capability=capability)
    where.any_of("subject_id", "domain", domain)
    where.any_of("verification", "kind", kind)
    where.any_of("environment", "environment", environment)
    where.any_of("verdict", "verdict", verdict)
    where.any_of("status", "status", status)
    if cut_off is not None:
        where.add("cut_off = :cut_off", cut_off=cut_off)
    for name, (low, high) in tokens.items():
        if low is not None:
            where.add(f"{name} >= :{name}_min", **{f"{name}_min": low})
        if high is not None:
            where.add(f"{name} <= :{name}_max", **{f"{name}_max": high})
    return where


def rows(connection: Connection, statement, params: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(row) for row in connection.execute(statement, params).mappings()]


def json_file(value: Any, filename: str, download: bool) -> Response:
    body = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'} if download else {}
    return Response(body, media_type="application/json", headers=headers)


def create_api(services: AppletServices) -> FastAPI:
    api = FastAPI()
    engine = services.engine()

    @api.get("/meta")
    def meta() -> dict[str, Any]:
        with engine.connect() as connection:
            stats = {row["key"]: row["value"] for row in rows(connection, text("SELECT key, value FROM run_stats"), {})}
            domains = rows(
                connection,
                text(
                    "SELECT subject_id, domain, COUNT(*) AS capabilities, SUM(slots_planned) AS slots "
                    "FROM capabilities GROUP BY subject_id, domain ORDER BY subject_id"
                ),
                {},
            )
            kinds = rows(
                connection,
                text("SELECT verification AS key, COUNT(*) AS n FROM proposals GROUP BY 1 ORDER BY 2 DESC"),
                {},
            )
            environments = rows(
                connection,
                text("SELECT environment AS key, COUNT(*) AS n FROM proposals GROUP BY 1 ORDER BY 2 DESC"),
                {},
            )
            counts = (
                connection.execute(
                    text(
                        "SELECT (SELECT COUNT(*) FROM capabilities) AS capabilities, "
                        "(SELECT COUNT(*) FROM proposals) AS proposals"
                    )
                )
                .mappings()
                .one()
            )
        return {"stats": stats, "domains": domains, "kinds": kinds, "environments": environments, "counts": dict(counts)}

    @api.get("/proposals")
    def proposals(
        where: Where = Depends(proposal_filter),
        sort: str = "capability",
        direction: str = Query(default="asc", alias="dir"),
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=50, ge=1, le=MAX_PAGE_SIZE),
    ) -> dict[str, Any]:
        if sort == "rank" and "q" not in where.params:
            sort = "capability"
        order = PROPOSAL_SORTS.get(sort)
        if order is None:
            raise HTTPException(400, f"unknown sort {sort!r}")
        order = order.format(dir="DESC" if direction == "desc" else "ASC")
        params = {**where.params, "limit": page_size, "offset": (page - 1) * page_size}
        with engine.connect() as connection:
            total = connection.execute(
                where.statement(f"SELECT COUNT(*) FROM proposals WHERE {where.sql()}"), where.params
            ).scalar_one()
            found = rows(
                connection,
                where.statement(
                    f"SELECT {LIST_COLUMNS} FROM proposals WHERE {where.sql()} "
                    f"ORDER BY {order} LIMIT :limit OFFSET :offset"
                ),
                params,
            )
        return {"total": total, "page": page, "page_size": page_size, "rows": found}

    @api.get("/stats")
    def stats(
        where: Where = Depends(proposal_filter),
    ) -> dict[str, Any]:
        clause, params = where.sql(), where.params

        def grouped(key: str) -> list[dict[str, Any]]:
            return rows(
                connection,
                where.statement(
                    f"SELECT {key} AS key, COUNT(*) AS n, "
                    "COUNT(*) FILTER (WHERE verdict = 'accept') AS accepted, "
                    "COUNT(*) FILTER (WHERE verdict = 'reject') AS rejected, "
                    "COUNT(*) FILTER (WHERE cut_off) AS cut_off "
                    f"FROM proposals WHERE {clause} GROUP BY 1 ORDER BY 2 DESC"
                ),
                params,
            )

        with engine.connect() as connection:
            totals = (
                connection.execute(
                    where.statement(
                        "SELECT COUNT(*) AS n, COUNT(DISTINCT capability_id) AS capabilities, "
                        "COUNT(*) FILTER (WHERE verdict = 'accept') AS accepted, "
                        "COUNT(*) FILTER (WHERE verdict = 'reject') AS rejected, "
                        "COUNT(*) FILTER (WHERE cut_off) AS cut_off, "
                        "COUNT(*) FILTER (WHERE completions > 1) AS repaired, "
                        "SUM(completion_tokens) AS completion_tokens, SUM(reasoning_tokens) AS reasoning_tokens, "
                        "SUM(prompt_tokens) AS prompt_tokens, "
                        "percentile_cont(0.5) WITHIN GROUP (ORDER BY completion_tokens) AS median_completion_tokens, "
                        "percentile_cont(0.5) WITHIN GROUP (ORDER BY reasoning_tokens) AS median_reasoning_tokens "
                        f"FROM proposals WHERE {clause}"
                    ),
                    params,
                )
                .mappings()
                .one()
            )
            by_domain = rows(
                connection,
                where.statement(
                    "SELECT subject_id AS key, MIN(domain) AS label, COUNT(*) AS n, "
                    "COUNT(*) FILTER (WHERE verdict = 'accept') AS accepted, "
                    "COUNT(*) FILTER (WHERE verdict = 'reject') AS rejected, "
                    "COUNT(*) FILTER (WHERE cut_off) AS cut_off "
                    f"FROM proposals WHERE {clause} GROUP BY 1 ORDER BY 1"
                ),
                params,
            )
            crosstab = rows(
                connection,
                where.statement(
                    "SELECT subject_id AS domain, verification AS kind, COUNT(*) AS n, "
                    "COUNT(*) FILTER (WHERE verdict = 'accept') AS accepted, "
                    "COUNT(*) FILTER (WHERE verdict IS NOT NULL) AS decided "
                    f"FROM proposals WHERE {clause} GROUP BY 1, 2 ORDER BY 1, 2"
                ),
                params,
            )
            histograms = {}
            for name in TOKEN_FIELDS:
                bounds = (
                    connection.execute(
                        where.statement(
                            f"SELECT MIN({name}) AS lo, MAX({name}) AS max, "
                            f"percentile_disc(0.99) WITHIN GROUP (ORDER BY {name}) AS hi FROM proposals WHERE {clause}"
                        ),
                        params,
                    )
                    .mappings()
                    .one()
                )
                lo, hi = bounds["lo"], bounds["hi"]
                bins: list[int] = [0] * HISTOGRAM_BINS
                if lo is not None:
                    width = max(1, -(-(hi - lo + 1) // HISTOGRAM_BINS))
                    for row in rows(
                        connection,
                        where.statement(
                            f"SELECT LEAST(({name} - :lo) / :width, {HISTOGRAM_BINS - 1}) AS bin, COUNT(*) AS n "
                            f"FROM proposals WHERE {clause} AND {name} IS NOT NULL GROUP BY 1"
                        ),
                        {**params, "lo": lo, "width": width},
                    ):
                        bins[int(row["bin"])] = row["n"]
                    # Bins span the minimum to the 99th percentile; the last bin also holds the tail above it.
                    histograms[name] = {"lo": lo, "hi": hi, "max": bounds["max"], "width": width, "bins": bins}
                else:
                    histograms[name] = {"lo": None, "hi": None, "max": None, "width": None, "bins": []}
            result = {
                "totals": dict(totals),
                "by_kind": grouped("verification"),
                "by_environment": grouped("environment"),
                "by_verdict": grouped("verdict"),
                "by_status": grouped("status"),
                "by_domain": by_domain,
                "crosstab": crosstab,
                "histograms": histograms,
            }
        return json.loads(json.dumps(result, default=float))

    @api.get("/proposals/{capability_id}/{slot}")
    def proposal(capability_id: str, slot: int) -> dict[str, Any]:
        with engine.connect() as connection:
            found = rows(
                connection,
                text("SELECT * FROM proposals WHERE capability_id = :capability AND slot = :slot"),
                {"capability": capability_id, "slot": slot},
            )
            if not found:
                raise HTTPException(404, "proposal not found")
            siblings = rows(
                connection,
                text(
                    "SELECT slot, title, status, verification, environment, verdict, cut_off, completion_tokens "
                    "FROM proposals WHERE capability_id = :capability ORDER BY slot"
                ),
                {"capability": capability_id},
            )
            capability = rows(
                connection,
                text(
                    "SELECT id, name, outcome, domain, subject_id, area, section_path, status, accepted, rejected, "
                    "cut_off, slots_planned FROM capabilities WHERE id = :capability"
                ),
                {"capability": capability_id},
            )
        record = found[0]
        record.pop("search", None)
        body = (record.get("proposal") or {}).get("body") or ""
        record["sections"] = body_sections(body)
        return {"proposal": record, "siblings": siblings, "capability": capability[0] if capability else None}

    @api.get("/proposals/{capability_id}/{slot}/raw/{kind}")
    def proposal_raw(capability_id: str, slot: int, kind: str, download: bool = False) -> Response:
        columns = {"proposal": "proposal", "verdict": "verdict_record"}
        if kind not in columns:
            raise HTTPException(404, "unknown file")
        with engine.connect() as connection:
            value = connection.execute(
                text(f"SELECT {columns[kind]} FROM proposals WHERE capability_id = :capability AND slot = :slot"),
                {"capability": capability_id, "slot": slot},
            ).scalar_one_or_none()
        if value is None:
            raise HTTPException(404, f"no {kind} file for this slot")
        return json_file(value, f"{capability_id}.slot-{slot:02d}.{kind}.json", download)

    @api.get("/capabilities")
    def capabilities(
        q: str | None = None,
        domain: list[str] = Query(default=[]),
        status: list[str] = Query(default=[]),
        sort: str = "id",
        direction: str = Query(default="asc", alias="dir"),
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=50, ge=1, le=MAX_PAGE_SIZE),
    ) -> dict[str, Any]:
        where = Where()
        if q and q.strip():
            like = "%" + q.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
            where.add(
                "(id ILIKE :like ESCAPE '\\' OR name ILIKE :like ESCAPE '\\' OR outcome ILIKE :like ESCAPE '\\')",
                like=like,
            )
        where.any_of("subject_id", "domain", domain)
        where.any_of("status", "status", status)
        order = CAPABILITY_SORTS.get(sort)
        if order is None:
            raise HTTPException(400, f"unknown sort {sort!r}")
        order = order.format(dir="DESC" if direction == "desc" else "ASC")
        with engine.connect() as connection:
            total = connection.execute(
                where.statement(f"SELECT COUNT(*) FROM capabilities WHERE {where.sql()}"), where.params
            ).scalar_one()
            found = rows(
                connection,
                where.statement(
                    "SELECT id, name, domain, subject_id, area, status, slots_planned, proposed, accepted, rejected, "
                    "cut_off, requests, completion_tokens, reasoning_tokens, prompt_tokens, wall_time "
                    f"FROM capabilities WHERE {where.sql()} ORDER BY {order} LIMIT :limit OFFSET :offset"
                ),
                {**where.params, "limit": page_size, "offset": (page - 1) * page_size},
            )
        return {"total": total, "page": page, "page_size": page_size, "rows": found}

    @api.get("/capabilities/{capability_id}")
    def capability(capability_id: str) -> dict[str, Any]:
        with engine.connect() as connection:
            found = rows(connection, text("SELECT * FROM capabilities WHERE id = :id"), {"id": capability_id})
            if not found:
                raise HTTPException(404, "capability not found")
            slots = rows(
                connection,
                text(
                    "SELECT slot, title, workflow, status, plan_status, planned_verification, planned_environment, "
                    "verification, environment, solver_machine, verdict, reason, cut_off, failure, completions, "
                    "prompt_tokens, completion_tokens, reasoning_tokens FROM proposals "
                    "WHERE capability_id = :id ORDER BY slot"
                ),
                {"id": capability_id},
            )
        return {"capability": found[0], "slots": slots}

    @api.get("/capabilities/{capability_id}/raw/{kind}")
    def capability_raw(capability_id: str, kind: str, download: bool = False) -> Response:
        columns = {"plan": "plan", "result": "result", "failure": "failure", "catalog": "catalog_entry"}
        if kind not in columns:
            raise HTTPException(404, "unknown file")
        with engine.connect() as connection:
            value = connection.execute(
                text(f"SELECT {columns[kind]} FROM capabilities WHERE id = :id"), {"id": capability_id}
            ).scalar_one_or_none()
        if value is None:
            raise HTTPException(404, f"no {kind} record for this capability")
        return json_file(value, f"{capability_id}.{kind}.json", download)

    return api

"""Result-set-equivalence scoring for the ``text_to_sql`` verifier environment.

The reward model's ``ground_truth`` is a JSON object::

    {
      "schema_sql":        "<CREATE TABLE ...; ...>",
      "insert_sql":        "<INSERT INTO ...; ...>",
      "reference_sql":     "<single SELECT>",
      "order_significant": bool,          # reference has a top-level ORDER BY
      "table_names":       ["t1", "t2"]   # informational; the perturbation reads sqlite_master directly
    }

``grade`` rebuilds an in-memory SQLite database from that DDL and compares the
candidate query's result set to the reference's on two databases: the seeded
one, and a copy with every third row of the database deleted. Both must match.
A query that returns a hard-coded literal passes the first and fails the second.

The candidate query runs under three independent read-only layers: a
single-statement check, a keyword blocklist matched against comment/string
stripped text, and a SQLite authorizer that permits only read operations. A
progress handler bounds runaway queries.

``grade`` returns a ``GradeOutcome`` — ``MATCH`` / ``MISMATCH`` / ``INFRA``.
``INFRA`` marks a broken task or a verifier fault (missing keys, un-loadable
schema, a reference query that will not run) — the dataset-preparation contract
rejects those rows up front, and the rollout environment scores them ``0``
without crashing a worker.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import time
from enum import StrEnum
from typing import Any


class GradeOutcome(StrEnum):
    MATCH = "match"
    MISMATCH = "mismatch"
    INFRA = "infra"  # broken task or a reference that will not run


INFRA = GradeOutcome.INFRA  # re-exported for callers and tests that only care about the sentinel
_QUERY_DEADLINE = 5.0
_MAX_RESULT_ROWS = 100_000
_PROGRESS_HANDLER_OPS = 100_000  # SQLite VM instructions between deadline checks

_STRING_RE = re.compile(r"'(?:[^']|'')*'")
_LINE_COMMENT_RE = re.compile(r"--[^\n]*")
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_SOLUTION_RE = re.compile(r"<solution>\s*(.*?)\s*</solution>", re.DOTALL | re.IGNORECASE)
_SQL_FENCE_RE = re.compile(r"```(?:sql)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)

_BLOCKED_KEYWORDS_RE = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|pragma|vacuum|reindex|analyze|"
    r"begin|commit|rollback|savepoint|release|grant|revoke|truncate|trigger|into)\b",
    re.IGNORECASE,
)
_NONDET_NOW_LITERAL_RE = re.compile(r"'\s*now\s*'", re.IGNORECASE)
_NONDET_FN_RE = re.compile(
    r"\b(current_date|current_time|current_timestamp|localtime|localtimestamp|now|curdate|curtime|"
    r"sysdate|random|randomblob|uuid|newid|rand)\s*\(?",
    re.IGNORECASE,
)
_CREATE_NAME_RE = re.compile(
    r"^\s*create\s+(?:temp(?:orary)?\s+)?table\s+(?:if\s+not\s+exists\s+)?([`\"\[]?)(?P<name>[A-Za-z0-9_$]+)\1\s*(?P<dot>\.)?",
    re.IGNORECASE,
)

_GROUND_TRUTH_KEYS = ("schema_sql", "insert_sql", "reference_sql", "order_significant")


# --------------------------------------------------------------------------- #
# static SQL analysis (shared with the dataset-preparation transform)          #
# --------------------------------------------------------------------------- #


def strip_sql_noise(sql: str) -> str:
    """Return ``sql`` with block/line comments and single-quoted literals blanked."""
    sql = _BLOCK_COMMENT_RE.sub(" ", sql)
    sql = _LINE_COMMENT_RE.sub(" ", sql)
    return _STRING_RE.sub("''", sql)


def split_statements(script: str) -> list[str]:
    """Split a SQL script on top-level semicolons, honoring quotes and comments."""
    out: list[str] = []
    buf: list[str] = []
    i, n = 0, len(script)
    while i < n:
        ch = script[i]
        if ch == "'":
            buf.append(ch)
            i += 1
            while i < n:
                buf.append(script[i])
                if script[i] == "'":
                    if i + 1 < n and script[i + 1] == "'":
                        buf.append(script[i + 1])
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            continue
        if ch == '"':
            buf.append(ch)
            i += 1
            while i < n:
                buf.append(script[i])
                if script[i] == '"':
                    i += 1
                    break
                i += 1
            continue
        if ch == "-" and i + 1 < n and script[i + 1] == "-":
            while i < n and script[i] != "\n":
                i += 1
            continue
        if ch == "/" and i + 1 < n and script[i + 1] == "*":
            i += 2
            while i + 1 < n and not (script[i] == "*" and script[i + 1] == "/"):
                i += 1
            i += 2
            continue
        if ch == ";":
            stmt = "".join(buf).strip()
            if stmt:
                out.append(stmt)
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    tail = "".join(buf).strip()
    if tail:
        out.append(tail)
    return out


def classify_statement(stmt: str) -> str:
    """One of ``create_table`` / ``insert`` / ``select`` / ``other_ddl`` / ``dml`` / ``unknown``."""
    s = strip_sql_noise(stmt).lstrip().lower()
    if s.startswith(("create table", "create temp table", "create temporary table")):
        return "create_table"
    if s.startswith(("insert into", "insert or")):
        return "insert"
    if s.startswith(("with ", "select ")):
        return "select"
    if s.startswith("create "):
        return "other_ddl"
    if s.startswith(("update ", "delete ", "replace ", "drop ", "alter ", "truncate ", "merge ")):
        return "dml"
    return "unknown"


def create_table_is_schema_qualified(stmt: str) -> bool:
    """True for ``CREATE TABLE db.tbl (...)`` — these cannot be rebuilt in a bare in-memory DB."""
    m = _CREATE_NAME_RE.match(strip_sql_noise(stmt))
    return bool(m and m.group("dot"))


def create_table_name(stmt: str) -> str | None:
    """Return the (unqualified) table name a ``CREATE TABLE`` statement declares, or ``None``."""
    m = _CREATE_NAME_RE.match(strip_sql_noise(stmt))
    return m.group("name") if m and not m.group("dot") else None


def is_nondeterministic(sql: str) -> bool:
    """True if the query text references clock/RNG functions."""
    if _NONDET_NOW_LITERAL_RE.search(sql):
        return True
    return bool(_NONDET_FN_RE.search(strip_sql_noise(sql)))


def has_top_level_order_by(sql: str) -> bool:
    """True iff ``sql`` has an ``ORDER BY`` not nested inside parentheses.

    Parenthesized spans (subqueries, CTE bodies, ``OVER (... ORDER BY ...)``) are removed first, leaving
    only a clause that governs the row order of the final result.
    """
    s = strip_sql_noise(sql)
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"\([^()]*\)", " ", s)
    return re.search(r"\border\s+by\b", s, re.IGNORECASE) is not None


def extract_sql(response: str) -> str:
    """Pull the candidate query out of a model response: ``<solution>`` tags, then a fenced block, then raw."""
    m = _SOLUTION_RE.search(response)
    if m:
        response = m.group(1)
    m = _SQL_FENCE_RE.search(response)
    if m:
        response = m.group(1)
    return response.strip()


# --------------------------------------------------------------------------- #
# execution + comparison                                                       #
# --------------------------------------------------------------------------- #


def _deadline_handler(deadline: float):
    def handler() -> int:
        return 1 if time.monotonic() > deadline else 0

    return handler


def build_db(create_stmts: list[str], insert_stmts: list[str]) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.text_factory = str
    try:
        cur = conn.cursor()
        for st in create_stmts:
            cur.execute(st)
        for st in insert_stmts:
            cur.execute(st)
        conn.commit()
    except Exception:
        conn.close()
        raise
    return conn


def perturb_db(conn: sqlite3.Connection, *, stride: int = 3) -> int:
    """Delete every ``stride``-th row of the whole database (table name, then rowid order), in place.

    Table names come from ``sqlite_master`` rather than a caller-supplied list, so a stale or empty
    ``table_names`` in the ground truth cannot silently turn the anti-hardcoding pass into a no-op.
    """
    cur = conn.cursor()
    table_names = [
        row[0]
        for row in cur.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    ]
    catalog: list[tuple[str, int]] = []
    for t in table_names:
        catalog.extend((t, r[0]) for r in cur.execute(f'SELECT rowid FROM "{t}" ORDER BY rowid'))
    victims = catalog[stride - 1 :: stride]
    by_table: dict[str, list[int]] = {}
    for t, rid in victims:
        by_table.setdefault(t, []).append(rid)
    for t, rids in by_table.items():
        placeholders = ",".join("?" * len(rids))
        cur.execute(f'DELETE FROM "{t}" WHERE rowid IN ({placeholders})', rids)
    conn.commit()
    return len(victims)


_ALLOWED_ACTIONS = {
    getattr(sqlite3, name)
    for name in ("SQLITE_SELECT", "SQLITE_READ", "SQLITE_FUNCTION", "SQLITE_RECURSIVE")
    if getattr(sqlite3, name, None) is not None
}


def _read_only_authorizer(action, _a1, _a2, _db, _src):
    return sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS else sqlite3.SQLITE_DENY


def _run_query(conn: sqlite3.Connection, sql: str, *, read_only: bool) -> tuple[int, list[tuple]]:
    """Execute ``sql`` under the deadline handler; ``read_only`` adds the authorizer for candidate queries."""
    if read_only:
        conn.set_authorizer(_read_only_authorizer)
    conn.set_progress_handler(_deadline_handler(time.monotonic() + _QUERY_DEADLINE), _PROGRESS_HANDLER_OPS)
    try:
        cur = conn.cursor()
        cur.execute(sql)
        rows = [tuple(r) for r in cur.fetchmany(_MAX_RESULT_ROWS + 1)]
        ncols = len(cur.description) if cur.description else 0
    finally:
        conn.set_progress_handler(None, 0)
        if read_only:
            conn.set_authorizer(None)
    return ncols, rows


def _norm_value(v: Any) -> tuple:
    if v is None:
        return ("~null",)
    if isinstance(v, bool):
        return ("~num", float(v))
    if isinstance(v, int):
        return ("~num", v)
    if isinstance(v, float):
        if not math.isfinite(v):
            return ("~nonfinite", repr(v))
        return ("~num", round(v, 6))
    if isinstance(v, bytes):
        return ("~bytes", v.hex())
    return ("~str", str(v))


def _norm_rows(rows: list[tuple]) -> list[tuple]:
    return [tuple(_norm_value(c) for c in row) for row in rows]


def results_equivalent(
    reference: tuple[int, list[tuple]],
    candidate: tuple[int, list[tuple]],
    *,
    order_significant: bool,
) -> tuple[bool, str]:
    """Column count and column order are significant; column names/aliases are not. Row order is
    compared only when ``order_significant`` is True."""
    ref_ncols, ref_rows = reference
    cand_ncols, cand_rows = candidate
    if ref_ncols != cand_ncols:
        return False, f"column count: reference {ref_ncols}, candidate {cand_ncols}"
    ref_n = _norm_rows(ref_rows)
    cand_n = _norm_rows(cand_rows)
    if len(ref_n) != len(cand_n):
        return False, f"row count: reference {len(ref_n)}, candidate {len(cand_n)}"
    if order_significant:
        return (ref_n == cand_n, "row order / values differ" if ref_n != cand_n else "ok (ordered)")
    if sorted(ref_n) != sorted(cand_n):
        return False, "row multiset differs"
    return True, "ok (unordered)"


def guard_candidate_sql(sql: str) -> tuple[bool, str]:
    """Static gate on an untrusted candidate query (layer 1 of 3)."""
    stmts = [s for s in split_statements(sql) if s.strip()]
    if len(stmts) != 1:
        return False, f"expected exactly 1 SQL statement, got {len(stmts)}"
    clean = strip_sql_noise(stmts[0]).strip()
    if not re.match(r"(?is)^(with|select)\b", clean):
        return False, "query must start with SELECT or WITH"
    m = _BLOCKED_KEYWORDS_RE.search(clean)
    if m:
        return False, f"blocked keyword: {m.group(0)!r}"
    return True, stmts[0]


def _clone_db(conn: sqlite3.Connection) -> sqlite3.Connection:
    """Page-level copy of a built in-memory database, so the seed inserts run only once per grade."""
    clone = sqlite3.connect(":memory:")
    clone.text_factory = str
    conn.backup(clone)
    return clone


def _compare_on(
    conn: sqlite3.Connection, reference_sql: str, candidate_stmt: str, order_significant: bool, label: str
) -> tuple[GradeOutcome, str] | None:
    """Run the reference and the candidate on ``conn`` and compare their result sets.

    Returns a ``(outcome, detail)`` tuple when the pass is decisive — ``INFRA`` for a broken reference,
    ``MISMATCH`` for a candidate error or a result-set mismatch — and ``None`` when the two agree.
    """
    try:
        reference = _run_query(conn, reference_sql, read_only=True)
    except sqlite3.Error as exc:
        return GradeOutcome.INFRA, f"reference query failed on {label} db: {exc}"
    if len(reference[1]) > _MAX_RESULT_ROWS:
        return GradeOutcome.INFRA, f"reference result exceeds {_MAX_RESULT_ROWS} rows on {label} db"
    try:
        candidate = _run_query(conn, candidate_stmt, read_only=True)
    except sqlite3.Error as exc:
        return GradeOutcome.MISMATCH, f"candidate query failed on {label} db: {exc}"
    if len(candidate[1]) > _MAX_RESULT_ROWS:
        return GradeOutcome.MISMATCH, f"candidate result exceeds {_MAX_RESULT_ROWS} rows on {label} db"
    equal, detail = results_equivalent(reference, candidate, order_significant=order_significant)
    if not equal:
        return GradeOutcome.MISMATCH, f"{label} db: {detail}"
    return None


# --------------------------------------------------------------------------- #
# public API                                                                   #
# --------------------------------------------------------------------------- #


def _validated_ground_truth(
    ground_truth: Any,
) -> tuple[dict[str, Any], list[str], list[str], str]:
    if isinstance(ground_truth, str):
        try:
            ground_truth = json.loads(ground_truth)
        except ValueError as exc:
            raise ValueError("text_to_sql ground_truth is not valid JSON.") from exc
    if not isinstance(ground_truth, dict):
        raise ValueError("text_to_sql ground_truth must be a JSON object.")
    missing = [key for key in _GROUND_TRUTH_KEYS if key not in ground_truth]
    if missing:
        raise ValueError(f"text_to_sql ground_truth missing keys: {missing}.")

    statements: dict[str, list[str]] = {}
    for field, expected_kind, label in (
        ("schema_sql", "create_table", "CREATE TABLE"),
        ("insert_sql", "insert", "INSERT"),
    ):
        script = ground_truth[field]
        if not isinstance(script, str) or not script.strip():
            raise ValueError(f"text_to_sql {field} must be a non-empty string.")
        field_statements = split_statements(script)
        if not field_statements or any(classify_statement(stmt) != expected_kind for stmt in field_statements):
            raise ValueError(f"text_to_sql {field} must contain only {label} statements.")
        statements[field] = field_statements

    reference_sql = ground_truth["reference_sql"]
    if not isinstance(reference_sql, str) or not reference_sql.strip():
        raise ValueError("text_to_sql reference_sql must be a non-empty string.")
    valid_reference, reference_or_error = guard_candidate_sql(reference_sql)
    if not valid_reference:
        raise ValueError(f"text_to_sql reference_sql is invalid: {reference_or_error}.")
    if is_nondeterministic(reference_or_error):
        raise ValueError("text_to_sql reference_sql must be deterministic.")
    if not isinstance(ground_truth["order_significant"], bool):
        raise ValueError("text_to_sql order_significant must be a boolean.")

    table_names = ground_truth.get("table_names", [])
    if not isinstance(table_names, list) or not all(isinstance(table, str) for table in table_names):
        raise ValueError("text_to_sql table_names must be a list of strings.")
    return ground_truth, statements["schema_sql"], statements["insert_sql"], reference_or_error


def parse_ground_truth(ground_truth: Any) -> dict[str, Any] | None:
    """Return a statically safe spec dict, or ``None`` if it cannot be used."""
    try:
        spec, _, _, _ = _validated_ground_truth(ground_truth)
    except ValueError:
        return None
    return spec


def normalize_ground_truth(ground_truth: Any) -> str:
    """Canonical verifier input for one row. Raises ``ValueError`` for a row the verifier cannot run."""
    spec, create_stmts, insert_stmts, reference_sql = _validated_ground_truth(ground_truth)
    try:
        conn = build_db(create_stmts, insert_stmts)
    except sqlite3.Error as exc:
        raise ValueError(f"text_to_sql seed DDL does not load in SQLite: {exc}.") from exc
    conn.close()
    canonical = {
        "schema_sql": ";\n".join(create_stmts) + ";",
        "insert_sql": ";\n".join(insert_stmts) + ";",
        "reference_sql": reference_sql,
        "order_significant": spec["order_significant"],
        "table_names": sorted(spec.get("table_names", [])),
    }
    return json.dumps(canonical, sort_keys=True)


def grade(ground_truth: Any, candidate_sql: str) -> tuple[GradeOutcome, str]:
    """Return ``(GradeOutcome, detail)`` — ``MATCH`` / ``MISMATCH`` / ``INFRA``."""
    try:
        spec, create_stmts, insert_stmts, reference_sql = _validated_ground_truth(ground_truth)
    except ValueError as exc:
        return GradeOutcome.INFRA, str(exc)
    ok, payload = guard_candidate_sql(candidate_sql)
    if not ok:
        return GradeOutcome.MISMATCH, f"guard rejected candidate query: {payload}"
    candidate_stmt = payload
    order_significant = spec["order_significant"]

    try:
        seeded = build_db(create_stmts, insert_stmts)
    except sqlite3.Error as exc:
        return GradeOutcome.INFRA, f"could not rebuild database: {exc}"
    try:
        outcome = _compare_on(seeded, reference_sql, candidate_stmt, order_significant, "seeded")
        if outcome is not None:
            return outcome
        try:
            perturbed = _clone_db(seeded)
        except sqlite3.Error as exc:
            return GradeOutcome.INFRA, f"could not build the perturbed database: {exc}"
        try:
            try:
                perturb_db(perturbed)
            except sqlite3.Error as exc:
                return GradeOutcome.INFRA, f"could not perturb database: {exc}"
            outcome = _compare_on(perturbed, reference_sql, candidate_stmt, order_significant, "perturbed")
            if outcome is not None:
                return outcome
        finally:
            perturbed.close()
    finally:
        seeded.close()
    return GradeOutcome.MATCH, "result sets match on seeded and perturbed databases"


def score(ground_truth: Any, response: str) -> tuple[float, dict[str, Any]]:
    """Rollout-time reward: ``1.0`` for a match, ``0.0`` otherwise (INFRA also scores 0 and is flagged)."""
    outcome, detail = grade(ground_truth, extract_sql(response))
    if outcome is GradeOutcome.INFRA:
        return 0.0, {"verifier_error": detail}
    return (1.0 if outcome is GradeOutcome.MATCH else 0.0), {"detail": detail}


def is_correct(response: str, normalized_ground_truth: str) -> bool:
    """Contract preflight check: does ``response`` satisfy the (already normalized) verifier input?"""
    return grade(normalized_ground_truth, extract_sql(response))[0] is GradeOutcome.MATCH

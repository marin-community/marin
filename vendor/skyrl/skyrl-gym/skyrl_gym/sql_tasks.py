"""SQL task sessions with candidate execution in Shellbox and private host grading."""

import json
import math
import re
import sqlite3
from concurrent.futures import Executor
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from uuid import uuid4

import pandas as pd
from rolloutengine.contracts import ModelTurn, SessionStart, Transition
from shellbox.machine import Command, ExitReason, Machine
from skyrl_gym.source_task import session_parameters
from taskcompendium.grading_result import GradeResult, Outcome
from rolloutengine.spec import LoweredTaskSpec
from rolloutengine.task_session import session_start
from verifyit.modes.grade_exact import grade_exact_candidate
from verifyit.spec import ExactSpec

from skyrl_gym.envs.sql.utils import final_sql
from skyrl_gym.envs.text_to_sql import scoring
from skyrl_gym.task_records import terminal_grade
from skyrl_gym.task_sessions import BlockingOperations

QUERY_OUTPUT_LIMIT_BYTES = 4 * 1024 * 1024
SQL_QUERY_TIMEOUT = 30.0
QUERY_DATABASE_FILENAME = "fixture.sqlite"
SQL_DATABASE_DIRECTORIES = {
    "synsql": "SynSQL-2.5M/databases",
    "spider": "spider/database",
    "bird": "bird/train/train_databases",
}


@dataclass(frozen=True)
class QueryResult:
    columns: int = 0
    rows: tuple[tuple, ...] = ()
    error: str | None = None


def _reference_result(
    connection: sqlite3.Connection, sql: str, *, verifyit: bool, timeout: float = scoring.QUERY_TIMEOUT
) -> scoring.QueryRows:
    result = scoring.query_result(connection, sql, timeout=timeout)
    if len(result.rows) > scoring._MAX_RESULT_ROWS:
        raise ValueError("Reference result exceeds the row limit")
    if verifyit and any(isinstance(value, float) and not math.isfinite(value) for row in result.rows for value in row):
        raise ValueError("Reference result contains a nonfinite value")
    return result


def _seeded_databases(spec: dict, root: Path, *, verifyit: bool) -> list[tuple[Path, scoring.QueryRows]]:
    connection = scoring.build_db(
        scoring.split_statements(spec["schema_sql"]), scoring.split_statements(spec["insert_sql"])
    )
    try:
        cases = []
        for name in ("seeded", "perturbed"):
            if name == "perturbed":
                scoring.perturb_db(connection)
            reference = _reference_result(connection, spec["reference_sql"], verifyit=verifyit)
            path = root / f"{name}.sqlite"
            with sqlite3.connect(path) as snapshot:
                connection.backup(snapshot)
            cases.append((path, reference))
        return cases
    finally:
        connection.close()


def _database_snapshot(database: Path, reference_sql: str, root: Path, *, verifyit: bool) -> scoring.QueryRows:
    if not database.is_file():
        raise FileNotFoundError(database)
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        with sqlite3.connect(root / QUERY_DATABASE_FILENAME) as snapshot:
            connection.backup(snapshot)
            return _reference_result(snapshot, reference_sql, verifyit=verifyit, timeout=SQL_QUERY_TIMEOUT)
    finally:
        connection.close()


async def _upload_queries(machine: Machine, databases: list[Path]) -> str:
    directory = f"/tmp/skyrl-sql-{uuid4().hex}"
    await machine.upload(Path(scoring.__file__), f"{directory}/query.py")
    for database in databases:
        await machine.upload(database, f"{directory}/{database.name}")
    return directory


async def _query(machine: Machine, directory: str, database: str, sql: str, *, timeout: float) -> QueryResult:
    result = await machine.run(
        Command(
            ("python", f"{directory}/query.py", f"{directory}/{database}"),
            stdin=json.dumps({"sql": sql, "timeout": timeout}).encode(),
            timeout=timeout + 10,
            output_limit_bytes=QUERY_OUTPUT_LIMIT_BYTES,
        )
    )
    if result.reason != ExitReason.EXITED or result.exit_code != 0:
        raise RuntimeError(f"SQLite runtime failed: {result.stderr.decode(errors='replace')}")
    if result.stdout_truncated:
        return QueryResult(error="Candidate result exceeds the output limit")
    payload = json.loads(result.stdout)
    if "error" in payload:
        return QueryResult(error=payload["error"])
    rows = tuple(
        tuple(
            bytes.fromhex(value["bytes"])
            if isinstance(value, dict) and "bytes" in value
            else float(value["float"])
            if isinstance(value, dict)
            else value
            for value in row
        )
        for row in payload["rows"]
    )
    return QueryResult(payload["columns"], rows)


def _exact_equal(reference: str, candidate: str) -> bool:
    specification = ExactSpec(
        expected=(reference,), ignore_case=False, ignore_whitespace=False, strip_outer_whitespace=False
    )
    return grade_exact_candidate(specification, candidate).reward == 1.0


def _seeded_equal(reference: scoring.QueryRows, candidate: QueryResult, *, ordered: bool, verifyit: bool) -> bool:
    actual = (candidate.columns, list(candidate.rows))
    if not verifyit:
        return scoring.results_equivalent(reference, actual, order_significant=ordered)[0]
    if any(isinstance(value, float) and not math.isfinite(value) for row in candidate.rows for value in row):
        return False
    encoded = []
    for columns, rows in (reference, actual):
        normalized = [
            tuple(value[1] if value[0] == "~num" else value for value in row) for row in scoring.normalized_rows(rows)
        ]
        encoded.append(json.dumps([columns, scoring.canonical_rows(normalized, multiset=True, ordered=ordered)]))
    return _exact_equal(*encoded)


def _set_equal(reference: list[tuple], candidate: QueryResult, *, verifyit: bool) -> bool:
    if candidate.error is not None:
        return False
    if not verifyit:
        return frozenset(reference) == frozenset(candidate.rows)
    if any(isinstance(value, float) and not math.isfinite(value) for row in candidate.rows for value in row):
        return False
    return _exact_equal(
        scoring.canonical_rows(reference, multiset=False, ordered=False),
        scoring.canonical_rows(list(candidate.rows), multiset=False, ordered=False),
    )


def _reference_failure(error: sqlite3.Error | ValueError) -> GradeResult:
    diagnostics = {"verifier_error": str(error)}
    return GradeResult(Outcome.INFRA_ERROR, None, "SQL verification failed", diagnostics=diagnostics)


class SeededSQLTaskSession:
    """Compare one query on seeded and perturbed public databases."""

    def __init__(self, lowered: LoweredTaskSpec, machine: Machine | None, *, executor: Executor | None = None):
        task = lowered.task
        parameters = session_parameters(task)
        self.task = task
        self.machine = machine
        self.blocking = BlockingOperations(executor)
        self.local_files = ExitStack()
        self.verifyit = bool(parameters.config.get("verifyit_enabled", False))
        self.spec = scoring.parse_ground_truth(parameters.extras.get("reward_model", {}).get("ground_truth"))
        self.directory = ""
        self.cases: list[tuple[str, scoring.QueryRows]] = []
        self.result = GradeResult(Outcome.UNAVAILABLE, None, "The task has no completed query")

    async def prepare(self) -> SessionStart:
        if self.spec is None:
            self.result = GradeResult(
                Outcome.INFRA_ERROR,
                None,
                "Invalid SQL task",
                diagnostics={"verifier_error": "invalid reward_model.ground_truth"},
            )
            return session_start(self.task, shell_tool=None, provider_tools=())
        assert self.machine is not None
        root = Path(self.local_files.enter_context(TemporaryDirectory(prefix="skyrl-sql-reference-")))
        try:
            cases = await self.blocking.run(_seeded_databases, self.spec, root, verifyit=self.verifyit)
        except (sqlite3.Error, ValueError) as error:
            self.result = _reference_failure(error)
        else:
            self.directory = await _upload_queries(self.machine, [path for path, _ in cases])
            self.cases = [(path.name, reference) for path, reference in cases]
        return session_start(self.task, shell_tool=None, provider_tools=())

    async def advance(self, turn: ModelTurn) -> Transition:
        if self.result.status != Outcome.UNAVAILABLE:
            return Transition(done=True, reward=0.0, grade=self.result, metrics=dict(self.result.diagnostics))
        assert self.machine is not None and self.spec is not None
        accepted, query = scoring.guard_candidate_sql(scoring.extract_sql(turn.text))
        reward = 0.0
        detail = "Candidate query was rejected"
        if accepted:
            for database, reference in self.cases:
                candidate = await _query(self.machine, self.directory, database, query, timeout=scoring.QUERY_TIMEOUT)
                if candidate.error is not None or not _seeded_equal(
                    reference, candidate, ordered=self.spec["order_significant"], verifyit=self.verifyit
                ):
                    detail = candidate.error or "Query result differs"
                    break
            else:
                reward = 1.0
                detail = "Result sets match on seeded and perturbed databases"
        self.result = GradeResult(Outcome.GRADED, reward, diagnostics={"detail": detail})
        return Transition(done=True, reward=reward, grade=self.result, metrics={"detail": detail})

    async def grade(
        self, messages: tuple[dict[str, Any], ...], *, stop_reason: str, outputs: dict[str, bytes]
    ) -> GradeResult:
        return self.result

    async def close(self) -> None:
        try:
            await self.blocking.close()
        finally:
            self.local_files.close()


class SQLTaskSession:
    """Execute SQL tools between model turns and grade the final result set."""

    def __init__(self, lowered: LoweredTaskSpec, machine: Machine | None, *, executor: Executor | None = None):
        task = lowered.task
        assert machine is not None
        parameters = session_parameters(task)
        config, extras = parameters.config, parameters.extras
        self.task = task
        self.machine = machine
        self.blocking = BlockingOperations(executor)
        self.local_files = ExitStack()
        self.database = (
            Path(config["db_path"])
            / SQL_DATABASE_DIRECTORIES[extras["data"]]
            / extras["db_id"]
            / f"{extras['db_id']}.sqlite"
        )
        self.reference_sql = extras["reward_spec"]["ground_truth"]
        self.verifyit = bool(config.get("verifyit_enabled", False))
        self.max_turns = lowered.session.max_turns
        self.directory = ""
        self.reference = scoring.QueryRows(0, [])
        self.failure: GradeResult | None = None
        self.transcript: list[str] = []
        self.grades: list[GradeResult] = []

    async def prepare(self) -> SessionStart:
        root = Path(self.local_files.enter_context(TemporaryDirectory(prefix="skyrl-sql-reference-")))
        try:
            self.reference = await self.blocking.run(
                _database_snapshot, self.database, self.reference_sql, root, verifyit=self.verifyit
            )
        except (sqlite3.Error, ValueError) as error:
            self.failure = _reference_failure(error)
        else:
            self.directory = await _upload_queries(self.machine, [root / QUERY_DATABASE_FILENAME])
        return session_start(self.task, shell_tool=None, provider_tools=())

    async def advance(self, turn: ModelTurn) -> Transition:
        if self.failure is not None:
            self.grades.append(self.failure)
            return Transition(done=True, reward=0.0, grade=self.failure, metrics=dict(self.failure.diagnostics))
        for tag in ("</sql>", "</solution>"):
            if tag in turn.text and turn.text.split(tag, 1)[1]:
                raise ValueError(f"The model response must end at {tag}")
        self.transcript.append(turn.text)
        remaining = self.max_turns - len(self.grades) - 1
        done = remaining == 0 or ("<solution>" in turn.text and "</solution>" in turn.text)
        if done:
            query = final_sql("".join(self.transcript))
            reward = -1.0
            if query is not None:
                candidate = await _query(
                    self.machine, self.directory, QUERY_DATABASE_FILENAME, query, timeout=SQL_QUERY_TIMEOUT
                )
                reward = float(_set_equal(self.reference.rows, candidate, verifyit=self.verifyit))
            grade = GradeResult(Outcome.GRADED, reward, passed=reward == 1.0)
            self.grades.append(grade)
            return Transition(done=True, reward=reward, grade=grade)
        match = re.search(r"<sql>(.*?)</sql>", turn.text, re.DOTALL)
        if match is None:
            observation = "Your previous action is invalid. Follow the format of outputting thinking process and sql tool, and try again."
        else:
            candidate = await _query(
                self.machine, self.directory, QUERY_DATABASE_FILENAME, match.group(1), timeout=scoring.QUERY_TIMEOUT
            )
            if candidate.error is not None:
                observation = candidate.error
            else:
                frame = pd.DataFrame(frozenset(candidate.rows))
                observation = frame.to_string(index=False)
                if len(observation) > 9000:
                    observation = "Truncated to 50 lines since returned response too long: " + frame.head(50).to_string(
                        index=False
                    )
        observation = f"\n\n<observation>{observation}\n<reminder>You have {remaining} turns left to complete the task.</reminder></observation>\n\n"
        self.transcript.append(observation)
        grade = GradeResult(Outcome.UNAVAILABLE, None, "Tool execution is not a terminal verdict")
        self.grades.append(grade)
        return Transition(done=False, reward=0.0, grade=grade, observations=({"role": "user", "content": observation},))

    async def grade(
        self, messages: tuple[dict[str, Any], ...], *, stop_reason: str, outputs: dict[str, bytes]
    ) -> GradeResult:
        return terminal_grade(self.grades)

    async def close(self) -> None:
        try:
            await self.blocking.close()
        finally:
            self.local_files.close()

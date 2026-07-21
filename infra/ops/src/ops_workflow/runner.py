# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Agent runner boundary and an in-memory contract fake."""

from dataclasses import dataclass
from typing import Protocol

MAX_EVIDENCE_BYTES = 256 * 1024
MAX_RESULT_BYTES = 64 * 1024


class RunnerConflict(RuntimeError):
    """The requested runner operation conflicts with durable session state."""


@dataclass(frozen=True)
class SessionSpec:
    name: str
    case_id: str
    title: str
    repo_revision: str


@dataclass(frozen=True)
class SessionRef:
    id: str
    name: str


@dataclass(frozen=True)
class TurnAck:
    number: int


@dataclass(frozen=True)
class TurnSnapshot:
    number: int
    active: bool
    terminal: bool
    permission_request_ids: tuple[str, ...]


@dataclass(frozen=True)
class ResultArtifact:
    revision: int
    content: bytes


class AgentRunner(Protocol):
    """Idempotent session and exact-turn operations for an agent runtime."""

    async def ensure_session(self, spec: SessionSpec) -> SessionRef: ...

    async def upload_evidence(self, session_id: str, content: bytes) -> None: ...

    async def start_turn(self, session_id: str, client_request_id: str, prompt: str) -> TurnAck: ...

    async def turn_snapshot(self, session_id: str, number: int) -> TurnSnapshot: ...

    async def result_artifact(self, session_id: str) -> ResultArtifact: ...

    async def interrupt_turn(self, session_id: str, number: int) -> None: ...

    async def archive_idle(self, session_id: str) -> None: ...


@dataclass
class _FakeSession:
    ref: SessionRef
    case_id: str
    title: str
    repo_revision: str
    evidence: bytes = b""
    next_turn_number: int = 1
    active_turn_number: int | None = None
    terminal_turns: set[int] | None = None
    request_turns: dict[str, int] | None = None
    result_revision: int = 0
    result_content: bytes | None = None
    archived: bool = False

    def __post_init__(self) -> None:
        if self.terminal_turns is None:
            self.terminal_turns = set()
        if self.request_turns is None:
            self.request_turns = {}


class FakeAgentRunner:
    """In-memory implementation of the idempotent runner contract."""

    def __init__(self) -> None:
        self._sessions_by_name: dict[str, _FakeSession] = {}
        self._sessions_by_id: dict[str, _FakeSession] = {}

    async def ensure_session(self, spec: SessionSpec) -> SessionRef:
        existing = self._sessions_by_name.get(spec.name)
        if existing is not None:
            if (
                existing.case_id != spec.case_id
                or existing.title != spec.title
                or existing.repo_revision != spec.repo_revision
            ):
                raise RunnerConflict("deterministic session name belongs to different inputs")
            if existing.archived:
                raise RunnerConflict("session is archived")
            return existing.ref

        ref = SessionRef(id=f"fake-{len(self._sessions_by_id) + 1}", name=spec.name)
        session = _FakeSession(
            ref=ref,
            case_id=spec.case_id,
            title=spec.title,
            repo_revision=spec.repo_revision,
        )
        self._sessions_by_name[spec.name] = session
        self._sessions_by_id[ref.id] = session
        return ref

    async def upload_evidence(self, session_id: str, content: bytes) -> None:
        if len(content) > MAX_EVIDENCE_BYTES:
            raise ValueError(f"evidence exceeds {MAX_EVIDENCE_BYTES} bytes")
        session = self._session(session_id)
        self._require_open(session)
        session.evidence = bytes(content)

    async def start_turn(self, session_id: str, client_request_id: str, prompt: str) -> TurnAck:
        session = self._session(session_id)
        self._require_open(session)
        assert session.request_turns is not None
        existing = session.request_turns.get(client_request_id)
        if existing is not None:
            return TurnAck(existing)
        if session.active_turn_number is not None:
            raise RunnerConflict("session already has an active turn")
        if not session.evidence:
            raise RunnerConflict("evidence must be uploaded before a turn starts")
        if not prompt:
            raise ValueError("prompt must not be empty")

        number = session.next_turn_number
        session.next_turn_number += 1
        session.active_turn_number = number
        session.request_turns[client_request_id] = number
        return TurnAck(number)

    async def turn_snapshot(self, session_id: str, number: int) -> TurnSnapshot:
        session = self._session(session_id)
        assert session.terminal_turns is not None
        if session.active_turn_number == number:
            return TurnSnapshot(number=number, active=True, terminal=False, permission_request_ids=())
        if number in session.terminal_turns:
            return TurnSnapshot(number=number, active=False, terminal=True, permission_request_ids=())
        raise KeyError(f"unknown turn {number}")

    async def result_artifact(self, session_id: str) -> ResultArtifact:
        session = self._session(session_id)
        if session.result_content is None:
            raise KeyError("ops-result artifact does not exist")
        return ResultArtifact(revision=session.result_revision, content=session.result_content)

    async def interrupt_turn(self, session_id: str, number: int) -> None:
        session = self._session(session_id)
        if session.active_turn_number != number:
            raise RunnerConflict("turn is not active")
        assert session.terminal_turns is not None
        session.active_turn_number = None
        session.terminal_turns.add(number)

    async def archive_idle(self, session_id: str) -> None:
        session = self._session(session_id)
        if session.active_turn_number is not None:
            raise RunnerConflict("cannot archive a session with an active turn")
        session.archived = True

    async def complete_turn(self, session_id: str, number: int, result_content: bytes) -> None:
        """Complete an exact fake turn and publish the next result revision."""

        if len(result_content) > MAX_RESULT_BYTES:
            raise ValueError(f"result exceeds {MAX_RESULT_BYTES} bytes")
        session = self._session(session_id)
        if session.active_turn_number != number:
            raise RunnerConflict("turn is not active")
        assert session.terminal_turns is not None
        session.active_turn_number = None
        session.terminal_turns.add(number)
        session.result_revision += 1
        session.result_content = bytes(result_content)

    def _session(self, session_id: str) -> _FakeSession:
        try:
            return self._sessions_by_id[session_id]
        except KeyError as error:
            raise KeyError(f"unknown session {session_id}") from error

    @staticmethod
    def _require_open(session: _FakeSession) -> None:
        if session.archived:
            raise RunnerConflict("session is archived")

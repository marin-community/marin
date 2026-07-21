# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Narrow server-side adapter for Loom's ACP session API."""

import json
import time
from dataclasses import dataclass
from typing import Protocol

import httpx


@dataclass(frozen=True)
class LoomSession:
    id: str
    url: str
    live_turn: int | None


@dataclass(frozen=True)
class ChatSnapshot:
    blocks: tuple[dict[str, object], ...]
    live_turn: int | None

    @property
    def complete(self) -> bool:
        return self.live_turn is None and any(block.get("kind") == "turn_end" for block in self.blocks)


@dataclass(frozen=True)
class AgentArtifact:
    content: str
    revision: int


class AgentGateway(Protocol):
    """The subset of Loom required by the ops workflow."""

    async def create_session(self, *, name: str, title: str, goal: str, case_id: str, turn_id: str) -> LoomSession: ...

    async def prompt(self, session_id: str, text: str, *, actor: str, case_id: str, turn_id: str) -> int | None: ...

    async def chat(self, session_id: str) -> ChatSnapshot: ...

    async def artifact(self, session_id: str, name: str) -> AgentArtifact: ...

    async def archive(self, session_id: str) -> None: ...

    async def interrupt(self, session_id: str) -> None: ...

    async def close(self) -> None: ...


class LoomGateway:
    """Authenticated Loom REST client; its bearer token never reaches the browser."""

    def __init__(
        self,
        *,
        api_url: str,
        token: str,
        repo_root: str,
        base: str,
        agent: str,
        model: str | None,
        effort: str | None,
    ) -> None:
        self._repo_root = repo_root
        self._base = base
        self._agent = agent
        self._model = model
        self._effort = effort
        self._client = httpx.AsyncClient(
            base_url=api_url.rstrip("/"),
            headers={"Authorization": f"Bearer {token}"},
            timeout=httpx.Timeout(30, read=60),
        )

    async def create_session(self, *, name: str, title: str, goal: str, case_id: str, turn_id: str) -> LoomSession:
        payload = {
            "cwd": self._repo_root,
            "title": title,
            "goal": goal,
            "name": name,
            "base": self._base,
            "agent": self._agent,
            "protocol": "acp",
            "mode": "plan",
        }
        if self._model:
            payload["model"] = self._model
        if self._effort:
            payload["effort"] = self._effort
        response = await self._client.post("/api/sessions", json=payload)
        if response.status_code == 409:
            response = await self._client.get(f"/api/sessions/{name}")
        response.raise_for_status()
        session = response.json()
        session_id = str(session["id"])
        url_response = await self._client.get(f"/api/sessions/{session_id}/url")
        url_response.raise_for_status()
        snapshot = await self.chat(session_id)
        return LoomSession(id=session_id, url=str(url_response.json()["url"]), live_turn=snapshot.live_turn)

    async def prompt(self, session_id: str, text: str, *, actor: str, case_id: str, turn_id: str) -> int | None:
        response = await self._client.post(
            f"/api/sessions/{session_id}/prompt",
            json={"text": text, "by": actor, "force_steer": False, "force_queued": False, "files": []},
        )
        response.raise_for_status()
        turn = response.json().get("turn")
        return int(turn) if isinstance(turn, int) else None

    async def chat(self, session_id: str) -> ChatSnapshot:
        response = await self._client.get(f"/api/sessions/{session_id}/chat")
        response.raise_for_status()
        payload = response.json()
        blocks = payload.get("blocks")
        if not isinstance(blocks, list):
            raise RuntimeError("Loom chat response has no blocks array")
        live_turn = payload.get("live_turn")
        return ChatSnapshot(
            blocks=tuple(block for block in blocks if isinstance(block, dict)),
            live_turn=int(live_turn) if isinstance(live_turn, int) else None,
        )

    async def artifact(self, session_id: str, name: str) -> AgentArtifact:
        response = await self._client.get(f"/api/sessions/{session_id}/artifacts/{name}")
        response.raise_for_status()
        payload = response.json()
        content = payload.get("content")
        meta = payload.get("meta")
        if not isinstance(content, str) or not isinstance(meta, dict) or not isinstance(meta.get("rev"), int):
            raise RuntimeError(f"Loom artifact {name!r} has an invalid response")
        return AgentArtifact(content=content, revision=int(meta["rev"]))

    async def archive(self, session_id: str) -> None:
        response = await self._client.post(f"/api/sessions/{session_id}/archive", json={})
        response.raise_for_status()

    async def interrupt(self, session_id: str) -> None:
        response = await self._client.post(f"/api/sessions/{session_id}/interrupt", json={})
        response.raise_for_status()

    async def close(self) -> None:
        await self._client.aclose()


@dataclass
class _StubSession:
    id: str
    url: str
    prompts: list[str]
    started_at: float
    turn: int
    case_id: str
    turn_id: str


class StubAgentGateway:
    """Deterministic local ACP substitute used by the browser spike and tests."""

    def __init__(self, *, completion_delay: float = 0.8) -> None:
        self._completion_delay = completion_delay
        self._sessions: dict[str, _StubSession] = {}
        self._sessions_by_name: dict[str, _StubSession] = {}

    async def create_session(self, *, name: str, title: str, goal: str, case_id: str, turn_id: str) -> LoomSession:
        existing = self._sessions_by_name.get(name)
        if existing is not None:
            return LoomSession(id=existing.id, url=existing.url, live_turn=existing.turn)
        session_id = f"stub-{len(self._sessions) + 1}"
        session = _StubSession(
            id=session_id,
            url=f"http://127.0.0.1:7878/s/{session_id}",
            prompts=[goal],
            started_at=time.monotonic(),
            turn=0,
            case_id=case_id,
            turn_id=turn_id,
        )
        self._sessions[session_id] = session
        self._sessions_by_name[name] = session
        return LoomSession(id=session.id, url=session.url, live_turn=0)

    async def prompt(self, session_id: str, text: str, *, actor: str, case_id: str, turn_id: str) -> int | None:
        session = self._sessions[session_id]
        session.turn += 1
        session.prompts.append(text)
        session.started_at = time.monotonic()
        session.case_id = case_id
        session.turn_id = turn_id
        return session.turn

    async def chat(self, session_id: str) -> ChatSnapshot:
        session = self._sessions[session_id]
        blocks: list[dict[str, object]] = []
        for turn, prompt in enumerate(session.prompts):
            blocks.append(
                {"turn": turn, "seq": 0, "kind": "user_message", "payload": {"text": prompt}, "created_at": ""}
            )
            complete = turn < session.turn or time.monotonic() - session.started_at >= self._completion_delay
            if complete:
                blocks.extend(_stub_reply(turn))
        current_complete = time.monotonic() - session.started_at >= self._completion_delay
        return ChatSnapshot(blocks=tuple(blocks), live_turn=None if current_complete else session.turn)

    async def archive(self, session_id: str) -> None:
        if session_id not in self._sessions:
            raise KeyError(session_id)

    async def artifact(self, session_id: str, name: str) -> AgentArtifact:
        if name != "ops-result":
            raise KeyError(name)
        session = self._sessions[session_id]
        return AgentArtifact(
            content=json.dumps(
                {
                    "schema_version": 2,
                    "case_id": session.case_id,
                    "ops_turn_id": session.turn_id,
                    "outcome": "no_action",
                    "summary": "The stub completed the read-only investigation.",
                    "evidence": [],
                    "action_taken": "none",
                    "recommended_next_step": "Review the case evidence if the alert remains firing.",
                    "escalation": None,
                }
            ),
            revision=session.turn + 1,
        )

    async def interrupt(self, session_id: str) -> None:
        session = self._sessions[session_id]
        session.started_at = 0

    async def close(self) -> None:
        return None


def _stub_reply(turn: int) -> list[dict[str, object]]:
    return [
        {
            "turn": turn,
            "seq": 1,
            "kind": "thought",
            "payload": {"text": "Correlating the Grafana fingerprint with read-only Kubernetes evidence."},
            "created_at": "",
        },
        {
            "turn": turn,
            "seq": 2,
            "kind": "tool_call",
            "payload": {"title": "kubectl get events", "status": "completed"},
            "created_at": "",
        },
        {
            "turn": turn,
            "seq": 3,
            "kind": "agent_message",
            "payload": {
                "text": (
                    "Validated the alert against the target cluster in read-only mode. "
                    "The alert is still visible, no automatic mutation was attempted, and an operator should "
                    "review the linked Grafana evidence if it remains firing."
                )
            },
            "created_at": "",
        },
        {"turn": turn, "seq": 4, "kind": "turn_end", "payload": {"stop_reason": "end_turn"}, "created_at": ""},
    ]

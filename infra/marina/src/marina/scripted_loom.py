# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A deterministic Loom HTTP/SSE boundary for Marina browser journeys."""

import contextlib
import json
import queue
import threading
import time
from dataclasses import dataclass, field

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from marina.journeys import KERNEL_START_TIMEOUT, LOOPBACK, free_port

SESSION_ID = "plantt-agent-1"
CREATED_AT = "2026-09-09T12:00:00Z"


@dataclass(frozen=True)
class RecordedRequest:
    method: str
    path: str
    body: dict[str, object]
    origin: str | None


@dataclass
class ScriptedLoom:
    """Mutable journey controller serving the Loom routes consumed by the panel."""

    requests: list[RecordedRequest] = field(default_factory=list)
    blocks: list[dict[str, object]] = field(default_factory=list)
    live_turn: int | None = None
    _events: queue.Queue[tuple[str, object]] = field(default_factory=queue.Queue)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _stopping: threading.Event = field(default_factory=threading.Event)

    def reset(self) -> None:
        with self._lock:
            self.requests.clear()
            self.blocks.clear()
            self.live_turn = None
        while True:
            try:
                self._events.get_nowait()
            except queue.Empty:
                break

    def recorded(self, path: str) -> tuple[RecordedRequest, ...]:
        with self._lock:
            return tuple(request for request in self.requests if request.path == path)

    def release(self, event: str, data: object) -> None:
        if event == "block":
            block = data
            assert isinstance(block, dict)
            key = (block["turn"], block["seq"])
            with self._lock:
                self.blocks = [item for item in self.blocks if (item["turn"], item["seq"]) != key]
                self.blocks.append(block)
        elif event == "turn":
            turn = data
            assert isinstance(turn, dict)
            self.live_turn = int(turn["turn"]) if turn.get("state") == "started" else None
        self._events.put((event, data))

    def stop(self) -> None:
        self._stopping.set()

    def _record(self, request: Request, body: dict[str, object]) -> None:
        with self._lock:
            self.requests.append(
                RecordedRequest(
                    method=request.method,
                    path=request.url.path,
                    body=body,
                    origin=request.headers.get("origin"),
                )
            )

    def _session(self) -> dict[str, object]:
        return {
            "id": SESSION_ID,
            "status": "running",
            "transition": None,
            "protocol": "acp",
            "profile": "marina",
            "current_mode": "default",
            "created_by": "journey-user",
        }

    def app(self) -> FastAPI:
        api = FastAPI()
        api.add_middleware(
            CORSMiddleware,
            allow_origin_regex=r"http://127\.0\.0\.1:\d+",
            allow_credentials=True,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Content-Type"],
        )

        @api.get("/healthz")
        def healthz() -> dict[str, bool]:
            return {"ok": True}

        @api.post("/api/auth/me")
        async def auth_me(request: Request) -> dict[str, object]:
            body = await request.json()
            self._record(request, body)
            return {"authenticated": True, "username": "journey-user"}

        @api.post("/api/sessions/launch")
        async def launch(request: Request) -> dict[str, object]:
            body = await request.json()
            self._record(request, body)
            with self._lock:
                self.blocks = [
                    {
                        "turn": 1,
                        "seq": 0,
                        "kind": "user_message",
                        "payload": {"text": body["goal"], "by": "journey-user"},
                        "created_at": CREATED_AT,
                    }
                ]
                self.live_turn = 1
            return self._session()

        @api.post("/api/sessions/get")
        async def get_session(request: Request) -> JSONResponse:
            body = await request.json()
            self._record(request, body)
            if body.get("session") != SESSION_ID or not self.blocks:
                return JSONResponse({"error": "session not found"}, status_code=404)
            return JSONResponse(self._session())

        @api.post("/api/sessions/chat")
        async def chat(request: Request) -> dict[str, object]:
            body = await request.json()
            self._record(request, body)
            with self._lock:
                return {
                    "blocks": list(self.blocks),
                    "older_cursor": None,
                    "live_turn": self.live_turn,
                    "effective_mode": "default" if self.live_turn is not None else None,
                    "pending_prompt": None,
                    "metadata": {"commands": [], "config_options": [], "modes": [], "steering_supported": False},
                }

        @api.get("/api/sessions/chat/stream")
        def chat_stream(request: Request, session: str):
            self._record(request, {"session": session})

            def events():
                while not self._stopping.is_set():
                    try:
                        event, data = self._events.get(timeout=0.2)
                    except queue.Empty:
                        yield ": keepalive\n\n"
                        continue
                    yield f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"

            return StreamingResponse(events(), media_type="text/event-stream")

        @api.post("/api/sessions/prompt/create")
        async def prompt(request: Request) -> dict[str, object]:
            body = await request.json()
            self._record(request, body)
            with self._lock:
                turn = max((int(block["turn"]) for block in self.blocks), default=0) + 1
                self.blocks.append(
                    {
                        "turn": turn,
                        "seq": 0,
                        "kind": "user_message",
                        "payload": {"text": body["text"], "by": "journey-user"},
                        "created_at": CREATED_AT,
                    }
                )
                self.live_turn = turn
            return {"queued": False, "turn": turn}

        @api.post("/api/sessions/interrupt")
        async def interrupt(request: Request) -> dict[str, bool]:
            body = await request.json()
            self._record(request, body)
            self.live_turn = None
            return {"interrupted": True}

        @api.post("/api/sessions/url")
        async def session_url(request: Request) -> dict[str, str]:
            body = await request.json()
            self._record(request, body)
            return {"url": f"http://{LOOPBACK}/s/{SESSION_ID}"}

        @api.post("/api/sessions/permissions/answer")
        async def answer_permission(request: Request) -> dict[str, object]:
            body = await request.json()
            self._record(request, body)
            return {"resolved": True, "option_id": body["option_id"]}

        return api


@dataclass
class ScriptedLoomServer:
    controller: ScriptedLoom = field(default_factory=ScriptedLoom)
    port: int = field(default_factory=free_port)
    _server: uvicorn.Server | None = None
    _thread: threading.Thread | None = None

    @property
    def origin(self) -> str:
        return f"http://{LOOPBACK}:{self.port}"

    def start(self) -> None:
        self._server = uvicorn.Server(
            uvicorn.Config(self.controller.app(), host=LOOPBACK, port=self.port, log_level="warning")
        )
        self._thread = threading.Thread(target=self._server.run, name="scripted-loom", daemon=True)
        self._thread.start()
        deadline = time.monotonic() + KERNEL_START_TIMEOUT
        while time.monotonic() < deadline:
            with contextlib.suppress(httpx.HTTPError):
                if httpx.get(f"{self.origin}/healthz", timeout=1).status_code == 200:
                    return
            time.sleep(0.05)
        raise RuntimeError(f"scripted Loom did not answer within {KERNEL_START_TIMEOUT}s")

    def stop(self) -> None:
        self.controller.stop()
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=10)

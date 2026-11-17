# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Client for NVIDIA NeMo Skills' local execution sandbox protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import requests
import threading


MAX_VERIFIER_OUTPUT_CHARACTERS = 65536
# Share the limit across clients so increasing Gym workers cannot exhaust the sandbox.
_SANDBOX_SLOTS = threading.BoundedSemaphore(16)


@dataclass(frozen=True)
class SandboxClient:
    host: str = "127.0.0.1"
    port: int = 6000
    _sessions: set[str] = field(default_factory=set, compare=False, repr=False)
    _requested_sessions: set[str] = field(default_factory=set, compare=False, repr=False)

    def execute(
        self,
        code: str,
        *,
        language: str,
        timeout_seconds: float,
        session_id: str | None = None,
        max_output_characters: int = 1000,
    ) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if session_id is not None:
            headers["X-Session-ID"] = session_id
            self._requested_sessions.add(session_id)
        with _SANDBOX_SLOTS:
            response = requests.post(
                f"http://{self.host}:{self.port}/execute",
                headers=headers,
                json={
                    "generated_code": code,
                    "language": language,
                    "timeout": timeout_seconds,
                    "max_output_characters": max_output_characters,
                    **({"traceback_verbosity": "Plain"} if language == "ipython" else {}),
                },
                timeout=timeout_seconds + 5.0,
            )
        response.raise_for_status()
        value = response.json()
        if not isinstance(value, dict):
            raise RuntimeError(f"sandbox returned a non-object response: {value!r}")
        if "<output cut>" in value.get("stdout", "") or "<output cut>" in value.get("stderr", ""):
            value["output_truncated"] = True
        if session_id is not None:
            if session_id in self._sessions and value.get("new_session_created") is True:
                raise RuntimeError(f"Sandbox lost stateful session {session_id}")
            self._sessions.add(session_id)
        return value

    def close_session(self, session_id: str) -> None:
        """Release a used session; already-deleted sessions are accepted."""
        if session_id not in self._requested_sessions:
            return
        response = requests.delete(
            f"http://{self.host}:{self.port}/sessions/{session_id}",
            headers={"X-Session-ID": session_id},
            timeout=5.0,
        )
        if response.status_code != 404:
            response.raise_for_status()
        self._requested_sessions.discard(session_id)
        self._sessions.discard(session_id)


def sandbox_output_text(result: dict[str, Any]) -> str:
    output = f"{result.get('stdout', '')}{result.get('stderr', '')}"
    return output[:-1] if output.endswith("\n") else output

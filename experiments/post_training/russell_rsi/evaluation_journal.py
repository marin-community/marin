# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reserve single-use supplementary attempts before model inference."""

import base64
import json
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass

import httpx
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.russell_rsi.bootstrap_loop import write_once
from experiments.post_training.russell_rsi.sources import compact_json_sha256


@dataclass
class AttemptJournal:
    directory: StoragePath
    binding: dict
    turn_index: int = 0

    def saved_result(self) -> dict | None:
        reservation = self.directory / "reservation.json"
        result = self.directory / "result.json"
        if not reservation.exists():
            if result.exists():
                raise ValueError("Completed supplementary attempt has no reservation")
            return None
        write_once(reservation, self.binding)
        if not result.exists():
            raise RuntimeError("Reserved supplementary attempt is incomplete; refuse to repeat inference")
        saved = json.loads(result.read_text())
        if saved["binding"] != self.binding:
            raise ValueError("Completed supplementary attempt has a different identity")
        return saved["result"]

    async def run(self, operation: Callable[[], Awaitable[dict]]) -> dict:
        saved = self.saved_result()
        if saved is not None:
            return saved
        write_once(self.directory / "reservation.json", self.binding)
        token = ACTIVE_ATTEMPT.set(self)
        try:
            value = await operation()
            write_once(self.directory / "result.json", {"binding": self.binding, "result": value})
            return value
        finally:
            ACTIVE_ATTEMPT.reset(token)

    async def post(self, client: httpx.AsyncClient, url: str, body: dict) -> httpx.Response:
        directory = self.directory / "turns" / f"{self.turn_index:03d}"
        self.turn_index += 1
        identity = {"attempt": self.binding, "request_sha256": compact_json_sha256(body)}
        if (directory / "issued.json").exists():
            raise RuntimeError("Supplementary model request was already issued; refuse to repeat inference")
        write_once(directory / "request.json", body)
        write_once(directory / "issued.json", identity)
        response = await client.post(url, json=body)
        write_once(
            directory / "response.json",
            {
                "identity": identity,
                "url": url,
                "status_code": response.status_code,
                "body_base64": base64.b64encode(response.content).decode("ascii"),
            },
        )
        return response


ACTIVE_ATTEMPT: ContextVar[AttemptJournal | None] = ContextVar("supplementary_attempt", default=None)


@dataclass(frozen=True)
class EvaluationJournal:
    directory: StoragePath
    binding: dict

    def seal(self) -> None:
        write_once(self.directory / "binding.json", self.binding)
        for kind, keys in self.binding["attempts"].items():
            for key, task_sha256 in keys.items():
                self.attempt(kind, key, task_sha256).saved_result()

    def complete(self) -> bool:
        return all(
            (self.directory / kind / key / "result.json").exists()
            for kind, keys in self.binding["attempts"].items()
            for key in keys
        )

    def attempt(self, kind: str, key: str, task_sha256: str) -> AttemptJournal:
        if self.binding["attempts"][kind].get(key) != task_sha256:
            raise ValueError("Supplementary attempt is outside the frozen plan")
        return AttemptJournal(
            self.directory / kind / key,
            {"evaluation": self.binding, "kind": kind, "key": key, "task_sha256": task_sha256},
        )

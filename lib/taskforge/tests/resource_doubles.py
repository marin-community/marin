# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Scripted environment, inference and web doubles for the Taskforge tests."""

import errno
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import override

from taskforge.effects.events import Fingerprint, ReplayPolicy
from taskforge.effects.issues import ContextExceeded, Hold, Restart, Transient
from taskforge.model.llm import Completion
from taskforge.resources.env import CommandResult, RecordedEnv
from taskforge.resources.execution import Notify
from taskforge.resources.inference import InferenceClient, InferenceRequest
from taskforge.resources.web import WebClient, WebFetch, WebSearch


@dataclass(frozen=True, slots=True)
class FakeCommand:
    result: CommandResult
    writes: Mapping[PurePosixPath, bytes]


class FakeEnv(RecordedEnv):
    """An in-memory filesystem and explicitly scripted shell effects, without a shell parser."""

    def __init__(self, image: str, commands: Mapping[str, FakeCommand | Transient | Restart], notify: Notify) -> None:
        super().__init__(image, notify)
        self.commands = commands
        self.files: dict[PurePosixPath, bytes] = {}

    @override
    async def run(self, cmd: str, *, timeout: float) -> CommandResult:
        self.check_open()
        scripted = self.commands.get(cmd)
        if scripted is None:
            raise ValueError(f"Unscripted fake command: {cmd}")
        if isinstance(scripted, Transient | Restart):
            raise scripted
        self.files.update(scripted.writes)
        self.record_command(cmd, timeout, scripted.result)
        return scripted.result

    @override
    async def read(self, path: PurePosixPath, *, offset: int = 0, limit: int | None = None) -> bytes:
        self.check_open()
        if offset < 0 or (limit is not None and limit < 0):
            raise ValueError("offset and limit must be nonnegative")
        try:
            return self.files[path][offset : None if limit is None else offset + limit]
        except KeyError as error:
            raise FileNotFoundError(errno.ENOENT, "No such file", str(path)) from error

    @override
    async def write(self, path: PurePosixPath, data: bytes) -> None:
        self.check_open()
        self.files[path] = data
        self.record_write(path, data)


class FakeInference(InferenceClient):
    def __init__(
        self, completions: Sequence[Completion | Transient | Restart | Hold | ContextExceeded], *, identity: str = "fake"
    ) -> None:
        self.script = tuple(completions)
        self.requests: list[InferenceRequest] = []
        self.identity = identity

    @override
    def fingerprint(self) -> Fingerprint:
        return Fingerprint("inference", self.identity, ReplayPolicy.CACHE)

    @override
    async def complete(self, request: InferenceRequest) -> Completion:
        index = len(self.requests)
        self.requests.append(request)
        if index >= len(self.script):
            raise RuntimeError("Fake inference script exhausted")
        result = self.script[index]
        if isinstance(result, Transient | Restart | Hold | ContextExceeded):
            raise result
        return result


class FakeWeb(WebClient):
    def __init__(self, searches: Mapping[str, str], pages: Mapping[str, str], *, identity: str = "fake") -> None:
        self.searches, self.pages, self.identity = dict(searches), dict(pages), identity
        self.requests: list[WebSearch | WebFetch] = []

    @override
    def fingerprint(self) -> Fingerprint:
        return Fingerprint("web", self.identity, ReplayPolicy.CACHE)

    @override
    async def search(self, query: str) -> str:
        self.requests.append(WebSearch(query))
        return self.searches[query]

    @override
    async def fetch(self, url: str) -> str:
        self.requests.append(WebFetch(url))
        return self.pages[url]

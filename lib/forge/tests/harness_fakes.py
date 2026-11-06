# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping, Sequence
from typing import override

from forge.effects.context import Context
from forge.harness.protocol import Harness, Tool
from forge.model.stage import Exit
from forge.model.workpiece import Conversation
from forge.resources.env import Env
from forge.resources.inference import InferenceClient


class FakeHarness(Harness):
    def __init__(self, exits: Sequence[Exit]) -> None:
        self.exits = tuple(exits)
        self.sessions: list[Conversation] = []

    @override
    async def run(
        self, ctx: Context, session: Conversation, env: Env, client: InferenceClient, tools: Mapping[str, Tool[object]]
    ) -> Exit:
        index = len(self.sessions)
        self.sessions.append(session)
        return self.exits[index]

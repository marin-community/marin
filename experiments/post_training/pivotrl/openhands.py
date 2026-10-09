# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade OpenHands candidates (:mod:`marin.rl.openhands_pivot`) by functional next action.

:class:`OpenHandsNextActionTask` grades replies with
:func:`verifyit.adapters.openhands_next_action.grade_next_action`. Its LLM-judge tier is off unless
the task names a judge model.
"""

import json
import os
from dataclasses import dataclass
from typing import Any

from verifyit.adapters.openhands_next_action import NextActionJudge, TurnState, grade_next_action
from verifyit.grade import Reward
from verifyit.modes.grade_judge import JudgeConnection

from experiments.post_training.pivotrl.task import row_id, sort_key, tool_call_message, without_nulls


@dataclass(frozen=True)
class OpenHandsNextActionTask:
    """OpenHands turns graded by functional next action; the judge tier is off unless a model is set."""

    judge_model: str | None = None
    judge_base_url: str = ""
    judge_api_key_env: str = ""

    components = ("tool_name", "same_operation", "functional", "undecided")
    row_id = staticmethod(row_id)
    sort_key = staticmethod(sort_key)

    def request(self, row: dict[str, Any]) -> dict[str, Any]:
        tools = json.loads(row["extra_info"]["openhands"]["tools_json"])
        return {"messages": [without_nulls(message) for message in row["prompt"]], "tools": tools}

    def grade(self, row: dict[str, Any], message: dict[str, Any]) -> Reward:
        turn = row["extra_info"]["openhands"]
        return grade_next_action(
            json.loads(turn["expected_call_json"]),
            message,
            TurnState(turn["repo_root"], turn["cwd"]),
            observation=turn["observation"],
            judge=self._judge(),
        )

    def reference(self, row: dict[str, Any]) -> dict[str, Any]:
        """The expert's call as an assistant turn, without the narration that preceded it."""
        call = json.loads(row["extra_info"]["openhands"]["expected_call_json"])
        return tool_call_message(call["name"], call["arguments"])

    def _judge(self) -> NextActionJudge | None:
        if self.judge_model is None:
            return None
        return NextActionJudge(
            model=self.judge_model,
            connection=JudgeConnection(base_url=self.judge_base_url, api_key=os.environ[self.judge_api_key_env]),
        )


OPENHANDS_NEXT_ACTION = OpenHandsNextActionTask()
# Tier 2 enabled. A run grading with this task must forward OPENROUTER_TOKEN to its jobs.
OPENHANDS_NEXT_ACTION_JUDGED = OpenHandsNextActionTask(
    judge_model="openai/gpt-oss-120b",
    judge_base_url="https://openrouter.ai/api/v1",
    judge_api_key_env="OPENROUTER_TOKEN",
)

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import io
import json
import tarfile
from pathlib import Path

from marin.datakit.download.rollout_transforms import TRAJECTORY_FAILED_TAG, TRAJECTORY_SOLVED_TAG
from marin.datakit.download.rts_glm53_rollouts import load_trial_shard, trial_conversation
from openai_harmony import Message

PROMPT = "You are an AI assistant tasked with solving command-line tasks. Task: create /app/out.txt"
FIRST_REPLY = json.dumps(
    {
        "analysis": "Nothing exists.",
        "plan": "Write the file.",
        "commands": [{"keystrokes": "touch /app/out.txt\n", "duration": 0.1}],
    },
    indent=2,
)
FINAL_REPLY = '{\n  "analysis": "Done.",\n  "plan": "None.",\n  "commands": [],\n  "task_complete": true\n}'


def _debug(final_reply: str = FINAL_REPLY) -> dict:
    return {
        "request": {
            "messages": [
                {"role": "user", "content": PROMPT},
                {"role": "assistant", "content": FIRST_REPLY, "reasoning_content": "Need create file."},
                {"role": "user", "content": "New Terminal Output:\nroot@box:/app# touch /app/out.txt\n"},
            ]
        },
        "response": {
            "choices": [{"message": {"role": "assistant", "content": final_reply, "reasoning": "It exists now."}}]
        },
    }


def _execution(execution_id: str, *, phase: str = "glm", reward: float | None = 1.0) -> dict:
    return {"execution_id": execution_id, "task_group_id": "rts_group_a", "phase": phase, "reward": reward}


def _write_shard(path: Path, trials: dict[str, dict[str, object]]) -> None:
    with tarfile.open(path, "w") as archive:
        for trial, members in trials.items():
            for name, payload in members.items():
                data = json.dumps(payload).encode()
                info = tarfile.TarInfo(f"{trial}/{name}")
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))


def test_conversation_keeps_every_turns_reasoning() -> None:
    messages = trial_conversation(_debug())

    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant"]
    assert [m.get("reasoning_content") for m in messages if m["role"] == "assistant"] == [
        "Need create file.",
        "It exists now.",
    ]
    assert json.loads(messages[-1]["content"])["task_complete"] is True


def test_conversation_ignores_serialized_null_reasoning() -> None:
    debug = _debug()
    debug["response"]["choices"][0]["message"]["reasoning"] = "None"

    assert "reasoning_content" not in trial_conversation(debug)[-1]


def test_conversation_rejects_malformed_terminus_reply() -> None:
    assert trial_conversation(_debug(final_reply='{\n  `analysis`: "Done."\n}')) is None


def test_shard_keeps_verified_model_trials_from_the_last_episode(tmp_path: Path) -> None:
    stale = _debug(final_reply='{"analysis": "stale", "plan": "", "commands": []}')
    shard = tmp_path / "trials-00000.tar"
    _write_shard(
        shard,
        {
            "solved": {
                "rts_execution.json": _execution("solved"),
                "agent/episode-2/debug.json": stale,
                "agent/episode-10/debug.json": _debug(),
            },
            "failed": {"rts_execution.json": _execution("failed", reward=0.0), "agent/episode-0/debug.json": _debug()},
            "errored": {"rts_execution.json": _execution("errored", reward=None)},
            "gold": {"rts_execution.json": _execution("gold", phase="oracle")},
        },
    )

    docs = list(load_trial_shard(str(shard)))

    assert [(doc["source_id"], doc["outcome"]) for doc in docs] == [
        ("solved", TRAJECTORY_SOLVED_TAG),
        ("failed", TRAJECTORY_FAILED_TAG),
    ]
    final = Message.from_dict(docs[0]["messages"][-1])
    assert json.loads(final.content[0].text)["analysis"] == "Done."

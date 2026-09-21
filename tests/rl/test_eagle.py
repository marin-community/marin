# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

from finestore.eval import ConversationType, ParticipantType, RolloutContentType
from marin.rl.eagle import eagle_replay_records


def _row(
    *,
    turn_id: int,
    participant: ParticipantType,
    content: str,
    prompt_token_ids: list[int] | None = None,
    completion_token_ids: list[int] | None = None,
) -> dict:
    return {
        "task": "aime24",
        "doc_id": "7",
        "trial_id": "trial-1",
        "turn_id": turn_id,
        "part_id": 0,
        "conversation_type": ConversationType.CHAT,
        "participant_type": participant,
        "participant_id": participant,
        "content_type": RolloutContentType.MESSAGE,
        "content": content,
        "prompt_token_ids": prompt_token_ids,
        "completion_token_ids": completion_token_ids,
    }


def test_eagle_replay_records_preserve_exact_harbor_tokens() -> None:
    rows = [
        _row(turn_id=0, participant=ParticipantType.USER, content="Solve this."),
        _row(
            turn_id=1,
            participant=ParticipantType.ASSISTANT,
            content="Answer: 42",
            prompt_token_ids=[1, 2, 3],
            completion_token_ids=[4, 5],
        ),
    ]

    records = list(eagle_replay_records(rows))

    assert len(records) == 1
    assert records[0].prompt_token_ids == (1, 2, 3)
    assert records[0].response_token_ids == (4, 5)
    assert records[0].group_id == "aime24/7/trial-1"


def test_eagle_replay_records_assemble_evalchemy_chat() -> None:
    rows = [
        _row(turn_id=0, participant=ParticipantType.SYSTEM, content="Think carefully."),
        _row(turn_id=1, participant=ParticipantType.USER, content="Solve this."),
        _row(turn_id=2, participant=ParticipantType.ASSISTANT, content="Answer: 42"),
    ]

    [record] = eagle_replay_records(rows)
    payload = json.loads(record.to_json())

    assert payload == {
        "group_id": "aime24/7/trial-1",
        "prompt": [
            {"role": "system", "content": "Think carefully."},
            {"role": "user", "content": "Solve this."},
        ],
        "response": "Answer: 42",
    }

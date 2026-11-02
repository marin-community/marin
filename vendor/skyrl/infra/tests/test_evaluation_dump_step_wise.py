"""Step-wise evaluation dump rows retain the scoring boundary."""

import json
from pathlib import Path

from skyrl_train.utils.trainer_utils import dump_per_dataset_eval_results


class _Tokenizer:
    def decode(self, tokens: list[int]) -> str:
        return ",".join(str(token) for token in tokens)


def test_step_wise_dump_marks_only_final_row_as_scored(tmp_path: Path) -> None:
    batch = {
        "prompt_token_ids": [[1], [1]],
        "response_ids": [[2], [3]],
        "rewards": [0.25, 1.0],
        "is_last_step": [False, True],
    }
    dump_per_dataset_eval_results(
        str(tmp_path), _Tokenizer(), batch, ["task", "task"], ["agent", "agent"], {}, ["uid", "uid"]
    )

    rows = [json.loads(line) for line in (tmp_path / "task.jsonl").read_text().splitlines()]
    assert [(row["score"], row["is_last_step"]) for row in rows] == [(0.25, False), (1.0, True)]

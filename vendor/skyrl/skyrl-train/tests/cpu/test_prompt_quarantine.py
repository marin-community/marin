"""Quarantine of prompts whose groups training can never use."""

import pytest

from skyrl_train.group_admission import AdmissionRejection
from skyrl_train.rollouts.loader import QUARANTINED_PROMPTS_METRIC, PromptLoader, SeededPasses

UNUSABLE = AdmissionRejection.BELOW_MINIMUM_GROUP_SIZE.value


class Prompts:
    def __init__(self, num_rows: int):
        self.num_rows = num_rows

    def __len__(self) -> int:
        return self.num_rows

    def __getitem__(self, index: int) -> int:
        return index

    def uid(self, index: int) -> str:
        return str(index)

    def collate_fn(self, items: list[int]) -> list[dict]:
        return [{"uid": str(item)} for item in items]


def loader(num_rows: int, batch_size: int) -> PromptLoader:
    return PromptLoader(
        Prompts(num_rows),
        SeededPasses(num_rows, seed=0, shuffle=False),
        batch_size=batch_size,
        max_stale_retries=1,
        max_unusable_groups=2,
    )


def offered(prompts: PromptLoader, count: int) -> list[str]:
    return [prompts.next_prompt(live=())["uid"] for _ in range(count)]


def test_unusable_prompt_stops_being_offered():
    prompts = loader(num_rows=4, batch_size=3)
    prompts.record_disposition("0", UNUSABLE)
    prompts.record_disposition("0", UNUSABLE)

    assert "0" not in offered(prompts, 6)
    assert prompts.observe([])[QUARANTINED_PROMPTS_METRIC] == 1.0


def test_usable_group_resets_the_count():
    prompts = loader(num_rows=4, batch_size=3)
    prompts.record_disposition("0", UNUSABLE)
    prompts.record_disposition("0", "consumed")
    prompts.record_disposition("0", AdmissionRejection.STALE.value)
    prompts.record_disposition("0", UNUSABLE)

    assert "0" in offered(prompts, 4)


def test_quarantine_below_one_batch_fails():
    prompts = loader(num_rows=4, batch_size=3)
    for uid in ("0", "1"):
        prompts.record_disposition(uid, UNUSABLE)
    prompts.record_disposition("0", UNUSABLE)

    with pytest.raises(RuntimeError, match="leaving 2 usable prompts"):
        prompts.record_disposition("1", UNUSABLE)

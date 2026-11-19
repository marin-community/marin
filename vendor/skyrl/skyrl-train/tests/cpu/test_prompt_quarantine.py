"""Quarantine of prompts whose groups training can never use."""

import pytest
from loguru import logger

from skyrl_train.group_admission import AdmissionRejection
from skyrl_train.rollouts.loader import QUARANTINE_STEPS, QUARANTINED_PROMPTS_METRIC, PromptLoader, SeededPasses

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

    def label(self, index: int) -> str:
        return f"task-{index}"

    def collate_fn(self, items: list[int]) -> list[dict]:
        return [{"uid": str(item)} for item in items]


def loader(num_rows: int, batch_size: int, max_unusable_groups: int = 2) -> PromptLoader:
    return PromptLoader(
        Prompts(num_rows),
        SeededPasses(num_rows, seed=0, shuffle=False),
        batch_size=batch_size,
        max_stale_retries=1,
        max_unusable_groups=max_unusable_groups,
    )


def offered(prompts: PromptLoader, count: int) -> list[str]:
    return [prompts.next_prompt(live=())["uid"] for _ in range(count)]


@pytest.mark.parametrize("max_unusable_groups", [1, 2])
def test_unusable_prompt_sits_out_then_returns(max_unusable_groups: int):
    prompts = loader(num_rows=4, batch_size=3, max_unusable_groups=max_unusable_groups)
    for _ in range(max_unusable_groups):
        prompts.record_disposition("0", UNUSABLE)

    for _ in range(QUARANTINE_STEPS - 1):
        assert "0" not in offered(prompts, 6)
        assert prompts.observe([])[QUARANTINED_PROMPTS_METRIC] == 1.0
    assert prompts.observe([])[QUARANTINED_PROMPTS_METRIC] == 0.0
    assert "0" in offered(prompts, 4)


def test_quarantine_warning_names_the_task(caplog):
    handler = logger.add(caplog.handler, format="{message}")
    prompts = loader(num_rows=4, batch_size=3, max_unusable_groups=1)
    prompts.record_disposition("2", UNUSABLE)
    logger.remove(handler)

    assert "Quarantined prompt 2 (task-2)" in caplog.text


def test_usable_group_resets_the_count():
    prompts = loader(num_rows=4, batch_size=3)
    prompts.record_disposition("0", UNUSABLE)
    prompts.record_disposition("0", "consumed")
    prompts.record_disposition("0", AdmissionRejection.STALE.value)
    prompts.record_disposition("0", UNUSABLE)

    assert "0" in offered(prompts, 4)


def test_quarantine_keeps_one_batch_of_prompts():
    # Every prompt fails, as in a sandbox outage: only the one spare prompt may leave the rotation.
    prompts = loader(num_rows=4, batch_size=3)
    for uid in ("0", "1", "2", "3"):
        prompts.record_disposition(uid, UNUSABLE)
        prompts.record_disposition(uid, UNUSABLE)

    assert sorted(set(offered(prompts, 6))) == ["1", "2", "3"]
    assert prompts.observe([])[QUARANTINED_PROMPTS_METRIC] == 1.0

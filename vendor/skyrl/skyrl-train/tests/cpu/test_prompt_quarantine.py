"""Quarantine of prompts whose groups training can never use."""

import pytest
from loguru import logger

from skyrl_train.adaptive_sampling import AdaptiveSourceOrder
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


def test_adaptive_loader_scans_past_live_rows_and_resumes_a_stale_retry():
    def make_loader():
        order = AdaptiveSourceOrder(
            ["busy", "free"],
            {"busy": 1e6, "free": 1},
            seed=37,
            window_size=1,
            decay=0.95,
            epsilon=1,
            prior_alpha=1,
            prior_beta=1,
        )
        return PromptLoader(Prompts(2), order, batch_size=1, max_stale_retries=1, max_unusable_groups=2)

    prompts = make_loader()
    group = prompts.next_prompt(live={"0"})
    assert group["uid"] == "1"
    assert prompts.observe([])["curriculum/busy/proposed_rows"] > 0
    prompts.record_disposition(group["uid"], AdmissionRejection.STALE.value)
    prompts.retry(group)
    resumed = make_loader()
    resumed.load_state_dict(prompts.state_dict())
    assert resumed.next_prompt(live={"0", "1"}) is None
    assert resumed.next_prompt(live={"0"}) == group
    assert resumed.observe([])["curriculum/busy/proposed_rows"] == 0


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


def test_quarantine_keeps_enough_prompts_for_each_open_batch():
    families = {str(index): "agentic" if index >= 7 else "math" for index in range(10)}
    prompts = PromptLoader(
        Prompts(len(families)),
        SeededPasses(len(families), seed=0, shuffle=False),
        batch_size=3,
        max_stale_retries=1,
        max_unusable_groups=1,
        families=families,
        family_minimums={"agentic": 2, "math": 4},
    )
    for uid in ("7", "8", "9"):
        prompts.record_disposition(uid, UNUSABLE)

    first = prompts.next_prompt(live=(), family="agentic")
    assert first is not None
    second = prompts.next_prompt(live={first["uid"]}, family="agentic")
    assert second is not None
    assert {first["uid"], second["uid"]} == {"8", "9"}


def test_family_prompt_passes_cover_each_pool_and_resume():
    families = {str(index): "agentic" if index >= 96 else "math" for index in range(100)}

    def make_loader():
        return PromptLoader(
            Prompts(len(families)),
            SeededPasses(len(families), seed=7, shuffle=True),
            batch_size=8,
            max_stale_retries=1,
            max_unusable_groups=2,
            families=families,
        )

    def draws(loader, family, count):
        return [loader.next_prompt(live=(), family=family)["uid"] for _ in range(count)]

    original = make_loader()
    first_math = draws(original, "math", 7)
    first_agentic = draws(original, "agentic", 1)
    restored = make_loader()
    restored.load_state_dict(original.state_dict())

    remaining_math = []
    remaining_agentic = []
    for count in (29, 30, 30):
        remaining_math.extend(draws(original, "math", count))
        remaining_agentic.extend(draws(original, "agentic", 1))
    assert set(first_math + remaining_math) == {str(index) for index in range(96)}
    assert set(first_agentic + remaining_agentic) == {str(index) for index in range(96, 100)}
    assert draws(restored, "math", 89) == remaining_math
    assert draws(restored, "agentic", 3) == remaining_agentic

    draws(original, "agentic", 1)
    restored.load_state_dict(original.state_dict())
    assert draws(restored, "agentic", 3) == draws(original, "agentic", 3)


def test_quarantine_keeps_one_batch_of_prompts():
    # Every prompt fails, as in a sandbox outage: only the one spare prompt may leave the rotation.
    prompts = loader(num_rows=4, batch_size=3)
    for uid in ("0", "1", "2", "3"):
        prompts.record_disposition(uid, UNUSABLE)
        prompts.record_disposition(uid, UNUSABLE)

    assert sorted(set(offered(prompts, 6))) == ["1", "2", "3"]
    assert prompts.observe([])[QUARANTINED_PROMPTS_METRIC] == 1.0

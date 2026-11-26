"""Quarantine of entries whose groups training can never use."""

import pytest
from loguru import logger

from skyrl_train.config.utils import get_default_config
from skyrl_train.group_admission import AdmissionRejection
from skyrl_train.rollouts.context import TrainingContext
from skyrl_train.rollouts.loader import QUARANTINE_STEPS, QUARANTINED_ENTRIES_METRIC, DatasetEntrySelector, SeededPasses

UNUSABLE = AdmissionRejection.BELOW_MINIMUM_GROUP_SIZE.value


class Entries:
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


def make_selector(
    num_rows: int, batch_size: int, max_unusable_groups: int = 2, *, open_slots: int | None = None
) -> DatasetEntrySelector:
    return DatasetEntrySelector(
        Entries(num_rows),
        SeededPasses(num_rows, seed=0, shuffle=False),
        batch_size=batch_size,
        minimum_pool_size=(batch_size if open_slots is None else open_slots) + 1,
        max_stale_retries=1,
        max_unusable_groups=max_unusable_groups,
    )


def offered(selector: DatasetEntrySelector, count: int) -> list[str]:
    return [selector.next_entry(live=())["uid"] for _ in range(count)]


@pytest.mark.parametrize("max_unusable_groups", [1, 2])
def test_unusable_entry_sits_out_then_returns(max_unusable_groups: int):
    selector = make_selector(num_rows=5, batch_size=3, max_unusable_groups=max_unusable_groups)
    for _ in range(max_unusable_groups):
        selector.record_disposition("0", UNUSABLE)

    for _ in range(QUARANTINE_STEPS - 1):
        assert "0" not in offered(selector, 6)
        assert selector.observe([])[QUARANTINED_ENTRIES_METRIC] == 1.0
    assert selector.observe([])[QUARANTINED_ENTRIES_METRIC] == 0.0
    assert "0" in offered(selector, 5)


def test_quarantine_warning_names_the_task(caplog):
    handler = logger.add(caplog.handler, format="{message}")
    selector = make_selector(num_rows=5, batch_size=3, max_unusable_groups=1)
    selector.record_disposition("2", UNUSABLE)
    logger.remove(handler)

    assert "Quarantined entry 2 (task-2)" in caplog.text


def test_usable_group_resets_the_count():
    selector = make_selector(num_rows=5, batch_size=3)
    selector.record_disposition("0", UNUSABLE)
    selector.record_disposition("0", "consumed")
    selector.record_disposition("0", AdmissionRejection.STALE.value)
    selector.record_disposition("0", UNUSABLE)

    assert "0" in offered(selector, 5)


def test_quarantine_keeps_a_spare_in_each_family():
    families = {str(index): "agentic" if index >= 7 else "math" for index in range(11)}
    selector = DatasetEntrySelector(
        Entries(len(families)),
        SeededPasses(len(families), seed=0, shuffle=False),
        batch_size=3,
        minimum_pool_size=7,
        max_stale_retries=1,
        max_unusable_groups=1,
        families=families,
        family_minimums={"agentic": 3, "math": 5},
    )
    for uid in ("7", "8", "9", "10"):
        selector.record_disposition(uid, UNUSABLE)

    live = set()
    for _ in range(3):
        entry = selector.next_entry(live=live, family="agentic")
        assert entry is not None
        live.add(entry["uid"])
    assert live == {"8", "9", "10"}


def test_family_entry_passes_cover_each_pool_and_resume():
    families = {str(index): "agentic" if index >= 96 else "math" for index in range(100)}

    def make_loader():
        return DatasetEntrySelector(
            Entries(len(families)),
            SeededPasses(len(families), seed=7, shuffle=True),
            batch_size=8,
            minimum_pool_size=9,
            max_stale_retries=1,
            max_unusable_groups=2,
            families=families,
        )

    def draws(selector, family, count):
        return [selector.next_entry(live=(), family=family)["uid"] for _ in range(count)]

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


def test_quarantine_keeps_a_spare_across_open_batches():
    selector = make_selector(num_rows=6, batch_size=2, open_slots=4)
    for uid in ("0", "1", "2", "3", "4", "5"):
        selector.record_disposition(uid, UNUSABLE)
        selector.record_disposition(uid, UNUSABLE)

    assert sorted(set(offered(selector, 10))) == ["1", "2", "3", "4", "5"]
    assert selector.observe([])[QUARANTINED_ENTRIES_METRIC] == 1.0


def test_training_context_requires_a_spare_across_open_batches():
    config = get_default_config()
    config.trainer.train_batch_size = 2
    config.trainer.rollout_buffer.max_staleness_steps = 1

    with pytest.raises(ValueError, match="has 4 distinct entries and needs 5"):
        TrainingContext.from_config(config, Entries(4), workers=None)

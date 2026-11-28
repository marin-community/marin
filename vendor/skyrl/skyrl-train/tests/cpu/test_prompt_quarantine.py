"""Quarantine of prompts whose groups training can never use."""

import asyncio
from collections import Counter
from dataclasses import replace

import pytest
import torch
from loguru import logger

from skyrl_train.adaptive_sampling import AdaptiveSourceOrder
from skyrl_train.config.rollout_validation import minimum_family_pool_sizes
from skyrlrecipe.declared import PromptAcquisition
from skyrl_train.config.utils import get_default_config
from skyrl_train.contextual_sampling import PromptThompsonOrder, PublicPrompt
from skyrl_train.dynamic_sampling import GroupSelectionResult
from skyrl_train.group_admission import AdmissionRejection
from skyrl_train.rollouts.buffer import (
    SAMPLING_SOURCE,
    BatchPolicy,
    GroupRewards,
    PayloadReference,
    RolloutBuffer,
    RolloutBufferConfig,
    RolloutVerdict,
)
from skyrl_train.rollouts.context import TrainingContext
from skyrl_train.rollouts.loader import (
    QUARANTINE_STEPS,
    QUARANTINED_PROMPTS_METRIC,
    STALE_RETRIES_EXHAUSTED_METRIC,
    PromptLoader,
    PromptLoaderState,
    SeededPasses,
)
from skyrl_train.telemetry import GeneratedWork

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


def loader(num_rows: int, batch_size: int, max_unusable_groups: int = 2, *, open_slots: int | None = None) -> PromptLoader:
    return PromptLoader(
        Prompts(num_rows),
        SeededPasses(num_rows, seed=0, shuffle=False),
        batch_size=batch_size,
        minimum_pool_size=(batch_size if open_slots is None else open_slots) + 1,
        max_stale_retries=1,
        max_unusable_groups=max_unusable_groups,
    )


def offered(prompts: PromptLoader, count: int) -> list[str]:
    return [prompts.next_prompt(live=())["uid"] for _ in range(count)]


def _family_thompson_loader(families, quotas, *, staleness=0, acquisition=PromptAcquisition.WITHIN_SOURCE_COST_AWARE):
    order = PromptThompsonOrder(
        [
            PublicPrompt(uid, f"source-{int(uid) % 2}", f"What is {uid} plus one?", family)
            for uid, family in families.items()
        ],
        set(),
        {},
        acquisition=acquisition,
        seed=31,
        window_size=sum(quotas.values()),
        decay=0.95,
        epsilon=0.25,
        reference_tokens=100,
        groups_per_family=quotas,
    )
    loader = PromptLoader(
        Prompts(len(families)),
        order,
        batch_size=sum(quotas.values()),
        minimum_pool_size=sum(quotas.values()) * (staleness + 1) + 1,
        max_stale_retries=1,
        max_unusable_groups=1,
        families=families,
        family_minimums=minimum_family_pool_sizes(quotas, staleness),
    )
    return loader, order


def test_seeded_loader_preserves_quarantine_and_retry_budget(tmp_path):
    original = loader(num_rows=7, batch_size=2, max_unusable_groups=2)
    original.record_disposition("0", UNUSABLE)
    original.record_disposition("0", UNUSABLE)
    original.record_disposition("1", UNUSABLE)
    retry = original.next_prompt({"0", "1"})
    original.retry(retry)
    original.observe([])

    checkpoint = tmp_path / "seeded.pt"
    torch.save(original.state_dict(), checkpoint)
    restored = loader(num_rows=7, batch_size=2, max_unusable_groups=2)
    restored.load_state_dict(torch.load(checkpoint, weights_only=False))
    for current in (original, restored):
        assert current.next_prompt(()) == retry
        current.retry(retry)
        current.record_disposition("1", UNUSABLE)
        metrics = current.observe([])
        assert metrics[STALE_RETRIES_EXHAUSTED_METRIC] == 1
        assert metrics[QUARANTINED_PROMPTS_METRIC] == 2
        assert not {"0", "1"}.intersection(offered(current, 9))
        for _ in range(QUARANTINE_STEPS - 2):
            current.observe([])
        assert "0" in offered(current, 9)
        assert "1" not in offered(current, 9)
        current.observe([])
        assert "1" in offered(current, 9)


def test_legacy_loader_checkpoint_restores_queued_retry_without_eligibility(tmp_path):
    original = loader(num_rows=5, batch_size=2, max_unusable_groups=1)
    retry = original.next_prompt(())
    state = PromptLoaderState(original.state_dict().order, [retry])
    # The original pickle contained only the order and the queued retries.
    object.__delattr__(state, "eligibility")
    checkpoint = tmp_path / "legacy-loader.pt"
    torch.save(state, checkpoint)

    restored = loader(num_rows=5, batch_size=2, max_unusable_groups=1)
    restored.record_disposition(retry["uid"], UNUSABLE)
    restored.load_state_dict(torch.load(checkpoint, weights_only=False))
    assert restored.next_prompt(()) == retry
    assert restored.observe([])[QUARANTINED_PROMPTS_METRIC] == 0


def test_weighted_loader_restores_fallback_scan_position():
    def make_loader():
        order = AdaptiveSourceOrder(
            ["busy", "free", "free", "free"],
            {"busy": 1, "free": 0.0001},
            seed=37,
            window_size=1,
            decay=0.95,
            epsilon=1,
            prior_alpha=1,
            prior_beta=1,
        )
        return PromptLoader(
            Prompts(4), order, batch_size=1, minimum_pool_size=2, max_stale_retries=1, max_unusable_groups=1
        )

    original = make_loader()
    assert original.next_prompt({"0"})["uid"] == "1"
    restored = make_loader()
    restored.load_state_dict(original.state_dict())
    assert original.next_prompt({"0"})["uid"] == restored.next_prompt({"0"})["uid"] == "2"


def test_order_without_family_leases_rejects_a_partitioned_loader():
    order = AdaptiveSourceOrder(
        ["one", "one", "two", "two"],
        {},
        seed=37,
        window_size=2,
        decay=0.95,
        epsilon=1,
        prior_alpha=1,
        prior_beta=1,
    )
    with pytest.raises(ValueError, match="does not support family leases"):
        PromptLoader(
            Prompts(4),
            order,
            batch_size=2,
            minimum_pool_size=3,
            max_stale_retries=1,
            max_unusable_groups=1,
            families={"0": "agentic", "1": "agentic", "2": "math", "3": "math"},
        )


def test_family_thompson_resumes_retries_quarantine_and_exhaustion(tmp_path):
    families = {str(row): "short" if row < 4 else "long" for row in range(8)}
    quotas = {"short": 2, "long": 2}
    loader, _ = _family_thompson_loader(families, quotas)
    retry = loader.next_prompt((), "short")
    loader.retry(retry)
    quarantined = next(uid for uid, family in families.items() if family == "short" and uid != retry["uid"])
    loader.record_disposition(quarantined, UNUSABLE)
    loader.observe([])

    checkpoint = tmp_path / "loader.pt"
    torch.save(loader.state_dict(), checkpoint)
    restored, _ = _family_thompson_loader(families, quotas)
    restored.load_state_dict(torch.load(checkpoint, weights_only=False))
    live = {uid for uid in families if uid != quarantined}
    for current in (loader, restored):
        assert current.next_prompt(live, "short") is None
        assert current.next_prompt((), "long")["family"] == "long"
        assert current.next_prompt((), "short") == retry
        current.retry(retry)
        metrics = current.observe([])
        assert metrics[STALE_RETRIES_EXHAUSTED_METRIC] == 1
        assert metrics[QUARANTINED_PROMPTS_METRIC] == 1
        for _ in range(QUARANTINE_STEPS - 2):
            current.observe([])
        assert current.next_prompt(live, "short")["uid"] == quarantined
    assert [loader.next_prompt((), "short") for _ in range(8)] == [
        restored.next_prompt((), "short") for _ in range(8)
    ]


def test_source_pass_loader_defers_stale_retries_and_restores_quarantine(tmp_path):
    prompts = [PublicPrompt(str(i), "math", f"Count from {i}.") for i in range(16)]

    def make_loader():
        order = PromptThompsonOrder(
            prompts, set(), {}, acquisition=PromptAcquisition.WITHIN_SOURCE_PASS_COST_AWARE,
            seed=19, window_size=4, decay=0.95, epsilon=0.05, reference_tokens=100,
        )
        return PromptLoader(Prompts(16), order, batch_size=4, minimum_pool_size=5, max_stale_retries=2, max_unusable_groups=1)

    loader = make_loader()
    first = loader.next_prompt(())
    loader.retry(first)
    second = loader.next_prompt(())
    assert first != second
    loader.record_disposition(second["uid"], UNUSABLE)
    live = {first["uid"]}
    path = tmp_path / "source-pass.pt"
    # The context checkpoint includes a previously dispatched, interrupted request for regeneration.
    torch.save(replace(loader.state_dict(), retries=[first]), path)
    restored = make_loader()
    restored.load_state_dict(torch.load(path, weights_only=False))
    assert restored.state_dict().retries == []
    for current in (loader, restored):
        assert current.next_prompt({p.uid for p in prompts}) is None
    seen = {first["uid"], second["uid"]}
    for _ in range(14):
        prompt = loader.next_prompt(live)
        assert prompt == restored.next_prompt(live) and prompt["uid"] not in seen
        seen.add(prompt["uid"])
    assert seen == {p.uid for p in prompts}
    # After one full pass, a leased and a quarantined row remain unavailable.
    for _ in range(14):
        prompt = loader.next_prompt(live)
        assert prompt == restored.next_prompt(live) and prompt["uid"] not in {first["uid"], second["uid"]}
    for current in (loader, restored):
        for _ in range(QUARANTINE_STEPS):
            current.observe([])
        assert {current.next_prompt(())["uid"] for _ in range(2)} == {first["uid"], second["uid"]}


@pytest.mark.asyncio
@pytest.mark.parametrize("acquisition", list(PromptAcquisition))
async def test_family_thompson_fills_exact_quotas_across_open_batches_and_resume(tmp_path, acquisition):
    quotas = {"short": 11, "long": 21}
    families = {str(row): "short" if row < 24 else "long" for row in range(70)}
    if acquisition is PromptAcquisition.WITHIN_SOURCE_PASS_COST_AWARE:
        with pytest.raises(ValueError, match="cannot partition a per-source prompt pass"):
            _family_thompson_loader(families, quotas, staleness=1, acquisition=acquisition)
        return
    config = RolloutBufferConfig(
        batch_size=32,
        max_in_flight=32,
        max_staleness_steps=1,
        batch_policy=BatchPolicy.FULL_BATCH,
        dynamic_sampling=None,
        max_candidate_groups=None,
        groups_per_family=quotas,
    )
    loader, order = _family_thompson_loader(families, quotas, staleness=1, acquisition=acquisition)
    buffer = RolloutBuffer(config)

    async def dispatch(buffer, loader, order, live, count):
        selected = []
        for _ in range(count):
            lease = await buffer.acquire_lease()
            prompt = loader.next_prompt(live, lease.family)
            uid = prompt["uid"]
            assert uid not in live
            assert Counter(families[row] for row in live | {uid})[lease.family] <= 2 * quotas[lease.family]
            live.add(uid)
            selected.append((lease.batch_id, lease.family, uid))
            prompt[SAMPLING_SOURCE] = order.assign(uid)
            verdict = RolloutVerdict(
                uid, (), GroupSelectionResult.KEEP, GroupRewards((0.0, 1.0), (0.0, 1.0), True), GeneratedWork(2, 2, 20)
            )
            await buffer.commit(lease.lease_id, prompt, verdict, PayloadReference(uid))
        return selected

    async with asyncio.timeout(5):
        await buffer.publish(1)
        live = set()
        await dispatch(buffer, loader, order, live, 33)
        first = await buffer.admit(1)
        assert first.selection is not None
        assert Counter(families[group.uid] for group in first.admitted) == quotas
        live.difference_update(group.uid for group in first.admitted)
        loader.observe(first.selection.judged)
        checkpoint = tmp_path / "queue.pt"
        torch.save((loader.state_dict(), buffer.snapshot()), checkpoint)

        restored_buffer = RolloutBuffer(config)
        restored_loader, restored_order = _family_thompson_loader(
            families, quotas, staleness=1, acquisition=acquisition
        )
        saved_loader, saved_buffer = torch.load(checkpoint, weights_only=False)
        restored_loader.load_state_dict(saved_loader)
        await restored_buffer.restore(saved_buffer)
        restored_live = {rollout.verdict.uid for rollout in saved_buffer.ready}

        await buffer.publish(2)
        await restored_buffer.publish(2)
        assert await dispatch(buffer, loader, order, live, 63) == await dispatch(
            restored_buffer, restored_loader, restored_order, restored_live, 63
        )
        second = await restored_buffer.admit(1)
        assert second.selection is not None
        assert Counter(families[group.uid] for group in second.admitted) == quotas
        await restored_buffer.publish(3)
        third = await restored_buffer.admit(1)
        assert third.selection is not None
        assert Counter(families[group.uid] for group in third.admitted) == quotas


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
        return PromptLoader(
            Prompts(2), order, batch_size=1, minimum_pool_size=2, max_stale_retries=1, max_unusable_groups=2,
        )

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
    prompts = loader(num_rows=5, batch_size=3, max_unusable_groups=max_unusable_groups)
    for _ in range(max_unusable_groups):
        prompts.record_disposition("0", UNUSABLE)

    for _ in range(QUARANTINE_STEPS - 1):
        assert "0" not in offered(prompts, 6)
        assert prompts.observe([])[QUARANTINED_PROMPTS_METRIC] == 1.0
    assert prompts.observe([])[QUARANTINED_PROMPTS_METRIC] == 0.0
    assert "0" in offered(prompts, 5)


def test_quarantine_warning_names_the_task(caplog):
    handler = logger.add(caplog.handler, format="{message}")
    prompts = loader(num_rows=5, batch_size=3, max_unusable_groups=1)
    prompts.record_disposition("2", UNUSABLE)
    logger.remove(handler)

    assert "Quarantined prompt 2 (task-2)" in caplog.text


def test_usable_group_resets_the_count():
    prompts = loader(num_rows=5, batch_size=3)
    prompts.record_disposition("0", UNUSABLE)
    prompts.record_disposition("0", "consumed")
    prompts.record_disposition("0", AdmissionRejection.STALE.value)
    prompts.record_disposition("0", UNUSABLE)

    assert "0" in offered(prompts, 5)


def test_quarantine_keeps_a_spare_in_each_family():
    families = {str(index): "agentic" if index >= 7 else "math" for index in range(11)}
    prompts = PromptLoader(
        Prompts(len(families)),
        SeededPasses(len(families), seed=0, shuffle=False),
        batch_size=3,
        minimum_pool_size=7,
        max_stale_retries=1,
        max_unusable_groups=1,
        families=families,
        family_minimums={"agentic": 3, "math": 5},
    )
    for uid in ("7", "8", "9", "10"):
        prompts.record_disposition(uid, UNUSABLE)

    live = set()
    for _ in range(3):
        prompt = prompts.next_prompt(live=live, family="agentic")
        assert prompt is not None
        live.add(prompt["uid"])
    assert live == {"8", "9", "10"}


def test_family_prompt_passes_cover_each_pool_and_resume():
    families = {str(index): "agentic" if index >= 96 else "math" for index in range(100)}

    def make_loader():
        return PromptLoader(
            Prompts(len(families)),
            SeededPasses(len(families), seed=7, shuffle=True),
            batch_size=8,
            minimum_pool_size=9,
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


def test_quarantine_keeps_a_spare_across_open_batches():
    prompts = loader(num_rows=6, batch_size=2, open_slots=4)
    for uid in ("0", "1", "2", "3", "4", "5"):
        prompts.record_disposition(uid, UNUSABLE)
        prompts.record_disposition(uid, UNUSABLE)

    assert sorted(set(offered(prompts, 10))) == ["1", "2", "3", "4", "5"]
    assert prompts.observe([])[QUARANTINED_PROMPTS_METRIC] == 1.0


def test_training_context_requires_a_spare_across_open_batches():
    config = get_default_config()
    config.trainer.train_batch_size = 2
    config.trainer.rollout_buffer.max_staleness_steps = 1

    with pytest.raises(ValueError, match="has 4 distinct prompts and needs 5"):
        TrainingContext.from_config(config, Prompts(4), workers=None)

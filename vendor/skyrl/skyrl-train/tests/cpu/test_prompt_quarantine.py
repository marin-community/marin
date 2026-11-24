"""Quarantine of prompts whose groups training can never use."""

import asyncio
from types import SimpleNamespace

import pytest
import ray
from loguru import logger

from skyrl_train.config.rollout_validation import BatchPolicy, FAMILY_COLUMN
from skyrl_train.dynamic_sampling import GroupSelectionPolicy, GroupSelectionResult
from skyrl_train.group_admission import AdmissionRejection, GroupAdmissionPolicy, GroupAdvantageInvariant
from skyrl_train.rollouts.buffer import (
    GroupRewards,
    PayloadReference,
    RolloutBuffer,
    RolloutBufferConfig,
    RolloutContentPolicy,
    RolloutTask,
    RolloutVerdict,
)
from skyrl_train.rollouts.context import RolloutRequestSpec, TrainingContext
from skyrl_train.rollouts.loader import QUARANTINE_STEPS, QUARANTINED_PROMPTS_METRIC, PromptLoader, SeededPasses
from skyrl_train.rollouts.payloads import MemoryPayloads, RolloutWriter
from skyrl_train.task_inputs import PreferenceInput, TrajectoryMetadata
from skyrl_train.telemetry import GeneratedWork

UNUSABLE = AdmissionRejection.BELOW_MINIMUM_GROUP_SIZE.value
USABLE = "consumed"


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
        return [
            {"uid": str(item), "input": PreferenceInput([item], {}, TrajectoryMetadata(str(item), "test"))}
            for item in items
        ]


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
    prompts.record_disposition("0", USABLE)
    prompts.record_disposition("0", AdmissionRejection.STALE.value)
    prompts.record_disposition("0", UNUSABLE)

    assert "0" in offered(prompts, 4)


@pytest.mark.parametrize("family", [None, "agentic"])
def test_recovered_prompt_fills_the_missing_group(family: str | None):
    max_unusable_groups = 3
    if family is None:
        prompts = loader(num_rows=3, batch_size=2, max_unusable_groups=max_unusable_groups)
        live = set()
    else:
        prompts = PromptLoader(
            Prompts(5),
            SeededPasses(5, seed=0, shuffle=False),
            batch_size=3,
            max_stale_retries=1,
            max_unusable_groups=max_unusable_groups,
            families={"0": family, "1": family, "2": family, "3": "math", "4": "math"},
            family_minimums={family: 2, "math": 1},
        )
        # A quarantine in another pool precedes the one that can fill this slot.
        for _ in range(max_unusable_groups):
            prompts.record_disposition("3", UNUSABLE)
        # The batch already has its math group.
        live = {"4"}

    for _ in range(max_unusable_groups):
        prompts.record_disposition("0", UNUSABLE)
    kept = prompts.next_prompt(live=live, family=family)
    assert kept is not None and kept["uid"] == "1"
    live.add(kept["uid"])

    # With one group missing, prompt 2 keeps failing and the cooldown clock cannot advance.
    for _ in range(max_unusable_groups):
        failed = prompts.next_prompt(live=live, family=family)
        assert failed is not None and failed["uid"] == "2"
        prompts.record_disposition(failed["uid"], UNUSABLE)

    recovered = prompts.next_prompt(live=live, family=family)
    assert recovered is not None and recovered["uid"] == "0"
    assert recovered.get(FAMILY_COLUMN) == family
    prompts.record_disposition(recovered["uid"], USABLE)
    assert prompts.observe([])[QUARANTINED_PROMPTS_METRIC] == (0.0 if family is None else 1.0)


@pytest.mark.asyncio
async def test_recovered_prompt_fills_the_current_full_batch_with_a_future_lease(monkeypatch: pytest.MonkeyPatch):
    max_unusable_groups = 3
    prompts = loader(num_rows=3, batch_size=2, max_unusable_groups=max_unusable_groups)
    for _ in range(max_unusable_groups):
        prompts.record_disposition("0", UNUSABLE)
    config = RolloutBufferConfig(
        batch_size=2,
        max_in_flight=4,
        max_staleness_steps=1,
        batch_policy=BatchPolicy.FULL_BATCH,
        dynamic_sampling=None,
        max_candidate_groups=None,
    )
    buffer = RolloutBuffer(config)
    rejections: asyncio.Queue[None] = asyncio.Queue()
    # Hold later attempts from the failing prompt while the first batch waits for its last group.
    for _ in range(max_unusable_groups):
        rejections.put_nowait(None)

    async def run_task(task: RolloutTask, _writer: RolloutWriter) -> int:
        uid = task.prompt["uid"]
        rejected = uid == "2"
        if rejected:
            await rejections.get()
        verdict = RolloutVerdict(
            uid,
            (AdmissionRejection.BELOW_MINIMUM_GROUP_SIZE,) if rejected else (),
            None if rejected else GroupSelectionResult.KEEP,
            None if rejected else GroupRewards((0.0,), (0.0,), False),
            GeneratedWork(1, 1, 1),
        )
        await buffer.commit(task.lease.lease_id, task.prompt, verdict, PayloadReference(None if rejected else uid))
        return verdict.work.generated_token_count

    actor = SimpleNamespace(
        publish=SimpleNamespace(remote=buffer.publish),
        acquire_lease=SimpleNamespace(remote=buffer.acquire_lease),
        admit=SimpleNamespace(remote=buffer.admit),
    )
    monkeypatch.setattr(ray, "kill", lambda _actor: None)
    content = RolloutContentPolicy(
        GroupAdmissionPolicy(
            GroupAdvantageInvariant.no_group_advantage(physical_group_size=1), rollout_logprobs_required=False
        ),
        GroupSelectionPolicy(None),
    )
    context = TrainingContext(
        prompts,
        config,
        actor,
        content,
        RolloutRequestSpec(samples_per_prompt=1, sampling_params={}),
        SimpleNamespace(run_task=run_task),
        MemoryPayloads(),
        rollout_spans=False,
    )
    context.start()
    try:
        async with asyncio.timeout(5):
            await context.publish(1)
            batch = await context.next_batch_metadata(stall_timeout=2)
        assert {group.uid for group in batch.groups} == {"0", "1"}
    finally:
        await context.close()


def test_early_retry_waits_for_the_quarantined_prompt_to_finish():
    prompts = loader(num_rows=3, batch_size=2, max_unusable_groups=1)
    prompts.record_disposition("0", UNUSABLE)
    prompts.record_disposition("2", UNUSABLE)
    first = prompts.next_prompt(live={"1"})
    assert first is not None and first["uid"] == "0"

    # Another rejection reaches the minimum while the first early retry is still generating.
    prompts.record_disposition("2", UNUSABLE)
    regular = prompts.next_prompt(live={"0", "1"})
    assert regular is not None and regular["uid"] == "2"
    prompts.record_disposition("0", UNUSABLE)
    retry = prompts.next_prompt(live={"1", "2"})
    assert retry is not None and retry["uid"] == "0"


def test_failed_early_retry_preserves_quarantine_and_rotates_candidates():
    prompts = loader(num_rows=4, batch_size=2, max_unusable_groups=1)
    for uid in ("0", "1", "3"):
        prompts.record_disposition(uid, UNUSABLE)

    live = {"2"}
    first = prompts.next_prompt(live=live)
    assert first is not None and first["uid"] in {"0", "1"}
    live.add(first["uid"])
    other = prompts.next_prompt(live=live)
    assert other is not None and other["uid"] == "3"

    live.remove(first["uid"])
    prompts.record_disposition(first["uid"], UNUSABLE)
    live.add(other["uid"])
    assert prompts.next_prompt(live=live) is None
    live.remove(other["uid"])
    prompts.record_disposition(other["uid"], UNUSABLE)
    next_retry = prompts.next_prompt(live=live)
    assert next_retry is not None and {first["uid"], next_retry["uid"]} == {"0", "1"}


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

    # Completing a batch clears its unused early retry.
    assert prompts.observe([])[QUARANTINED_PROMPTS_METRIC] == 1.0
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

    # Completing a batch clears its unused early retry.
    assert prompts.observe([])[QUARANTINED_PROMPTS_METRIC] == 1.0
    assert sorted(set(offered(prompts, 6))) == ["1", "2", "3"]

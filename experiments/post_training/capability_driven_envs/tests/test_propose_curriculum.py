# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

# These tests run in the marin-taskforge package environment; environments without it skip them.
pytest.importorskip("taskforge")

import asyncio
import itertools
import json
import logging
from dataclasses import replace
from datetime import UTC, datetime, timedelta

from taskforge.llm.client import ModelRequestRejected, ModelUnavailable
from taskforge.stage10_proposal.model import Environment

from experiments.post_training.capability_driven_envs import propose_curriculum
from experiments.post_training.capability_driven_envs.propose_curriculum import (
    CONTROL_FILE,
    FAILURE_FILE,
    MAX_REPAIRS,
    RESULT_FILE,
    STACK_POLICY,
    VERIFICATION_RULE,
    ControlState,
    OutputRoot,
    ProvenanceMismatch,
    RequestLimit,
    ShardSpec,
    apply_control,
    capability_window,
    run_shard,
    shard_ids,
)
from experiments.post_training.capability_driven_envs.tests.scripted_model import ScriptedModel
from experiments.post_training.capability_driven_envs.tests.test_proposal_source import (
    CAPABILITY_ID,
    filled_document,
    ideas,  # noqa: F401 (fixture)
    plan,
    slot,
)

SLOTS = 2


def spec(concurrency: int = 4, verification_rule: str = VERIFICATION_RULE) -> ShardSpec:
    return ShardSpec(
        shard=0,
        slots=SLOTS,
        concurrency=concurrency,
        environments=frozenset(Environment),
        max_tokens=1000,
        verification_rule=verification_rule,
        policy_name=STACK_POLICY,
        catalog="catalog.json",
        git_sha="0" * 40,
    )


def run(model, ideas, out, shard_spec=None, capability_ids=(CAPABILITY_ID,)):  # noqa: F811
    return asyncio.run(run_shard(model, ideas, list(capability_ids), out, shard_spec or spec()))


def null_then_proposed_plan() -> str:
    return json.dumps(plan([slot(1, "reasoning", "llm_as_judge", "null"), slot(2, "shellsim", "code")], []))


def write_control(root, **fields) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / CONTROL_FILE).write_text(json.dumps(fields))


def now_iso(age: float = 0.0) -> str:
    return (datetime.now(UTC) - timedelta(seconds=age)).isoformat()


def test_a_shard_writes_each_slot_and_a_provenanced_result_and_a_rerun_skips_it(ideas, tmp_path):  # noqa: F811
    idea = ideas[CAPABILITY_ID]
    model = ScriptedModel()
    model.reply(null_then_proposed_plan())
    model.reply(filled_document(idea, 2))
    out = OutputRoot(str(tmp_path / "run"))

    summary = run(model, ideas, out)

    capability = tmp_path / "run" / CAPABILITY_ID
    result = json.loads((capability / RESULT_FILE).read_text())
    assert set(result["slots"]) == {"1", "2"} and result["requests"] == 2
    assert result["provenance"] == spec().provenance().record(idea)
    assert json.loads((capability / "slot-01.verdict.json").read_text())["reasons"] == ["null proposal: no grader"]
    assert json.loads((capability / "slot-02.proposal.json").read_text())["header"]["id"] == f"{CAPABILITY_ID}/2"
    assert json.loads((capability / "plan.json").read_text())["slots"][1]["status"] == "propose"
    progress = json.loads((tmp_path / "run" / "shards" / "progress-0000.json").read_text())
    assert progress["capabilities_done"] == 1 and progress["requests"] == 2 and progress["halted"] is None
    assert summary.failed == []

    rerun = run(ScriptedModel(), ideas, out)
    assert rerun.ran == 0


def test_a_failed_plan_writes_a_failure_with_its_kind_and_no_result(ideas, tmp_path):  # noqa: F811
    model = ScriptedModel()
    model.reply("not json")
    model.reply("still not json")

    summary = run(model, ideas, OutputRoot(str(tmp_path / "run")))

    assert summary.failed == [CAPABILITY_ID]
    failure = json.loads((tmp_path / "run" / CAPABILITY_ID / FAILURE_FILE).read_text())
    assert failure["kind"] == ["StructuredOutputError"]
    assert not (tmp_path / "run" / CAPABILITY_ID / RESULT_FILE).exists()


@pytest.mark.parametrize(
    "error",
    [
        ModelUnavailable("retries exhausted", ()),
        ModelRequestRejected(401, "bad token", ()),
        RuntimeError("The GLM relay has no registered endpoint"),
    ],
    ids=["unavailable", "rejected", "relay"],
)
def test_a_slot_model_or_relay_error_fails_the_capability_not_the_shard(ideas, tmp_path, error):  # noqa: F811
    """Regression for the probe that crashed run_shard with an ExceptionGroup from one slot's ModelUnavailable."""
    model = ScriptedModel()
    model.reply(null_then_proposed_plan())
    model.fail(error)

    summary = run(model, ideas, OutputRoot(str(tmp_path / "run")))

    capability = tmp_path / "run" / CAPABILITY_ID
    assert summary.failed == [CAPABILITY_ID]
    assert not (capability / RESULT_FILE).exists()
    failure = json.loads((capability / FAILURE_FILE).read_text())
    assert str(error) in failure["error"]
    assert (capability / "plan.json").exists() and (capability / "slot-01.verdict.json").exists()


def test_a_storage_error_on_a_slot_file_fails_the_capability_not_the_shard(ideas, tmp_path):  # noqa: F811
    class FailingSlotWrites(OutputRoot):
        async def write(self, files):
            if any("slot-" in name for name in files):
                raise OSError("connection reset")
            await super().write(files)

    model = ScriptedModel()
    model.reply(null_then_proposed_plan())
    model.reply(filled_document(ideas[CAPABILITY_ID], 2))

    summary = run(model, ideas, FailingSlotWrites(str(tmp_path / "run")))

    assert summary.failed == [CAPABILITY_ID]
    failure = json.loads((tmp_path / "run" / CAPABILITY_ID / FAILURE_FILE).read_text())
    assert failure["kind"] == ["OSError"]


def test_document_repairs_stop_after_the_cap_and_the_slot_failure_is_final(ideas, tmp_path):  # noqa: F811
    uncited = filled_document(ideas[CAPABILITY_ID], 2).replace("(research[0])", "")
    model = ScriptedModel()
    model.reply(null_then_proposed_plan())
    for _ in range(MAX_REPAIRS + 1):
        model.reply(uncited)

    summary = run(model, ideas, OutputRoot(str(tmp_path / "run")))

    assert len(model.requests) == 1 + MAX_REPAIRS + 1
    result = json.loads((tmp_path / "run" / CAPABILITY_ID / RESULT_FILE).read_text())
    assert result["slots"]["2"].endswith(f"no valid document after {MAX_REPAIRS} repairs")
    assert summary.failed == []
    assert summary.progress.repairs["slot"] == MAX_REPAIRS
    assert summary.progress.slot_failures == {"repairs_exhausted": 1}


def test_a_root_holding_results_of_another_prompt_refuses_to_start(ideas, tmp_path):  # noqa: F811
    model = ScriptedModel()
    model.reply(null_then_proposed_plan())
    model.reply(filled_document(ideas[CAPABILITY_ID], 2))
    out = OutputRoot(str(tmp_path / "run"))
    run(model, ideas, out)

    other = ScriptedModel()
    with pytest.raises(ProvenanceMismatch, match=CAPABILITY_ID):
        run(other, ideas, out, spec(verification_rule="Grade everything with code."))
    assert other.requests == []


def test_a_root_written_by_another_driver_commit_with_the_same_prompts_resumes(ideas, tmp_path):  # noqa: F811
    model = ScriptedModel()
    model.reply(null_then_proposed_plan())
    model.reply(filled_document(ideas[CAPABILITY_ID], 2))
    out = OutputRoot(str(tmp_path / "run"))
    run(model, ideas, out)

    other = ScriptedModel()
    rerun = run(other, ideas, out, replace(spec(), git_sha="1" * 40))
    assert rerun.ran == 0 and other.requests == []


def test_the_circuit_breaker_halts_a_shard_whose_capabilities_keep_failing(ideas, tmp_path):  # noqa: F811
    many = {f"c{i:02d}": ideas[CAPABILITY_ID].model_copy(update={"capability_id": f"c{i:02d}"}) for i in range(30)}
    model = ScriptedModel()
    for _ in many:
        model.fail(ModelRequestRejected(401, "expired token", ()))

    summary = run(model, many, OutputRoot(str(tmp_path / "run")), capability_ids=sorted(many))

    assert summary.halted is not None
    assert propose_curriculum.BREAKER_MIN_FAILURES <= len(summary.failed) < len(many)
    progress = json.loads((tmp_path / "run" / "shards" / "progress-0000.json").read_text())
    assert progress["halted"] == summary.halted


def test_the_capability_window_grows_with_the_controlled_limit(tmp_path):
    write_control(tmp_path / "run", concurrency_per_shard=40, written_at=now_iso())

    async def scenario() -> tuple[int, int]:
        limit, window = RequestLimit(2), RequestLimit(capability_window(2, SLOTS))
        await apply_control(OutputRoot(str(tmp_path / "run")), limit, window, SLOTS, ControlState())
        return limit.limit, window.limit

    assert capability_window(2, SLOTS) == 4
    assert asyncio.run(scenario()) == (40, 2 * 20 + 2)


@pytest.mark.parametrize(
    ("control", "message"),
    [
        (None, "no control file, holding 4"),
        ({"concurrency_per_shard": 9, "written_at": "age:400"}, "controller dead, holding 4"),
        ({"concurrency_per_shard": 0, "written_at": "age:0"}, "refused (below 1), holding 4"),
        ({"concurrency_per_shard": 9}, "control file unreadable, holding 4"),
    ],
    ids=["missing", "stale", "zero", "no-timestamp"],
)
def test_a_missing_stale_or_refused_control_file_is_logged_and_the_limit_held(tmp_path, caplog, control, message):
    root = tmp_path / "run"
    root.mkdir()
    if control is not None:
        if "written_at" in control:
            control["written_at"] = now_iso(float(control["written_at"].removeprefix("age:")))
        write_control(root, **control)

    async def scenario() -> int:
        limit, window = RequestLimit(4), RequestLimit(capability_window(4, SLOTS))
        await apply_control(OutputRoot(str(root)), limit, window, SLOTS, ControlState())
        return limit.limit

    with caplog.at_level(logging.INFO):
        assert asyncio.run(scenario()) == 4
    assert message in caplog.text


def test_pause_stops_new_requests_until_the_controller_resumes(ideas, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(propose_curriculum, "CONTROL_POLL", 0.01)
    root = tmp_path / "run"
    write_control(root, pause=True, written_at=now_iso())
    model = ScriptedModel()
    model.reply(null_then_proposed_plan())
    model.reply(filled_document(ideas[CAPABILITY_ID], 2))

    async def scenario():
        shard = asyncio.create_task(run_shard(model, ideas, [CAPABILITY_ID], OutputRoot(str(root)), spec()))
        await asyncio.sleep(0.2)
        paused_requests = len(model.requests)
        write_control(root, concurrency_per_shard=4, written_at=now_iso())
        return paused_requests, await shard

    paused_requests, summary = asyncio.run(scenario())
    assert paused_requests == 0
    assert summary.failed == [] and (root / CAPABILITY_ID / RESULT_FILE).exists()


def test_shards_partition_the_sorted_ids():
    ids = {f"c{i}": None for i in range(7)}
    shards = [shard_ids(ids, s, 3) for s in range(3)]
    assert sorted(itertools.chain.from_iterable(shards)) == sorted(ids)
    assert shards[0] == ["c0", "c3", "c6"]


def test_a_raised_limit_releases_waiting_requests():
    async def scenario() -> list[int]:
        limit = RequestLimit(1)
        seen = []
        release = asyncio.Event()

        async def request() -> None:
            async with limit:
                seen.append(limit.in_flight)
                await release.wait()

        tasks = [asyncio.create_task(request()) for _ in range(3)]
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert limit.in_flight == 1 and limit.waiting == 2
        await limit.set(3)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        release.set()
        await asyncio.gather(*tasks)
        return seen

    assert max(asyncio.run(scenario())) == 3

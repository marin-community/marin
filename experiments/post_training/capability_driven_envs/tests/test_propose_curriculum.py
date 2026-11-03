# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from click.testing import CliRunner

# These tests run in the marin-taskforge package environment; environments without it skip them.
pytest.importorskip("taskforge")

import asyncio
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
    OutputRoot,
    ProvenanceMismatch,
    RequestLimit,
    ShardSpec,
    apply_control,
    capability_window,
    run_shard,
)
from experiments.post_training.capability_driven_envs.tests.scripted_model import ScriptedModel
from experiments.post_training.capability_driven_envs.tests.test_proposer import (
    CAPABILITY_ID,
    filled_document,
    ideas,  # noqa: F401 (fixture)
    plan,
    slot,
)

SLOTS = 2
N25_CATALOG = (
    "s3://marin-us-east-02a/marin/task-curriculum/catalogs/2026.09.20-subject-local-v3-0ce32038771d/curriculum.json"
)
N25_PROVENANCE = {
    "slots": 25,
    "catalog": N25_CATALOG,
    "environments": ["container", "reasoning", "shellsim"],
    "policy": "determinate_first_preferred",
    "system_prompt_sha256": "1dfec5ddc36cc8bcb63767d00f19741dc7fc87c3d7ebc67982c1bed23ee76b6d",
    "verification_rule_sha256": "84259144193ba1104e4370c541c6be3821006f6d84d4179cf0b0ae8c6fab5b75",
    "plan_template_sha256": "e9479eeb4e2d18ef99a08ee3cc9ddafa4b1bce1306451ebdbcb2c187b0fa6156",
    "proposal_template_sha256": "2f6e9224f877e405e3f09795a69c554f0ce662fe45f7670c757449f11a1f48fd",
    "git_sha": "f83978b4e500d599564a9cd43fd04e2086dd3a0d",
    "max_tokens": 131072,
    "max_repairs": 4,
}
"""The provenance every ``result.json`` of the 2026-11-02 n=25 run records, less its capability hash."""


def spec(concurrency: int = 4, **changes) -> ShardSpec:
    base = ShardSpec(
        slots=SLOTS,
        concurrency=concurrency,
        environments=frozenset(Environment),
        max_tokens=1000,
        catalog="catalog.json",
        git_sha="0" * 40,
    )
    return replace(base, **changes)


def run(model, ideas, root, shard_spec=None):  # noqa: F811
    return asyncio.run(run_shard(model, ideas, root, shard_spec or spec()))


def null_then_proposed_plan() -> str:
    return json.dumps(plan([slot(1, "reasoning", "llm_as_judge", "null"), slot(2, "shellsim", "code")], []))


def proposing_model(ideas) -> ScriptedModel:  # noqa: F811
    model = ScriptedModel()
    model.reply(null_then_proposed_plan())
    model.reply(filled_document(ideas[CAPABILITY_ID], 2))
    return model


def write_control(root, **fields) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / CONTROL_FILE).write_text(json.dumps(fields))


def now_iso(age: float = 0.0) -> str:
    return (datetime.now(UTC) - timedelta(seconds=age)).isoformat()


def test_the_default_policy_reproduces_the_n25_runs_provenance():
    n25 = spec(slots=25, max_tokens=131072, catalog=N25_CATALOG, git_sha=N25_PROVENANCE["git_sha"])
    assert n25.provenance() == N25_PROVENANCE


def test_a_shard_writes_each_slot_and_a_provenanced_result_and_a_rerun_skips_it(ideas, tmp_path):  # noqa: F811
    idea = ideas[CAPABILITY_ID]
    root = OutputRoot(str(tmp_path / "run"))

    assert run(proposing_model(ideas), ideas, root) == []

    capability = tmp_path / "run" / CAPABILITY_ID
    result = json.loads((capability / RESULT_FILE).read_text())
    assert set(result["slots"]) == {"1", "2"} and result["requests"] == 2
    assert result["provenance"] == spec().provenance() | {"capability_hash": idea.capability_hash}
    assert json.loads((capability / "slot-01.verdict.json").read_text())["reasons"] == ["null proposal: no grader"]
    assert json.loads((capability / "slot-02.proposal.json").read_text())["header"]["id"] == f"{CAPABILITY_ID}/2"
    assert json.loads((capability / "plan.json").read_text())["slots"][1]["status"] == "propose"
    rerun = ScriptedModel()
    assert run(rerun, ideas, root) == [] and rerun.requests == []


def test_a_successful_rerun_deletes_the_failed_attempts_files(ideas, tmp_path):  # noqa: F811
    capability = tmp_path / "run" / CAPABILITY_ID
    capability.mkdir(parents=True)
    (capability / FAILURE_FILE).write_text("{}")
    (capability / "slot-02.failure.txt").write_text("cut off")

    assert run(proposing_model(ideas), ideas, OutputRoot(str(tmp_path / "run"))) == []

    assert not (capability / FAILURE_FILE).exists() and not (capability / "slot-02.failure.txt").exists()
    assert (capability / "slot-02.proposal.json").exists() and (capability / RESULT_FILE).exists()


def test_a_failed_plan_writes_a_failure_and_no_result(ideas, tmp_path):  # noqa: F811
    model = ScriptedModel()
    model.reply("not json")
    model.reply("still not json")

    assert run(model, ideas, OutputRoot(str(tmp_path / "run"))) == [CAPABILITY_ID]

    failure = json.loads((tmp_path / "run" / CAPABILITY_ID / FAILURE_FILE).read_text())
    assert failure["error"].startswith("StructuredOutputError")
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
    model = ScriptedModel()
    model.reply(null_then_proposed_plan())
    model.fail(error)

    assert run(model, ideas, OutputRoot(str(tmp_path / "run"))) == [CAPABILITY_ID]

    capability = tmp_path / "run" / CAPABILITY_ID
    assert not (capability / RESULT_FILE).exists()
    assert str(error) in json.loads((capability / FAILURE_FILE).read_text())["error"]


def test_a_storage_error_on_a_slot_file_fails_the_capability_not_the_shard(ideas, tmp_path):  # noqa: F811
    class FailingSlotWrites(OutputRoot):
        async def write(self, files):
            if any("slot-02" in name for name in files):
                raise OSError("connection reset")
            await super().write(files)

    assert run(proposing_model(ideas), ideas, FailingSlotWrites(str(tmp_path / "run"))) == [CAPABILITY_ID]

    failure = json.loads((tmp_path / "run" / CAPABILITY_ID / FAILURE_FILE).read_text())
    assert failure["error"] == "OSError('connection reset')"


def test_document_repairs_stop_after_the_cap_and_the_slot_failure_is_final(ideas, tmp_path):  # noqa: F811
    uncited = filled_document(ideas[CAPABILITY_ID], 2).replace("(research[0])", "")
    model = ScriptedModel()
    model.reply(null_then_proposed_plan())
    for _ in range(MAX_REPAIRS + 1):
        model.reply(uncited)

    assert run(model, ideas, OutputRoot(str(tmp_path / "run"))) == []

    assert len(model.requests) == 1 + MAX_REPAIRS + 1
    result = json.loads((tmp_path / "run" / CAPABILITY_ID / RESULT_FILE).read_text())
    assert result["slots"]["2"].endswith(f"no valid document after {MAX_REPAIRS} repairs")


def test_a_root_holding_results_of_another_rule_refuses_and_another_commit_resumes(ideas, tmp_path):  # noqa: F811
    root = OutputRoot(str(tmp_path / "run"))
    run(proposing_model(ideas), ideas, root)

    other = ScriptedModel()
    with pytest.raises(ProvenanceMismatch, match=CAPABILITY_ID):
        run(other, ideas, root, spec(rule="proposer.research_criterion"))
    assert run(other, ideas, root, spec(git_sha="1" * 40)) == [] and other.requests == []


def test_resume_reads_only_the_shards_own_results(ideas, tmp_path):  # noqa: F811
    """Regression: the run listed the whole tree to resume, which outlived the storage timeout once it was complete;
    a result of another shard, even of another provenance, is never read."""
    stray = tmp_path / "run" / "other-capability" / RESULT_FILE
    stray.parent.mkdir(parents=True)
    stray.write_text(json.dumps({"provenance": {"policy": "another"}}))

    assert run(proposing_model(ideas), ideas, OutputRoot(str(tmp_path / "run"))) == []


def apply(root, limit: RequestLimit) -> RequestLimit:
    window = RequestLimit(capability_window(limit.limit, SLOTS))
    asyncio.run(apply_control(OutputRoot(str(root)), limit, window, SLOTS))
    return window


def test_the_control_file_sets_the_limit_and_the_window_follows_it(tmp_path):
    write_control(tmp_path / "run", concurrency_per_shard=40, written_at=now_iso())
    limit = RequestLimit(2)

    window = apply(tmp_path / "run", limit)

    assert (limit.limit, window.limit) == (40, 4 * 20 + 2)


@pytest.mark.parametrize(
    "control",
    [
        None,
        {"concurrency_per_shard": 9, "written_at": now_iso(400)},
        {"concurrency_per_shard": 0, "written_at": now_iso()},
        {"concurrency_per_shard": 9},
        {"concurrency_per_shard": 9, "written_at": datetime.now().isoformat()},
    ],
    ids=["missing", "stale", "zero", "no-timestamp", "zoneless"],
)
def test_a_missing_stale_or_refused_control_file_is_logged_as_an_error_and_the_limit_held(tmp_path, caplog, control):
    root = tmp_path / "run"
    root.mkdir()
    if control is not None:
        write_control(root, **control)
    limit = RequestLimit(4)

    apply(root, limit)

    assert limit.limit == 4
    assert [record.levelno for record in caplog.records] == [logging.ERROR]


def test_pause_stops_new_requests_until_the_controller_resumes(ideas, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(propose_curriculum, "CONTROL_POLL", 0.01)
    root = tmp_path / "run"
    write_control(root, pause=True, written_at=now_iso())
    model = proposing_model(ideas)

    async def scenario():
        shard = asyncio.create_task(run_shard(model, ideas, OutputRoot(str(root)), spec()))
        await asyncio.sleep(0.2)
        paused_requests = len(model.requests)
        write_control(root, concurrency_per_shard=4, written_at=now_iso())
        return paused_requests, await shard

    paused_requests, failed = asyncio.run(scenario())
    assert paused_requests == 0
    assert failed == [] and (root / CAPABILITY_ID / RESULT_FILE).exists()


def test_a_raised_limit_releases_waiting_requests():
    async def scenario() -> list[int]:
        limit, seen, release = RequestLimit(1), [], asyncio.Event()

        async def request() -> None:
            async with limit:
                seen.append(limit.in_flight)
                await release.wait()

        tasks = [asyncio.create_task(request()) for _ in range(3)]
        await asyncio.sleep(0)
        assert limit.in_flight == 1
        await limit.set(3)
        await asyncio.sleep(0)
        release.set()
        await asyncio.gather(*tasks)
        return seen

    assert max(asyncio.run(scenario())) == 3


@pytest.mark.parametrize(("failed", "code"), [([], 0), ([CAPABILITY_ID], 1)])
def test_the_process_exits_without_joining_threads_and_with_the_shards_status(
    ideas,  # noqa: F811
    monkeypatch,
    failed,
    code,
):
    """Regression: the n=25 run's driver called ``os._exit(0)`` before its failure check, so no shard ever exited 1."""
    exits = []
    monkeypatch.setattr(propose_curriculum, "read_catalog", lambda url: ideas)
    monkeypatch.setattr(propose_curriculum, "endpoint_source", lambda *args: None)

    async def ran(*args):
        return failed

    monkeypatch.setattr(propose_curriculum, "_run", ran)
    monkeypatch.setattr(propose_curriculum.os, "_exit", exits.append)
    args = ["--catalog", "c.json", "--output-root", "out", "--shard", "0", "--shards", "1", "--concurrency", "1"]

    CliRunner().invoke(propose_curriculum.main, [*args, "--git-sha", "0" * 40, "--base-url", "http://x/v1"])

    assert exits == [code]

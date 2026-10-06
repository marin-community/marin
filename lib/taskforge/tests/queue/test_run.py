# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio

from taskforge.ledger.jsonl import JsonlLedger
from taskforge.llm.client import GlmUnavailable
from taskforge.loop.events import Terminal
from taskforge.loop.program import LEDGER_DIR
from taskforge.queue.run import FailedItems, item_terminal
from taskforge.triage.verdict import TriageDecision
from taskforge.validate.outcome import Cause

ACCEPT, REJECT = TriageDecision.ACCEPT, TriageDecision.REJECT


async def test_a_run_takes_every_item_to_a_terminal_and_a_relaunch_runs_none_again(
    queue_run, author_replies, fake_glm, fakes
):
    author_replies(2)
    run = queue_run(rubric=fakes.rubric(ACCEPT, decisions={"b/0": REJECT}))
    policy = fakes.policy()

    first = await run({"a": "a", "b": "b", "c": "c"}, policy, width=8)

    assert first.items == {"a--0": Terminal.ACCEPTED, "b--0": Terminal.REJECTED, "c--0": Terminal.ACCEPTED}
    assert first.failed == {}
    assert len(fake_glm.requests) == 2
    calls = (list(run.source.calls), list(run.rubric.assessed), run.model.requests)

    second = await run({"a": "a", "b": "b", "c": "c"}, policy, width=8)

    assert second.items == first.items
    assert (run.source.calls, run.rubric.assessed, run.model.requests) == calls
    assert len(fake_glm.requests) == 2


async def test_width_bounds_the_phases_running_at_once(queue_run, fakes):
    ideas = {f"i{n}": f"i{n}" for n in range(6)}
    policy = fakes.policy()

    narrow = queue_run(rubric=fakes.rubric(REJECT))
    summary = await narrow(ideas, policy, width=2)

    assert set(summary.items.values()) == {Terminal.REJECTED}
    assert narrow.rubric.peak == 2


async def test_the_whole_width_runs_at_once(queue_run, fakes):
    ideas = {f"i{n}": f"i{n}" for n in range(6)}
    wide = queue_run(rubric=fakes.rubric(REJECT, barrier=asyncio.Barrier(6)))

    summary = await asyncio.wait_for(wide(ideas, fakes.policy(), width=6), timeout=30)

    assert len(summary.items) == 6
    assert wide.rubric.peak == 6


async def test_a_failed_item_is_skipped_until_a_launch_retries_it(queue_run, fakes):
    rubric = fakes.rubric(REJECT, failing={"b/0"})
    run = queue_run(rubric=rubric)
    policy = fakes.policy()

    first = await run({"a": "a", "b": "b"}, policy, width=4)

    assert first.items == {"a--0": Terminal.REJECTED, "b--0": Terminal.FAILED}
    assert first.failed == {"b--0": "RuntimeError"}
    assert item_terminal(run.root / LEDGER_DIR, "b--0") is Terminal.FAILED

    rubric.failing.clear()
    skipped = await run({"a": "a", "b": "b"}, policy, width=4, failed=FailedItems.SKIP)

    assert skipped.items["b--0"] is Terminal.FAILED
    assert skipped.failed == {}
    assert rubric.assessed.count("b/0") == 1

    retried = await run({"a": "a", "b": "b"}, policy, width=4, failed=FailedItems.RETRY)

    assert retried.items == {"a--0": Terminal.REJECTED, "b--0": Terminal.REJECTED}
    assert rubric.assessed.count("b/0") == 2


async def test_an_abandoned_item_re_enters_validation_on_the_next_launch_without_rebuilding(
    queue_run, author_replies, fake_glm, fakes
):
    author_replies(1)
    model = fakes.model(unavailable=lambda: GlmUnavailable("router drained", ()))
    run = queue_run(model=model)
    policy = fakes.policy(max_validation_retries=1)

    first = await run({"a": "a"}, policy, width=4)

    assert first.items == {"a--0": Terminal.ABANDONED}
    assert first.ungraded_causes[Cause.MODEL_UNAVAILABLE] > 0
    assert first.failed == {}

    model.unavailable = None
    second = await run({"a": "a"}, policy, width=4)

    assert second.items == {"a--0": Terminal.ACCEPTED}
    assert len(fake_glm.requests) == 1


async def test_a_run_killed_mid_validation_resumes_without_reproposing_or_rebuilding(
    queue_run, author_replies, fake_glm, fakes
):
    author_replies(1)
    model = fakes.model(hang=True)
    run = queue_run(model=model)
    policy = fakes.policy()

    task = asyncio.create_task(run({"a": "a"}, policy, width=4))
    await asyncio.wait_for(model.started.wait(), timeout=30)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert item_terminal(run.root / LEDGER_DIR, "a--0") is None

    model.hang = False
    resumed = await run({"a": "a"}, policy, width=4)

    assert resumed.items == {"a--0": Terminal.ACCEPTED}
    assert run.source.calls == ["a"]
    assert run.rubric.assessed == ["a/0"]
    assert len(fake_glm.requests) == 1


async def test_an_idea_whose_source_fails_is_recorded_and_its_siblings_finish(queue_run, fakes):
    run = queue_run(rubric=fakes.rubric(REJECT), source=fakes.source(failing=frozenset({"bad"})))

    summary = await run({"bad": "bad", "good": "good"}, fakes.policy(), width=4)

    assert summary.items == {"good--0": Terminal.REJECTED}
    assert summary.failed == {"idea--bad": "RuntimeError"}


async def test_an_item_with_an_inconsistent_log_is_recorded_and_its_siblings_finish(queue_run, fakes):
    run = queue_run(rubric=fakes.rubric(REJECT))
    policy = fakes.policy()
    await run({"a": "a", "b": "b"}, policy, width=4)
    log = JsonlLedger(run.root / LEDGER_DIR).path_for("b--0")
    log.write_text(log.read_text() + log.read_text().splitlines(keepends=True)[-1])

    summary = await run({"a": "a", "b": "b"}, policy, width=4)

    assert summary.items == {"a--0": Terminal.REJECTED, "b--0": Terminal.FAILED}
    assert summary.failed == {"b--0": "ValueError"}

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest
from evalrecords.records import JudgeCallBudget
from finestore.store import DataStore
from marin.rollouts.judge_budget import JudgeBudget
from taskcompendium.grading_result import GradeResult, Outcome


@pytest.mark.asyncio
async def test_unsettled_requests_remain_reserved_on_resume(tmp_path):
    limits = JudgeCallBudget(max_calls=6, max_microdollars=7, microdollars_per_call=2)
    with DataStore.open(str(tmp_path / "results")) as store:
        budget = JudgeBudget(store, limits)
        failure = GradeResult(Outcome.INFRA_ERROR, None, diagnostics={"grader_started": False})
        await budget.finish("actor-startup", failure)
        assert not JudgeBudget(store, limits).entries
        await budget.reserve("first", 3)
        second = asyncio.create_task(budget.reserve("second", 3))
        await asyncio.sleep(0)
        assert not second.done()
        await budget.finish(
            "first",
            GradeResult(Outcome.GRADED, 1.0, detail={"judge_receipts": [{"id": "a", "usage": {"cost": 0.0000008}}]}),
        )
        await second
        await budget.finish("second", None)
    with DataStore.open(str(tmp_path / "results")) as store:
        restored = JudgeBudget(store, limits)
        with pytest.raises(RuntimeError, match="exhausted"):
            await restored.reserve("third", 1)
        assert restored.entries["first"] == {"calls": 1, "microdollars": 1, "settled": True}
        assert restored.entries["second"] == {"calls": 3, "microdollars": 6, "settled": False}


@pytest.mark.asyncio
async def test_judge_concurrency_limits_active_graders_with_budget_to_spare(tmp_path):
    limits = JudgeCallBudget(max_calls=30, max_microdollars=100, microdollars_per_call=2)
    with DataStore.open(str(tmp_path / "results")) as store:
        budget = JudgeBudget(store, limits, concurrency=1)
        await budget.reserve("first", 3)
        second = asyncio.create_task(budget.reserve("second", 3))
        await asyncio.sleep(0)
        assert not second.done()
        assert "second" not in budget.entries
        await budget.finish("first", None)
        await second
        assert budget.entries["first"]["microdollars"] == 6

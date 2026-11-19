# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import runpy

import pytest
from taskcompendium.models import format_conversation
from taskcompendium.runtime.resources import resource_bytes

from experiments.post_training.task_curation.datasets.polaris.tasks import sources
from experiments.post_training.task_curation.tests.conversion import converted_task


def test_polaris_keeps_complete_problem_and_private_original_reference():
    problem = "Prove the formula for a tetrahedron with side lengths $a,b,c$."
    task = converted_task(sources()[0].config, {"problem": problem, "answer": "hidden-formula", "difficulty": "5/8"})
    public = format_conversation(task.context.events)
    assert problem in public
    assert "hidden-formula" not in public
    files = {r.path: resource_bytes(r) for r in task.resources.verifier}
    assert json.loads(files["config.json"]) == {"problem": problem, "answer": "hidden-formula"}
    assert not task.resources.oracle and not task.resources.worker


@pytest.mark.parametrize(
    ("candidate", "reference", "expected"),
    [
        (r"\boxed{9375}", "9375", 0.0),
        (r"<think>work</think>\boxed{9375}", "9375", 1.0),
        (r"<think>work</think>9375", "9375", 0.0),
        (r"<think>work</think>\boxed{1/2}", r"\frac{1}{2}", 1.0),
        (r"<think>work</think>\boxed{9376}", "9375", 0.0),
        (r"<think>work</think>wrong</think>\boxed{9375}", "9375", 0.0),
        (r"<think>work</think>\boxed{x+x}", "2*x", 1.0),
        (r"<think>work</think>\boxed{(2,1)}", "(1,2)", 0.0),
    ],
)
def test_polaris_packaged_reward_preserves_native_format_and_math(candidate, reference, expected, tmp_path, monkeypatch):
    # The grader's declared environment supplies pylatexenc. Run these cases with
    # that extra installed; converter-only test environments need not contain it.
    pytest.importorskip("pylatexenc")
    task = converted_task(sources()[0].config, {"problem": "Solve.", "answer": reference, "difficulty": "1/8"})
    for resource in task.resources.verifier:
        path = tmp_path / resource.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(resource_bytes(resource))
    monkeypatch.syspath_prepend(str(tmp_path))
    grader = runpy.run_path(str(tmp_path / "grade.py"))
    assert grader["score"](candidate, json.loads((tmp_path / "config.json").read_text())) == expected

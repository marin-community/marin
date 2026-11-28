# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import runpy
from pathlib import Path

from taskcompendium.models import CommandSemantics, EnvironmentRequirements, Source
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import NormalizedTask, RawRow
from taskcompendium.runtime.resources import resource_bytes

from experiments.post_training.task_curation.datasets.litmus.convert import convert, golden


def test_litmus_scores_the_last_formatted_answer_with_the_rows_tolerance(tmp_path: Path, monkeypatch):
    prompt = "Report the measured value as [[value]]."
    data = {
        "agent_ref": {"name": "litmus_agent_agent"},
        "responses_create_params": {"input": [{"role": "user", "content": prompt}]},
        "question_family": "tier5_admet_direct",
        "question_license": "OpenMDW-1.1",
        "output_regex": r"\[\[([-\d.]+)\]\]",
        "expected_answer": 5.0,
        "answer_type": "float",
        "match": {"rule": "abs_window", "abs_tol": 0.2},
    }
    row = RawRow("test-row", Source(dataset="fixture", revision="pin", row="0", importer_revision="1"), data)
    context = ConversionContext(
        {}, EnvironmentRequirements(command_semantics=CommandSemantics.LINUX_PROCESS, packages_lock="/fixture/lock")
    )
    result = convert(row, context)
    assert isinstance(result, NormalizedTask)
    task = result.task
    assert [event.model_dump(include={"role", "content"}) for event in task.context.events] == [
        {"role": "user", "content": prompt}
    ]
    for resource in task.resources.verifier:
        (tmp_path / resource.path).write_bytes(resource_bytes(resource))
    contract = json.loads((tmp_path / "config.json").read_text())
    assert contract["question_license"] == data["question_license"]
    monkeypatch.syspath_prepend(tmp_path)
    score = runpy.run_path(str(tmp_path / "grade.py"))["score"]
    assert score("[[0]] corrected to [[5.1]]", contract) == 1.0
    assert score("[[5.1]] corrected to [[0]]", contract) == 0.0
    assert score("5.0", contract) == 0.0
    assert score(golden(task).event.content, contract) == 1.0
    assert score("", contract) == 0.0

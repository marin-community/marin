# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""ARC-AGI-2 publication sends ARC Prize's prompt per test input and weights rows by task."""

import io
import json
import tarfile

from taskcompendium.runtime.resources import resource_bytes

from experiments.evaluation.catalog import evaluation_benchmarks
from experiments.evaluation.publication_cases import published_cases

TASKS = {
    "bbbb0002": {
        "train": [{"input": [[1]], "output": [[2]]}],
        "test": [{"input": [[3]], "output": [[4]]}, {"input": [[5, 6]], "output": [[7, 8]]}],
    },
    "aaaa0001": {
        "train": [{"input": [[0, 1]], "output": [[1, 0]]}, {"input": [[2]], "output": [[2]]}],
        "test": [{"input": [[9, 0], [0, 9]], "output": [[0, 9], [9, 0]]}],
    },
}
# Rendered by arc-agi-benchmarking's convert_task_pairs_to_prompt for task aaaa0001.
AAAA0001_PROMPT = (
    "You are participating in a puzzle solving competition. You are an expert at solving puzzles.\n\n"
    "Below is a list of input and output pairs with a pattern. Your goal is to identify the pattern or "
    "transformation in the training examples that maps the input to the output, then apply that pattern to "
    "the test input to give a final output.\n\n"
    "Respond in the format of the training output examples\n\n"
    "--Training Examples--\n"
    "--Example 0-- \n\n INPUT: \n\n[[0, 1]]\n\nOUTPUT: \n\n[[1, 0]]\n\n"
    "--Example 1-- \n\n INPUT: \n\n[[2]]\n\nOUTPUT: \n\n[[2]]\n\n"
    "\n--End of Training Examples--\n\n"
    "--Test Input--\n[[9, 0], [0, 9]]\n--End of Test Input--\n\n"
    "Your response:"
)


def write_archive(file):
    with file.open("wb") as stream, tarfile.open(fileobj=stream, mode="w:gz") as archive:
        members = {
            "ARC-AGI-2-sha/README.md": b"not a task",
            "ARC-AGI-2-sha/data/training/cccc0003.json": json.dumps(TASKS["aaaa0001"]).encode(),
            **{f"ARC-AGI-2-sha/data/evaluation/{name}.json": json.dumps(task).encode() for name, task in TASKS.items()},
        }
        for name, content in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))


def test_publication_renders_each_test_input_with_task_weights(tmp_path, monkeypatch):
    [source] = evaluation_benchmarks()["arc-agi-2"].resolve()
    cases = sorted(published_cases(source, write_archive, tmp_path, monkeypatch, 3), key=lambda c: c.item.source_ordinal)
    assert {case.item.operation.response_selection for case in cases} == {"final_only"}
    configs = [
        json.loads(resource_bytes(next(r for r in case.task.resources.verifier if r.path == "config.json")))
        for case in cases
    ]
    assert configs == [
        {"expected": [[0, 9], [9, 0]], "weight": 1.0},
        {"expected": [[4]], "weight": 0.5},
        {"expected": [[7, 8]], "weight": 0.5},
    ]
    [event] = cases[0].task.context.events
    assert (event.role, event.content) == ("user", AAAA0001_PROMPT)
    [second_test] = cases[2].task.context.events
    assert second_test.content.endswith("--Test Input--\n[[5, 6]]\n--End of Test Input--\n\nYour response:")

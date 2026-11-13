# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish code rows while keeping executable tests in verifier-only resources."""

import json
import subprocess
import sys
import time
from functools import partial

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from shellbox.machine import PackageSpec
from zephyr.context import ZephyrContext

from taskcompendium.evaluation.corpus import evaluation_cases
from taskcompendium.evaluation.spec import (
    ChatGeneration,
    EvaluationProtocol,
    Metrics,
    OperationKind,
    Sampling,
    SummaryKind,
    TextGeneration,
    TrialPolicy,
)
from taskcompendium.importers.evalchemy.assets import code_protocol
from taskcompendium.importers.evalchemy.code import convert_python, python_item, read_crux
from taskcompendium.models import CommandSemantics, EnvironmentRequirements
from taskcompendium.pipeline.evaluation import publish_evaluation_corpus
from taskcompendium.pipeline.inputs import SourceFiles, SourceFormat
from taskcompendium.pipeline.models import IntendedUse, SourceRecipe
from taskcompendium.runtime.resources import resource_bytes

INPUTS = [
    {
        "suite": "humaneval",
        "rows": [
            {
                "task_id": "HumanEval/private",
                "prompt": "def identity(value):\n    ",
                "canonical_solution": "return value\n",
                "test": "def check(candidate):\n    assert candidate(7) == 7\n",
                "entry_point": "identity",
            }
        ],
    },
    {
        "suite": "cruxeval",
        "rows": [{"id": "identity", "code": "def f(x): return x", "input": "7", "output": "7"}],
    },
]


@pytest.mark.parametrize("case", INPUTS, ids=lambda case: case["suite"])
def test_python_publication_keeps_tests_private_and_expands_crux_directions(case, tmp_path):
    suite = case["suite"]
    filename = "test.parquet" if suite == "humaneval" else "test.jsonl"
    root = tmp_path / "source"
    path = root / filename
    path.parent.mkdir(parents=True)
    if path.suffix == ".parquet":
        pq.write_table(pa.Table.from_pylist(case["rows"]), path)
    else:
        path.write_text("".join(json.dumps(row) + "\n" for row in case["rows"]))
    environment = EnvironmentRequirements(
        command_semantics=CommandSemantics.LINUX_PROCESS,
        package_spec=PackageSpec(pypi_packages=("numpy==2.3.5",)),
    )
    recipe = SourceRecipe(
        name="python-publication",
        version="fixture",
        source=SourceFiles(
            dataset="python-fixture",
            revision="fixture",
            patterns=(filename,),
            format=SourceFormat.PARQUET if suite == "humaneval" else SourceFormat.JSONL,
            read=read_crux if suite == "cruxeval" else None,
        ),
        convert=partial(convert_python, suite=suite),
        intended_use=IntendedUse.EVAL,
        rubric=None,
        controls=None,
        grader_environment=environment,
    )
    protocol = EvaluationProtocol(
        benchmark=suite,
        version="fixture",
        operation=OperationKind.TEXT_GENERATION if suite == "humaneval" else OperationKind.CHAT_GENERATION,
        metrics=(Metrics.ACCURACY,),
        primary_metric="accuracy",
        sampling=Sampling(temperature=0, top_p=None, top_k=None, max_tokens=1024, stop=(), chat_template_kwargs={}),
        trials=TrialPolicy(count=1, seed=None),
        summary=SummaryKind.MEAN,
        expected_population=len(case["rows"]) * (2 if suite == "cruxeval" else 1),
    )
    output = str(tmp_path / "published")
    client = LocalClient()
    try:
        with (
            set_current_client(client),
            ZephyrContext(
                client=client,
                max_workers=1,
                chunk_storage_prefix=str(tmp_path / "chunks"),
            ) as context,
        ):
            publish_evaluation_corpus(
                recipe,
                protocol,
                partial(python_item, generation=TextGeneration() if suite == "humaneval" else ChatGeneration()),
                context,
                str(root),
                output,
            )
        published = list(evaluation_cases(output))
    finally:
        client.shutdown()
    assert len(published) == protocol.expected_population
    if case["suite"] == "cruxeval":
        assert [item.item.groups["direction"] for item in published] == ["input", "output"]
        assert len({item.task.id for item in published}) == 2
    for item in published:
        assert not item.task.resources.worker
        config = next(resource for resource in item.task.resources.verifier if resource.path == "config.json")
        private = json.loads(resource_bytes(config))
        assert private["problem"]["code" if case["suite"] == "cruxeval" else "test"]
        if case["suite"] == "humaneval":
            assert private["problem"]["test"] not in item.task.context.events[0].content


@pytest.mark.parametrize(
    ("direction", "response", "expected"),
    [
        (
            "input",
            "Use [ANSWER] and [/ANSWER] tags.\n[ANSWER]assert f(16) == 17[/ANSWER]",
            "f(16)",
        ),
        (
            "input",
            "Use [ANSWER] and [/ANSWER] tags.\n[ANSWER]assert f(16) == 17[/ANSWER]"
            "\nExample: [ANSWER]assert f(1) == 2[/ANSWER]",
            "f(16)",
        ),
        (
            "output",
            "Use [ANSWER] and [/ANSWER] tags.\n[ANSWER]assert f(16) == 17[/ANSWER]",
            "17",
        ),
    ],
)
def test_crux_answer_skips_restated_tag_instruction(direction, response, expected):
    assert code_protocol.crux_answer(response, direction) == expected


CONCAT = "def f(text, value):\n    return text + value"


def _passes(program: str) -> bool:
    """Whether a test program runs without raising, the grading wrapper's pass rule."""
    try:
        exec(compile(program, "<candidate>", "exec"), {})
    except Exception:
        return False
    return True


@pytest.mark.parametrize(
    ("direction", "arguments", "output", "reply", "passed"),
    [
        # A ")" inside the input does not end the call.
        ("output", "'(ac', ')'", "'(ac)'", "[ANSWER]\nassert f('(ac', ')') == '(ac)'\n[/ANSWER]", True),
        ("input", "'(ac', ')'", "'(ac)'", "[ANSWER]\nassert f('(ac', ')') == '(ac)'\n[/ANSWER]", True),
        # An "=" inside the output stays.
        ("output", "'h', '=l'", "'h=l'", "[ANSWER]\nassert f('h', '=l') == 'h=l'\n[/ANSWER]", True),
        # A wrong answer fails.
        ("output", "'a', 'b'", "'ab'", "[ANSWER]\nassert f('a', 'b') == 'ba'\n[/ANSWER]", False),
        ("input", "'a', 'b'", "'ab'", "[ANSWER]\nassert f('b', 'a') == 'ab'\n[/ANSWER]", False),
        # An output answer containing f(<input>), or an input answer without a call, is not an answer.
        ("output", "'a', 'b'", "'ab'", "[ANSWER]\nassert f('a', 'b') == f('a', 'b')\n[/ANSWER]", False),
        ("input", "'a', 'b'", "'ab'", "[ANSWER]\n'ab'\n[/ANSWER]", False),
    ],
)
def test_crux_reads_answers_by_the_harness_rule(direction, arguments, output, reply, passed):
    problem = {"code": CONCAT, "input": arguments, "output": output}

    assert _passes(code_protocol.test_program(problem, reply, "cruxeval", direction)) is passed


def _execute(tmp_path, program: str, timeout: int) -> tuple[float, float]:
    """Run the grader command's script as the grading machine does; return its reward and duration."""
    (tmp_path / "program.py").write_text(code_protocol.execution_program(program, timeout))
    start = time.monotonic()
    subprocess.run([sys.executable, "program.py"], cwd=tmp_path, check=True, timeout=timeout + 30)
    return json.loads((tmp_path / "reward.json").read_text())["reward"], time.monotonic() - start


def test_evalplus_budget_passes_a_test_program_longer_than_three_seconds(tmp_path):
    # The candidate's own workload runs past the source's 3 seconds, as HumanEval/139's reference solution does.
    reward, _ = _execute(tmp_path, "import time\ntime.sleep(3.5)", code_protocol.EXECUTION_TIMEOUT["humanevalplus"])
    assert reward == 1.0


@pytest.mark.parametrize(
    "program",
    [
        "while True:\n    pass",
        # The alarm cannot stop a program that catches it, so the child is killed one second later.
        "while True:\n    try:\n        while True:\n            pass\n    except BaseException:\n        pass",
    ],
)
def test_execution_program_fails_a_nonterminating_program_within_its_budget(tmp_path, program):
    reward, duration = _execute(tmp_path, program, 1)
    assert reward == 0.0
    assert duration < 10

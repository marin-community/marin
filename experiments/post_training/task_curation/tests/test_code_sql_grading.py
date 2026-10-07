# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior fixtures exercise the pinned scorers; these are not source-row witnesses."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest
from taskcompendium.datasets.code_contracts import normalize_apps
from taskcompendium.datasets.gretel_text_to_sql import normalize as normalize_gretel
from taskcompendium.grader import grader_config
from taskcompendium.models import Source, TaskSpec
from taskcompendium.pipeline.models import ImportFailureKind, ImportRejection, RawRow
from taskcompendium.runtime.resources import resource_bytes

from experiments.post_training.task_curation.datasets.skyrl.code_sql.binding import (
    APPS_TESTING_UTIL_SHA256,
    SOURCE_PINS,
    normalize_isolated,
    original_package,
    positive_witnesses,
)


@pytest.fixture(scope="module")
def apps_upstream_source() -> Path:
    source = os.environ.get("APPS_TESTING_UTIL_SOURCE")
    if source is None:
        pytest.skip("APPS integration needs APPS_TESTING_UTIL_SOURCE pointing to pinned testing_util.py")
    path = Path(source)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == APPS_TESTING_UTIL_SHA256
    return path


def run_original(
    tmp_path: Path, evaluator: str, contract: dict, answer: str, *, apps_source: Path | None = None
) -> dict:
    package = original_package({"evaluator": evaluator, "contract": contract})
    for resource in package.resources:
        (tmp_path / resource.path).write_bytes(resource_bytes(resource))
    (tmp_path / "answer.txt").write_text(answer)
    if evaluator == "apps":
        assert apps_source is not None
        subprocess.run(
            [
                "uv",
                "run",
                "--no-project",
                "--python",
                "3.10",
                "--with",
                "pyext==0.7",
                "--with",
                "numpy==1.23.5",
                "python",
                str(tmp_path / "grade_apps.py"),
                str(apps_source),
                str(tmp_path / "config.json"),
                str(tmp_path / "answer.txt"),
                str(tmp_path / "reward.json"),
            ],
            capture_output=True,
            timeout=30,
            check=True,
        )
        return {"status": "scored", "reward": json.loads((tmp_path / "reward.json").read_text())["reward"]}
    image = os.environ.get("SKYRL_CODE_SQL_IMAGE")
    if image is None:
        pytest.skip("Code/SQL source controls need SKYRL_CODE_SQL_IMAGE pinned to the original scorer image")
    completed = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "-v",
            f"{tmp_path}:/tests:ro",
            "-v",
            f"{tmp_path}:/app:ro",
            "-v",
            f"{tmp_path}:/logs/verifier",
            image,
            "python3",
            "/tests/grade_sql.py" if evaluator == "gretel_text_to_sql" else "/tests/grade_lcb.py",
            "/tests/config.json",
            "/app/answer.txt",
            "/logs/verifier/score.json",
        ],
        capture_output=True,
        timeout=360,
        check=False,
    )
    if completed.returncode != 0:
        return {"status": "invalid_task", "reward": 0.0}
    score = json.loads((tmp_path / "score.json").read_text())
    return {"status": "scored", "reward": score["reward"], "detail": score["detail"]}


@pytest.mark.parametrize(
    "tests,answer,reward",
    [
        ({"inputs": [""], "outputs": ["3.0\n"]}, "```python\nprint(' 3.000 ')\n```", 1.0),
        ({"inputs": [""], "outputs": ["3\n3\n"]}, "```python\nprint(3)\n```", 1.0),
        ({"inputs": [""], "outputs": ["3\n"]}, "print(3)", 0.0),
        ({"inputs": [""], "outputs": ["3\n"]}, "```python\nwhile True:\n    pass\n```", 0.0),
        (
            {"inputs": [""], "outputs": ["3\n"]},
            "```python\nprint(0)\n```\nThen:\n```python\nprint(3)\n```",
            1.0,
        ),
        (
            {"inputs": ["1\n", "2\n"], "outputs": ["1\n", "999\n"]},
            "```python\nprint(input())\n```",
            0.0,
        ),
        (
            {"fn_name": "solve", "inputs": [[1, 2]], "outputs": [[1, 2]]},
            "```python\nclass Solution:\n    def solve(self, a, b):\n        return (a, b)\n```",
            1.0,
        ),
    ],
)
@pytest.mark.integration
def test_original_apps_dispatch_extraction_comparison_and_binary_scoring(
    tmp_path, apps_upstream_source, tests, answer, reward
):
    verdict = run_original(
        tmp_path, "apps", {"input_output": json.dumps(tests)}, answer, apps_source=apps_upstream_source
    )
    assert (verdict["status"], verdict["reward"]) == ("scored", reward)


@pytest.mark.parametrize("evaluator", ["eurus2_code", "verifiable_code"])
@pytest.mark.integration
def test_original_code_sources_execute_their_native_fixtures(tmp_path, evaluator):
    tests = {"inputs": ["2 3\n"], "outputs": ["5\n"]}
    contract = (
        {"reward_model": {"ground_truth": json.dumps(tests)}}
        if evaluator == "eurus2_code"
        else {
            "verification_info": {
                "language": "python",
                "test_cases": [{"input": "2 3\n", "output": "5\n", "type": "stdin_stdout", "fn_name": None}],
            }
        }
    )
    verdict = run_original(tmp_path, evaluator, contract, "```python\nprint(sum(map(int, input().split())))\n```")
    assert (verdict["status"], verdict["reward"]) == ("scored", 1.0)


@pytest.mark.parametrize(
    "tests",
    [
        {"inputs": [""], "outputs": [["a", "b"]]},
        {"inputs": ["not json"], "outputs": ["1"], "fn_name": "solve"},
    ],
)
@pytest.mark.integration
def test_original_apps_empty_response_scores_zero(tmp_path, apps_upstream_source, tests):
    verdict = run_original(tmp_path, "apps", {"input_output": json.dumps(tests)}, "", apps_source=apps_upstream_source)
    assert (verdict["status"], verdict["reward"]) == ("scored", 0.0)


@pytest.mark.parametrize(
    "evaluator,contract",
    [
        ("eurus2_code", {"reward_model": {"ground_truth": "[]"}}),
        ("verifiable_code", {"verification_info": {"language": "python", "test_cases": []}}),
    ],
)
@pytest.mark.integration
def test_original_empty_code_suite_cannot_receive_vacuous_passing_reward(tmp_path, evaluator, contract):
    verdict = run_original(tmp_path, evaluator, contract, "```python\nprint(3)\n```")
    assert (verdict["status"], verdict["reward"]) == ("invalid_task", 0.0)


@pytest.fixture
def sql_contract():
    return {
        "sql_context": "CREATE TABLE t (v INTEGER); INSERT INTO t VALUES (1), (1), (2), (3);",
        "sql": "SELECT v FROM t",
    }


@pytest.mark.parametrize(
    "reference,answer,reward",
    [
        ("SELECT v FROM t", "<solution>SELECT v AS renamed FROM t ORDER BY v DESC</solution>", 1.0),
        ("SELECT v FROM t ORDER BY v", "```sql\nSELECT v FROM t ORDER BY v DESC\n```", 0.0),
        ("SELECT v FROM t", "SELECT DISTINCT v FROM t", 0.0),
        ("SELECT COUNT(*) FROM t", "SELECT 4", 0.0),
        ("SELECT v FROM t", "DROP TABLE t", 0.0),
    ],
)
@pytest.mark.integration
def test_original_sql_order_multiplicity_and_perturbed_fixtures(tmp_path, sql_contract, reference, answer, reward):
    sql_contract["sql"] = reference
    verdict = run_original(tmp_path, "gretel_text_to_sql", sql_contract, answer)
    assert (verdict["status"], verdict["reward"]) == ("scored", reward)


@pytest.mark.integration
def test_original_sql_broken_reference_fails_even_for_empty_candidate(tmp_path, sql_contract):
    sql_contract["sql"] = "SELECT missing_column FROM t"
    verdict = run_original(tmp_path, "gretel_text_to_sql", sql_contract, "")
    assert verdict["status"] == "invalid_task"


@pytest.mark.parametrize("seeded", [True, False])
def test_gretel_preparation_binds_seeded_tasks_and_rejects_schema_only_as_unsupported(sql_contract, seeded):
    dataset, revision = SOURCE_PINS["gretel_text_to_sql"]
    context = sql_contract["sql_context"] if seeded else "CREATE TABLE t (v INTEGER);"
    row = RawRow(
        id="authored-sql-preparation-fixture",
        source=Source(dataset=dataset, revision=revision, row="fixture", importer_revision="test"),
        data={
            "sql_prompt": "List every value in t, including duplicates.",
            "sql_context": context,
            "sql": sql_contract["sql"],
            "sql_explanation": "Return each stored value.",
            "sql_complexity": "basic",
            "sql_task_type": "select",
            "id": "fixture",
        },
    )
    result = normalize_isolated(row, normalize_task=normalize_gretel, image="authored-sql-fixture@sha256:" + "0" * 64)
    if not seeded:
        assert isinstance(result, ImportRejection)
        assert (result.kind, result.reason) == (
            ImportFailureKind.UNSUPPORTED,
            "original_gretel_preparation_unsupported",
        )
        return
    assert isinstance(result, TaskSpec)
    config = grader_config(result)
    assert result.verifier.kind == "native_command"
    assert config["contract"]["sql"] == "SELECT v FROM t"


def test_apps_alternative_stdout_outputs_bind_to_original_evaluator():
    dataset, revision = SOURCE_PINS["apps"]
    row = RawRow(
        id="authored-contract-fixture",
        source=Source(dataset=dataset, revision=revision, row="fixture", importer_revision="test"),
        data={
            "question": "Print one of the accepted alternatives.",
            "input_output": json.dumps({"inputs": [""], "outputs": [["a", "b"]]}),
            "solutions": "[]",
            "problem_id": 1,
        },
    )
    result = normalize_isolated(row, normalize_task=normalize_apps, image="authored-native@sha256:" + "0" * 64)
    assert isinstance(result, TaskSpec)
    assert result.verifier.kind == "native_command"


@pytest.mark.parametrize("input_output", ['{"inputs": [], "outputs": []}', "not JSON"])
def test_apps_invalid_source_cases_rejected_before_native_binding(input_output):
    dataset, revision = SOURCE_PINS["apps"]
    row = RawRow(
        id="invalid-apps-cases",
        source=Source(dataset=dataset, revision=revision, row="fixture", importer_revision="test"),
        data={"question": "Return one.", "input_output": input_output},
    )
    result = normalize_isolated(row, normalize_task=normalize_apps, image="authored-native@sha256:" + "0" * 64)
    assert isinstance(result, ImportRejection)
    assert result.reason == "invalid_test_contract"


@pytest.mark.parametrize("solution", ["print(3)", "```python\nprint(3)\n```"])
@pytest.mark.integration
def test_source_solution_witness_preserves_original_code_extraction(tmp_path, apps_upstream_source, solution):
    contract = {
        "input_output": json.dumps({"inputs": [""], "outputs": ["3\n"]}),
        "solutions": json.dumps(["", solution]),
    }
    witnesses = positive_witnesses({"evaluator": "apps", "contract": contract})
    verdict = run_original(tmp_path, "apps", contract, witnesses[0].answer, apps_source=apps_upstream_source)
    assert (verdict["status"], verdict["reward"]) == ("scored", 1.0)


@pytest.mark.integration
def test_original_lcb_functional_comparison_rejects_scalar_for_list_output(tmp_path):
    tests = {"fn_name": "solve", "inputs": [[1]], "outputs": [[6]]}
    verdict = run_original(
        tmp_path,
        "eurus2_code",
        {"reward_model": {"ground_truth": json.dumps(tests)}},
        "```python\ndef solve(x): return 6\n```",
    )
    assert (verdict["status"], verdict["reward"]) == ("scored", 0.0)


@pytest.mark.integration
def test_original_apps_functional_comparison_accepts_first_output(tmp_path, apps_upstream_source):
    tests = {"fn_name": "solve", "inputs": [[1]], "outputs": [[6]]}
    verdict = run_original(
        tmp_path,
        "apps",
        {"input_output": json.dumps(tests)},
        "```python\ndef solve(x): return 6\n```",
        apps_source=apps_upstream_source,
    )
    assert (verdict["status"], verdict["reward"]) == ("scored", 1.0)

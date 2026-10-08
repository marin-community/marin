# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The SkyRL grader scripts turn their source scorers' verdicts into binary rewards.

The scorers live only in the grader image, so each test stands in a small scorer with the same
interface and runs the packaged script as the grader machine does.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.post_training.task_curation.datasets.skyrl import code

APPS_TESTING_UTIL = """
def run_test(*, problem, test):
    assert problem["input_output"]["inputs"] == ["1 2\\n"]
    return [1, 1] if "print(a + b)" in test else [1, -1]
"""

LCB_MODULE = """
FUNCTIONAL_TEST_TYPE = "functional"


class TestExecutionMode:
    stop_on_failure = "stop_on_failure"


def normalize_lcb_ground_truth(value):
    return value


def extract_code_from_model(answer):
    start = answer.find("```python\\n")
    return answer[start + 10 : answer.rindex("```")] if start >= 0 else None


def lcb_execution_result(tests, code, execution_mode):
    assert execution_mode == TestExecutionMode.stop_on_failure
    return [test["output"] in code for test in tests], {"cases": len(tests)}
"""

SQL_MODULE = """
import enum
import sqlite3


class GradeOutcome(enum.Enum):
    INFRA = "infra"
    MATCH = "match"
    MISMATCH = "mismatch"


def split_statements(text):
    return [part.strip() for part in text.split(";") if part.strip()]


def classify_statement(text):
    lowered = text.lower()
    if lowered.startswith("create table"):
        return "create_table"
    return "insert" if lowered.startswith("insert") else "select"


def create_table_is_schema_qualified(text):
    return False


def create_table_name(text):
    return text.split()[2]


def is_nondeterministic(text):
    return False


def has_top_level_order_by(text):
    return "order by" in text.lower()


def extract_sql(text):
    return text.removeprefix("<solution>").removesuffix("</solution>")


def grade(spec, query):
    database = sqlite3.connect(":memory:")
    database.executescript(spec["schema_sql"] + spec["insert_sql"])
    try:
        expected = sorted(database.execute(spec["reference_sql"]).fetchall())
    except sqlite3.Error as error:
        return GradeOutcome.INFRA, str(error)
    try:
        actual = sorted(database.execute(query).fetchall())
    except sqlite3.Error as error:
        return GradeOutcome.MISMATCH, str(error)
    return (GradeOutcome.MATCH if actual == expected else GradeOutcome.MISMATCH), {"rows": len(actual)}
"""


def run_script(tmp_path: Path, script: str, scorer: Path, config: dict, answer: str) -> subprocess.CompletedProcess:
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "answer.txt").write_text(answer)
    return subprocess.run(
        [
            sys.executable,
            str(Path(code.__file__).with_name(script)),
            str(scorer),
            str(tmp_path / "config.json"),
            str(tmp_path / "answer.txt"),
            str(tmp_path / "score.json"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def reward(tmp_path: Path, completed: subprocess.CompletedProcess) -> float:
    assert completed.returncode == 0, completed.stderr
    return json.loads((tmp_path / "score.json").read_text())["reward"]


def skyrl_gym_root(tmp_path: Path, module: str, source: str) -> Path:
    """A SkyRL checkout holding only ``module``, laid out as the image installs it."""
    root = tmp_path / "skyrl_gym_root"
    path = root / (module.replace(".", "/") + ".py")
    path.parent.mkdir(parents=True)
    for package in path.relative_to(root).parents:
        if package != Path("."):
            (root / package / "__init__.py").write_text("")
    path.write_text(source)
    return root


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("```python\na, b = map(int, input().split())\nprint(a + b)\n```", 1.0),
        ("```python\nprint(a - b)\n```", 0.0),
        ("print(a + b)", 0.0),
    ],
)
def test_apps_grade_requires_every_case_to_pass(tmp_path, answer, expected):
    scorer = tmp_path / "testing_util.py"
    scorer.write_text(APPS_TESTING_UTIL)
    config = {
        "apps_source_sha256": hashlib.sha256(scorer.read_bytes()).hexdigest(),
        "input_output": json.dumps({"inputs": ["1 2\n"], "outputs": ["3\n"]}),
    }
    assert reward(tmp_path, run_script(tmp_path, code.APPS_GRADE, scorer, config, answer)) == expected


def test_apps_grade_refuses_a_changed_evaluator(tmp_path):
    scorer = tmp_path / "testing_util.py"
    scorer.write_text(APPS_TESTING_UTIL)
    config = {"apps_source_sha256": "0" * 64, "input_output": json.dumps({"inputs": ["1 2\n"], "outputs": ["3\n"]})}
    completed = run_script(tmp_path, code.APPS_GRADE, scorer, config, "```python\nprint(a + b)\n```")
    assert completed.returncode != 0
    assert not (tmp_path / "score.json").exists()


@pytest.mark.parametrize(
    ("answer", "expected"),
    [("```python\nprint(3)  # 3\n```", 1.0), ("```python\nprint(4)\n```", 0.0), ("print(3)", 0.0)],
)
def test_lcb_grade_requires_every_case_to_pass(tmp_path, answer, expected):
    root = skyrl_gym_root(tmp_path, "skyrl_gym.envs.lcb.livecodebench", LCB_MODULE)
    config = {"test_cases": json.dumps([{"testtype": "stdin", "input": "", "output": "3"}])}
    assert reward(tmp_path, run_script(tmp_path, code.LCB_GRADE, root, config, answer)) == expected


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("<solution>SELECT v AS renamed FROM t</solution>", 1.0),
        ("<solution>SELECT DISTINCT v FROM t</solution>", 0.0),
        (code.FAILING_SQL, 0.0),
    ],
)
def test_sql_grade_compares_results_on_the_seeded_database(tmp_path, answer, expected):
    root = skyrl_gym_root(tmp_path, "skyrl_gym.envs.text_to_sql.scoring", SQL_MODULE)
    config = {
        "reference_sql": "SELECT v FROM t",
        "context_sql": "CREATE TABLE t (v INTEGER); INSERT INTO t VALUES (1), (1), (2);",
    }
    assert reward(tmp_path, run_script(tmp_path, code.SQL_GRADE, root, config, answer)) == expected


def test_sql_grade_fails_on_a_reference_that_does_not_run(tmp_path):
    root = skyrl_gym_root(tmp_path, "skyrl_gym.envs.text_to_sql.scoring", SQL_MODULE)
    config = {
        "reference_sql": "SELECT missing FROM t",
        "context_sql": "CREATE TABLE t (v INTEGER); INSERT INTO t VALUES (1);",
    }
    completed = run_script(tmp_path, code.SQL_GRADE, root, config, "<solution>SELECT v FROM t</solution>")
    assert completed.returncode != 0

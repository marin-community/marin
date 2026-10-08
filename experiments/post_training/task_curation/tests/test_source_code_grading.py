# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The code and SQL grader scripts reduce original evaluator case results without replacing them."""

import hashlib
import json
import sys

from experiments.post_training.task_curation.datasets.skyrl.code_sql import grade_apps, grade_lcb, grade_sql


def test_lcb_mode_uses_original_case_results_for_binary_reward(tmp_path, monkeypatch):
    source = tmp_path / "skyrl_gym/envs/lcb/livecodebench.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "FUNCTIONAL_TEST_TYPE = 'functional'\n"
        "class TestExecutionMode:\n"
        "    stop_on_failure = 'stop_on_failure'\n"
        "def normalize_lcb_ground_truth(value):\n"
        "    return value\n"
        "def extract_code_from_model(answer):\n"
        "    return answer if answer.startswith('```python') else None\n"
        "def lcb_execution_result(tests, code, execution_mode):\n"
        "    assert execution_mode == TestExecutionMode.stop_on_failure\n"
        "    return [code.endswith(test['expected']) for test in tests], {'cases': len(tests)}\n"
    )
    monkeypatch.setattr(grade_lcb, "SOURCE_ROOT", tmp_path)
    monkeypatch.setattr(grade_lcb, "LCB_PATH", source)
    config = {"test_cases": json.dumps([{"testtype": "stdio", "expected": "pass"}])}

    assert grade_lcb.code_score(config, "```python\npass") == {"reward": 1.0, "detail": {"cases": 1}}
    assert grade_lcb.code_score(config, "```python\nfail") == {"reward": 0.0, "detail": {"cases": 1}}


def test_sql_mode_uses_original_comparison_outcome(tmp_path, monkeypatch):
    source = tmp_path / "skyrl_gym/envs/text_to_sql/scoring.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "from enum import Enum\n"
        "class GradeOutcome(Enum):\n"
        "    INFRA = 'infra'\n"
        "    MATCH = 'match'\n"
        "    MISMATCH = 'mismatch'\n"
        "def split_statements(text):\n"
        "    return [part.strip() for part in text.split(';') if part.strip()]\n"
        "def classify_statement(text):\n"
        "    if text.lower().startswith('create table'): return 'create_table'\n"
        "    if text.lower().startswith('insert'): return 'insert'\n"
        "    return 'select'\n"
        "def create_table_is_schema_qualified(text): return False\n"
        "def create_table_name(text): return 't'\n"
        "def is_nondeterministic(text): return False\n"
        "def has_top_level_order_by(text): return False\n"
        "def extract_sql(text): return text\n"
        "def grade(spec, query):\n"
        "    outcome = GradeOutcome.MATCH if query == spec['reference_sql'] else GradeOutcome.MISMATCH\n"
        "    return outcome, {'query': query}\n"
    )
    monkeypatch.setattr(grade_sql, "SOURCE_ROOT", tmp_path)
    monkeypatch.setattr(grade_sql, "SQL_PATH", source)
    config = {"reference_sql": "SELECT v FROM t", "context_sql": "CREATE TABLE t(v); INSERT INTO t VALUES (1);"}

    assert grade_sql.sql_score(config, "SELECT v FROM t") == {
        "reward": 1.0,
        "detail": {"comparison": {"query": "SELECT v FROM t"}},
    }
    assert grade_sql.sql_score(config, "SELECT 2") == {
        "reward": 0.0,
        "detail": {"comparison": {"query": "SELECT 2"}},
    }


def test_apps_mode_uses_original_case_results(tmp_path, monkeypatch):
    source = tmp_path / "testing_util.py"
    source.write_text(
        "def run_test(*, problem, test):\n"
        "    assert problem['input_output']['inputs'] == ['x']\n"
        "    return [1, 1] if test == 'pass' else [1, 0]\n"
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "apps_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "input_output": json.dumps({"inputs": ["x"]}),
            }
        )
    )
    answer = tmp_path / "answer.txt"
    result = tmp_path / "reward.json"
    monkeypatch.setattr(sys, "argv", ["grade_apps.py", str(source), str(config), str(answer), str(result)])

    answer.write_text("```python\npass\n```")
    grade_apps.main()
    assert json.loads(result.read_text()) == {"reward": 1.0}

    answer.write_text("```python\nfail\n```")
    grade_apps.main()
    assert json.loads(result.read_text()) == {"reward": 0.0}

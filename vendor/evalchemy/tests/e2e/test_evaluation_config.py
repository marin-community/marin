"""Public contract tests for the portable Evalchemy evaluation configuration."""

import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from evalchemy_config import (
    EvaluationConfig,
    TaskOptions,
    apply_evaluation_patch,
    canonical_json,
    fingerprint,
    load_evaluation_config,
    materialize_eval_args,
)

_ROOT = Path(__file__).parents[2]


def _validate_config(path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "eval.serve_eval.cli", "validate-config", str(path)],
        cwd=_ROOT,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("task", ["gsm8k", "IFEval", "LABBenchProtocolQA", "LABBenchSeqQA", "LABBenchCloning"])
def test_validate_config_accepts_portable_yaml_without_starting_an_evaluation(tmp_path, task):
    config = tmp_path / "evaluation.yaml"
    config.write_text(f"tasks:\n  - {task}\nmax_tokens: 512\n")

    result = _validate_config(config)

    assert result.returncode == 0, result.stderr


def test_validate_config_fails_when_catalog_dependency_is_broken(tmp_path):
    config = tmp_path / "evaluation.yaml"
    config.write_text("tasks:\n  - gsm8k\n")
    broken_package = tmp_path / "lm_eval"
    broken_package.mkdir()
    (broken_package / "__init__.py").write_text("")
    (broken_package / "tasks.py").write_text("import evalchemy_missing_catalog_dependency\n")
    script = (
        "import runpy, sys; "
        f"sys.path.insert(0, {str(tmp_path)!r}); "
        "runpy.run_module('eval.serve_eval.cli', run_name='__main__')"
    )

    result = subprocess.run(
        [sys.executable, "-c", script, "validate-config", str(config)],
        cwd=_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "ModuleNotFoundError: No module named 'evalchemy_missing_catalog_dependency'" in result.stderr


@pytest.mark.parametrize(
    "contents, expected",
    [
        ("tasks:\n  - gsm8kk\n", "gsm8k"),
        ("tasks:\n  - gsm8k\nunsupported_option: true\n", "unsupported_option"),
    ],
)
def test_validate_config_rejects_invalid_portable_yaml(tmp_path, contents, expected):
    config = tmp_path / "evaluation.yaml"
    config.write_text(contents)

    result = _validate_config(config)

    assert result.returncode != 0
    assert expected in result.stderr


def test_evaluation_config_rejects_conflicting_legacy_output_alias():
    with pytest.raises(ValidationError, match="conflicting max_tokens"):
        EvaluationConfig.model_validate({"max_tokens": 512, "gen_kwargs": "max_gen_toks=1024"})


def test_evaluation_config_patch_revalidates_canonical_limits():
    config = load_evaluation_config({"tasks": ["gsm8k"], "max_tokens": 512})
    updated = apply_evaluation_patch(config, {"max_tokens": 768})

    assert updated.max_tokens == 768
    assert "max_gen_toks=768" in (updated.gen_kwargs or "")
    assert materialize_eval_args(updated)[-2:] == ["--max_tokens", "768"]


def test_evaluation_config_carries_task_level_client_routing():
    config = EvaluationConfig.model_validate(
        {
            "tasks": ["humaneval", "gsm8k"],
            "task_options": {
                "humaneval": {
                    "num_fewshot": 0,
                    "task_alias": "humaneval_0shot",
                    "generation": True,
                    "unsafe_code": True,
                    "completion_only": True,
                }
            },
        }
    )

    assert config.task_options["humaneval"] == TaskOptions(
        num_fewshot=0,
        task_alias="humaneval_0shot",
        generation=True,
        unsafe_code=True,
        completion_only=True,
    )
    with pytest.raises(ValidationError, match="task_options refer"):
        EvaluationConfig.model_validate({"tasks": ["gsm8k"], "task_options": {"aime24": {}}})


def test_canonical_json_and_schema_fingerprint_are_stable():
    config = EvaluationConfig.model_validate({"tasks": ["gsm8k", "math500"], "max_tokens": 512})
    same_config = EvaluationConfig.model_validate({"max_tokens": 512, "tasks": ["gsm8k", "math500"]})

    assert canonical_json(config) == canonical_json(same_config)
    assert len(fingerprint()) == 64


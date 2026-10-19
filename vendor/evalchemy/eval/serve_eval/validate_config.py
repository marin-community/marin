"""Validate portable evaluation YAML without constructing a model or task."""

from __future__ import annotations

import argparse
import difflib
from pathlib import Path
from typing import Sequence

from pydantic import ValidationError

from evalchemy_config import EvaluationConfig, load_evaluation_config

_CHAT_BENCHMARKS_DIR = Path(__file__).parents[1] / "chat_benchmarks"
_LM_EVAL_TASKS_DIR = Path(__file__).parents[1] / "lm_eval_tasks"


def _available_tasks() -> set[str]:
    """Return task names registered by the installed Evalchemy and lm-eval catalogs."""
    chat_tasks = {
        path.name for path in _CHAT_BENCHMARKS_DIR.iterdir() if (path / "eval_instruct.py").is_file()
    }
    try:
        from lm_eval.tasks import TaskManager
    except ModuleNotFoundError:
        lm_eval_tasks = {
            path.parent.name if path.name == "default.yaml" else path.stem
            for path in _LM_EVAL_TASKS_DIR.rglob("*.yaml")
        }
    else:
        lm_eval_tasks = set(TaskManager("ERROR", include_path=[str(_LM_EVAL_TASKS_DIR)]).all_tasks)
    return chat_tasks | lm_eval_tasks


def validate_config(path: Path) -> EvaluationConfig:
    """Load a portable config and reject task names absent from the installed catalog."""
    config = load_evaluation_config(path)
    available_tasks = _available_tasks()
    unknown_tasks = [task for task in config.tasks if task not in available_tasks]
    if unknown_tasks:
        suggestions = {
            task: difflib.get_close_matches(task, available_tasks, n=3) for task in unknown_tasks
        }
        suggestion_text = "; ".join(
            f"{task} (did you mean: {', '.join(matches)})" for task, matches in suggestions.items() if matches
        )
        message = f"Unknown evaluation tasks: {', '.join(unknown_tasks)}."
        if suggestion_text:
            message = f"{message} Close matches: {suggestion_text}."
        raise ValueError(message)
    return config


def main(argv: Sequence[str] | None = None) -> None:
    """Validate one portable evaluation YAML file."""
    parser = argparse.ArgumentParser(description="Validate a portable Evalchemy evaluation config.")
    parser.add_argument("config", type=Path, help="Portable evaluation YAML file")
    args = parser.parse_args(argv)
    try:
        config = validate_config(args.config)
    except (OSError, ValidationError, ValueError) as error:
        parser.error(str(error))
    print(f"valid evaluation config: {args.config} ({len(config.tasks)} task(s))")

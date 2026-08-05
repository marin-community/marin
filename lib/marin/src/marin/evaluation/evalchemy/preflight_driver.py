# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned-environment side of Evalchemy configuration preflight."""

import importlib
import json
import re
import sys
from importlib import metadata
from pathlib import Path

_EXTRA_NORMALIZER = re.compile(r"[-_.]+")


def preflight(request_path: Path) -> None:
    """Validate requested files and print their normalized evaluator metadata."""
    evalchemy_config = importlib.import_module("evalchemy_config")
    eval_package = importlib.import_module("eval")
    task_manager_module = importlib.import_module("lm_eval.tasks")

    requested_paths = json.loads(request_path.read_text())
    if not isinstance(requested_paths, list) or not all(isinstance(path, str) for path in requested_paths):
        raise ValueError("Evalchemy preflight request must be a list of config paths")

    native_tasks = task_manager_module.TaskManager()
    custom_root = Path(eval_package.__file__).parent / "chat_benchmarks"
    available_extras = set(metadata.metadata("evalchemy").get_all("Provides-Extra") or ())
    response: list[dict[str, object]] = []
    for raw_path in requested_paths:
        path = Path(raw_path)
        config = evalchemy_config.load_evaluation_config(path)
        runtime_extras: list[str] = []
        unknown_tasks: list[str] = []
        for task in config.tasks:
            is_custom = (custom_root / task / "eval_instruct.py").is_file()
            is_native = bool(native_tasks.match_tasks([task]))
            if is_custom and is_native:
                raise ValueError(f"task {task!r} is ambiguous between Evalchemy and lm-eval catalogs")
            if not is_custom and not is_native:
                unknown_tasks.append(task)
            extra = _EXTRA_NORMALIZER.sub("-", task).lower()
            if extra in available_extras and extra not in runtime_extras:
                runtime_extras.append(extra)
        if unknown_tasks:
            raise ValueError(f"tasks are not recognized by pinned Evalchemy: {unknown_tasks}")
        response.append(
            {
                "config": config.model_dump(mode="json", exclude_none=True),
                "runtime_extras": runtime_extras,
            }
        )
    print(json.dumps(response, separators=(",", ":")))


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: preflight_driver.py REQUESTS.json")
    preflight(Path(sys.argv[1]))


if __name__ == "__main__":
    main()

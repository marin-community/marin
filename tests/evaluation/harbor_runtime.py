# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execute Harbor policy checks in its independent locked runtime."""

import json
import os
import subprocess
from pathlib import Path

from marin.evaluation.harbor.driver_config import _driver_environment

_ROOT = Path(__file__).parents[2]
_DRIVER = _ROOT / "lib/marin/src/marin/evaluation/harbor/trial_driver.py"
_RUNTIME_PROJECT = _ROOT / "vendor/harbor"


def external_python(
    *args: str,
    hash_seed: str = "0",
    check: bool = True,
    extra_python_path: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    environment = _driver_environment()
    environment["PYTHONHASHSEED"] = hash_seed
    if extra_python_path is not None:
        environment["PYTHONPATH"] = os.pathsep.join((str(extra_python_path), environment["PYTHONPATH"]))
    completed = subprocess.run(
        [
            "uv",
            "run",
            "--isolated",
            "--project",
            str(_RUNTIME_PROJECT),
            "--extra",
            "runtime",
            "--no-default-groups",
            "--frozen",
            "python",
            *args,
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    if check:
        completed.check_returncode()
    return completed


def preflight(
    tmp_path: Path,
    requests: list[tuple[Path, dict[str, object]]],
    *,
    hash_seed: str = "0",
    check: bool = True,
    extra_python_path: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    request_path = tmp_path / f"requests-{hash_seed}.json"
    request_path.write_text(
        json.dumps(
            [{"path": str(path), "model_agent_kwargs": kwargs} for path, kwargs in requests],
            separators=(",", ":"),
        )
    )
    return external_python(
        str(_DRIVER),
        "preflight",
        str(request_path),
        hash_seed=hash_seed,
        check=check,
        extra_python_path=extra_python_path,
    )

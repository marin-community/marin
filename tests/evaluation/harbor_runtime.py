# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise Harbor's driver protocol through its isolated, locked environment."""

import json
import subprocess
from pathlib import Path

from marin.evaluation.harbor.driver_config import _driver_environment, harbor_runtime_command

_DRIVER = Path(__file__).parents[2] / "lib/marin/src/marin/evaluation/harbor/trial_driver.py"


def preflight(
    tmp_path: Path,
    requests: list[tuple[Path, dict[str, object]]],
    *,
    hash_seed: str = "0",
    check: bool = True,
    judge_model: str | None = None,
) -> subprocess.CompletedProcess[str]:
    request_path = tmp_path / f"requests-{hash_seed}.json"
    request_path.write_text(
        json.dumps(
            [{"path": str(path), "model_agent_kwargs": kwargs, "judge_model": judge_model} for path, kwargs in requests]
        )
    )
    return subprocess.run(
        harbor_runtime_command(_DRIVER, "preflight", str(request_path)),
        check=check,
        capture_output=True,
        text=True,
        env=_driver_environment({"PYTHONHASHSEED": hash_seed}),
    )

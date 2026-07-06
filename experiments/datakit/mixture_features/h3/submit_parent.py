# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit the H3 parent Iris job, mirroring marin.run.iris_run but stamping the true
iris BUILD_DATE into the filtered workspace.

Why: marin.run.iris_run strips .git from the staged workspace, so the built marin-iris
wheel loses its git-derived client revision date and the prod controller rejects it as
"too old". The worktree's real lib/iris commit date is 2026-06-28 (>= the 2026-06-22
floor); we write that truthfully into _build_info.py so both the local submitting client
AND the parent job on the cluster report the correct date.

Run from the worktree root (/home/rav/mve-swarm-launch) via `uv run python <this>`.
Env: source wandb.env first (WANDB_API_KEY auto-forwarded by iris; WANDB_ENTITY/MARIN_PREFIX
passed via -e). Secrets are never printed.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

from marin.run.iris_run import DEFAULT_WORKING_DIR_EXCLUDES, _create_filtered_workspace

TRUE_IRIS_DATE = "2026-06-28"  # worktree's real `git log -1 --format=%cs -- lib/iris`
CONFIG = "lib/iris/config/marin.yaml"
JOBNAME = sys.argv[1]
ENTITY = os.environ["WANDB_ENTITY"]

workspace = Path.cwd().resolve()
temp = _create_filtered_workspace(workspace, list(DEFAULT_WORKING_DIR_EXCLUDES))
try:
    build_info = temp / "lib/iris/src/iris/_build_info.py"
    build_info.write_text(
        "# Copyright The Marin Authors\n# SPDX-License-Identifier: Apache-2.0\n" f'BUILD_DATE = "{TRUE_IRIS_DATE}"\n'
    )
    child = [
        "python",
        "experiments/domain_phase_mix/launch_h3_mve.py",
        "--prefix",
        "gs://marin-us-east5",
        "--tpu-region",
        "us-east5",
        "--tpu-zone",
        "us-east5-a",
    ]
    iris_cmd = [
        "uv",
        "run",
        "iris",
        "--config",
        CONFIG,
        "job",
        "run",
        "--job-name",
        JOBNAME,
        "--cpu",
        "2",
        "--memory",
        "8GB",
        "--disk",
        "20GB",
        "--enable-extra-resources",
        "--region",
        "us-east5",
        "--zone",
        "us-east5-a",
        "-e",
        "MARIN_PREFIX",
        "gs://marin-us-east5",
        "-e",
        "WANDB_ENTITY",
        ENTITY,
        "--no-wait",
        "--",
        *child,
    ]
    result = subprocess.run(iris_cmd, cwd=temp, check=False)
    sys.exit(result.returncode)
finally:
    shutil.rmtree(temp, ignore_errors=True)

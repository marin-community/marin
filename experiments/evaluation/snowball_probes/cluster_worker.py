# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the conversation driver on an Iris CPU worker with renewable service credentials."""

import json
import os
from pathlib import Path

from iris.cluster.client.job_info import get_job_info

from experiments.evaluation.snowball_probes.worker import main

CLUSTER_STATE_DIR = Path("/tmp/snowball-probes-state")


def run(arguments: list[str], inspiration: list[dict]) -> None:
    info = get_job_info()
    assert info is not None
    key = info.env["GLM_API_TOKEN"]
    path = Path("/tmp/snowball-probes-glm.env")
    CLUSTER_STATE_DIR.mkdir(exist_ok=True)
    (CLUSTER_STATE_DIR / "wildchat.json").write_text(json.dumps(inspiration, ensure_ascii=False))
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(f"GLM_API_TOKEN={key}\n")
        main([*arguments, "--glm-env", str(path), "--state-dir", str(CLUSTER_STATE_DIR)])
    finally:
        path.unlink(missing_ok=True)

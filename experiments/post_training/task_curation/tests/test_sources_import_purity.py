# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Importing the source catalog performs no network I/O.

A pending source's module reads its pinned registry document at first use, not at
import. The test re-imports ``sources`` and ``driver`` in a subprocess whose
``hf_hub_download`` raises, so any registry read the import chain still performs
fails the test instead of silently reaching the network.
"""

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]

BLOCKED_IMPORT = """
import sys
from huggingface_hub import hf_hub_download

def blocked(*args, **kwargs):
    raise AssertionError("importing the source catalog performed network I/O")

sys.modules["huggingface_hub"].hf_hub_download = blocked
import experiments.post_training.task_curation.sources  # noqa: F401,E402
import experiments.post_training.task_curation.driver  # noqa: F401,E402
print("clean")
"""


def test_importing_sources_and_driver_does_no_network_io() -> None:
    result = subprocess.run(
        [sys.executable, "-c", BLOCKED_IMPORT],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": f"{ROOT}:{ROOT / 'lib'}", "PATH": "/usr/bin:/bin"},
    )
    detail = json.dumps({"stdout": result.stdout[-400:], "stderr": result.stderr[-800:]})
    assert result.returncode == 0 and "clean" in result.stdout, detail

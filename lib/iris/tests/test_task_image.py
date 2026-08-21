# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import subprocess

import pytest


@pytest.mark.docker
def test_task_image_can_preload_jemalloc() -> None:
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--env",
            "LD_PRELOAD=libjemalloc.so.2",
            "iris-task:latest",
            "python",
            "-c",
            "from pathlib import Path; assert 'libjemalloc.so.2' in Path('/proc/self/maps').read_text()",
        ],
        check=True,
    )

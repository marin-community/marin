# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import ast
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from fray.cluster import ResourceConfig
from levanter.optim.config import AdamConfig
from levanter.tracker import NoopConfig

from experiments.grug.base.launch import GRUG_130M_MODEL, GrugBaseLaunchConfig, build_grug_run_config

_DUMMY_DATA: Any = object()
_GRUG_LAUNCHERS = [
    Path("experiments/grug/base/launch.py"),
    Path("experiments/grug/moe/launch.py"),
]


def test_build_grug_run_config_sets_temporary_checkpoint_base_path():
    """``build_grug_run_config`` wires the checkpointer's ``base_path`` and
    ``temporary_base_path`` to paths derived from the launch config's ``output_path``,
    so a run gets stable, predictable checkpoint locations under the region its output
    path lives in.
    """
    output_path = "gs://marin-us-east5/experiments/grug/base-trial"
    with (
        patch("rigging.filesystem.cluster_config.urllib.request.urlopen", side_effect=OSError("not on GCP")),
        patch.dict(os.environ, {"MARIN_PREFIX": "gs://marin-us-central1/scratch"}),
    ):
        run_config = build_grug_run_config(
            GrugBaseLaunchConfig(
                model=GRUG_130M_MODEL,
                data=_DUMMY_DATA,
                output_path=output_path,
                run_id="grug-temp-path-test",
                resources=ResourceConfig.with_cpu(),
                steps=1,
                batch_size=1,
                seed=0,
                mp="params=float32,compute=bfloat16,output=bfloat16",
                tracker=NoopConfig(),
                optimizer=AdamConfig(),
                eval_batch_size=None,
            ),
        )

    checkpointer = run_config.trainer.trainer.checkpointer
    assert checkpointer.base_path == "gs://marin-us-east5/experiments/grug/base-trial/checkpoints"
    assert checkpointer.temporary_base_path == (
        "gs://marin-us-east5/tmp/ttl=14d/" "checkpoints-temp/marin-us-east5/experiments/grug/base-trial/checkpoints"
    )
    assert run_config.trainer.trainer.checkpoint_search_paths("grug-temp-path-test") == [
        "gs://marin-us-east5/experiments/grug/base-trial/checkpoints",
        "gs://marin-us-east5/tmp/ttl=14d/" "checkpoints-temp/marin-us-east5/experiments/grug/base-trial/checkpoints",
    ]
    assert checkpointer.keep is None


@pytest.mark.parametrize("launcher_path", _GRUG_LAUNCHERS)
def test_grug_launchers_use_final_only_permanent_retention(launcher_path: Path):
    tree = ast.parse(launcher_path.read_text())
    keep_values = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        func = node.func
        if not isinstance(func, ast.Name) or func.id != "CheckpointerConfig":
            continue

        for keyword in node.keywords:
            if keyword.arg == "keep":
                keep_values.append(keyword.value)

    assert keep_values
    assert all(isinstance(value, ast.Constant) and value.value is None for value in keep_values)

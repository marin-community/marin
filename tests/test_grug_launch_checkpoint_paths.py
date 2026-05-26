# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import ast
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from fray.cluster import ResourceConfig
from levanter.optim import AdamConfig
from levanter.tracker import NoopConfig

from experiments.grug.base.launch import GRUG_130M_MODEL, GrugBaseLaunchConfig, resolve_grug_run_config

_DUMMY_DATA: Any = object()
_GRUG_LAUNCHERS = [
    Path("experiments/grug/base/launch.py"),
    Path("experiments/grug/moe/launch.py"),
    Path("experiments/grug/modular_opt/launch.py"),
]


def test_resolve_grug_run_config_sets_temporary_checkpoint_base_path():
    """``resolve_grug_run_config`` wires the checkpointer's ``base_path`` and
    ``temporary_base_path`` to paths derived from the resolved output_path,
    so callers that pin ``override_output_path`` get stable, predictable
    checkpoint locations. The resolution runs under the *current* region's
    ``marin_prefix()``, which is what makes cross-region preemption work.
    """
    output_path = "gs://marin-us-east5/experiments/grug/base-trial"
    with (
        patch("rigging.filesystem.urllib.request.urlopen", side_effect=OSError("not on GCP")),
        patch.dict(os.environ, {"MARIN_PREFIX": "gs://marin-us-central1/scratch"}),
    ):
        run_config = resolve_grug_run_config(
            "grug-temp-path-test",
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
                eval=None,
            ),
            override_output_path=output_path,
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
    assert checkpointer.keep == []


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
    assert all(isinstance(value, ast.List) and len(value.elts) == 0 for value in keep_values)

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any
from unittest.mock import patch

from fray.cluster import ResourceConfig
from levanter.optim import AdamConfig
from levanter.tracker import NoopConfig

import experiments.grug.modular_opt.launch as modular_opt_launch
import experiments.grug.moe.launch as moe_launch
from experiments.grug.base.launch import GRUG_130M_MODEL, GrugBaseLaunchConfig, resolve_grug_run_config

_DUMMY_DATA: Any = object()


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


def test_moe_grug_launch_uses_final_only_permanent_retention():
    captured_run_config: Any = None

    def capture_run_config(run_config):
        nonlocal captured_run_config
        captured_run_config = run_config

    with patch.object(moe_launch, "run_grug", side_effect=capture_run_config):
        moe_launch.run_grug_moe_trial(
            moe_launch.GrugMoeLaunchConfig(
                model=moe_launch.GRUG_MOE_TRIAL_MODEL,
                data=_DUMMY_DATA,
                output_path="gs://marin-us-east5/experiments/grug/moe-trial",
                run_id="grug-moe-retention-test",
                resources=ResourceConfig.with_cpu(),
                steps=1,
                batch_size=1,
                seed=0,
                mp="params=float32,compute=bfloat16,output=bfloat16",
                tracker=NoopConfig(),
                optimizer=AdamConfig(),
                eval=None,
            )
        )

    assert captured_run_config.trainer.trainer.checkpointer.keep == []


def test_modular_opt_grug_launch_uses_final_only_permanent_retention():
    captured_run_config: Any = None

    def capture_run_config(run_config):
        nonlocal captured_run_config
        captured_run_config = run_config

    with patch.object(modular_opt_launch, "run_grug", side_effect=capture_run_config):
        modular_opt_launch.run_grug_modular_opt_trial(
            modular_opt_launch.GrugModularOptLaunchConfig(
                model=modular_opt_launch.GRUG_130M_MODEL,
                data=_DUMMY_DATA,
                output_path="gs://marin-us-east5/experiments/grug/modular-opt-trial",
                run_id="grug-modular-opt-retention-test",
                resources=ResourceConfig.with_cpu(),
                steps=1,
                batch_size=1,
                seed=0,
                mp="params=float32,compute=bfloat16,output=bfloat16",
                tracker=NoopConfig(),
                optimizer=AdamConfig(),
                eval=None,
            )
        )

    assert captured_run_config.trainer.trainer.checkpointer.keep == []

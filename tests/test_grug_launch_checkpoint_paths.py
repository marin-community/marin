# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any
from unittest.mock import patch

from fray.cluster import ResourceConfig
from levanter.optim import AdamConfig
from levanter.tracker import NoopConfig

from experiments.grug.base.launch import GRUG_130M_MODEL, GrugBaseLaunchConfig, run_grug_base_trial

_DUMMY_DATA: Any = object()


def test_grug_base_launch_sets_temporary_checkpoint_base_path():
    with (
        patch("rigging.filesystem.urllib.request.urlopen", side_effect=OSError("not on GCP")),
        patch.dict(os.environ, {"MARIN_PREFIX": "gs://marin-us-central1/scratch"}),
        patch("experiments.grug.base.launch.run_grug") as run_grug,
    ):
        run_grug_base_trial(
            GrugBaseLaunchConfig(
                model=GRUG_130M_MODEL,
                data=_DUMMY_DATA,
                output_path="gs://marin-us-east5/experiments/grug/base-trial",
                run_id="grug-temp-path-test",
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

    run_config = run_grug.call_args.args[0]
    checkpointer = run_config.trainer.trainer.checkpointer
    assert checkpointer.base_path == "gs://marin-us-east5/experiments/grug/base-trial/checkpoints"
    assert checkpointer.temporary_base_path == (
        "gs://marin-us-east5/tmp/ttl=14d/" "checkpoints-temp/marin-us-east5/experiments/grug/base-trial/checkpoints"
    )
    assert run_config.trainer.trainer.checkpoint_search_paths("grug-temp-path-test") == [
        "gs://marin-us-east5/experiments/grug/base-trial/checkpoints",
        "gs://marin-us-east5/tmp/ttl=14d/" "checkpoints-temp/marin-us-east5/experiments/grug/base-trial/checkpoints",
    ]

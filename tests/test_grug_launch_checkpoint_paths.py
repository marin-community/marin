# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any
from unittest.mock import patch

from fray.cluster import ResourceConfig
from levanter.optim import AdamConfig
from levanter.tracker import NoopConfig

from experiments.grug.base.launch import GRUG_130M_MODEL, GrugBaseLaunchConfig, prepare_grug_trial

_DUMMY_DATA: Any = object()


def test_prepare_grug_trial_sets_temporary_checkpoint_base_path():
    """``prepare_grug_trial`` wires the checkpointer's ``base_path`` and
    ``temporary_base_path`` to paths derived from the resolved output_path,
    so callers that pin ``override_output_path`` get stable, predictable
    checkpoint locations.
    """
    output_path = "gs://marin-us-east5/experiments/grug/base-trial"
    with (
        patch("rigging.filesystem.urllib.request.urlopen", side_effect=OSError("not on GCP")),
        patch.dict(os.environ, {"MARIN_PREFIX": "gs://marin-us-central1/scratch"}),
    ):
        plan = prepare_grug_trial(
            name="grug-temp-path-test",
            launch=GrugBaseLaunchConfig(
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

    run_config = plan.train_config
    checkpointer = run_config.trainer.trainer.checkpointer
    assert checkpointer.base_path == "gs://marin-us-east5/experiments/grug/base-trial/checkpoints"
    assert checkpointer.temporary_base_path == (
        "gs://marin-us-east5/tmp/ttl=14d/" "checkpoints-temp/marin-us-east5/experiments/grug/base-trial/checkpoints"
    )
    assert run_config.trainer.trainer.checkpoint_search_paths("grug-temp-path-test") == [
        "gs://marin-us-east5/experiments/grug/base-trial/checkpoints",
        "gs://marin-us-east5/tmp/ttl=14d/" "checkpoints-temp/marin-us-east5/experiments/grug/base-trial/checkpoints",
    ]

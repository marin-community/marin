# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import io
import json
import os
from typing import Any
from unittest.mock import patch

from fray.cluster import ResourceConfig
from levanter.optim import AdamConfig
from levanter.tracker import NoopConfig

from experiments.grug.base.launch import GRUG_130M_MODEL, GrugBaseLaunchConfig, resolve_grug_run_config
from experiments.grug.moe.launch import _find_checkpoint_across_regions

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


def test_find_checkpoint_across_regions_finds_temporary_checkpoint(monkeypatch):
    output_path = "gs://marin-us-central1/grug/moe-trial"
    temp_root = "marin-us-east5/tmp/ttl=14d/checkpoints-temp/marin-us-east5/grug/moe-trial/checkpoints"
    temp_step = f"{temp_root}/step-720"

    class FakeGCSFileSystem:
        def ls(self, path: str):
            if path == temp_root:
                return [temp_step]
            raise FileNotFoundError(path)

        def exists(self, path: str) -> bool:
            return path in {
                f"{temp_step}/metadata.json",
                f"{temp_step}/manifest.ocdbt",
            }

        def open(self, path: str):
            assert path == f"{temp_step}/metadata.json"
            return io.StringIO(json.dumps({"step": 720}))

    monkeypatch.setattr("gcsfs.GCSFileSystem", FakeGCSFileSystem)
    monkeypatch.setattr(
        "rigging.filesystem.REGION_TO_DATA_BUCKET",
        {
            "us-central1": "marin-us-central1",
            "us-east5": "marin-us-east5",
        },
    )

    assert _find_checkpoint_across_regions(output_path) == f"gs://{temp_root}"

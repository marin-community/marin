# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import yaml
from marin.execution.build_context import BuildContext, VersionCodex, build_context
from marin.execution.lazy import StepContext

from experiments.post_training import taskforge_rl

SPLIT = "s3://marin-us-east-02a/marin/taskforge/gen1000/rl/split-v1"


def _launch(step) -> dict:
    config = step.build_config(
        StepContext.for_run(
            output_path="s3://durable/run",
            prefix="s3://durable",
            runtime_args=step.runtime_args,
            deps=step.deps,
        )
    )
    return yaml.safe_load(config.launch_config_yaml)


def test_baseline_evaluates_the_pool_on_engines_only() -> None:
    with build_context(BuildContext(versions=VersionCodex(default="2026.10.20"))):
        launch = _launch(taskforge_rl.build_run(taskforge_rl.Stage.BASELINE))

    assert launch["skyrl"]["entrypoint"] == "taskcompendium_generate"
    assert launch["inputs"]["train_data"] == []
    assert launch["inputs"]["validation_data"][0]["relative_path"] == "pool.parquet"
    assert launch["skyrl"]["trainer"]["placement"]["colocate_all"] is True
    assert launch["iris"]["container_profile"] == "CONTAINER_PROFILE_PRIVILEGED"


def test_rl_trains_on_two_policy_and_four_inference_nodes_sharing_the_split_directory() -> None:
    with build_context(BuildContext(versions=VersionCodex(default="2026.10.20"))):
        step = taskforge_rl.build_run(
            taskforge_rl.Stage.RL, train_uri=f"{SPLIT}/train.parquet", heldout_uri=f"{SPLIT}/heldout.parquet"
        )
    launch = _launch(step)

    assert len(step.deps) == 2  # model mirror and one split directory
    assert launch["iris"]["allocation"]["num_nodes"] == 6
    assert launch["skyrl"]["generator"]["num_inference_engines"] == 16
    assert launch["skyrl"]["generator"]["eval_n_samples_per_prompt"] == 4
    assert launch["skyrl"]["trainer"]["eval_interval"] == taskforge_rl.EVAL_INTERVAL

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import yaml
from marin.execution.lazy import StepContext

from experiments.post_training.snowball_inventory_compare import Initializer, comparison_step


def configured_arm(initializer):
    step = comparison_step(
        initializer,
        version="2026.10.20.900",
        model_uri=f"s3://marin-us-east-02a/test/models/{initializer}",
        pool_uri=f"s3://marin-us-east-02a/test/pools/{initializer}",
        pool_version="2026.10.20.2",
        pool_sha256="a" * 64,
        validation_uri=f"s3://marin-us-east-02a/test/panels/{initializer}",
        validation_version="2026.10.20.1",
        validation_sha256="b" * 64,
        judge_base_url="https://judge.example/v1",
        judge_model="test-judge",
    )
    config = step.build_config(
        StepContext.for_run(
            output_path=f"s3://marin-us-east-02a/test/{step.name}/{step.version}",
            prefix="s3://marin-us-east-02a/test",
            runtime_args=step.runtime_args,
            deps=step.deps,
        )
    )
    return config, yaml.safe_load(config.training.launch_config_yaml)


@pytest.mark.parametrize(
    "initializer,repository,revision",
    [
        (
            Initializer.SFT,
            "open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21",
            "fe666478454e79fe6027a0bab0388d9abd68f51a",
        ),
        (
            Initializer.DRDOOM,
            "open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom",
            "5b5705e8345bb111de03b3f0f9a00c8041e57db4",
        ),
    ],
)
def test_composed_arm_binds_initializer_and_complete_paired_panel(initializer, repository, revision):
    # The exact wire identities prevent the old step14 tokenizer and243-row
    # validation defaults from silently surviving in a newly initialized arm.
    config, document = configured_arm(initializer)
    assert config.model_source is not None
    assert (config.model_source.repository, config.model_source.revision) == (repository, revision)
    model = document["inputs"]["model"]
    assert model["uri"] == f"s3://marin-us-east-02a/test/models/{initializer}"
    assert model["tokenizer_uri"] == repository
    assert model["tokenizer_revision"] == revision
    sessions = document["skyrl"]["environment"]["task_sessions"]
    assert "token_prefix" not in sessions
    assert sessions["remote_judge"] == {"base_url": "https://judge.example/v1", "api_key_env": "JUDGE_API_KEY"}
    assert document["runtime"]["task_env"]["VERIFYIT_JUDGE_MODEL"] == "test-judge"
    assert document["inputs"]["train_data"][0]["relative_path"] == "train.parquet"
    assert document["inputs"]["validation_data"][0]["relative_path"] == "validation.parquet"
    assert config.validation_family_counts == {
        "calendar": 167,
        "competitive_coding": 167,
        "reasoning_gym": 167,
        "stem": 166,
        "structural": 243,
    }
    recipe = document["skyrl"]
    assert recipe["trainer"]["evaluation_runner"] == "task"
    assert recipe["trainer"]["eval_num_prompts"] is None
    assert recipe["trainer"]["eval_required_score_coverage"] == 1.0
    assert recipe["trajectory_runner"]["eval_max_concurrent_tasks"] == 8
    assert recipe["environment"]["task_sessions"]["daytona"]["api_key_env"] == "DAYTONA_EVAL_API_KEY"
    evaluation = next(row for row in recipe["trainer"]["callbacks"] if row["type"] == "evaluation")
    assert evaluation == {"type": "evaluation", "eval_steps": 25, "eval_before_train": True}
    best = next(row for row in recipe["trainer"]["callbacks"] if row["type"] == "best_checkpoint")
    assert best["expected_tasks"] == best["minimum_scored_tasks"] == 910
    assert recipe["generator"]["eval_n_samples_per_prompt"] == 1
    assert recipe["generator"]["eval_sampling_params"]["temperature"] == 0
    assert recipe["generator"]["eval_sampling_params"]["max_generate_length"] == 8_192
    assert recipe["context_budget"] == {
        "request_window_tokens": 24_576,
        "max_new_tokens_per_turn": 8_192,
        "max_turns": 1,
    }
    assert recipe["trainer"]["max_steps"] == 50
    export = next(row for row in recipe["trainer"]["callbacks"] if row["type"] == "hf_model_save")
    assert export["save_steps"] == 50


def test_arms_share_optimizer_recipe_but_cannot_resume_each_others_state():
    sft, sft_document = configured_arm(Initializer.SFT)
    drdoom, drdoom_document = configured_arm(Initializer.DRDOOM)

    assert sft_document["artifacts"]["checkpoint_root"] != drdoom_document["artifacts"]["checkpoint_root"]
    assert sft_document["artifacts"]["export_root"] != drdoom_document["artifacts"]["export_root"]
    assert sft.training.run_id != drdoom.training.run_id
    assert (
        sft_document["skyrl"]["trainer"]["policy"]["optimizer_config"]
        == drdoom_document["skyrl"]["trainer"]["policy"]["optimizer_config"]
    )
    assert sft_document["skyrl"]["trainer"]["algorithm"] == drdoom_document["skyrl"]["trainer"]["algorithm"]
    assert sft_document["skyrl"]["trainer"]["rollout_buffer"] == drdoom_document["skyrl"]["trainer"]["rollout_buffer"]
    assert (
        sft_document["skyrl"]["trainer"]["resume_mode"] == drdoom_document["skyrl"]["trainer"]["resume_mode"] == "latest"
    )

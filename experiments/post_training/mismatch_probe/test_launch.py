# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import yaml

from experiments.post_training.mismatch_probe.launch import ARMS, ProbeSettings, probe_recipe


def test_probe_recipe_loads_warm_checkpoint_and_resets_relative_updates():
    settings = ProbeSettings(
        seed=17,
        prompt_count=2,
        samples_per_prompt=2,
        updates=(0, 1, 2),
        keep_fraction=0.5,
        cache_mode="off",
        reuse_probe=None,
        resume_path=None,
    )

    def recipe(resume_path):
        return yaml.safe_load(
            probe_recipe(
                ARMS["native-layout"],
                replace(settings, resume_path=resume_path),
                warmup=False,
                marin_commit="a" * 40,
                skyrl_commit="b" * 40,
            )
        )["trainer"]

    warm_path = "s3://bucket/warm/checkpoints/global_step_1"
    resumed = recipe(warm_path)
    assert resumed["resume_mode"] == "from_path"
    assert resumed["resume_path"] == warm_path
    assert resumed["reset_global_step_on_resume"] is True
    assert resumed["max_steps"] == 2

    fresh = recipe(None)
    assert fresh["resume_mode"] == "latest"
    assert fresh["reset_global_step_on_resume"] is False

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import runpy
import shutil
from dataclasses import replace

import yaml
from marin.execution.build_context import BuildContext, VersionCodex, build_context
from marin.execution.lazy import StepContext
from rigging.provenance import LAUNCH_PROVENANCE_ENV, Provenance

from experiments.post_training.mismatch_probe import launch
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


def test_probe_bundle_carries_submission_commit_without_a_git_checkout(tmp_path, monkeypatch):
    bundled_source = tmp_path / "experiments/post_training/mismatch_probe/launch.py"
    bundled_source.parent.mkdir(parents=True)
    shutil.copyfile(launch.__file__, bundled_source)
    bundle = runpy.run_path(str(bundled_source), run_name="bundled_probe")
    provenance = Provenance(
        tree_hash="submitted-tree",
        base_commit="a" * 40,
        dirty=False,
        branch="main",
        built_by="test-user",
    )
    monkeypatch.setenv(LAUNCH_PROVENANCE_ENV, provenance.to_json())
    settings = ProbeSettings(17, 2, 2, (0, 1, 2), 0.5, "off", None, None)
    with build_context(BuildContext(VersionCodex("2026.09.28"))):
        artifact = bundle["build_arms"](
            arms=(ARMS["native-layout"],),
            settings=settings,
            model_uri="s3://fixture/model",
            data_uri="s3://fixture/data",
            fixture_version="2026.09.26",
            runtime_commit="b" * 40,
            warmup=False,
        )["native-layout"]
    config = artifact.build_config(StepContext.for_fingerprint(artifact.runtime_args, artifact.deps))
    recipe = yaml.safe_load(config.launch_config_yaml)["skyrl"]
    assert recipe["trainer"]["mismatch_probe"]["marin_commit"] == provenance.base_commit

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
from pathlib import Path

import pytest
import yaml
from marin.execution.build_context import BuildContext, VersionCodex, build_context
from marin.execution.lazy import StepContext
from marin.experiment.namespacing import user_owned_name
from marin.rl.skyrl import SkyRLRuntime, SkyRLRuntimeProfile

from experiments.post_training.smartie.launch import build_configured_run

COMPOSE_LAUNCH = """
import importlib.metadata
import json
import sys
from cloud.iris.launch_config import load_launch_config
from marinskyrl.runtime_options import reference_model_required

launch = load_launch_config(sys.argv[1])
installed = json.loads(importlib.metadata.distribution("marinskyrl").read_text("direct_url.json"))
assert launch.runtime.launcher_commit == installed["vcs_info"]["commit_id"]
print(json.dumps({
    "entrypoint": launch.runtime.entrypoint,
    "training_type": launch.runtime.training_type,
    "reference_required": reference_model_required(launch.skyrl.trainer.algorithm),
    "advantage_estimator": launch.skyrl.trainer.algorithm.advantage_estimator,
    "off_policy_correction": launch.skyrl.trainer.algorithm.off_policy_correction,
    "resume_mode": launch.skyrl.trainer.resume_mode,
    "resume_path": launch.skyrl.trainer.resume_path,
    "restore_dataloader_state": launch.skyrl.trainer.restore_dataloader_state,
    "reset_global_step_on_resume": launch.skyrl.trainer.reset_global_step_on_resume,
}))
"""


@pytest.fixture
def candidate_config(tmp_path):
    # Historical run configs are immutable provenance, not fixtures tied to today's
    # pin. Exercise the current launcher with a separate candidate and fresh identity.
    source = yaml.safe_load(Path(__file__).with_name("configs").joinpath("2026.10.15.32.yaml").read_text())
    source["run"]["id"] = f"{user_owned_name('models/snowball-smartie')}-dev"
    source["runtime"]["launcher_commit"] = SkyRLRuntime(profile=SkyRLRuntimeProfile.MEGATRON).commit
    path = tmp_path / "candidate.yaml"
    path.write_text(yaml.safe_dump(source))
    return path


def test_saved_config_survives_artifact_binding_without_training_overrides(candidate_config):
    # Regression: the former entrypoint replaced saved async and batch settings
    # with campaign constants. Non-default values must reach the launch boundary.
    source = yaml.safe_load(candidate_config.read_text())
    source["skyrl"]["trainer"]["policy"]["optimizer_config"]["lr"] = 3.5e-6
    source["skyrl"]["trainer"]["rollout_buffer"]["max_staleness_steps"] = 3
    source["skyrl"]["trainer"]["train_batch_size"] = 256
    source["skyrl"]["trainer"]["policy_mini_batch_size"] = 256
    candidate_config.write_text(yaml.safe_dump(source))
    original = candidate_config.read_bytes()
    with build_context(BuildContext(versions=VersionCodex(default="dev"))):
        run = build_configured_run(candidate_config)
    bound = run.build_config(StepContext.for_fingerprint(run.runtime_args, run.deps))
    launch = yaml.safe_load(bound.launch_config_yaml)
    assert launch["skyrl"]["trainer"] == source["skyrl"]["trainer"]
    assert launch["skyrl"]["generator"] == source["skyrl"]["generator"]
    assert launch["skyrl"]["context_budget"] == source["skyrl"]["context_budget"]
    assert launch["skyrl"]["data"] == source["skyrl"]["data"]
    assert candidate_config.read_bytes() == original


@pytest.mark.timeout(180)  # Cold installs resolve the pinned CPU launcher in its own environment.
@pytest.mark.parametrize("resume", [False, True])
def test_saved_candidate_crosses_pinned_launcher_boundary(tmp_path, candidate_config, resume):
    source = yaml.safe_load(candidate_config.read_text())
    resume_path = str(tmp_path / "global_step_2") if resume else None
    source["skyrl"]["trainer"].update(
        resume_mode="from_path" if resume else "none",
        resume_path=resume_path,
        restore_dataloader_state=True,
        reset_global_step_on_resume=False,
    )
    candidate_config.write_text(yaml.safe_dump(source))
    with build_context(BuildContext(versions=VersionCodex(default="dev"))):
        run = build_configured_run(candidate_config)
    launch = run.build_config(StepContext.for_fingerprint(run.runtime_args, run.deps))
    path = tmp_path / "launch.yaml"
    path.write_text(launch.launch_config_yaml)
    project = Path(__file__).resolve().parents[3] / "config/external/MarinSkyRL"
    result = subprocess.run(
        ["uv", "run", "--frozen", "--project", str(project), "python", "-c", COMPOSE_LAUNCH, str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == {
        "entrypoint": "skyrl_train.entrypoints.taskcompendium",
        "training_type": "async",
        "reference_required": False,
        "advantage_estimator": "rloo_n",
        "off_policy_correction": "tis",
        "resume_mode": "from_path" if resume else "none",
        "resume_path": resume_path,
        "restore_dataloader_state": True,
        "reset_global_step_on_resume": False,
    }

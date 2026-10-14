# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
from pathlib import Path

import pytest
from marin.execution.build_context import BuildContext, VersionCodex, build_context
from marin.execution.lazy import StepContext

from experiments.post_training.smartie.launch import build_run

COMPOSE_LAUNCH = """
import json
import sys
from cloud.iris.launch_config import load_launch_config
from marinskyrl.runtime_options import reference_model_required
from skyrl_train.config.ftpo import ftpo_config, validate_ftpo

launch = load_launch_config(sys.argv[1])
validate_ftpo(launch.skyrl)
objective = ftpo_config(launch.skyrl.trainer.algorithm)
print(json.dumps({
    "entrypoint": launch.runtime.entrypoint,
    "training_type": launch.runtime.training_type,
    "reference_required": reference_model_required(launch.skyrl.trainer.algorithm),
    "auxiliary_active": objective.auxiliary_weight > 0,
    "verified_boundaries": objective.boundary.uses_verified_boundaries,
    "sampling_kind": launch.skyrl.data.sampling.kind,
    "domain_weights": dict(launch.skyrl.data.sampling.domain_weights),
    "resume_mode": launch.skyrl.trainer.resume_mode,
    "resume_path": launch.skyrl.trainer.resume_path,
    "restore_dataloader_state": launch.skyrl.trainer.restore_dataloader_state,
    "reset_global_step_on_resume": launch.skyrl.trainer.reset_global_step_on_resume,
}))
"""


@pytest.mark.timeout(180)  # Cold installs resolve the pinned CPU launcher in its own environment.
@pytest.mark.parametrize("resume", [False, True])
def test_joint_recipe_crosses_pinned_launcher_boundary(tmp_path, resume):
    resume_path = str(tmp_path / "global_step_2") if resume else None
    with build_context(BuildContext(versions=VersionCodex(default="dev"))):
        run = build_run(
            model_uri=str(tmp_path / "model"),
            data_uri=str(tmp_path / "data"),
            data_version="dev",
            max_steps=6 if resume else 2,
            resume_path=resume_path,
        )
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
        "training_type": "sync",
        "reference_required": True,
        "auxiliary_active": True,
        "verified_boundaries": True,
        "sampling_kind": "domain-weighted",
        "domain_weights": {
            "calendar": 0.5,
            "competitive_code": 0.5,
            "reasoning_gym": 0.5,
            "stem_mcqa": 0.5,
            "structural": 1.0,
        },
        "resume_mode": "from_path" if resume else "none",
        "resume_path": resume_path,
        "restore_dataloader_state": True,
        "reset_global_step_on_resume": False,
    }

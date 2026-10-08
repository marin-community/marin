# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pytest

from experiments.rl_data_reviews.grading_identity import ExecutionModules, verified_execution_grading
from infra.marina.applets.rl_data_catalog.server.grading_code import python_grading_program
from infra.marina.applets.rl_data_catalog.server.grading_routes import skyrl_grading_routes


def test_execution_identity_detects_current_grader_changes_before_publication(tmp_path) -> None:
    checkout = tmp_path / "checkout"
    package = checkout / "skyrl-gym/skyrl_gym"
    module = package / "envs/toy.py"
    module.parent.mkdir(parents=True)
    module.write_text(
        'class Env:\n def __init__(self):\n  self.expected="correct"\n def step(self,x):\n  return x == self.expected\n'
    )
    (package / "envs/base_text_env.py").write_text(
        "class BaseTextEnv:\n def init(self,x):\n  return x\n def close(self):\n  pass\n"
        " def set_rollout_evidence(self,x):\n  pass\n"
    )
    train = checkout / "skyrl-train/skyrl_train"
    contracts = train / "trajectory_runners/skyrl_gym_contracts.py"
    contracts.parent.mkdir(parents=True)
    contracts.write_text("def verification_from_env_step(x):\n return x\ndef fold_verification_results(x):\n return x\n")
    packages = {"skyrl_gym": package, "skyrl_train": train}
    row = {
        "id": "MarinSkyRL:toy",
        "dataset_revision": "data1",
        "revision": "repo1",
        "environment": "toy",
        "gym_entrypoint": "skyrl_gym.envs.toy:Env",
        "verifier_mode": "legacy",
        "grading_scope": {"agents": []},
        "grading_repositories": {name: {} for name in packages},
    }
    source = ExecutionModules(packages)
    route = skyrl_grading_routes(row, source, ())[0]
    program = python_grading_program(source, route.roots, tuple(packages), route.bindings)
    manifest = {
        "schema_version": 1,
        "routes": {"toy": {"program_revision": program.digest, "resources": {}, "locked_packages": {}}},
    }
    row["grading_manifest"] = manifest
    row["grading_revision"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(json.dumps(row))
    config = {
        "source": {"source_id": row["id"], "revision": "data1", "grading_snapshot": snapshot.name},
        "runtime": {
            "marinskyrl_checkout": str(checkout),
            "harbor_checkout": str(tmp_path / "unused-harbor"),
            "gym_python": "unused",
            "harbor_python": "unused",
        },
    }
    original = verified_execution_grading(config, tmp_path)
    module.write_text("# Documentation change\n" + module.read_text())
    assert verified_execution_grading(config, tmp_path) == original
    module.write_text(module.read_text().replace('="correct"', '="different"'))
    with pytest.raises(ValueError, match="Executed grading code differs"):
        verified_execution_grading(config, tmp_path)

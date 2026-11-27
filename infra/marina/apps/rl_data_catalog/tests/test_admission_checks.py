# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Admission requires declared environment artifacts and one runnable backend."""

import os
import subprocess
import sys
from pathlib import Path

from finestore.rl.curation import ActorCensusInput, GradingEnvironmentEvidence
from taskcompendium.backend_requirements import BackendFit, BackendSourceMap

from infra.marina.apps.rl_data_catalog.server.admission import (
    population_admission_blocker,
)

BINDING = ActorCensusInput(
    dataset_id="fixture/dataset",
    dataset_revision="r1",
    converter_version="1",
    converter_revision="abc123",
    measured_tasks=3,
    task_specs_sha256="a" * 64,
    evidence_url="s3://fixture/actor.json",
)


def grading_evidence(missing: dict[str, int]) -> GradingEnvironmentEvidence:
    return GradingEnvironmentEvidence(
        assessed_tasks=3,
        required_counts={"declared:bounded": 3},
        declared_counts={"declared:bounded": 3},
        status_counts={"matched": 3},
        reason_counts={
            "The typed grader reads the declared submitted answer or captured files in its own environment": 3
        },
        missing_artifact_counts=missing,
        evidence_url="s3://fixture/receipt.json",
    )


def source_map(supports: dict[str, str]) -> BackendSourceMap:
    return BackendSourceMap(
        source_id="Fixture:source",
        basis="converted_spec_census",
        task_count=3,
        inputs={},
        demands={},
        resource_ranges={},
        backends={
            name: BackendFit(
                support=support,
                task_counts={},
                roles={},
                blockers={},
                not_applicable={},
                validation_errors={},
            )
            for name, support in supports.items()
        },
        role_backend_assignments={},
        grading_environment=None,
        integrity_contracts=None,
    )


def test_missing_declared_environment_artifact_blocks_admission():
    blocker = population_admission_blocker(
        BINDING, 3, grading_evidence({"s3://marin-us-east-02a/marin/images/env-4a5dde6cb8723181/lock": 3})
    )
    assert blocker == (
        "declared environment artifacts exist in no storage region: "
        "s3://marin-us-east-02a/marin/images/env-4a5dde6cb8723181/lock"
    )


def test_verified_environment_artifacts_do_not_block_admission():
    assert population_admission_blocker(BINDING, 3, grading_evidence({})) is None


def test_packaged_admission_imports_without_taskcompendium():
    """Marina's base image has no taskcompendium; the packaged module must not need it."""
    probe = "\n".join(
        [
            "import importlib, importlib.abc, sys",
            "class Blocked(importlib.abc.MetaPathFinder):",
            "    def find_spec(self, name, path=None, target=None):",
            "        if name == 'taskcompendium' or name.startswith('taskcompendium.'):",
            "            raise ModuleNotFoundError(f'blocked: {name}')",
            "        return None",
            "sys.meta_path.insert(0, Blocked())",
            "importlib.import_module('infra.marina.apps.rl_data_catalog.server.admission')",
            "print('ok')",
        ]
    )
    root = Path(__file__).parents[5]
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=root,
        check=False,
        env={name: value for name, value in os.environ.items() if name != "PYTHONPATH"},
    )
    assert result.returncode == 0, f"import probe failed: {result.stderr[-500:]}"
    assert "ok" in result.stdout

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Schema evolution and the production lock-grader incompatibility."""

import ast
import json
import sys
from dataclasses import fields
from functools import partial
from pathlib import Path
from typing import get_args, get_type_hints

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from finestore.rl.status_taxonomy import taxonomy_tables
from rolloutengine.models import MachineRuntimeSpec
from shellbox.backends.daytona.machine import DaytonaMachineFactory
from shellbox.backends.docker.machine import DockerMachineFactory
from shellbox.backends.iris.machine import IrisMachineFactory
from shellbox.backends.local.machine import check_local_environment
from shellbox.machine import (
    BACKEND_DEPLOYMENT,
    Command,
    MachineSpec,
    Mount,
    UnsupportedMachineSpec,
)
from shellbox.requirements import (
    COMMAND_FIELDS,
    MACHINE_FIELDS,
    SOURCE_FIELDS,
    Requirement,
    Support,
)
from shellbox.survey import Checker, LocalResourceEvidence, capability_matrix
from taskcompendium.backend_fields import (
    ENVIRONMENT_FIELDS,
    EXECUTION_FIELDS,
    HARBOR_AGENT_FIELDS,
    HARBOR_ENVIRONMENT_FIELDS,
    HARBOR_VERIFIER_FIELDS,
    RESOURCE_FIELDS,
    RUNTIME_FIELDS,
    SCRIPT_GRADER_FIELDS,
    TASK_FIELDS,
)
from taskcompendium.backend_requirements import BackendSurvey, FitSupport, RuntimeProfile
from taskcompendium.models import EnvironmentRequirements, ExecutionRequirements, ScriptGrader, TaskResource, TaskSpec
from taskcompendium.runtime.resources import inline_resource

from experiments.post_training.task_curation.backend_capabilities import (
    build_survey,
    main,
    merge_surveys,
    production_backends,
    production_matrix,
    source_map,
    verify_backend_blocker_reasons,
)
from infra.marina.apps.rl_data_catalog.server.source_identity import canonical_source_id

ROOT = Path(__file__).resolve().parents[4]
RUNTIME = RuntimeProfile(
    network="deny", cpus=None, memory_mb=None, storage_mb=None, gpus=0, user=None, startup_timeout=60, cleanup_timeout=30
)


def test_backend_demand_classification_tracks_owning_schemas():
    # A new field at any of these boundaries must be reviewed for backend demands.
    for schema, classified in (
        (EnvironmentRequirements, ENVIRONMENT_FIELDS),
        (ExecutionRequirements, EXECUTION_FIELDS),
        (TaskResource, RESOURCE_FIELDS),
        (MachineRuntimeSpec, RUNTIME_FIELDS),
        (TaskSpec, TASK_FIELDS),
        (ScriptGrader, SCRIPT_GRADER_FIELDS),
    ):
        assert set(schema.model_fields) == set(classified), schema.__name__
    assert {field.name for field in fields(MachineSpec)} == set(MACHINE_FIELDS)
    assert {field.name for field in fields(Command)} == set(COMMAND_FIELDS)
    assert set(RuntimeProfile.model_fields) == set(MachineRuntimeSpec.model_fields) - {"backend"}
    sources = (*get_args(get_type_hints(MachineSpec)["source"]), Mount)
    assert {source.__name__ for source in sources} == set(SOURCE_FIELDS)
    for source in sources:
        assert {field.name for field in fields(source)} == set(SOURCE_FIELDS[source.__name__])

    # Read Harbor's canonical standalone schema without importing its optional runtime.
    tree = ast.parse((ROOT / "vendor/harbor/src/harbor_config/models/task/config.py").read_text())
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    for name, classified in (
        ("EnvironmentConfig", HARBOR_ENVIRONMENT_FIELDS),
        ("AgentConfig", HARBOR_AGENT_FIELDS),
        ("VerifierConfig", HARBOR_VERIFIER_FIELDS),
    ):
        actual = {
            node.target.id
            for node in classes[name].body
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        }
        assert actual == set(classified), name


def test_lock_grader_storage_has_no_backend_and_quota_removal_preserves_mixed_deployment():
    checkers = {"docker": DockerMachineFactory().check_environment, "local": check_local_environment}
    matrix = capability_matrix(checkers)
    document = {
        "task": {
            "environment_requirements": {
                "command_semantics": "linux_process",
                "docker_image": "example/image@sha256:" + "0" * 64,
            },
            "grader": {"environment": {"command_semantics": "linux_process", "packages_lock": "private.lock"}},
        },
        "runtime": {"task_machine": {"storage_mb": 1024}, "verifier_machine": {"storage_mb": 1024}},
    }
    defaults = {"agent": RUNTIME, "grader": RUNTIME}
    before = source_map("cohort", [document], matrix, checkers, {}, defaults)
    assert before.backends["local"].roles["grader"] == {FitSupport.UNSUPPORTED: 1}
    assert before.backends["docker"].roles["grader"] == {FitSupport.NOT_APPLICABLE: 1}
    assert before.backends["docker"].not_applicable == {"grader:host_image": 1}
    assert before.demands["grader"][Requirement.HOST_IMAGE] == 1
    assert before.resource_ranges["grader"]["storage_mb"] == (1024, 1024)
    assert before.role_backend_assignments == {"agent": {"daytona": 1}, "grader": {"local": 1}}
    document["runtime"]["verifier_machine"]["storage_mb"] = None
    after = source_map("cohort-v2", [document], matrix, checkers, {}, defaults)
    assert after.backends["local"].roles["grader"] == {Support.VERIFY: 1}
    assert after.backends["docker"].roles["agent"] == {Support.VERIFY: 1}
    assert after.backends["docker"].roles["grader"] == {FitSupport.NOT_APPLICABLE: 1}
    assert after.backends["docker"].task_counts == {Support.VERIFY: 1}
    assert after.backends["local"].roles["agent"] == {Support.UNSUPPORTED: 1}


def test_typed_local_requirement_preserves_limited_backend_support():
    checkers = _verified_local_checkers()
    document = {
        "task": {
            "environment_requirements": {
                "required_backend": "local",
                "command_semantics": "linux_process",
                "docker_image": "example/image@sha256:" + "0" * 64,
            },
            "grader": {"kind": "none"},
        },
        "runtime": {},
    }
    mapped = source_map(
        "local-tool", [document], production_matrix(checkers), checkers, {}, {"agent": RUNTIME, "grader": RUNTIME}
    )
    assert mapped.role_backend_assignments == {"agent": {"local": 1}}
    assert mapped.backends["local"].roles["agent"] == {FitSupport.SUPPORTED: 1}
    assert mapped.backends["daytona"].roles["agent"] == {FitSupport.UNSUPPORTED: 1}
    assert mapped.backends["daytona"].blockers == {"agent:required_backend": 1}


def test_grader_only_host_image_population_fits_local_and_excludes_image_backends():
    """A grader-only HostImage+lock population (Evalchemy:aime24) is not a backend-wide failure.

    LOCAL runs these graders in production, so its fit is supported; backends that cannot
    host a HostImage environment are not candidates at all, not failures.
    """
    checkers = _unit_checkers()
    matrix = production_matrix(checkers)
    documents = [
        {
            "task": {
                "grader": {
                    "environment": {"command_semantics": "linux_process", "packages_lock": "math.lock"},
                    "timeout": 600,
                },
                "resources": {"verifier": [{"path": "grader-files.tar", "bytes": "AAAA"}]},
            },
            "runtime": {},
        }
        for _ in range(3)
    ]
    mapped = source_map("aime24", documents, matrix, checkers, {}, {"agent": RUNTIME, "grader": RUNTIME})
    local = mapped.backends["local"]
    assert local.support == FitSupport.SUPPORTED
    assert local.task_counts == {FitSupport.SUPPORTED: 3}
    assert local.roles["agent"] == {FitSupport.ROLE_ABSENT: 3}
    assert local.roles["grader"] == {FitSupport.SUPPORTED: 3}
    assert local.blockers == {} and local.not_applicable == {}
    assert local.validation_errors == {}
    for backend, fit in mapped.backends.items():
        if backend == "local":
            continue
        assert fit.support == FitSupport.NOT_APPLICABLE, backend
        assert fit.task_counts == {FitSupport.NOT_APPLICABLE: 3}, backend
        assert fit.roles["agent"] == {FitSupport.ROLE_ABSENT: 3}, backend
        assert fit.roles["grader"] == {FitSupport.NOT_APPLICABLE: 3}, backend
        assert fit.not_applicable == {"grader:host_image": 3}, backend
        assert fit.blockers == {} and fit.validation_errors == {}, backend


def test_underspecified_environment_stays_a_typed_blocker_on_every_backend():
    """Applicability never hides a role that resolves to no machine environment."""
    checkers = _unit_checkers()
    document = {
        "task": {"grader": {"environment": {"command_semantics": "linux_process"}}},
        "runtime": {},
    }
    mapped = source_map(
        "underspecified", [document], production_matrix(checkers), checkers, {}, {"agent": RUNTIME, "grader": RUNTIME}
    )
    for backend, fit in mapped.backends.items():
        assert fit.roles["grader"] == {FitSupport.UNSUPPORTED: 1}, backend
        assert fit.not_applicable == {}, backend
        assert fit.validation_errors, backend


def _verified_local_checkers() -> dict[str, Checker]:
    """The LOCAL checker as the privileged worker presents it: probes accepted, machinery verified."""

    def accepting(requirements, spec, **kwargs) -> None:
        pass

    return {
        **_unit_checkers(),
        "local": partial(
            accepting,
            resource_evidence=LocalResourceEvidence(storage_enforced=True, cgroup_verified=True, cgroup_parent=None),
        ),
    }


PDB_SHAPE = "docker.io/library/python@sha256:" + "0" * 64


def _pdb_grader_document() -> dict:
    """One converted PDBThink-shaped task: the grader profile its admission rules require."""
    return {
        "task": {
            "grader": {
                "environment": {
                    "command_semantics": "linux_process",
                    "docker_image": PDB_SHAPE,
                    "setup_commands": ["mkdir -p /opt/grade"],
                    "working_directory": "/workspace",
                },
                "timeout": 600,
            },
            "resources": {
                "verifier": [
                    {
                        "path": "grade.py",
                        "source": {"kind": "inline_file", "content_base64": "cHJpbnQoJ2dyYWRlZCcp"},
                        "mode": "640",
                        "mtime_ns": 1700000000123456789,
                    }
                ]
            },
        },
        "runtime": {
            "verifier_machine": {
                "network": "deny",
                "cpus": 1,
                "memory_mb": 512,
                "storage_mb": 1024,
                "gpus": 0,
                "user": None,
                "startup_timeout": 60,
                "cleanup_timeout": 30,
            }
        },
    }


def test_local_resource_cells_reflect_the_survey_evidence():
    """Resource cells upgrade only when the worker's enforcement machinery verified."""
    unverified = production_matrix(_unit_checkers())
    verified = production_matrix(_verified_local_checkers())
    for requirement in (Requirement.CPUS, Requirement.MEMORY, Requirement.STORAGE):
        base = next(cell for cell in unverified if cell.backend == "local" and cell.requirement == requirement)
        # without verified worker machinery the cells wait for a live probe
        assert base.support == Support.VERIFY, requirement
        upgraded = next(cell for cell in verified if cell.backend == "local" and cell.requirement == requirement)
        assert upgraded.support == Support.SUPPORTED, requirement
        assert "verified on this worker" in upgraded.note, requirement


def test_local_orchestrated_requirements_carry_production_evidence():
    """Setup, uploads, deadlines and file metadata are production guarantees on LOCAL."""
    matrix = production_matrix(_unit_checkers())
    for requirement in (
        Requirement.SETUP,
        Requirement.FILE_UPLOAD,
        Requirement.COMMAND_TIMEOUT,
        Requirement.FILE_MODE,
        Requirement.FILE_MTIME,
    ):
        cell = next(cell for cell in matrix if cell.backend == "local" and cell.requirement == requirement)
        assert cell.support == Support.SUPPORTED, requirement
        assert any("_prepare_machine" in path or "taskcompendium" in path for path in cell.evidence), requirement
    docker_setup = next(cell for cell in matrix if cell.backend == "docker" and cell.requirement == Requirement.SETUP)
    # the offline survey cannot probe setup on the real image; a live probe grades it
    assert docker_setup.support == Support.VERIFY


def test_pdb_shaped_grader_fits_local_with_verified_enforcement_machinery():
    """The native PDBThink grader profile fits LOCAL once storage and cgroups verify."""
    checkers = _verified_local_checkers()
    mapped = source_map(
        "pdbthink",
        [_pdb_grader_document() for _ in range(3)],
        production_matrix(checkers),
        checkers,
        {},
        {"agent": RUNTIME, "grader": RUNTIME},
    )
    local = mapped.backends["local"]
    assert local.support == FitSupport.SUPPORTED
    assert local.task_counts == {FitSupport.SUPPORTED: 3}
    assert local.roles["grader"] == {FitSupport.SUPPORTED: 3}
    assert local.blockers == {} and local.validation_errors == {}
    # the measured resource ranges record the profile exactly as declared
    grader_ranges = mapped.resource_ranges["grader"]
    assert grader_ranges["cpus"] == (1, 1)
    assert grader_ranges["memory_mb"] == (512, 512)
    assert grader_ranges["storage_mb"] == (1024, 1024)

    # the survey carries the typed deployment classes
    survey = _survey_with_source(mapped, checkers)
    assert survey.deployment_classes["docker-gvisor"] == "dev_only"
    assert survey.deployment_classes["iris"] == "production"


def _survey_with_source(mapped, checkers):
    """A minimal bound survey carrying one measured source and the typed deployment classes."""
    return BackendSurvey(
        catalog_root="test:catalog",
        snapshot_id="snap",
        generated_at="2026-11-17T00:00:00Z",
        git_revision="rev",
        schema_sha256="0" * 64,
        runtime_defaults={"agent": RUNTIME, "grader": RUNTIME},
        matrix=production_matrix(checkers),
        deployment_limits={},
        deployment_classes={
            backend.value: deployment_class.value for backend, deployment_class in BACKEND_DEPLOYMENT.items()
        },
        sources=[mapped],
    )


def test_dev_only_supported_cell_can_never_satisfy_admission():
    """A dev-only backend's SUPPORTED verdict never counts for admission.

    The admission view (``production_backends``) drops dev-only backends, so a
    population whose only SUPPORTED verdict comes from local Docker or
    Docker+runsc still lacks production support and stays excluded.
    """
    checkers = _verified_local_checkers()
    mapped = source_map(
        "pdbthink",
        [_pdb_grader_document() for _ in range(3)],
        production_matrix(checkers),
        checkers,
        {},
        {"agent": RUNTIME, "grader": RUNTIME},
    )
    survey = _survey_with_source(mapped, checkers)
    production = production_backends(survey)
    # docker and docker-gvisor measured SUPPORTED here but are dev-only; the
    # admission view must not contain them at all
    assert "docker" not in production and "docker-gvisor" not in production
    assert set(production) == {"iris", "daytona", "local"}
    # production backends keep their own measured verdicts untouched
    assert production["local"].support == FitSupport.SUPPORTED


def test_conversation_only_census_needs_no_backend_despite_deployment_defaults():
    checkers = {"docker": DockerMachineFactory().check_environment}
    mapped = source_map(
        "conversation",
        [
            {
                "environment_requirements": {},
                "grader": {"kind": "none"},
                "execution_requirements": {
                    "cpus": 2,
                    "memory_mb": 4096,
                    "storage_mb": 10240,
                    "agent_timeout_sec": 900,
                    "verifier_timeout_sec": 600,
                },
            }
        ],
        capability_matrix(checkers),
        checkers,
        {},
        {"agent": RUNTIME, "grader": RUNTIME},
    )
    assert mapped.demands == {}
    assert mapped.backends["docker"].support == FitSupport.ROLE_ABSENT
    assert mapped.backends["docker"].task_counts == {FitSupport.ROLE_ABSENT: 1}
    assert mapped.backends["docker"].roles == {
        "agent": {FitSupport.ROLE_ABSENT: 1},
        "grader": {FitSupport.ROLE_ABSENT: 1},
    }


def test_declared_capacity_reaches_backend_checks_and_conflicting_runtime_fails():
    checkers = {"docker": DockerMachineFactory().check_environment}
    document = {
        "id": "budgeted",
        "environment_requirements": {
            "command_semantics": "linux_process",
            "docker_image": "example/image@sha256:" + "0" * 64,
        },
        "grader": {"kind": "none"},
        "execution_requirements": {
            "cpus": 2,
            "memory_mb": 4096,
            "storage_mb": 10240,
            "agent_timeout_sec": 900,
            "verifier_timeout_sec": 600,
        },
    }
    matrix = capability_matrix(checkers)
    defaults = {"agent": RUNTIME, "grader": RUNTIME}
    mapped = source_map("budgeted", [document], matrix, checkers, {}, defaults)
    assert mapped.resource_ranges["agent"] == {
        "cpus": (2, 2),
        "memory_mb": (4096, 4096),
        "storage_mb": (10240, 10240),
        "gpus": (0, 0),
    }
    assert mapped.demands["agent"][Requirement.COMMAND_TIMEOUT] == 1
    assert mapped.demands["agent"][Requirement.STORAGE] == 1
    with pytest.raises(ValueError, match="cpus conflicts"):
        source_map(
            "budgeted", [{"task": document, "runtime": {"task_machine": {"cpus": 3}}}], matrix, checkers, {}, defaults
        )


def test_unclassified_capability_counts_a_misfit_on_every_backend():
    checkers = {"docker": DockerMachineFactory().check_environment}
    document = {
        "environment_requirements": {
            "command_semantics": "linux_process",
            "capabilities": ["nemotron-agent:ns_tools_simple_agent"],
        },
        "grader": {"kind": "none"},
    }
    mapped = source_map(
        "unclassified", [document], capability_matrix(checkers), checkers, {}, {"agent": RUNTIME, "grader": RUNTIME}
    )
    fit = mapped.backends["docker"]
    # the vocabulary gap is pending evidence, not a fit failure (VERIFY ruling)
    assert fit.task_counts == {Support.VERIFY: 1}
    assert fit.roles["agent"] == {Support.VERIFY: 1}
    assert fit.blockers == {"agent:nemotron-agent:ns_tools_simple_agent": 1}


def test_unclassified_capability_preserves_the_other_role_and_known_requirements():
    checkers = {"docker": DockerMachineFactory().check_environment}
    task = {
        "environment_requirements": {
            "command_semantics": "linux_process",
            "docker_image": "example/image@sha256:" + "0" * 64,
            "capabilities": ["nemotron-agent:ns_tools_simple_agent"],
        },
        "grader": {"environment": {"command_semantics": "linux_process", "packages_lock": "private.lock"}},
    }
    mapped = source_map(
        "mixed",
        [{"task": task, "runtime": {}}],
        capability_matrix(checkers),
        checkers,
        {},
        {"agent": RUNTIME, "grader": RUNTIME},
    )
    fit = mapped.backends["docker"]
    assert fit.task_counts == {Support.VERIFY: 1}
    assert fit.roles["agent"] == {Support.VERIFY: 1}
    assert fit.roles["grader"] == {FitSupport.NOT_APPLICABLE: 1}
    assert fit.not_applicable == {"grader:host_image": 1}
    assert fit.blockers == {"agent:nemotron-agent:ns_tools_simple_agent": 1}
    assert mapped.demands["agent"][Requirement.DOCKER_IMAGE] == 1
    assert mapped.demands["grader"][Requirement.PACKAGES_LOCK] == 1
    assert mapped.demands["grader"][Requirement.HOST_IMAGE] == 1
    assert mapped.role_backend_assignments["grader"] == {"local": 1}


def _docker_task(task_id: str) -> dict:
    return {
        "id": task_id,
        "environment_requirements": {
            "command_semantics": "linux_process",
            "docker_image": "example/image@sha256:" + "0" * 64,
        },
        "grader": {"kind": "none"},
    }


def _census_entry(source_id: str, path: Path) -> dict:
    return {
        "source_id": source_id,
        "paths": [str(path)],
        "column": "task_spec",
        "runtime_column": "runtime",
        "expected_tasks": 1,
    }


def _shard_survey(
    tmp_path: Path,
    name: str,
    source_ids: list[str],
    *,
    measured_ids: list[str] | None = None,
    checkers: dict[str, Checker] | None = None,
) -> BackendSurvey:
    """Build a real shard: every listed source appears, measured ones with inputs."""
    measured_ids = measured_ids if measured_ids is not None else source_ids
    inputs = []
    for source_id in measured_ids:
        path = tmp_path / f"{name}-{source_id}.parquet"
        _write_spec_parquet(path, [{"task": _docker_task(f"{name}-task"), "runtime": {}}])
        inputs.append(_census_entry(source_id, path))
    profile = RUNTIME.model_dump(mode="json")
    manifest = {
        "catalog_root": "test:catalog",
        "snapshot_id": name,
        "source_ids": sorted(source_ids),
        "runtime_defaults": {"agent": profile, "grader": profile},
        "inputs": inputs,
    }
    if set(source_ids) - set(measured_ids):
        manifest["unassessed_reasons"] = {
            canonical_source_id(source_id): "No converted population is bound to this source in the test fixture."
            for source_id in set(source_ids) - set(measured_ids)
        }
    return build_survey(manifest, checkers or _unit_checkers(), "rev")


def _rejecting_checkers() -> dict[str, Checker]:
    def rejecting(requirements, spec) -> None:
        raise UnsupportedMachineSpec("The drift shard rejects every requirement")

    return {name: rejecting for name in ("docker", "docker-gvisor", "qemu", "shellsim", "daytona", "iris", "local")}


def test_unassessed_inputs_require_their_own_declared_reason(tmp_path):
    path = tmp_path / "specs.parquet"
    _write_spec_parquet(path, [{"task": _docker_task("measured"), "runtime": {}}])
    profile = RUNTIME.model_dump(mode="json")
    manifest = {
        "catalog_root": "test:catalog",
        "snapshot_id": "test-snapshot",
        "source_ids": ["cohort-measured", "cohort-omitted"],
        "runtime_defaults": {"agent": profile, "grader": profile},
        "inputs": [_census_entry("cohort-measured", path)],
    }
    with pytest.raises(ValueError, match="cohort-omitted"):
        build_survey(manifest, _unit_checkers(), "rev")
    manifest["unassessed_reasons"] = {"cohort-omitted": "The pinned revision has no converted population yet."}
    survey = build_survey(manifest, _unit_checkers(), "rev")
    row = {source.source_id: source for source in survey.sources}["cohort-omitted"]
    assert row.basis == "unassessed"
    assert row.reason == manifest["unassessed_reasons"]["cohort-omitted"]
    assert row.task_count == 0 and not row.backends and not row.demands
    manifest["unassessed_reasons"]["cohort-measured"] = "unused"
    with pytest.raises(ValueError, match="converted inputs"):
        build_survey(manifest, _unit_checkers(), "rev")


def test_local_disk_pool_provenance_comes_from_the_checkers(tmp_path):
    path = tmp_path / "specs.parquet"
    _write_spec_parquet(path, [{"task": _docker_task("measured"), "runtime": {}}])
    profile = RUNTIME.model_dump(mode="json")
    manifest = {
        "catalog_root": "test:catalog",
        "snapshot_id": "test-snapshot",
        "source_ids": ["cohort-measured"],
        "runtime_defaults": {"agent": profile, "grader": profile},
        "inputs": [_census_entry("cohort-measured", path)],
        "local_disk_root": "/tmp/never-verified",
    }
    survey = build_survey(manifest, _unit_checkers(), "rev")
    assert survey.local_disk_pool is None


def test_merge_reconciles_binding_and_rejects_bad_shards(tmp_path):
    first = _shard_survey(tmp_path, "first", ["cohort-a"])
    second = _shard_survey(tmp_path, "second", ["cohort-b"])
    merged = merge_surveys([first, second], catalog_root="selected:catalog", snapshot_id="selected-snapshot")
    assert [source.source_id for source in merged.sources] == ["cohort-a", "cohort-b"]
    assert merged.snapshot_id == "selected-snapshot"
    assert merged.matrix == first.matrix
    duplicate = _shard_survey(tmp_path, "third", ["cohort-a", "cohort-y"])
    with pytest.raises(ValueError, match="more than one shard"):
        merge_surveys([first, duplicate], catalog_root="selected:catalog", snapshot_id="selected-snapshot")
    drifted = _shard_survey(tmp_path, "drift", ["cohort-c"], checkers=_rejecting_checkers())
    with pytest.raises(ValueError, match="survey identity"):
        merge_surveys([first, drifted], catalog_root="selected:catalog", snapshot_id="selected-snapshot")


def test_merge_preserves_globally_unassessed_sources(tmp_path):
    first = _shard_survey(tmp_path, "first", ["cohort-a", "cohort-z"], measured_ids=["cohort-a"])
    second = _shard_survey(tmp_path, "second", ["cohort-b", "cohort-z"], measured_ids=["cohort-b"])
    merged = merge_surveys([first, second], catalog_root="selected:catalog", snapshot_id="selected-snapshot")
    rows = {source.source_id: source for source in merged.sources}
    assert rows["cohort-z"].basis == "unassessed"
    assert rows["cohort-z"].reason == "No converted population is bound to this source in the test fixture."
    assert rows["cohort-z"].task_count == 0
    assert rows["cohort-a"].basis == "converted_spec_census"
    assert rows["cohort-b"].basis == "converted_spec_census"
    measured_overrides = _shard_survey(tmp_path, "later", ["cohort-a", "cohort-z"], measured_ids=["cohort-a"])
    merged = merge_surveys(
        [measured_overrides, second], catalog_root="selected:catalog", snapshot_id="selected-snapshot"
    )
    rows = {source.source_id: source for source in merged.sources}
    assert rows["cohort-a"].basis == "converted_spec_census"
    assert rows["cohort-z"].basis == "unassessed"


def test_merge_cli_writes_the_bound_artifact(tmp_path, monkeypatch, capsys):
    first = _shard_survey(tmp_path, "first", ["cohort-a"])
    second = _shard_survey(tmp_path, "second", ["cohort-b"])
    for index, survey in enumerate((first, second), start=1):
        (tmp_path / f"shard-{index}.json").write_text(survey.model_dump_json())
    output = tmp_path / "backend-map.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "backend_capabilities",
            "--merge",
            str(tmp_path / "shard-1.json"),
            str(tmp_path / "shard-2.json"),
            "--catalog-root",
            "selected:catalog",
            "--snapshot-id",
            "selected-snapshot",
            "--output",
            str(output),
        ],
    )
    main()
    survey = BackendSurvey.model_validate_json(output.read_text())
    assert survey.snapshot_id == "selected-snapshot"
    assert len(survey.sources) == 2
    assert json.loads(capsys.readouterr().out)["shards"] == 2


def test_merge_projects_historical_spellings_before_dedup(tmp_path, monkeypatch):
    canonical = _shard_survey(tmp_path, "first", ["HuggingFaceH4:aime24"])
    (tmp_path / "shard-1.json").write_text(canonical.model_dump_json())
    historical = _shard_survey(tmp_path, "second", ["MarinSkyRL:aime24"])
    document = json.loads(historical.model_dump_json())
    # An older producer recorded the same source under its original spelling.
    document["sources"][0]["source_id"] = "MarinSkyRL:aime24"
    (tmp_path / "shard-2.json").write_text(json.dumps(document))
    output = tmp_path / "backend-map.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "backend_capabilities",
            "--merge",
            str(tmp_path / "shard-1.json"),
            str(tmp_path / "shard-2.json"),
            "--catalog-root",
            "selected:catalog",
            "--snapshot-id",
            "selected-snapshot",
            "--output",
            str(output),
        ],
    )
    with pytest.raises(ValueError, match="more than one shard"):
        main()


def test_actor_capacity_preserves_independent_separate_grader_profile():
    environment = {"command_semantics": "linux_process", "docker_image": "example/image@sha256:" + "0" * 64}
    task = {
        "id": "independent",
        "environment_requirements": environment,
        "grader": {"environment": environment},
        "execution_requirements": {
            "cpus": 2,
            "memory_mb": 4096,
            "storage_mb": 10240,
            "agent_timeout_sec": 900,
            "verifier_timeout_sec": 600,
        },
    }
    task["verifier_execution_requirements"] = {
        **task["execution_requirements"],
        "cpus": 4,
        "memory_mb": 8192,
        "storage_mb": 20480,
    }
    runtime = {
        "task_machine": {"cpus": 2, "memory_mb": 4096, "storage_mb": 10240},
        "verifier_machine": {"network": "deny"},
    }
    document = {"task": task, "runtime": runtime}
    checkers = {"docker": DockerMachineFactory().check_environment}
    mapped = source_map(
        "independent", [document], capability_matrix(checkers), checkers, {}, {"agent": RUNTIME, "grader": RUNTIME}
    )
    assert mapped.resource_ranges["agent"]["memory_mb"] == (4096, 4096)
    assert mapped.resource_ranges["grader"]["memory_mb"] == (8192, 8192)

    conflicting = {"task": task, "runtime": {**runtime, "verifier_machine": {"memory_mb": 16384}}}
    with pytest.raises(ValueError, match="memory_mb conflicts"):
        source_map(
            "conflicting",
            [conflicting],
            capability_matrix(checkers),
            checkers,
            {},
            {"agent": RUNTIME, "grader": RUNTIME},
        )


def test_production_matrix_records_local_runtime_and_startup_evidence():
    """LOCAL file transfers and command deadlines carry production trial evidence; the
    runtime startup deadline is rolloutengine-owned wherever the validator accepted it."""
    checkers = _unit_checkers()
    cells = {(cell.backend, cell.requirement): cell for cell in production_matrix(checkers)}
    for requirement in (Requirement.FILE_UPLOAD, Requirement.COMMAND_TIMEOUT):
        assert cells["local", requirement].support == Support.SUPPORTED
        # the offline survey cannot execute a runtime probe on docker; a live probe grades it
        assert cells["docker", requirement].support == Support.VERIFY
    for backend in ("docker", "local"):
        # The validator accepted the probe, so its measured PARTIAL downgrade is replaced.
        assert cells[backend, Requirement.STARTUP_TIMEOUT].support == Support.SUPPORTED, backend
    # A validator that rejected the startup probe keeps its measured rejection.
    assert cells["qemu", Requirement.STARTUP_TIMEOUT].support == Support.UNSUPPORTED


def _unit_checkers() -> dict[str, Checker]:
    """Real Docker and LOCAL checks; optional backends reject without their CI-absent packages."""

    def rejecting(requirements, spec) -> None:
        raise UnsupportedMachineSpec("The unit lane installs no optional backends")

    return {
        "docker": DockerMachineFactory().check_environment,
        "local": check_local_environment,
        "docker-gvisor": rejecting,
        "qemu": rejecting,
        "shellsim": rejecting,
        "daytona": rejecting,
        "iris": rejecting,
    }


def _write_spec_parquet(path: Path, documents: list[dict]) -> None:
    table = pa.Table.from_pylist(
        [
            {"task_spec": json.dumps(document["task"]), "runtime": json.dumps(document.get("runtime") or {})}
            for document in documents
        ]
    )
    pq.write_table(table, path)


def test_runtime_column_applies_published_runtime_allocations(tmp_path):
    task = {
        "id": "budgeted",
        "environment_requirements": {
            "command_semantics": "linux_process",
            "docker_image": "example/image@sha256:" + "0" * 64,
        },
        "grader": {"kind": "none"},
    }
    runtime = {"task_machine": {"cpus": 2, "memory_mb": 4096, "storage_mb": 10240}, "verifier_machine": {}}
    path = tmp_path / "specs.parquet"
    _write_spec_parquet(path, [{"task": task, "runtime": runtime}])
    profile = RUNTIME.model_dump(mode="json")
    manifest = {
        "catalog_root": "test:catalog",
        "snapshot_id": "test-snapshot",
        "source_ids": ["cohort"],
        "runtime_defaults": {"agent": profile, "grader": profile},
        "inputs": [
            {
                "source_id": "cohort",
                "paths": [str(path)],
                "column": "task_spec",
                "runtime_column": "runtime",
                "expected_tasks": 1,
            }
        ],
    }
    survey = build_survey(manifest, _unit_checkers(), "rev")
    measured = survey.sources[0]
    assert measured.task_count == 1
    assert measured.resource_ranges["agent"]["storage_mb"] == (10240, 10240)
    assert measured.demands["agent"][Requirement.STORAGE] == 1
    assert measured.backends["docker"].task_counts == {Support.VERIFY: 1}


def test_unclassified_capability_is_verify_not_unsupported():
    """A capability outside the survey vocabulary is a vocabulary gap, not a backend gap."""
    document = _pdb_grader_document()
    document["task"]["environment_requirements"] = {
        "command_semantics": "linux_process",
        "docker_image": PDB_SHAPE,
        "capabilities": ("nemotron-agent:ns_tools_simple_agent",),
    }
    checkers = _verified_local_checkers()
    mapped = source_map(
        "nemotron-math-tir",
        [document],
        production_matrix(checkers),
        checkers,
        {},
        {"agent": RUNTIME, "grader": RUNTIME},
    )
    for backend, fit in mapped.backends.items():
        # the agent role is VERIFY on every backend; the overall fit may still
        # be worse where the grader environment kind is not hostable
        assert fit.roles["agent"] == {FitSupport.VERIFY: 1}, backend
        assert fit.blockers.get("agent:nemotron-agent:ns_tools_simple_agent") == 1
    for backend in ("daytona", "iris", "local"):
        assert mapped.backends[backend].support == FitSupport.VERIFY, backend


def _delivery_document(contract_id: str | None, *, grader_build: bool = True, agent_build: bool = False) -> dict:
    """A Multi/RST-shaped task: contract-delivered roles with shipped build contexts."""
    dockerfile = inline_resource("Dockerfile", b"FROM busybox:1.36\nCOPY lock .\n")
    build = {"files": [dockerfile.model_dump(mode="json")]}
    contract = (
        {"id": contract_id, "version": 1, "dependency_state": "agent_mutable"} if contract_id is not None else None
    )
    if contract_id == "root-delivery":
        contract["verifier_files"] = [{"name": "test.sh", "mode": 493, "sha256": "0" * 64}]
    if contract_id == "multi-swe-delivery":
        contract["archive_sha256"] = "0" * 64
    agent_environment = (
        {
            "command_semantics": "linux_process",
            "docker_build": build,
        }
        if agent_build
        else {"command_semantics": "linux_process", "docker_image": PDB_SHAPE}
    )
    grader_environment = (
        {"command_semantics": "linux_process", "docker_build": build}
        if grader_build
        else {
            "command_semantics": "linux_process",
            "docker_image": PDB_SHAPE,
        }
    )
    return {
        "task": {
            "environment_requirements": agent_environment,
            "grader": {
                "environment": grader_environment,
                "timeout": 600,
            },
            "integrity_contract": contract,
        },
        "runtime": {},
    }


@pytest.mark.parametrize("contract_id", ["root-delivery", "multi-swe-delivery"])
@pytest.mark.parametrize("agent_user", [None, "root", "0", "agent"])
def test_delivery_fit_requires_another_user_only_for_a_demoted_actor(contract_id, agent_user):
    document = _delivery_document(contract_id, grader_build=False)
    document["task"]["integrity_contract"]["agent_user"] = agent_user
    checkers = _unit_checkers()
    mapped = source_map(
        "delivered-root",
        [document],
        production_matrix(checkers),
        checkers,
        {},
        {"agent": RUNTIME, "grader": RUNTIME},
    )
    fit = mapped.backends["docker"]
    assert Requirement.USER not in mapped.demands["grader"]
    assert fit.blockers.get("grader:non_root_user", 0) == 0
    if agent_user == "agent":
        assert fit.blockers["agent:non_root_user"] == 1
        assert fit.roles["agent"] == {FitSupport.PARTIAL: 1}
    else:
        assert Requirement.USER not in mapped.demands["agent"]
        assert fit.blockers.get("agent:non_root_user", 0) == 0
        assert fit.support is not FitSupport.PARTIAL


def test_multi_shaped_contract_rows_resolve_through_the_shipped_build():
    """A complete multi-swe delivery contract resolves the grader's shipped build."""
    checkers = _unit_checkers()
    mapped = source_map(
        "multi-shaped",
        [_delivery_document("multi-swe-delivery") for _ in range(3)],
        production_matrix(checkers),
        checkers,
        {},
        {"agent": RUNTIME, "grader": RUNTIME},
        None,
        {"absent_contract_tasks": 0, "groups": [{"id": "multi-swe-delivery", "present": True, "count": 3}]},
    )
    docker = mapped.backends["docker"]
    # the shipped build resolves: neither role carries the docker_build refusal
    assert FitSupport.UNSUPPORTED not in docker.roles["agent"]
    assert FitSupport.UNSUPPORTED not in docker.roles["grader"]
    assert not any("Unresolved Docker build context" in error for error in docker.validation_errors)


def test_rst_shaped_contract_rows_resolve_both_roles():
    """A complete root-delivery contract resolves agent and grader shipped builds."""
    checkers = _unit_checkers()
    mapped = source_map(
        "rst-shaped",
        [_delivery_document("root-delivery", agent_build=True) for _ in range(3)],
        production_matrix(checkers),
        checkers,
        {},
        {"agent": RUNTIME, "grader": RUNTIME},
        None,
        {"absent_contract_tasks": 0, "groups": [{"id": "root-delivery", "present": True, "count": 3}]},
    )
    docker = mapped.backends["docker"]
    assert FitSupport.UNSUPPORTED not in docker.roles["agent"]
    assert FitSupport.UNSUPPORTED not in docker.roles["grader"]
    assert not any("Unresolved Docker build context" in error for error in docker.validation_errors)


def test_docker_build_without_a_contract_still_fails_closed():
    checkers = _unit_checkers()
    mapped = source_map(
        "uncovered",
        [_delivery_document(None) for _ in range(3)],
        production_matrix(checkers),
        checkers,
        {},
        {"agent": RUNTIME, "grader": RUNTIME},
    )
    docker = mapped.backends["docker"]
    assert docker.roles["grader"] == {FitSupport.UNSUPPORTED: 3}
    assert any("Unresolved Docker build context" in error for error in docker.validation_errors)


def test_partial_contract_fails_closed():
    """A role covered on some tasks only keeps the refusal for every task."""
    checkers = _unit_checkers()
    documents = [_delivery_document("root-delivery"), _delivery_document(None), _delivery_document("root-delivery")]
    mapped = source_map(
        "partial",
        documents,
        production_matrix(checkers),
        checkers,
        {},
        {"agent": RUNTIME, "grader": RUNTIME},
        None,
        {"absent_contract_tasks": 1, "groups": [{"id": "root-delivery", "present": True, "count": 2}]},
    )
    docker = mapped.backends["docker"]
    assert docker.roles["grader"] == {FitSupport.UNSUPPORTED: 3}


def _production_checkers() -> dict[str, Checker]:
    """Every surveyed backend with the real production checks; the CI-absent dev-only backends reject."""

    def rejecting(requirements, spec) -> None:
        raise UnsupportedMachineSpec("The unit lane installs no optional backends")

    return {
        "docker": DockerMachineFactory().check_environment,
        "local": check_local_environment,
        "docker-gvisor": rejecting,
        "qemu": rejecting,
        "shellsim": rejecting,
        "daytona": DaytonaMachineFactory().check_environment,
        "iris": IrisMachineFactory(controller_url="http://survey.invalid").check_environment,
    }


def _kind_documents() -> dict[str, dict]:
    """One converted task per environment kind the converted roles resolve to."""
    dockerfile = inline_resource("Dockerfile", b"FROM busybox:1.36\nCOPY lock .\n")
    build = {"files": [dockerfile.model_dump(mode="json")]}
    environments = {
        "docker_image": {"command_semantics": "linux_process", "docker_image": PDB_SHAPE},
        "docker_build": {"command_semantics": "linux_process", "docker_build": build},
        "host_image": {"command_semantics": "linux_process", "packages_lock": "grade.lock"},
        "package_spec": {"command_semantics": "linux_process", "package_spec": {"pypi_packages": ["numpy==2.3.5"]}},
    }
    return {
        kind: {"environment_requirements": environment, "grader": {"kind": "none"}}
        for kind, environment in environments.items()
    }


def test_every_environment_kind_on_every_production_backend_labels_every_blocker():
    """R41 (R40 item 47): the derivation labels every blocker the evidence page renders.

    Derives fit over every environment kind (Docker image, Docker build, host image,
    package spec) on every production backend and asserts each blocker carries a
    reason the served taxonomy labels, so the Backend evidence page can never meet
    an unmapped reason (R40's crash class, found only at the final browser sweep of
    140 of 321 sources).
    """
    served_reasons = set(taxonomy_tables([requirement.value for requirement in Requirement])["backend-reason"])
    checkers = _production_checkers()
    matrix = production_matrix(checkers)
    for kind, document in _kind_documents().items():
        mapped = source_map(f"kind-{kind}", [document], matrix, checkers, {}, {"agent": RUNTIME, "grader": RUNTIME})
        for backend, fit in mapped.backends.items():
            stored = {**fit.blockers, **fit.not_applicable}
            assert set(fit.blocker_reasons) == set(stored), (kind, backend, fit.blocker_reasons, stored)
            assert set(fit.blocker_reasons.values()) <= served_reasons, (kind, backend, fit.blocker_reasons)
            # The page derives the display fit: the stored keys plus the
            # not_applicable environment kinds, spelled as the R40 crash key.
            display = {**fit.blockers}
            for key, count in fit.not_applicable.items():
                role, _, environment_kind_value = key.partition(":")
                display[f"{role}:unsupported_environment_kind:{environment_kind_value}"] = count
            for key in display:
                emitted = fit.blocker_reasons.get(key)
                capability = ":".join(key.split(":")[1:])
                if emitted and emitted in served_reasons:
                    continue
                assert (
                    capability == "required_backend"
                    or capability.startswith("unsupported_environment_kind:")
                    or capability in served_reasons
                ), (kind, backend, key)
            # The page's not-measured and incomplete-support blockers stay
            # unreachable: both roles are fully measured and partial support
            # always carries its refused demand.
            for role in ("agent", "grader"):
                assert sum(fit.roles.get(role, {}).values()) == mapped.task_count, (kind, backend, role)
            if fit.support == FitSupport.PARTIAL:
                assert stored, (kind, backend)
        # the derived survey passes the catalog load's fail-closed verification
        verify_backend_blocker_reasons(_survey_with_source(mapped, checkers))


def test_unclassified_capability_blockers_carry_the_canonical_reason():
    """An unclassified capability renders as the taxonomy's reason, not a crash."""
    document = {
        "environment_requirements": {
            "command_semantics": "linux_process",
            "capabilities": ["nemotron-agent:ns_tools_simple_agent"],
        },
        "grader": {"kind": "none"},
    }
    checkers = _unit_checkers()
    mapped = source_map(
        "unclassified", [document], production_matrix(checkers), checkers, {}, {"agent": RUNTIME, "grader": RUNTIME}
    )
    fit = mapped.backends["docker"]
    assert fit.blockers == {"agent:nemotron-agent:ns_tools_simple_agent": 1}
    assert fit.blocker_reasons == {"agent:nemotron-agent:ns_tools_simple_agent": "unclassified-capability"}
    verify_backend_blocker_reasons(_survey_with_source(mapped, checkers))


def _host_image_survey(checkers: dict[str, Checker]) -> tuple[BackendSurvey, object]:
    """The aime24-shaped grader-only host-image population, derived and sealed."""
    documents = [
        {
            "task": {
                "grader": {
                    "environment": {"command_semantics": "linux_process", "packages_lock": "math.lock"},
                    "timeout": 600,
                },
                "resources": {"verifier": [{"path": "grader-files.tar", "bytes": "AAAA"}]},
            },
            "runtime": {},
        }
        for _ in range(3)
    ]
    mapped = source_map(
        "aime24", documents, production_matrix(checkers), checkers, {}, {"agent": RUNTIME, "grader": RUNTIME}
    )
    return _survey_with_source(mapped, checkers), mapped


def test_blocker_reason_verification_refuses_an_unlabelled_blocker():
    checkers = _unit_checkers()
    survey, mapped = _host_image_survey(checkers)
    verify_backend_blocker_reasons(survey)
    docker_fit = mapped.backends["docker"]
    assert docker_fit.not_applicable == {"grader:host_image": 3}
    stripped = _survey_with_source(
        mapped.model_copy(
            update={"backends": {**mapped.backends, "docker": docker_fit.model_copy(update={"blocker_reasons": {}})}}
        ),
        checkers,
    )
    with pytest.raises(ValueError, match="no mapped reason"):
        verify_backend_blocker_reasons(stripped)


def test_blocker_reason_verification_refuses_a_reason_outside_the_taxonomy():
    checkers = _unit_checkers()
    _survey, mapped = _host_image_survey(checkers)
    docker_fit = mapped.backends["docker"]
    relabelled = _survey_with_source(
        mapped.model_copy(
            update={
                "backends": {
                    **mapped.backends,
                    "docker": docker_fit.model_copy(update={"blocker_reasons": {"grader:host_image": "mystery-reason"}}),
                }
            }
        ),
        checkers,
    )
    with pytest.raises(ValueError, match="outside the served taxonomy"):
        verify_backend_blocker_reasons(relabelled)


def test_blocker_reason_verification_refuses_an_unmeasured_role():
    checkers = _unit_checkers()
    _survey, mapped = _host_image_survey(checkers)
    docker_fit = mapped.backends["docker"]
    unmeasured = _survey_with_source(
        mapped.model_copy(
            update={"backends": {**mapped.backends, "docker": docker_fit.model_copy(update={"roles": {}})}}
        ),
        checkers,
    )
    with pytest.raises(ValueError, match="unmeasured"):
        verify_backend_blocker_reasons(unmeasured)


def test_blocker_reason_verification_refuses_partial_support_without_blockers():
    checkers = _unit_checkers()
    _survey, mapped = _host_image_survey(checkers)
    local_fit = mapped.backends["local"]
    assert local_fit.blockers == {} and local_fit.not_applicable == {}
    partial = _survey_with_source(
        mapped.model_copy(
            update={
                "backends": {**mapped.backends, "local": local_fit.model_copy(update={"support": FitSupport.PARTIAL})}
            }
        ),
        checkers,
    )
    with pytest.raises(ValueError, match="incomplete-support"):
        verify_backend_blocker_reasons(partial)

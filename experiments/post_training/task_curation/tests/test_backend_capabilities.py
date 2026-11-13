# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Schema evolution and the production lock-grader incompatibility."""

import ast
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import get_args, get_type_hints

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from rolloutengine.models import MachineRuntimeSpec
from shellbox.backends.docker.machine import DockerMachineFactory
from shellbox.backends.local.machine import check_local_environment
from shellbox.machine import Command, MachineSpec, Mount, UnsupportedMachineSpec
from shellbox.requirements import (
    COMMAND_FIELDS,
    MACHINE_FIELDS,
    SOURCE_FIELDS,
    Requirement,
    Support,
)
from shellbox.survey import Checker, capability_matrix
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
from taskcompendium.backend_requirements import BackendSurvey, RuntimeProfile
from taskcompendium.models import EnvironmentRequirements, ExecutionRequirements, ScriptGrader, TaskResource, TaskSpec

from experiments.post_training.task_curation.backend_capabilities import (
    build_survey,
    main,
    merge_surveys,
    source_map,
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
    assert all(fit.roles["grader"] == {Support.UNSUPPORTED: 1} for fit in before.backends.values())
    assert before.demands["grader"][Requirement.HOST_IMAGE] == 1
    assert before.resource_ranges["grader"]["storage_mb"] == (1024, 1024)
    assert before.role_backend_assignments == {"agent": {"daytona": 1}, "grader": {"local": 1}}
    document["runtime"]["verifier_machine"]["storage_mb"] = None
    after = source_map("cohort-v2", [document], matrix, checkers, {}, defaults)
    assert after.backends["local"].roles["grader"] == {Support.PARTIAL: 1}
    assert after.backends["docker"].roles["agent"] == {Support.PARTIAL: 1}
    assert after.backends["docker"].roles["grader"] == {Support.UNSUPPORTED: 1}
    assert after.backends["local"].roles["agent"] == {Support.UNSUPPORTED: 1}


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
    assert mapped.backends["docker"].task_counts == {Support.SUPPORTED: 1}


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
    assert fit.task_counts == {Support.UNSUPPORTED: 1}
    assert fit.roles["agent"] == {Support.UNSUPPORTED: 1}
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
    assert fit.task_counts == {Support.UNSUPPORTED: 1}
    assert fit.roles["agent"] == {Support.UNSUPPORTED: 1}
    assert fit.roles["grader"] == {Support.UNSUPPORTED: 1}
    assert fit.blockers == {
        "agent:nemotron-agent:ns_tools_simple_agent": 1,
        "grader:cleanup_timeout": 1,
        "grader:startup_timeout": 1,
        "grader:host_image": 1,
        "grader:packages_lock": 1,
    }
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

    return {name: rejecting for name in ("docker", "gvisor", "qemu", "shellsim", "daytona", "iris", "local")}


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
    runtime = {
        "task_machine": {"cpus": 2, "memory_mb": 4096, "storage_mb": 10240},
        "verifier_machine": {"cpus": 4, "memory_mb": 8192, "storage_mb": 20480},
    }
    document = {"task": task, "runtime": runtime}
    checkers = {"docker": DockerMachineFactory().check_environment}
    mapped = source_map(
        "independent", [document], capability_matrix(checkers), checkers, {}, {"agent": RUNTIME, "grader": RUNTIME}
    )
    assert mapped.resource_ranges["agent"]["memory_mb"] == (4096, 4096)
    assert mapped.resource_ranges["grader"]["memory_mb"] == (8192, 8192)


def _unit_checkers() -> dict[str, Checker]:
    """Real Docker and LOCAL checks; optional backends reject without their CI-absent packages."""

    def rejecting(requirements, spec) -> None:
        raise UnsupportedMachineSpec("The unit lane installs no optional backends")

    return {
        "docker": DockerMachineFactory().check_environment,
        "local": check_local_environment,
        "gvisor": rejecting,
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
    assert measured.backends["docker"].task_counts == {Support.PARTIAL: 1}

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Contracts for source isolation and behavioral acceptance."""

import asyncio
import hashlib
import json
import subprocess
import sys
from argparse import Namespace
from dataclasses import asdict

import pytest
from shellbox.machine import ExitReason, NetworkPolicy, Result
from taskcompendium.environment import ShellVerifierSpec
from taskcompendium.parquet import read_tasks

from experiments.post_training.russell_rsi.corpus import CommitRecord
from experiments.post_training.russell_rsi.sources import SourceSnapshot, source_group_id
from experiments.post_training.russell_rsi.tasks import (
    CONTROL_SOURCE,
    PROBE_RUNNER,
    GeneratedRepair,
    InvalidRepair,
    ObservationCase,
    VerifierReport,
    accept_candidate,
    accept_candidates,
    build_task,
    control_files,
    generate_candidates,
    generation_request,
    source_partition,
    write_partitions,
)


def seed():
    return SourceSnapshot(
        repository="example/math",
        parent_sha="a" * 40,
        commit_sha="b" * 40,
        parent_files={"maths.py": "def add(a,b): return a-b\n", "LICENSE": "MIT license"},
        reference_files={"maths.py": "def add(a,b): return a+b\n", "LICENSE": "MIT license"},
        license_paths=("LICENSE",),
        split="train",
    )


def repair():
    return GeneratedRepair(
        problem_statement="Repair addition for positive integers.",
        cases=(
            ObservationCase(probe_python="from maths import add\nobservation = add(2,3)\n", expected_json=5),
            ObservationCase(probe_python="from maths import add\nobservation = add(-2,4)\n", expected_json=2),
        ),
        editable_paths=("maths.py",),
    )


def test_task_solver_contains_only_parent_and_private_tests_stay_in_verifier():
    snapshot = seed()
    task = build_task(snapshot, repair(), image="python-git", timeout=10)
    solver = (
        task.environment.model_dump_json()
        + task.context.model_dump_json()
        + task.id
        + json.dumps(task.metadata)
        + "".join(file.content.decode() for file in task.environment.files)
    )
    assert "return a-b" in solver
    assert "return a+b" not in solver
    assert "observation = add" not in solver
    assert snapshot.commit_sha not in solver
    verifier = ShellVerifierSpec.model_validate_json(task.verifier.parameters_json)
    assert any(b"observation = add" in file.content for file in verifier.files)
    assert verifier.environment == task.environment
    assert not task.environment.network


class ResultMachine:
    def __init__(self, metrics):
        self.metrics = VerifierReport(
            tests=metrics["tests"],
            failures=metrics["failures"],
            errors=metrics["errors"],
            observations=tuple(
                metrics.get("observations", [None if metrics["errors"] else (4 if metrics["failures"] else 5), 2])
            ),
            case_errors=("execution_error" if metrics["errors"] else None, None),
            case_diagnostics=("", ""),
        )
        self.files = {}
        self.closed = False

    async def upload(self, source, target):
        self.files[target] = source.read_bytes()

    async def run(self, command):
        if command.argv != ("python", "-I", "/tmp/taskcompendium/runner.py"):
            return Result(0, b"", b"", False, False, ExitReason.EXITED)
        successful = self.metrics.tests > 0 and self.metrics.failures == self.metrics.errors == 0
        return Result(
            0 if successful else 1,
            ("RSI_RESULT=" + self.metrics.model_dump_json()).encode(),
            b"",
            False,
            False,
            ExitReason.EXITED,
        )

    async def close(self):
        self.closed = True


class ResultFactory:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.machines = []
        self.specs = []

    async def create(self, spec):
        self.specs.append(spec)
        machine = ResultMachine(next(self.outcomes))
        self.machines.append(machine)
        return machine


@pytest.mark.parametrize(
    "parent,reference,accepted",
    [
        ({"tests": 2, "failures": 1, "errors": 0}, {"tests": 2, "failures": 0, "errors": 0}, True),
        ({"tests": 2, "failures": 0, "errors": 1}, {"tests": 2, "failures": 0, "errors": 0}, False),
        ({"tests": 2, "failures": 0, "errors": 0}, {"tests": 2, "failures": 0, "errors": 0}, False),
        ({"tests": 2, "failures": 1, "errors": 0}, {"tests": 2, "failures": 1, "errors": 0}, False),
    ],
)
def test_acceptance_requires_behavioral_failure_and_reference_success(parent, reference, accepted):
    snapshot = seed()
    task = build_task(snapshot, repair(), image="python-git", timeout=10)
    factory = ResultFactory([parent, parent, reference, reference])
    result = asyncio.run(accept_candidate(task, snapshot, factory=factory))
    assert result.accepted is accepted
    assert len(factory.machines) == 4
    assert all(machine.closed for machine in factory.machines)
    assert all(spec.network == NetworkPolicy.DENY for spec in factory.specs)
    assert all(b"return a+b" not in machine.files["/workspace/maths.py"] for machine in factory.machines[:2])
    assert all(b"return a+b" in machine.files["/workspace/maths.py"] for machine in factory.machines[2:])
    assert all("/tmp/taskcompendium/cases.json" in machine.files for machine in factory.machines)


def stored_generation(directory, snapshot, summary, content):
    directory.mkdir(parents=True)
    request = generation_request(snapshot, failure_summary=summary, max_tokens=1024)
    (directory / "snapshot.json").write_text(snapshot.model_dump_json())
    (directory / "generation.json").write_text(
        json.dumps(
            {
                "request": request,
                "request_sha256": hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest(),
                "relay_job": "/relay",
                "response": {"choices": [{"message": {"content": content}}]},
            }
        )
    )
    return Namespace(
        output=directory.parent,
        relay_job="/relay",
        token_env="UNAVAILABLE_TOKEN",
        failure_summary=summary,
        max_tokens=1024,
        max_candidates=2,
    )


def test_generation_resume_rejects_changed_request_without_overwriting_source(tmp_path):
    snapshot = seed()
    directory = tmp_path / "candidates" / source_group_id(snapshot)
    args = stored_generation(directory, snapshot, "addition", repair().model_dump_json())
    args.snapshots = tmp_path / "snapshots.jsonl"
    args.snapshots.write_text(snapshot.model_dump_json() + "\n")
    asyncio.run(generate_candidates(args))
    original = (directory / "snapshot.json").read_bytes()
    args.failure_summary = "multiplication"
    with pytest.raises(ValueError, match="request identity"):
        asyncio.run(generate_candidates(args))
    assert (directory / "snapshot.json").read_bytes() == original
    assert GeneratedRepair.model_validate_json((directory / "repair.json").read_text()) == repair()


def test_generation_rejects_bad_schema_and_continues_to_next_candidate(tmp_path):
    first = seed()
    second = first.model_copy(update={"commit_sha": "c" * 40})
    args = stored_generation(tmp_path / "candidates" / source_group_id(first), first, "addition", '{"invalid": true}')
    stored_generation(tmp_path / "candidates" / source_group_id(second), second, "addition", repair().model_dump_json())
    args.snapshots = tmp_path / "snapshots.jsonl"
    args.snapshots.write_text(first.model_dump_json() + "\n" + second.model_dump_json() + "\n")
    asyncio.run(generate_candidates(args))
    assert json.loads((args.output / source_group_id(first) / "generation_rejected.json").read_text())["rejected"]
    assert (
        GeneratedRepair.model_validate_json((args.output / source_group_id(second) / "repair.json").read_text())
        == repair()
    )


def test_distinct_source_scopes_at_one_commit_keep_separate_candidates(tmp_path):
    first = seed()
    second = first.model_copy(
        update={
            "parent_files": {"other.py": "def add(a,b): return a-b\n", "LICENSE": "MIT license"},
            "reference_files": {"other.py": "def add(a,b): return a+b\n", "LICENSE": "MIT license"},
        }
    )
    repairs = [
        repair(),
        repair().model_copy(
            update={
                "editable_paths": ("other.py",),
                "cases": tuple(
                    case.model_copy(
                        update={"probe_python": case.probe_python.replace("from maths import", "from other import")}
                    )
                    for case in repair().cases
                ),
            }
        ),
    ]
    for snapshot, generated in zip((first, second), repairs, strict=True):
        args = stored_generation(
            tmp_path / "candidates" / source_group_id(snapshot), snapshot, "addition", generated.model_dump_json()
        )
    args.snapshots = tmp_path / "snapshots.jsonl"
    args.snapshots.write_text(first.model_dump_json() + "\n" + second.model_dump_json() + "\n")
    asyncio.run(generate_candidates(args))
    outputs = sorted(args.output.glob("*/repair.json"))
    assert len(outputs) == 2
    tasks = [
        build_task(snapshot, generated, image="python-git", timeout=10)
        for snapshot, generated in zip((first, second), repairs, strict=True)
    ]
    assert tasks[0].id != tasks[1].id
    assert tasks[0].source.row != tasks[1].source.row


def source_record(snapshot, family):
    return CommitRecord(
        sha=snapshot.commit_sha,
        tree_sha="d" * 40,
        author={},
        author_login="example-author",
        committer={},
        message="Repair addition",
        parents=[snapshot.parent_sha],
        repositories=[snapshot.repository],
        discovery_queries=[],
        family=family,
        split=snapshot.split,
    )


def test_partitions_obey_inventory_families_and_keep_test_sources_sealed(tmp_path):
    train = seed()
    development = train.model_copy(update={"repository": "example/dev", "commit_sha": "c" * 40, "split": "dev"})
    sealed = train.model_copy(update={"repository": "example/test", "commit_sha": "d" * 40, "split": "test"})
    snapshots = (train, development, sealed)
    inventory = {snapshot.commit_sha: source_record(snapshot, snapshot.repository) for snapshot in snapshots}
    pairs = []
    for snapshot in snapshots:
        task = build_task(snapshot, repair(), image="python-git", timeout=10)
        pairs.append((snapshot, task.model_copy(update={"metadata": {"split": "train", "family": "forged"}})))
    write_partitions(tmp_path, pairs, inventory)
    training = list(read_tasks(str(tmp_path / "train.parquet")))
    development_tasks = list(read_tasks(str(tmp_path / "development.parquet")))
    assert [task.source.row for task in training] == [f"{train.repository}:{train.parent_sha}:{source_group_id(train)}"]
    assert [task.source.row for task in development_tasks] == [
        f"{development.repository}:{development.parent_sha}:{source_group_id(development)}"
    ]
    assert training[0].metadata["family"] == train.repository
    assert all(sealed.repository not in task.source.row for task in training + development_tasks)
    with pytest.raises(ValueError, match="source-family split"):
        source_partition(sealed.model_copy(update={"split": "train"}), inventory[sealed.commit_sha])


def test_acceptance_records_invalid_probe_and_writes_empty_partitions(tmp_path):
    snapshot = seed()
    candidates = tmp_path / "candidates"
    directory = candidates / source_group_id(snapshot)
    directory.mkdir(parents=True)
    (directory / "snapshot.json").write_text(snapshot.model_dump_json())
    truncated = repair().model_copy(
        update={"cases": (ObservationCase(probe_python='value = "truncated', expected_json=5),)}
    )
    (directory / "repair.json").write_text(truncated.model_dump_json())
    inventory = tmp_path / "inventory.jsonl"
    inventory.write_text(json.dumps(asdict(source_record(snapshot, snapshot.repository))) + "\n")
    output = tmp_path / "accepted"
    (tmp_path / "manifest.json").write_text(json.dumps({"repository_wheels": {snapshot.repository: None}}))
    asyncio.run(
        accept_candidates(
            Namespace(
                output=output,
                inventory=inventory,
                backend="docker",
                skopeo=tmp_path / "skopeo",
                image_cache=tmp_path / "cache",
                candidates=candidates,
                max_candidates=1,
                image="python-git",
                timeout=10,
                dependency_bundles=tmp_path,
            )
        )
    )
    report = json.loads((directory / "acceptance.json").read_text())
    assert report["accepted"] is False
    assert report["stage"] == "task_validation"
    assert list(read_tasks(str(output / "train.parquet"))) == []
    assert list(read_tasks(str(output / "development.parquet"))) == []


def test_acceptance_rejects_unstable_behavioral_results():
    snapshot = seed()
    task = build_task(snapshot, repair(), image="python-git", timeout=10)
    failure = {"tests": 2, "failures": 1, "errors": 0}
    success = {"tests": 2, "failures": 0, "errors": 0}
    factory = ResultFactory([failure, success, success, success])
    result = asyncio.run(accept_candidate(task, snapshot, factory=factory))
    assert not result.accepted
    assert all(machine.closed for machine in factory.machines)


@pytest.mark.parametrize("leak", ["b" * 40, "return a + b + sum([0, 0, 0, 0])"])
def test_task_rejects_generated_prompt_with_reference_content(leak):
    snapshot = seed().model_copy(
        update={
            "reference_files": {
                "maths.py": "def add(a,b):\n    return a + b + sum([0, 0, 0, 0])\n",
                "LICENSE": "MIT license",
            }
        }
    )
    generated = repair().model_copy(update={"problem_statement": "Repair addition. " + leak})
    with pytest.raises(InvalidRepair):
        build_task(snapshot, generated, image="python-git", timeout=10)


def test_admission_rejects_changed_observations_with_identical_failure_counts():
    snapshot = seed()
    task = build_task(snapshot, repair(), image="python-git", timeout=10)
    first = {"tests": 2, "failures": 1, "errors": 0, "observations": [4, 2]}
    second = {**first, "observations": [3, 2]}
    success = {"tests": 2, "failures": 0, "errors": 0}
    result = asyncio.run(accept_candidate(task, snapshot, factory=ResultFactory([first, second, success, success])))
    assert not result.accepted


def test_probe_child_runs_real_source_without_expected_values_or_unittest_scoring(tmp_path):
    source = tmp_path / "maths.py"
    source.write_text(CONTROL_SOURCE["unittest_monkeypatch"] + "def add(a, b): return a-b\n")
    case = repair().cases[0]
    result = subprocess.run(
        [sys.executable, "-I", "-c", PROBE_RUNNER],
        input=case.probe_python,
        text=True,
        capture_output=True,
        cwd=tmp_path,
        check=True,
    )
    assert json.loads(result.stdout) == -1
    source.write_text("def add(a, b): return a+b\n")
    result = subprocess.run(
        [sys.executable, "-I", "-c", PROBE_RUNNER],
        input=case.probe_python,
        text=True,
        capture_output=True,
        cwd=tmp_path,
        check=True,
    )
    assert json.loads(result.stdout) == case.expected_json


def test_monkeypatch_control_attacks_new_modules(tmp_path):
    snapshot = seed().model_copy(update={"parent_files": {"LICENSE": "MIT license"}})
    files = control_files(snapshot, "unittest_monkeypatch")
    (tmp_path / "maths.py").write_text(files["maths.py"])
    probe = (
        "import maths, unittest\n"
        "result = unittest.TestResult()\n"
        "result.failures.append(('broken', 'failure'))\n"
        "observation = {'has_add': hasattr(maths, 'add'), 'forged_success': result.wasSuccessful()}\n"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", PROBE_RUNNER],
        input=probe,
        text=True,
        capture_output=True,
        cwd=tmp_path,
        check=True,
    )
    assert json.loads(result.stdout) == {"has_add": False, "forged_success": True}


def test_forged_metrics_are_an_observation_not_a_success_report(tmp_path):
    source = tmp_path / "maths.py"
    source.write_text("import os\n" 'os.write(1, b\'{"tests":1,"failures":0,"errors":0}\')\n' "os._exit(0)\n")
    case = repair().cases[0]
    result = subprocess.run(
        [sys.executable, "-I", "-c", PROBE_RUNNER],
        input=case.probe_python,
        text=True,
        capture_output=True,
        cwd=tmp_path,
        check=True,
    )
    assert json.loads(result.stdout) != case.expected_json


def test_probe_rejects_inprocess_assertions_and_edit_paths_outside_source_scope():
    predicate = repair().model_copy(
        update={
            "cases": (
                ObservationCase(
                    probe_python="from maths import add\nassert add(2,3) == 5\nobservation = True", expected_json=True
                ),
            )
        }
    )
    with pytest.raises(InvalidRepair, match="scoring predicates"):
        build_task(seed(), predicate, image="python-git", timeout=10)
    outside = repair().model_copy(update={"editable_paths": ("outside.py",)})
    with pytest.raises(InvalidRepair, match="source change scope"):
        build_task(seed(), outside, image="python-git", timeout=10)


def test_wheel_dependencies_are_immutable_assets_in_both_machine_environments(tmp_path):
    wheel = tmp_path / "example-1.0-py3-none-any.whl"
    wheel.write_bytes(b"wheel-content")
    task = build_task(
        seed(),
        repair(),
        image="python-git",
        timeout=10,
        dependency_wheels=tmp_path,
        dependency_wheels_uri="s3://example/dependencies/math",
    )
    assert len(task.environment.assets) == 1
    asset = task.environment.assets[0]
    assert asset.uri == "s3://example/dependencies/math/" + wheel.name
    assert asset.sha256 == hashlib.sha256(wheel.read_bytes()).hexdigest()
    assert asset.size_bytes == wheel.stat().st_size
    assert all(file.content != wheel.read_bytes() for file in task.environment.files)
    verifier = ShellVerifierSpec.model_validate_json(task.verifier.parameters_json)
    assert verifier.environment.assets == task.environment.assets

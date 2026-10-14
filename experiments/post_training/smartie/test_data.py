# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from rigging.filesystem.storage_path import StoragePath
from rolloutengine.spec import LoweredTaskSpec
from rolloutengine.test_support import arithmetic_task, lowered, machine_runtime
from shellbox.backends.daytona.machine import DaytonaMachineFactory
from taskcompendium.models import CommandSemantics, EnvironmentRequirements, ScriptGrader

from experiments.post_training.smartie.data import (
    AdmittedInput,
    build_real_dataset,
    convert_admitted_source,
    split_real_tasks,
)
from experiments.post_training.smartie.deployment import ImageDeployment, LockImage
from experiments.post_training.smartie.filter_structural import filter_structural
from experiments.post_training.smartie.mixture import build_mixture


@pytest.mark.parametrize("overlap", [False, True])
def test_mixture_preserves_payloads_and_rejects_cross_split_leakage(tmp_path, overlap):
    real = tmp_path / "real"
    structural = tmp_path / "structural"
    real.mkdir()
    structural.mkdir()
    real_manifest = {
        "failures": [],
        "sources": [{"root": "fixture/calendar"}, {"root": "fixture/competitive_code"}],
    }
    structural_manifest = {"outputs": {}}
    expected = {}
    for split in ("train", "validation"):
        expected[split] = []
        for route, root in (("real", real), ("structural", structural)):
            identities = [f"calendar-{split}", f"code-{split}"] if route == "real" else [f"structural-{split}"]
            if overlap and route == "structural" and split == "validation":
                identities = ["calendar-train"]
            payloads = [lowered(arithmetic_task(id=identity)).model_dump_json() for identity in identities]
            expected[split].extend(payloads)
            path = root / f"{split}.parquet"
            pq.write_table(pa.table({"lowered_task_spec": payloads}), path)
            if route == "real":
                real_manifest[f"{split}_ids"] = identities
                for source, identity in zip(real_manifest["sources"], identities, strict=True):
                    source[f"{split}_ids"] = [identity]
            else:
                structural_manifest["outputs"][split] = {
                    "rows": len(identities),
                    "native_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
    (real / "manifest.json").write_text(json.dumps(real_manifest))
    (structural / "manifest.json").write_text(json.dumps(structural_manifest))
    output = tmp_path / "mixed"
    mixing = {
        "source_routes": {"fixture/calendar": "calendar", "fixture/competitive_code": "competitive_code"},
        "domain_weights": {"calendar": 1.0, "competitive_code": 1.0, "structural": 1.0},
    }
    if overlap:
        with pytest.raises(ValueError, match="overlapping task identities"):
            build_mixture(StoragePath(str(real)), structural, StoragePath(str(output)), **mixing)
        assert not output.exists()
        return
    build_mixture(StoragePath(str(real)), structural, StoragePath(str(output)), **mixing)
    for split in expected:
        actual = pq.read_table(output / f"{split}.parquet").to_pydict()
        assert actual == {
            "lowered_task_spec": expected[split],
            "teacher_route": ["calendar", "competitive_code", "structural"],
        }
    assert (output / "manifest.json").exists()


@pytest.mark.parametrize("broken_native", [False, True])
def test_structural_filter_preserves_holdout_and_removes_text_or_token_matches(tmp_path, broken_native):
    tokens = tmp_path / "tokens"
    native = tmp_path / "native"
    tokens.mkdir()
    native.mkdir()
    definitions = {
        "train": [("text-match", "shared", [1]), ("token-match", "different", [2]), ("keep", "unique", [3])],
        "validation": [("heldout", "shared", [2])],
    }
    for split, cases in definitions.items():
        rows = [
            dict(
                case_id=case,
                task_id=case,
                prompt_sha256=digest,
                prompt_token_ids=ids,
                family="code_fence",
                control="close_required",
            )
            for case, digest, ids in cases
        ]
        (tokens / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        payloads = [arithmetic_task(id="structure-closure:" + case) for case, _, _ in reversed(cases)]
        if broken_native and split == "validation":
            payloads = [arithmetic_task(id="unrelated")]
        pq.write_table(
            pa.table({"lowered_task_spec": [lowered(task).model_dump_json() for task in payloads]}),
            native / f"{split}.parquet",
        )
    output = tmp_path / "filtered"
    if broken_native:
        with pytest.raises(ValueError, match="identity mismatch"):
            filter_structural(tokens, native, output)
        assert not (output / "manifest.json").exists()
        return
    report = filter_structural(tokens, native, output)
    assert report["excluded_train_case_ids"] == ["text-match", "token-match"]
    assert report["outputs"]["train"]["rows"] == 1
    assert json.loads((output / "train.jsonl").read_text())["case_id"] == "keep"
    actual = pq.read_table(output / "train.parquet")["lowered_task_spec"].to_pylist()
    assert [LoweredTaskSpec.model_validate_json(row).task.id for row in actual] == ["structure-closure:keep"]
    assert (output / "validation.parquet").read_bytes() == (native / "validation.parquet").read_bytes()
    assert (output / "validation.jsonl").read_bytes() == (tokens / "validation.jsonl").read_bytes()


def test_lock_image_deployment_preserves_task_contract_and_rejects_changed_lock(tmp_path):
    lock = tmp_path / "requirements.lock"
    lock.write_text("example==1.0\n")
    lock_hash = hashlib.sha256(lock.read_bytes()).hexdigest()
    (tmp_path / ".artifact.json").write_text(
        json.dumps({"result": {"lock_sha256": lock_hash, "data": ["nltk:wordnet"]}})
    )
    environment = EnvironmentRequirements(
        capabilities=("shell", "filesystem"),
        command_semantics=CommandSemantics.LINUX_PROCESS,
        packages_lock=str(lock),
        working_directory="/app",
        setup_commands=("mkdir -p /app/output",),
        environment_variables={"LANG": "C.UTF-8"},
    )
    task = arithmetic_task(grader=ScriptGrader(argv=("python3", "/tests/grade.py"), environment=environment))
    image = LockImage(lock_hash, ("nltk:wordnet",), "registry.example/grader@sha256:" + "a" * 64)
    deployment = ImageDeployment((image,))
    deployed = deployment.task(task)
    expected = task.model_dump()
    expected["grader"]["environment"].update(packages_lock=None, docker_image=image.image)
    assert deployed.model_dump() == expected
    assert task.grader.environment.packages_lock == str(lock)
    assert deployment.evidence == {str(lock): image}
    converted = convert_admitted_source(
        {
            "mode": "full",
            "status": "completed",
            "intended_use": "train",
            "unprocessed_rows": 0,
            "source_dataset": "fixture",
            "source_revision": "1",
            "admission_counts": {"admitted": 1},
        },
        [{"task_id": task.id, "task_json": task.model_dump_json()}],
        machine=machine_runtime(backend="daytona"),
        session=lowered(task).session,
        factories={"daytona": DaytonaMachineFactory()},
        deployment=deployment,
    )
    assert not converted.failures
    assert converted.tasks[0].task == deployed
    with pytest.raises(ValueError, match="Expected one image deployment"):
        ImageDeployment(()).task(task)
    lock.write_text("example==2.0\n")
    with pytest.raises(ValueError, match="differs from its artifact record"):
        ImageDeployment((image,)).task(task)


def test_admitted_tasks_roundtrip_and_split_without_order_leakage():
    tasks = [arithmetic_task(id=f"task-{index}") for index in range(10)]
    rows = [{"task_id": task.id, "task_json": task.model_dump_json()} for task in tasks]
    manifest = {
        "mode": "full",
        "status": "completed",
        "intended_use": "train",
        "unprocessed_rows": 0,
        "source_dataset": "fixture",
        "source_revision": "1",
        "admission_counts": {"admitted": 10},
    }
    converted = convert_admitted_source(
        manifest, rows, machine=machine_runtime(), session=lowered(tasks[0]).session, factories={}
    )
    assert not converted.failures
    restored = tuple(LoweredTaskSpec.model_validate_json(task.model_dump_json()) for task in converted.tasks)
    assert [task.task for task in restored] == tasks
    train, validation = split_real_tasks(restored, validation_count=2, seed=42)
    assert len(train) == 8 and len(validation) == 2
    assert {task.task.id for task in train}.isdisjoint(task.task.id for task in validation)
    assert split_real_tasks(reversed(restored), validation_count=2, seed=42) == (train, validation)
    with pytest.raises(ValueError, match="Duplicate task identities"):
        split_real_tasks((*restored, restored[0]), validation_count=2, seed=42)


def test_conversion_reports_all_invalid_rows_and_population_mismatch():
    task = arithmetic_task()
    manifest = {
        "mode": "full",
        "status": "completed",
        "intended_use": "train",
        "unprocessed_rows": 0,
        "source_dataset": "fixture",
        "source_revision": "1",
        "admission_counts": {"admitted": 4},
    }
    rows = [
        {"task_id": "bad-json", "task_json": "{}"},
        {"task_id": "wrong-id", "task_json": task.model_dump_json()},
        {"task_id": task.id, "task_json": task.model_dump_json()},
    ]
    result = convert_admitted_source(
        manifest, rows, machine=machine_runtime(), session=lowered(task).session, factories={}
    )
    assert [failure.task_id for failure in result.failures] == ["bad-json", "wrong-id", "<source>"]
    assert len(result.tasks) == 1
    with pytest.raises(ValueError, match="completed FULL"):
        convert_admitted_source(
            {**manifest, "mode": "quick"},
            rows,
            machine=machine_runtime(),
            session=lowered(task).session,
            factories={},
        )


@pytest.mark.parametrize("invalid", [False, True])
def test_source_files_publish_native_splits_only_after_all_sources_pass(tmp_path, invalid):
    inputs = []
    expected = {}
    for source_index in range(2):
        root = tmp_path / f"source-{source_index}"
        (root / "final").mkdir(parents=True)
        tasks = [arithmetic_task(id=f"source-{source_index}-task-{index}") for index in range(4)]
        expected.update({task.id: task for task in tasks})
        rows = [{"task_id": task.id, "task_json": task.model_dump_json()} for task in tasks]
        if invalid:
            rows[0]["task_json"] = "{}"
        pq.write_table(pa.Table.from_pylist(rows), root / "final/part-00000.parquet")
        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "mode": "full",
                    "status": "completed",
                    "intended_use": "train",
                    "unprocessed_rows": 0,
                    "source_dataset": "fixture",
                    "source_revision": "1",
                    "admission_counts": {"admitted": 4},
                    "datasets": {"final": str(root / "final")},
                }
            )
        )
        inputs.append(AdmittedInput(str(root), 4, 1))
    output = tmp_path / "dataset"
    kwargs = dict(
        sources=tuple(inputs),
        output=StoragePath(str(output)),
        seed=42,
        machine=machine_runtime(),
        session=lowered(arithmetic_task()).session,
        factories={},
    )
    if invalid:
        with pytest.raises(ValueError, match="publication rejected"):
            build_real_dataset(**kwargs)
        report = json.loads((output / "conversion-report.json").read_text())
        assert len(report["failures"]) == 2
        assert {failure["source"] for failure in report["failures"]} == {source.root for source in inputs}
        assert not (output / "train.parquet").exists()
        assert not (output / "manifest.json").exists()
        return
    report = build_real_dataset(**kwargs)
    assert len(report["train_ids"]) == 6 and len(report["validation_ids"]) == 2
    assert set(report["train_ids"]).isdisjoint(report["validation_ids"])
    assert {task_id for source in report["sources"] for task_id in source["train_ids"]} == set(report["train_ids"])
    recovered = {}
    for name in ("train", "validation"):
        for row in pq.read_table(output / f"{name}.parquet").to_pylist():
            task = LoweredTaskSpec.model_validate_json(row["lowered_task_spec"]).task
            recovered[task.id] = task
    assert recovered == expected
    with pytest.raises(FileExistsError):
        build_real_dataset(**kwargs)

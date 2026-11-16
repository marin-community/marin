# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Multi-SWE export keeps source inputs private and actor declarations intact."""

import hashlib
import io
import json
import tarfile
import tomllib

import pyarrow as pa
import pyarrow.parquet as pq
from taskcompendium.harbor.adapter import declared_execution_limits
from taskcompendium.harbor.negative_control import public_task_preparation
from taskcompendium.models import DockerBuildExecution, Source, TaskSpec
from taskcompendium.pipeline.models import RawRow
from taskcompendium.runtime.resources import resource_bytes

from experiments.post_training.task_curation.datasets.prime_intellect.capture_models import ImageBinding
from experiments.post_training.task_curation.datasets.prime_intellect.multiswe.convert import (
    DATASET,
    REVISION,
    VERSION,
    CapturedImage,
    task_from_capture,
)
from experiments.post_training.task_curation.task_spec_harbor import prepare


def test_multiswe_frozen_row_exports_with_private_parser_and_original_actor_limits(tmp_path):
    raw = {
        "org": "fixture",
        "repo": "repository",
        "number": 7,
        "instance_id": "fixture__repository-7",
        "title": "Return the requested value",
        "body": "Replace the incorrect value.",
        "base": {"sha": "a" * 40},
        "resolved_issues": [],
        "fix_patch": "source gold patch",
        "test_patch": "hidden test patch",
    }
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, content in {"fix-run.sh": b"source harness", "test.patch": b"hidden test patch"}.items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            member.mode = 0o644
            member.mtime = 1744227900
            archive.addfile(member, io.BytesIO(content))
    data = buffer.getvalue()
    capture = CapturedImage(
        ImageBinding(
            original_references=("mswebench/fixture_m_repository:pr-7",),
            pinned_reference="mswebench/fixture_m_repository@sha256:" + "b" * 64,
            task_ids=(raw["instance_id"],),
        ),
        data,
        hashlib.sha256(data).hexdigest(),
        '{"fixture":true}',
    )
    row = RawRow(
        id="multiswe-fixture",
        source=Source(dataset=DATASET, revision=REVISION, row="0", importer_revision=VERSION),
        data=raw,
    )
    task = task_from_capture(row, capture)
    assert isinstance(task, TaskSpec)
    cohort = tmp_path / "cohort.parquet"
    pq.write_table(pa.Table.from_pylist([{"id": task.id, "task_spec": task.model_dump_json(), "runtime": "{}"}]), cohort)
    output = tmp_path / "export"
    prepare(cohort, output, source_revision="c" * 40, docker_build=DockerBuildExecution.DECLARATIVE)
    packet = output / "tasks" / task.id
    restored = TaskSpec.model_validate_json((packet / "task-spec.json").read_text())
    assert restored == task
    assert declared_execution_limits(restored) == {"cpus": 4, "memory_mb": 4096, "storage_mb": 10240}
    config = tomllib.loads((packet / "task.toml").read_text())
    assert {key: config["environment"][key] for key in ("cpus", "memory_mb", "storage_mb")} == {
        "cpus": 4,
        "memory_mb": 4096,
        "storage_mb": 10240,
    }
    assert restored.environment_requirements.docker_image == capture.binding.pinned_reference
    private = {r.path: resource_bytes(r) for r in restored.resources.verifier}
    assert private["home-grading.tar.gz"] == data
    assert json.loads(private["row.json"]) == raw
    assert private["multiswe/host_grade.py"]
    assert public_task_preparation(restored)["resources"] == {"all": [], "worker": []}
    assert restored.grader.timeout is None
    context = {r.path: resource_bytes(r) for r in restored.grader.environment.docker_build.files}
    provenance = json.loads(context["provenance.json"])
    assert hashlib.sha256(context["requirements.lock"]).hexdigest() == provenance["requirements_lock_sha256"]
    assert provenance["base_image"].encode() in context["Dockerfile"]

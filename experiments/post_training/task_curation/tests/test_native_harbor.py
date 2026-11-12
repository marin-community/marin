# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Native QUICK imports preserve the package instead of capturing selected workspace files."""

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from finestore.rl.curation import DaytonaSafety
from marin.execution.lazy import run
from taskcompendium.grading_environment import assess_grading_environment
from taskcompendium.models import Source, TaskSpec
from taskcompendium.pipeline.conversion import convert_row
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import NormalizedTask, RawRow
from taskcompendium.pipeline.source_processing import SourceProcessingMode
from taskcompendium.runtime.resources import resource_bytes

from experiments.post_training.task_curation import native_release
from experiments.post_training.task_curation.campaign import CampaignRuntime
from experiments.post_training.task_curation.config import PipelineOptions
from experiments.post_training.task_curation.datasets import native_releases
from experiments.post_training.task_curation.datasets.native_harbor import native_task, unpack_native
from experiments.post_training.task_curation.prebuilt import PrebuiltCohort
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, SourceOrigin
from experiments.post_training.task_curation.within_source_binding import bind_within_source_dedup, read_dedup_receipts


@pytest.mark.parametrize("separate_solution", [False, True])
def test_native_package_round_trip_keeps_state_boundary_private_files_and_modes(
    separate_solution, tmp_path, monkeypatch
):
    files = {
        "instruction.md": (b"Repair /work and preserve the service state.", 0o644),
        "task.toml": (
            b"[environment]\ncpus = 3\nmemory_mb = 6144\nallow_internet = false\n"
            b'[environment.setup]\ncommand = "start-service"\ntimeout_sec = 321\n'
            b'[verifier]\nenvironment_mode = "shared"\n',
            0o644,
        ),
        "environment/Dockerfile": (b"FROM ubuntu:24.04\nWORKDIR /work\n", 0o644),
        "environment/entrypoint.sh": (b"#!/bin/sh\nexec service\n", 0o755),
        "tests/test.sh": (b"#!/bin/sh\ncheck-service-and-links\n", 0o750),
        "solution/solve.sh": (b"#!/bin/sh\nrepair-service\n", 0o700),
    }
    buffer = io.BytesIO()
    solution_buffer = io.BytesIO()
    with (
        tarfile.open(fileobj=buffer, mode="w:gz") as archive,
        tarfile.open(fileobj=solution_buffer, mode="w:gz") as solution_archive,
    ):
        for name, (content, mode) in files.items():
            member = tarfile.TarInfo(name)
            member.size, member.mode, member.mtime = len(content), mode, 123
            target = solution_archive if separate_solution and name.startswith("solution/") else archive
            target.addfile(member, io.BytesIO(content))
    context = ConversionContext({}, None)
    data = unpack_native(
        {
            "path": "native",
            "task_binary": buffer.getvalue(),
            "solution_binary": solution_buffer.getvalue() if separate_solution else None,
        },
        context,
    )
    raw = RawRow("native", Source(dataset="fixture", revision="pin", row="0", importer_revision="1"), data)
    result = convert_row(raw, native_task, context)
    assert isinstance(result, NormalizedTask)
    task = TaskSpec.model_validate_json(result.task.model_dump_json())
    assert task.resources.worker == ()
    assert {r.path for r in task.resources.oracle} == {"solution/solve.sh"}
    resources = (*task.resources.verifier, *task.resources.oracle)
    assert {r.path: (resource_bytes(r), int(r.mode, 8)) for r in resources} == files
    assert {r.mtime_ns for r in resources} == {123_000_000_000}
    assert task.context.events[0].content == files["instruction.md"][0].decode()
    assert assess_grading_environment(task).status == "matched"

    published = tmp_path / "published"
    published.mkdir()
    serialized = task.model_dump_json()
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "path": "native",
                    "task_spec": serialized,
                    "task_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
                }
            ]
        ),
        published / "tasks.parquet",
    )
    (published / "manifest.json").write_text('{"mode": "quick"}')
    monkeypatch.setattr(native_release, "hf_hub_download", lambda _repo, filename, **_kwargs: str(published / filename))
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "artifacts"))
    monkeypatch.setattr("rigging.filesystem.cluster_config.region_from_metadata", lambda: None)
    source = RlDataSource(
        info=SourceInfo(
            origin=SourceOrigin.MARIN,
            upstream_dataset="native",
            upstream_name="Native",
            build_path="native import fixture",
        ),
        config=PrebuiltCohort(
            name="native", version="2026.11.12.1", repo="fixture/native", revision="pin", expected_rows=1
        ),
        pipeline=native_release.import_native_quick,
    )
    artifact = run(source.pipeline(source, PipelineOptions(SourceProcessingMode.QUICK, CampaignRuntime())))[0]
    output = Path(artifact.path)
    imported = pq.read_table(next((output / "normalize").glob("*.parquet"))).to_pylist()
    assert imported[0]["task_json"] == serialized
    assert (output / "tasks.parquet").read_bytes() == (published / "tasks.parquet").read_bytes()
    assert not (output / "final").exists()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["mode"] == "quick"
    assert manifest["grading_environment"]["assessed_tasks"] == manifest["converted_rows"] == 1

    incomplete = task.model_copy(
        update={
            "resources": task.resources.model_copy(
                update={"verifier": tuple(r for r in task.resources.verifier if r.path != "tests/test.sh")}
            )
        }
    ).model_dump_json()
    pq.write_table(
        pa.table(
            {
                "path": ["native"],
                "task_spec": [incomplete],
                "task_sha256": [hashlib.sha256(incomplete.encode()).hexdigest()],
            }
        ),
        published / "tasks.parquet",
    )
    with pytest.raises(ValueError, match="Incomplete native Harbor package"):
        native_release.run_native_import(native_release.NativeQuickImport(str(tmp_path / "incomplete"), source.config))


@pytest.mark.parametrize(
    "source_id",
    [
        "Harbor Hub:skill2env/skill2env",
        "TaskTrove:DCAgent__exp_rle_adversarial-v6",
        "TaskTrove:smartie-swesmith-admitted",
    ],
)
def test_native_registration_requires_actor_receipt_for_its_private_pin(source_id, tmp_path, monkeypatch):
    entry = next(
        entry for entry in json.loads(native_releases.LEDGER.read_text())["entries"] if entry["id"] == source_id
    )
    release = entry["disposition_release"]
    release.pop("daytona_measurement", None)
    ledger = tmp_path / "ledger.json"
    monkeypatch.setattr(native_releases, "LEDGER", ledger)
    ledger.write_text(json.dumps({"entries": [entry]}))
    assert native_releases.sources() == []

    measurement = {
        "dataset_id": release["repo"],
        "dataset_revision": "0" * 40,
        "daytona_snapshot_count": 1,
        "daytona_machine_slots": release["converted_rows"],
        "daytona_execution_failures": 1,
        "daytona_count_basis": "fixture actor census over the exact private pin",
        "daytona_unsafe_reasons": ["execution_requirement"],
        "daytona_unsafe_details": {"execution_requirement": "fixture unsupported actor requirement"},
        "daytona_unsafe_counts": {"execution_requirement": 1},
    }
    release["daytona_measurement"] = measurement
    ledger.write_text(json.dumps({"entries": [entry]}))
    with pytest.raises(ValueError, match="does not cover the private source pin"):
        native_releases.sources()

    measurement["dataset_revision"] = release["revision"]
    ledger.write_text(json.dumps({"entries": [entry]}))
    (source,) = native_releases.sources()
    receipts = read_dedup_receipts(Path(__file__).parents[1] / "datasets/within_source_dedup.json")
    source = bind_within_source_dedup(source, receipts[source.info.id])
    declaration = source.declaration
    assert declaration.id == source_id
    assert declaration.current_release.repo == release["repo"]
    assert declaration.current_release.revision == release["revision"]
    assert declaration.daytona_safety is DaytonaSafety.UNSAFE
    assert declaration.daytona_unsafe_counts == {"execution_requirement": 1}
    assert "excluded" not in source.info.tags

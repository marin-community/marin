# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Durable teacher requests and independently recorded admission provenance."""

import asyncio
import json
import shutil
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from taskcompendium.parquet import read_tasks, write_tasks

from experiments.post_training.russell_rsi import repair_tasks
from experiments.post_training.russell_rsi.corpus import CommitRecord
from experiments.post_training.russell_rsi.repair_tasks import (
    QualifiedUnionConfig,
    canonical_sha256,
    download_evidence,
    generate_one_repair,
    prepare_qualified_union,
    repair_request,
    sha256,
)
from experiments.post_training.russell_rsi.settings import GLM_TOKEN_ENV
from experiments.post_training.russell_rsi.sources import SourceSnapshot, source_group_id
from experiments.post_training.russell_rsi.tasks import CONTROL_REWARDS, GeneratedRepair, ObservationCase, build_task


@pytest.fixture
def provider(monkeypatch):
    requests = []
    content = [
        json.dumps(
            {
                "problem_statement": "Repair addition.",
                "cases": [{"probe_python": "from maths import add\nobservation = add(2,3)", "expected_json": 5}],
                "editable_paths": ["maths.py"],
            }
        )
    ]

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            data = json.dumps(
                {
                    "id": "completion",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "teacher",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": content[0]}, "finish_reason": "stop"}
                    ],
                }
            ).encode()
            self.send_response(content[1] if len(content) > 1 else 200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        monkeypatch.setattr(
            repair_tasks, "resolve_glm_base_url", lambda relay: f"http://127.0.0.1:{server.server_port}/v1"
        )
        monkeypatch.setenv(GLM_TOKEN_ENV, "test-token")
        try:
            yield requests, content
        finally:
            server.shutdown()
            thread.join()


def test_request_response_are_durable_and_resume_makes_no_second_call(tmp_path, provider):
    requests, content = provider
    directory = tmp_path / "local"
    directory.mkdir()
    remote = tmp_path / "remote"
    remote.mkdir()
    request = {"model": "teacher", "messages": [{"role": "user", "content": "source"}]}

    async def persist(path):
        if path.name == "generation.json":
            assert (remote / "request-start.json").exists()
            assert not (directory / "repair.json").exists()
        shutil.copyfile(path, remote / path.name)

    async def run():
        await generate_one_repair(
            request, relay_job="relay", manifest_sha256="sealed", directory=directory, persist=persist
        )

    asyncio.run(run())
    shutil.rmtree(directory)
    shutil.copytree(remote, directory)
    asyncio.run(run())
    assert len(requests) == 1
    assert (
        json.loads((remote / "generation.json").read_text())["response"]["choices"][0]["message"]["content"]
        == content[0]
    )
    assert json.loads((remote / "repair.json").read_text())["cases"][0]["expected_json"] == 5
    with pytest.raises(ValueError):
        asyncio.run(
            generate_one_repair(
                request, relay_job="relay", manifest_sha256="changed", directory=directory, persist=persist
            )
        )
    assert len(requests) == 1


def test_invalid_response_is_preserved_and_terminal(tmp_path, provider):
    requests, content = provider
    content[0] = '{"cases": "invalid"}'

    async def persist(path):
        assert path.exists()

    request = {"model": "teacher", "messages": [{"role": "user", "content": "source"}]}
    for _ in range(2):
        asyncio.run(
            generate_one_repair(
                request, relay_job="relay", manifest_sha256="sealed", directory=tmp_path, persist=persist
            )
        )
    assert len(requests) == 1
    assert (
        json.loads((tmp_path / "generation.json").read_text())["response"]["choices"][0]["message"]["content"]
        == content[0]
    )
    assert (tmp_path / "generation-rejection.json").exists()
    assert not (tmp_path / "repair.json").exists()


def test_started_request_without_response_blocks_resume(tmp_path, provider):
    requests, _ = provider
    request = {"model": "teacher", "messages": [{"role": "user", "content": "source"}]}
    identity = {
        "request": request,
        "request_sha256": canonical_sha256(request),
        "relay_job": "relay",
        "original_manifest_sha256": "sealed",
    }
    (tmp_path / "request-start.json").write_text(json.dumps(identity))

    async def persist(path):
        assert path.name == "generation-outcome.json"

    asyncio.run(
        generate_one_repair(request, relay_job="relay", manifest_sha256="sealed", directory=tmp_path, persist=persist)
    )
    assert not requests
    assert json.loads((tmp_path / "generation-outcome.json").read_text())["stage"] == "ambiguous_request"
    (tmp_path / "request-start.json").unlink()
    with pytest.raises(ValueError):
        asyncio.run(
            generate_one_repair(
                request, relay_job="relay", manifest_sha256="sealed", directory=tmp_path, persist=persist
            )
        )
    assert not requests


def snapshot(commit):
    return SourceSnapshot(
        repository="example/math",
        parent_sha="a" * 40,
        commit_sha=commit * 40,
        parent_files={"maths.py": "def add(a,b): return a-b\n", "LICENSE": "MIT"},
        reference_files={"maths.py": "def add(a,b): return a+b\n", "LICENSE": "MIT"},
        license_paths=("LICENSE",),
        split="train",
    )


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def admitted(root, seed, inputs, code):
    identifier = source_group_id(seed)
    directory = root / "generation" / identifier
    repair = GeneratedRepair(
        problem_statement="Repair addition.",
        cases=(ObservationCase(probe_python="from maths import add\nobservation=add(2,3)", expected_json=5),),
        editable_paths=("maths.py",),
    )
    write_json(directory / "snapshot.json", seed.model_dump(mode="json"))
    write_json(directory / "repair.json", repair.model_dump(mode="json"))
    task = build_task(seed, repair, image=inputs["image"], timeout=120)
    generation = {
        "request": {"model": "teacher", "messages": []},
        "request_sha256": canonical_sha256({"model": "teacher", "messages": []}),
    }
    write_json(directory / "generation.json", generation)
    identity_inputs = {
        "generation_request_sha256": generation["request_sha256"],
        "admission_code_sha256": code,
        "snapshot_sha256": sha256((directory / "snapshot.json").read_bytes()),
        "repair_sha256": sha256((directory / "repair.json").read_bytes()),
        "task_spec_sha256": sha256(task.model_dump_json().encode()),
        "source_split": "train",
        "source_commit": seed.commit_sha,
        "source_family": "example/math",
        "image": inputs["image"],
        "dependency_manifest_sha256": inputs["dependency_manifest"]["sha256"],
        "prepared_runtime": inputs["runtime_bundle"]["archive_sha256"],
    }
    identity = {"inputs": identity_inputs, "sha256": canonical_sha256(identity_inputs)}
    write_json(directory / "admission-identity.json", identity)
    parent = {
        "tests": 1,
        "errors": 0,
        "failures": 1,
        "observations": [-1],
        "case_errors": [None],
        "case_diagnostics": [""],
    }
    reference = {**parent, "failures": 0, "observations": [5]}
    controls = {key: {"status": "graded", "reward": reward} for key, reward in CONTROL_REWARDS.items()}
    acceptance = {
        "accepted": True,
        "completed": True,
        "behavioral_acceptance": True,
        "attempt": 1,
        "identity": identity["sha256"],
        "parent": parent,
        "reference": reference,
        "patch_controls": controls,
    }
    attempt = directory / "attempts" / "0001"
    write_json(directory / "acceptance.json", acceptance)
    write_json(attempt / "result.json", acceptance)
    for label, report in (("parent", parent), ("reference", reference)):
        for number in (1, 2):
            write_json(attempt / "records" / f"{label}-{number}.json", {"completed": True, "report": report})
    for key, result in controls.items():
        write_json(attempt / "records" / f"control-{key}.json", {"completed": True, "result": result})
    exported = task.model_copy(update={"metadata": {**task.metadata, "family": "example/math"}})
    return {"id": identifier, "accepted": True, "admission_identity": identity}, exported


def seal(root, manifest, filename):
    manifest["original_artifact_uri"] = str(root)
    manifest["files"] = {
        path.relative_to(root).as_posix(): sha256(path.read_bytes())
        for path in root.rglob("*")
        if path.is_file() and path.name != filename
    }
    path = root / filename
    write_json(path, manifest)
    return path, sha256(path.read_bytes())


@pytest.fixture
def union_evidence(tmp_path):
    original, repaired = tmp_path / "original", tmp_path / "repaired"
    original.mkdir()
    repaired.mkdir()
    seeds = [snapshot("b"), snapshot("c")]
    source_file = tmp_path / "snapshots.jsonl"
    source_file.write_text("".join(seed.model_dump_json() + "\n" for seed in seeds))
    inventory_file = tmp_path / "inventory.jsonl"
    records = [
        CommitRecord(
            seed.commit_sha,
            "tree",
            {},
            None,
            {},
            "repair",
            [seed.parent_sha],
            [seed.repository],
            [],
            family="example/math",
            split="train",
        )
        for seed in seeds
    ]
    inventory_file.write_text("".join(json.dumps(record.__dict__) + "\n" for record in records))
    inputs = {
        "parent_development_identity": "frozen-parent",
        "source_pool": {"uri": str(source_file), "sha256": sha256(source_file.read_bytes())},
        "inventory": {"uri": str(inventory_file), "sha256": sha256(inventory_file.read_bytes())},
        "image": "pinned-python",
        "dependency_manifest": {"sha256": "locked-wheels"},
        "runtime_bundle": {"archive_sha256": "runtime"},
    }
    first, first_task = admitted(original, seeds[0], inputs, "original-code")
    second, second_task = admitted(repaired, seeds[1], inputs, "repair-code")
    write_tasks(str(original / "train.parquet"), [first_task])
    (repaired / "accepted").mkdir()
    write_tasks(str(repaired / "accepted" / "train.parquet"), [second_task])
    original_manifest = {
        "version": 1,
        "inputs": inputs,
        "candidates": [first, {"id": second["id"], "accepted": False}],
        "accepted_source_groups": [first["id"]],
        "repair_source_groups": [second["id"]],
        "admission_code_identities": {first["id"]: "original-code"},
    }
    path, digest = seal(original, original_manifest, "manifest.json")
    repair_manifest = {
        "version": 1,
        "completed": True,
        "inputs": inputs,
        "original_manifest_sha256": digest,
        "candidates": [second],
        "accepted_source_groups": [second["id"]],
        "admission_code_identities": {second["id"]: "repair-code"},
    }
    seal(repaired, repair_manifest, "repair-manifest.json")
    config = QualifiedUnionConfig(str(path), digest, str(repaired), str(tmp_path / "union"), 3, "frozen-parent")
    return config, repaired, repair_manifest, [first_task, second_task]


def test_union_verifies_each_recorded_code_and_publishes_partial_before_gate(union_evidence):
    config, _, _, expected = union_evidence
    with pytest.raises(ValueError, match="Qualified union has 2"):
        prepare_qualified_union(config)
    output = Path(config.output_path)
    actual = list(read_tasks(str(output / "train.parquet")))
    assert [task.model_dump_json() for task in actual] == [
        task.model_dump_json() for task in sorted(expected, key=lambda task: task.id)
    ]
    assert json.loads((output / "summary.json").read_text())["train_rows"] == 2
    assert {
        row["admission_identity"]["inputs"]["admission_code_sha256"]
        for row in json.loads((output / "provenance.json").read_text())
    } == {"original-code", "repair-code"}
    prepare_qualified_union(replace(config, minimum_train_rows=2))


@pytest.mark.parametrize("mutation", ["report", "row", "duplicate"])
def test_union_rejects_inconsistent_qualification_even_when_files_are_resealed(union_evidence, mutation):
    config, repaired, manifest, expected = union_evidence
    if mutation == "report":
        record = next((repaired / "generation").glob("*/attempts/0001/records/reference-2.json"))
        value = json.loads(record.read_text())
        value["report"]["observations"] = [99]
        write_json(record, value)
    elif mutation == "row":
        task = expected[1].model_copy(update={"metadata": {**expected[1].metadata, "extra": "changed"}})
        write_tasks(str(repaired / "accepted" / "train.parquet"), [task])
    else:
        write_tasks(str(repaired / "accepted" / "train.parquet"), [expected[1], expected[1]])
    seal(repaired, manifest, "repair-manifest.json")
    with pytest.raises(ValueError):
        prepare_qualified_union(replace(config, minimum_train_rows=2))
    assert not (Path(config.output_path) / "train.parquet").exists()


def test_sealed_hash_mismatch_blocks_evidence_use(union_evidence, tmp_path):
    config, _, _, _ = union_evidence
    source = Path(config.original_manifest_uri).parent / "train.parquet"
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="digest mismatch"):
        download_evidence(config.original_manifest_uri, config.original_manifest_sha256, tmp_path / "download")


def test_repair_prompt_includes_original_proposal_and_scope_qc_as_data():
    request = repair_request(
        snapshot("b"),
        "fixed abstract skill",
        {"qc_feedback": {"reason": "wrong SQLite fixture", "reference": {"observations": [5]}}},
        {"choices": [{"message": {"content": "original-proposal"}}]},
    )
    data = json.loads(request["messages"][-1]["content"])
    assert data["original_proposal"] == "original-proposal"
    assert data["task_qc"]["reason"] == "wrong SQLite fixture"
    assert data["task_qc"]["reference"]["observations"] == [5]
    assert request["extra_body"]["prompt_cache_key"].startswith("russell-rsi-repair-1-")


def test_failed_scope_does_not_block_other_scopes_or_retry_after_relay_rename(tmp_path, provider):
    requests, content = provider
    failed = tmp_path / "failed"
    passed = tmp_path / "passed"
    failed.mkdir()
    passed.mkdir()
    request = {"model": "teacher", "messages": [{"role": "user", "content": "source"}]}

    async def persist(path):
        assert path.exists()

    content.append(503)
    asyncio.run(
        generate_one_repair(
            request, relay_job="original-relay", manifest_sha256="sealed", directory=failed, persist=persist
        )
    )
    assert json.loads((failed / "generation-outcome.json").read_text())["stage"] == "provider_error"
    assert not (failed / "generation.json").exists()
    assert len(requests) == 1
    content.pop()
    asyncio.run(
        generate_one_repair(
            request, relay_job="original-relay", manifest_sha256="sealed", directory=passed, persist=persist
        )
    )
    for directory in (failed, passed):
        asyncio.run(
            generate_one_repair(
                request, relay_job="renamed-relay", manifest_sha256="sealed", directory=directory, persist=persist
            )
        )
    assert len(requests) == 2
    assert (passed / "repair.json").exists()
    assert json.loads((failed / "request-start.json").read_text())["relay_job"] == "original-relay"


def test_pinned_wheel_inventory_requires_every_file_and_exact_bytes(tmp_path):
    source = tmp_path / "bundle" / "python"
    source.mkdir(parents=True)
    first = source / "first.whl"
    first.write_bytes(b"first wheel bytes")
    manifest = {
        "repository_wheels": {"example/math": "python"},
        "wheel_files": {
            "python/first.whl": {"sha256": sha256(first.read_bytes())},
            "python/second.whl": {"sha256": sha256(b"second wheel bytes")},
        },
    }
    output = tmp_path / "download"
    with pytest.raises(FileNotFoundError):
        repair_tasks.download_wheels(str(source.parent), json.dumps(manifest).encode(), output)
    second = source / "second.whl"
    second.write_bytes(b"different bytes")
    with pytest.raises(ValueError, match="digest mismatch"):
        repair_tasks.download_wheels(str(source.parent), json.dumps(manifest).encode(), output)
    second.write_bytes(b"second wheel bytes")
    repair_tasks.download_wheels(str(source.parent), json.dumps(manifest).encode(), output)
    assert (output / "python" / "first.whl").read_bytes() == first.read_bytes()
    assert (output / "python" / "second.whl").read_bytes() == second.read_bytes()

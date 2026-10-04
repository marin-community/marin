# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fixed expectations, one provider call, and review-bound admission."""

import asyncio
import hashlib
import json
import threading
from dataclasses import asdict, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import rigging.runtime_bundle
import shellbox.backends.qemu.machine
from rigging.runtime_bundle import RuntimeBundle
from shellbox.machine import MachineStartupError
from taskcompendium.parquet import read_tasks, write_tasks

from experiments.post_training.russell_rsi import contract_tasks, tasks
from experiments.post_training.russell_rsi.contract_tasks import (
    FixedContract,
    TeacherStatement,
    contract_attempt,
    digest,
    saved_statement,
    statement_request,
)
from experiments.post_training.russell_rsi.settings import GLM_TOKEN_ENV
from experiments.post_training.russell_rsi.sources import SourceSnapshot, source_group_id
from experiments.post_training.russell_rsi.test_tasks import AdmissionFactory, repair


@pytest.fixture
def source():
    return SourceSnapshot(
        repository="example/math",
        parent_sha="a" * 40,
        commit_sha="b" * 40,
        parent_files={"maths.py": "def add(a,b): return a-b\n", "LICENSE": "MIT license"},
        reference_files={"maths.py": "def add(a,b): return a+b\n", "LICENSE": "MIT license"},
        license_paths=("LICENSE",),
        split="train",
    )


@pytest.fixture
def contract():
    return FixedContract(
        "math.add",
        ("Add two integers; preserve negative operands.",),
        ("from maths import add\nobservation = add(2,3)", "from maths import add\nobservation = add(-2,4)"),
        ("maths.py",),
        "def add(a,b): ...",
        ("types",),
    )


@pytest.fixture
def provider(monkeypatch):
    calls = []
    status = [200]

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            calls.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            body = json.dumps(
                {
                    "id": "completion",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "teacher",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": json.dumps(
                                    {"problem_statement": "Repair integer addition while preserving negative operands."}
                                ),
                            },
                            "finish_reason": "stop",
                        }
                    ],
                }
            ).encode()
            self.send_response(status[0])
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args):  # noqa: A002
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        monkeypatch.setattr(
            contract_tasks, "resolve_glm_base_url", lambda relay: f"http://127.0.0.1:{server.server_port}/v1"
        )
        monkeypatch.setenv(GLM_TOKEN_ENV, "test-token")
        yield calls, status
        server.shutdown()
        thread.join()


async def persisted(path):
    assert path.exists()


def test_provider_failure_consumes_only_that_contract_call(tmp_path, provider, contract):
    calls, status = provider
    request = statement_request(contract, {"skills": []})
    status[0] = 500
    assert (
        asyncio.run(
            saved_statement(request, identity="frozen", relay_job="relay", directory=tmp_path, persist=persisted)
        )
        is None
    )
    status[0] = 200
    assert (
        asyncio.run(
            saved_statement(request, identity="frozen", relay_job="renamed-relay", directory=tmp_path, persist=persisted)
        )
        is None
    )
    other = tmp_path / "other"
    other.mkdir()
    assert asyncio.run(
        saved_statement(request, identity="frozen", relay_job="relay", directory=other, persist=persisted)
    )
    assert len(calls) == 2
    assert json.loads((tmp_path / "generation-outcome.json").read_text())["stage"] == "provider_error"


def test_ambiguous_call_cannot_be_reissued(tmp_path, provider, contract):
    request = statement_request(contract, {"skills": []})
    (tmp_path / "request-start.json").write_text(
        json.dumps({"identity": "frozen", "request_sha256": digest(request), "request": request})
    )
    assert (
        asyncio.run(
            saved_statement(request, identity="frozen", relay_job="relay", directory=tmp_path, persist=persisted)
        )
        is None
    )
    assert not provider[0]
    assert json.loads((tmp_path / "generation-outcome.json").read_text())["stage"] == "ambiguous_request"


def configure_observations(monkeypatch, source, failures):
    calls = []

    async def run(files, verifier, **kwargs):
        calls.append(files)
        if failures:
            error = failures.pop(0)
            if error:
                raise error
        values = (-1, -6) if files == source.parent_files else (5, 2)
        return tasks.VerifierReport(
            tests=2, failures=2, errors=0, observations=values, case_errors=(None, None), case_diagnostics=("", "")
        )

    monkeypatch.setattr(tasks, "run_verifier", run)
    return calls


def prepared_statement(directory):
    directory.mkdir(exist_ok=True)
    statement = TeacherStatement(problem_statement="Repair integer addition while preserving negative operands.")
    (directory / "statement.json").write_text(statement.model_dump_json())
    return statement


def execute(contract, source, tmp_path, stage, review=None):
    return asyncio.run(
        contract_attempt(
            contract,
            source,
            directory=tmp_path,
            identity={"frozen": "source-probes-runtime-code"},
            stage=stage,
            review=review or {},
            request=statement_request(contract, {"skills": []}),
            relay_job="relay",
            build=lambda repair: tasks.build_task(source, repair, image="python-git", timeout=10),
            factory=None,
            persist=persisted,
        )
    )


def test_prepare_resume_does_not_repeat_observations_and_rejects_changed_review(tmp_path, monkeypatch, source, contract):
    calls = configure_observations(monkeypatch, source, [])
    prepared_statement(tmp_path)
    first = execute(contract, source, tmp_path, "prepare")
    assert first["stage"] == "review_ready"
    assert len(calls) == 4
    assert execute(contract, source, tmp_path, "prepare") == first
    assert len(calls) == 4
    with pytest.raises(ValueError, match="does not match"):
        execute(
            contract, source, tmp_path, "admit", {"math.add": {"statement_sha256": "changed", "decision": "approve"}}
        )
    assert len(calls) == 4


def test_shared_startup_budget_is_not_reset_on_resume(tmp_path, monkeypatch, source, contract):
    calls = configure_observations(
        monkeypatch, source, [MachineStartupError("startup1"), MachineStartupError("startup2")]
    )
    prepared_statement(tmp_path)
    result = execute(contract, source, tmp_path, "prepare")
    assert result["stage"] == "infrastructure_exhausted"
    assert len(calls) == 2
    assert execute(contract, source, tmp_path, "prepare") == result
    assert len(calls) == 2


def test_nonstartup_error_propagates_and_is_recorded(tmp_path, monkeypatch, source, contract):
    configure_observations(monkeypatch, source, [RuntimeError("setup failure")])
    prepared_statement(tmp_path)
    with pytest.raises(RuntimeError, match="setup failure"):
        execute(contract, source, tmp_path, "prepare")
    exception = json.loads((tmp_path / "attempts/0001/exception.json").read_text())
    assert exception["exception_type"] == "RuntimeError"
    assert not (tmp_path / "result.json").exists()


def test_behavioral_rejection_skips_controls_after_exact_statement_review(tmp_path, monkeypatch, source, contract):
    calls = configure_observations(monkeypatch, source, [])
    prepared_statement(tmp_path)
    result = execute(contract, source, tmp_path, "prepare")

    async def passed(files, verifier, **kwargs):
        calls.append(files)
        return tasks.VerifierReport(
            tests=2, failures=0, errors=0, observations=(5, 2), case_errors=(None, None), case_diagnostics=("", "")
        )

    async def unexpected_controls(*args, **kwargs):
        raise AssertionError("Behavioral rejection must not start controls")

    monkeypatch.setattr(tasks, "run_verifier", passed)
    monkeypatch.setattr(tasks, "patch_controls", unexpected_controls)
    final = execute(
        contract,
        source,
        tmp_path,
        "admit",
        {"math.add": {"statement_sha256": result["statement_sha256"], "decision": "approve"}},
    )
    assert final["stage"] == "behavioral_rejection"
    assert len(calls) == 8
    assert json.loads((tmp_path / "attempts/0001/acceptance.json").read_text())["parent"]["errors"] == 0


def test_retry_cannot_replace_frozen_reference_values(tmp_path, monkeypatch, source, contract):
    configure_observations(monkeypatch, source, [])
    prepared_statement(tmp_path)
    prepared = execute(contract, source, tmp_path, "prepare")
    failures = [MachineStartupError("final admission startup")]
    calls = []

    async def changed(files, verifier, **kwargs):
        calls.append(files)
        if failures:
            raise failures.pop()
        values = (-1, -6) if files == source.parent_files else (6, 2)
        return tasks.VerifierReport(
            tests=2, failures=2, errors=0, observations=values, case_errors=(None, None), case_diagnostics=("", "")
        )

    monkeypatch.setattr(tasks, "run_verifier", changed)
    final = execute(
        contract,
        source,
        tmp_path,
        "admit",
        {"math.add": {"statement_sha256": prepared["statement_sha256"], "decision": "approve"}},
    )
    assert final["stage"] == "behavioral_rejection"
    assert json.loads((tmp_path / "expected.json").read_text())["observations"] == [5, 2]
    assert len(calls) == 5
    assert len(list((tmp_path / "attempts").iterdir())) == 2


def test_wrapper_prepares_then_admits_into_distinct_output_with_inherited_proofs(
    tmp_path, monkeypatch, provider, source, contract
):
    artifact = tmp_path / "input"
    artifact.mkdir()
    prior = tasks.build_task(
        source.model_copy(update={"commit_sha": "c" * 40}), repair(), image="python-git", timeout=10
    )
    write_tasks(str(artifact / "train.parquet"), [prior])
    proof = {"task_sha256": digest(prior.model_dump(mode="json")), "source_group": "prior-source"}
    proof_bytes = json.dumps(proof, sort_keys=True).encode()
    proof_hash = hashlib.sha256(proof_bytes).hexdigest()
    path = artifact / "evidence" / proof_hash / "proposal.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(proof_bytes)
    bank = {
        "tasks": [
            {
                "task_id": prior.id,
                "task_sha256": proof["task_sha256"],
                "admission_sha256": proof_hash,
                "source_id": "prior-source",
                "capability": "types",
                "contract_id": "math.prior",
                "relation": "new_contract",
            }
        ],
        "feedback_identity": "prior-feedback",
    }
    (artifact / "bank.json").write_text(json.dumps(bank))
    (artifact / "snapshots.jsonl").write_text(source.model_dump_json() + "\n")
    wheels = tmp_path / "wheels"
    (wheels / "math").mkdir(parents=True)
    wheel = wheels / "math" / "fixture.whl"
    wheel.write_bytes(b"wheel fixture used only at the machine I/O boundary")
    dependencies = {
        "base_uri": str(wheels),
        "repository_wheels": {source.repository: "math"},
        "wheel_files": {"math/fixture.whl": {"sha256": hashlib.sha256(wheel.read_bytes()).hexdigest()}},
    }
    dependency_file = wheels / "manifest.json"
    dependency_file.write_text(json.dumps(dependencies))
    runtime = RuntimeBundle(
        manifest_uri="fixture", manifest_sha256="fixture", archive_uri="fixture", archive_sha256="fixture"
    )
    manifest = {
        "version": 1,
        "original_artifact_uri": str(artifact),
        "image": "python-git",
        "runtime_bundle": asdict(runtime),
        "dependency_manifest": {
            "uri": str(dependency_file),
            "sha256": hashlib.sha256(dependency_file.read_bytes()).hexdigest(),
        },
        "dependency_wheels_uri": str(wheels),
        "contracts": [
            {
                "contract_id": contract.contract_id,
                "source_id": source_group_id(source),
                "obligations": contract.obligations,
                "probes": contract.probes,
                "editable_paths": contract.editable_paths,
                "source_excerpt": contract.source_excerpt,
                "capability_labels": contract.capability_labels,
            }
        ],
        "files": {
            p.relative_to(artifact).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in artifact.rglob("*")
            if p.is_file()
        },
    }
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(manifest))
    capabilities = tmp_path / "capabilities.json"
    capabilities.write_text(json.dumps({"skills": []}))
    factory = AdmissionFactory()
    monkeypatch.setattr(
        rigging.runtime_bundle, "install_runtime_bundle", lambda config: {"directory_name": "fixture-runtime"}
    )
    monkeypatch.setattr(shellbox.backends.qemu.machine, "QemuMachineFactory", lambda *args, **kwargs: factory)
    config = contract_tasks.ContractTasksConfig(
        manifest_uri=str(manifest_file),
        manifest_sha256=hashlib.sha256(manifest_file.read_bytes()).hexdigest(),
        output_path=str(tmp_path / "prepare"),
        relay_job="fixture-relay",
        stage="prepare",
        capabilities_uri=str(capabilities),
        capabilities_sha256=hashlib.sha256(capabilities.read_bytes()).hexdigest(),
        response_cap=1,
        admission_concurrency=1,
        statement_review_uri="",
        statement_review_sha256="",
        prepared_manifest_uri="",
        prepared_manifest_sha256="",
    )
    contract_tasks.prepare_contract_tasks(config)
    prepared = tmp_path / "prepare" / "contracts" / contract.contract_id / "prepared.json"
    prepared_record = json.loads(prepared.read_text())
    review_file = tmp_path / "review.json"
    review_file.write_text(
        json.dumps(
            {contract.contract_id: {"statement_sha256": prepared_record["statement_sha256"], "decision": "approve"}}
        )
    )
    handoff = tmp_path / "prepare" / "repair-manifest.json"
    admitted = replace(
        config,
        stage="admit",
        output_path=str(tmp_path / "admit"),
        statement_review_uri=str(review_file),
        statement_review_sha256=hashlib.sha256(review_file.read_bytes()).hexdigest(),
        prepared_manifest_uri=str(handoff),
        prepared_manifest_sha256=hashlib.sha256(handoff.read_bytes()).hexdigest(),
    )
    contract_tasks.prepare_contract_tasks(admitted)
    result_bank = json.loads((tmp_path / "admit" / "bank.json").read_text())
    assert len(result_bank["tasks"]) == 2
    assert result_bank["tasks"][0] == bank["tasks"][0]
    assert (tmp_path / "admit" / "evidence" / proof_hash / "proposal.json").read_bytes() == proof_bytes
    new = result_bank["tasks"][1]
    assert new["capability"] == "types"
    actual_tasks = {task.id: task for task in read_tasks(str(tmp_path / "admit" / "train.parquet"))}
    assert actual_tasks[prior.id] == prior
    new_proof_path = tmp_path / "admit" / "evidence" / new["admission_sha256"] / "proposal.json"
    assert hashlib.sha256(new_proof_path.read_bytes()).hexdigest() == new["admission_sha256"]
    new_proof = json.loads(new_proof_path.read_text())
    assert new_proof["source_group"] == source_group_id(source)
    assert new_proof["task_sha256"] == digest(actual_tasks[new["task_id"]].model_dump(mode="json"))
    controls = json.loads(
        (tmp_path / "admit" / "contracts" / contract.contract_id / "attempts/0001/controls.json").read_text()
    )
    assert tasks.controls_pass(controls)
    assert len(provider[0]) == 1
    assert factory.created == 28


def test_cancelled_prepared_attempts_consume_the_shared_budget(tmp_path, monkeypatch, source, contract):
    configure_observations(monkeypatch, source, [])
    prepared_statement(tmp_path)
    prepared = execute(contract, source, tmp_path, "prepare")
    review = {"math.add": {"statement_sha256": prepared["statement_sha256"], "decision": "approve"}}
    calls = configure_observations(monkeypatch, source, [asyncio.CancelledError(), asyncio.CancelledError()])
    for _ in range(2):
        with pytest.raises(asyncio.CancelledError):
            execute(contract, source, tmp_path, "admit", review)
    assert len(calls) == 2
    assert execute(contract, source, tmp_path, "admit", review)["stage"] == "infrastructure_exhausted"
    assert len(calls) == 2
    assert len(list((tmp_path / "attempts").iterdir())) == 2

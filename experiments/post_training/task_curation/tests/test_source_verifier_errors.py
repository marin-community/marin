# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Source parser errors require unchanged parser, logs and failed-trial evidence."""

import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest
from finestore.rl.curation import ContentEvidenceReference
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import ResourceGroups, Source, TaskSpec
from taskcompendium.pipeline.oracle_filter import OracleTaskReceipt, OracleTrialReceipt, SourceVerifierError
from taskcompendium.runtime.resources import inline_resource

from experiments.post_training.task_curation.datasets.prime_intellect.multiswe.convert import HERE, PARSER_FILES
from experiments.post_training.task_curation.datasets.prime_intellect.multiswe.environment import report_environment
from experiments.post_training.task_curation.source_verifier_errors import (
    multiswe_parser_sha256,
    require_source_verifier_error,
)
from experiments.post_training.task_curation.tests.prebuilt_fixtures import prebuilt_task
from lib.taskcompendium.tests.oracle_filter_fixtures import oracle_receipt

FIXTURES = Path(__file__).parent / "fixtures/multiswe_source_error"


def write_evidence(path: Path, payload: bytes) -> ContentEvidenceReference:
    path.write_bytes(payload)
    return ContentEvidenceReference(uri=str(path), sha256=hashlib.sha256(payload).hexdigest())


def tar_bytes(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, payload in files.items():
            item = tarfile.TarInfo(name)
            item.size = len(payload)
            archive.addfile(item, io.BytesIO(payload))
    return buffer.getvalue()


@pytest.fixture
def source_error(tmp_path):
    task = TaskSpec.model_validate(json.loads(prebuilt_task("upstream-error")["lowered_task_spec"])["task"])
    parser_resources = tuple(inline_resource("multiswe/" + name, (HERE / name).read_bytes()) for name in PARSER_FILES)
    task = task.model_copy(
        update={
            "source": Source(
                dataset="PrimeIntellect/Multi-SWE-RL-Verified",
                revision="64b3ebf62750b1373e26106cfc658a1861809d0f",
                row="fixture",
                importer_revision="fixture",
            ),
            "grader": task.grader.model_copy(update={"environment": report_environment()}),
            "resources": ResourceGroups(
                verifier=(*parser_resources, inline_resource("row.json", b'{"repo":"go-zero"}')),
                oracle=(inline_resource("solution/solve.sh", b"true\n"),),
            ),
        }
    )
    original = oracle_receipt([task]).tasks[task.id].model_dump(mode="json")
    result = original["trials"][0]["result"]
    result["exception_info"] = {"exception_type": "VerifierRuntimeError"}
    result["verifier_result"] = {"rewards": {}}
    trial = OracleTrialReceipt(
        path="/tmp/source-proof/jobs/oracle/result.json",
        result=result,
        result_sha256=canonical_sha256(result),
        verdict={"status": "infra_error", "reward": 0, "detail": {"error": "Grader did not write a reward file"}},
        oracle_exit_code=0,
        execution_requirements=task.execution_requirements,
    )
    source_files = {
        "multiswe-test-output.txt": b"PASS test_name\nFAIL test_name\n",
        "multiswe-read-exit-code.txt": b"0\n",
        "multiswe-harness-stdout.txt": b"",
        "multiswe-harness-stderr.txt": b"",
    }
    archive_files, index = {}, []
    directory = "source-proof/jobs/oracle/verifier/source-artifacts/"
    for name, data in source_files.items():
        payload = tar_bytes({name: data})
        digest = hashlib.sha256(payload).hexdigest()
        archive_files[directory + digest + ".tar"] = payload
        index.append({"file": digest + ".tar", "sha256": digest, "bytes": len(payload)})
    archive_files[directory + "index.json"] = json.dumps(index).encode()
    archive = write_evidence(tmp_path / "production.tar.gz", tar_bytes(archive_files))
    fixture_files = {
        resource.path: (HERE / resource.path.removeprefix("multiswe/")).read_bytes() for resource in parser_resources
    }
    fixture_files["row.json"] = b'{"repo":"go-zero"}'
    fixture_files.update(
        ("parser-context/" + name, (HERE / "parser" / name).read_bytes())
        for name in ("Dockerfile", "requirements.lock", "provenance.json")
    )
    fixture_files.update(("source-artifacts/" + name, data) for name, data in source_files.items())
    fixture_files["fixture-provenance.json"] = b"{}"
    fixture = write_evidence(tmp_path / "fixture.tar.gz", tar_bytes(fixture_files))
    traceback = write_evidence(
        tmp_path / "stderr", b"ValueError: Passed tests and failed tests should not have common items: {'test_name'}\n"
    )
    reproduction = {
        "task_id": task.id,
        "returncode": 1,
        "reward_written": False,
        "source_archive_sha256": archive.sha256,
        "fixture": fixture.model_dump(mode="json"),
        "files": {"parser.stderr": traceback.model_dump(mode="json")},
    }
    proof = SourceVerifierError(
        kind="multiswe_overlapping_test_results",
        source_task_sha256=original["source_task_sha256"],
        source_task_checksum=original["source_task_checksum"],
        runtime_sha256=original["runtime_sha256"],
        consumed_task_checksum=original["harbor_task_checksum"],
        result_sha256=trial.result_sha256,
        parser_sha256=multiswe_parser_sha256(task),
        log_sha256=hashlib.sha256(source_files["multiswe-test-output.txt"]).hexdigest(),
        source=write_evidence(tmp_path / "taskset.py", gzip.decompress((FIXTURES / "taskset.py.gz").read_bytes())),
        archive=archive,
        fixture=fixture,
        traceback=traceback,
        reproduction=write_evidence(tmp_path / "reproduction.json", json.dumps(reproduction).encode()),
        decision=write_evidence(
            tmp_path / "decision.json",
            b'{"kind":"multiswe_overlapping_test_results","authority_message":"7d38152f01f4c573c67e5f9a8f8a41e2"}',
        ),
    )
    attempts = []
    for index in range(2):
        earlier = {**trial.result, "id": f"infrastructure-{index}"}
        attempts.append(
            trial.model_copy(update={"result": earlier, "result_sha256": canonical_sha256(earlier)}).model_dump(
                mode="json"
            )
        )
    attempts.append(trial.model_dump(mode="json"))
    original.update(
        trials=[trial.model_dump(mode="json")],
        calibration={**original["calibration"], "attempts": attempts},
        negative_control=None,
        exclusion_reason="source_verifier_error",
        source_verifier_error=proof.model_dump(mode="json"),
    )
    return task, trial, proof, original, fixture_files, reproduction


def test_proved_upstream_error_is_excluded_without_changing_the_trial(source_error):
    task, trial, proof, document, _, _ = source_error
    require_source_verifier_error(task, trial, proof)
    entry = OracleTaskReceipt.model_validate(document)
    assert not entry.passed
    assert not entry.original_oracle_passed
    assert entry.calibration.attempts[-1].result == trial.result
    assert entry.calibration.attempts[-1].verdict == trial.verdict
    assert entry.source_verifier_error == proof


def test_reproduced_log_cannot_replace_the_original_copied_log(source_error):
    task, trial, proof, _, files, reproduction = source_error
    files["source-artifacts/multiswe-test-output.txt"] = b"a different log that happens to raise\n"
    fixture = write_evidence(Path(proof.fixture.uri), tar_bytes(files))
    reproduction["fixture"] = fixture.model_dump(mode="json")
    proof = proof.model_copy(
        update={
            "fixture": fixture,
            "reproduction": write_evidence(Path(proof.reproduction.uri), json.dumps(reproduction).encode()),
        }
    )
    with pytest.raises(ValueError, match="parser or logs differ"):
        require_source_verifier_error(task, trial, proof)


@pytest.mark.parametrize("change", ["passing", "resource_failure", "unrelated_parser_error"])
def test_proof_cannot_reclassify_another_outcome(source_error, change):
    task, trial, proof, document, _, reproduction = source_error
    if change == "unrelated_parser_error":
        traceback = write_evidence(Path(proof.traceback.uri), b"ModuleNotFoundError: unrelated missing dependency\n")
        reproduction["files"]["parser.stderr"] = traceback.model_dump(mode="json")
        proof = proof.model_copy(
            update={
                "traceback": traceback,
                "reproduction": write_evidence(Path(proof.reproduction.uri), json.dumps(reproduction).encode()),
            }
        )
        with pytest.raises(ValueError, match="approved exception"):
            require_source_verifier_error(task, trial, proof)
        return
    if change == "passing":
        result = {**trial.result, "exception_info": None, "verifier_result": {"rewards": {"reward": 1}}}
        trial = trial.model_copy(
            update={
                "result": result,
                "result_sha256": canonical_sha256(result),
                "verdict": {"status": "scored", "reward": 1},
            }
        )
    else:
        trial = trial.model_copy(
            update={"verifier_resource_usage": trial.verifier_resource_usage.model_copy(update={"oom_kills": 1})}
        )
    document["source_verifier_error"]["result_sha256"] = trial.result_sha256
    document["trials"] = [trial.model_dump(mode="json")]
    document["calibration"]["attempts"][-1] = trial.model_dump(mode="json")
    with pytest.raises(ValueError, match="failed original source trial"):
        OracleTaskReceipt.model_validate(document)

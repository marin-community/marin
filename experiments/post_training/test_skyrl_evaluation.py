# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from evalrecords.records import RunStatus, record_path
from marin.evaluation.model_config import ModelConfig
from marin.evaluation.runner import _record
from marin.execution.artifact import RECORD_FILENAME, ArtifactRecord
from marin.execution.lazy import ArtifactStep, materialized_config
from marin.rl.skyrl import ArtifactHfModel, SkyRLPolicyModel, SkyRLRun

import experiments.evaluation.launch as launch_module
import experiments.evaluation.pipeline as pipeline_module
from experiments.evaluation.evals import resolve_evaluations
from experiments.post_training.rl_tests.fixtures import write_skyrl_run_record
from experiments.post_training.skyrl_evaluation import evaluate


def evaluate_smoke(source, serving, *, timeout=None):
    return evaluate(
        source,
        serving,
        resolve_evaluations("smoke"),
        version=source.step.version,
        accelerator="H100x1",
        submission_cluster="marin",
        federated_cluster="cw-rno2a",
        job_timeout_seconds=timeout,
    )


def remove_rl_run_ids(path, *, legacy=False):
    receipt = ArtifactRecord.model_validate_json(path.read_text())
    assert receipt.result is not None
    del receipt.result["run_id"], receipt.result["attempt_id"]
    if legacy:
        receipt = ArtifactRecord(result_type=receipt.result_type, result=receipt.result)
    path.write_text(receipt.model_dump_json())


@pytest.mark.parametrize(
    ("model_fields", "job_timeout_seconds", "error"),
    [
        ({"revision": "starter-commit"}, None, "come from the model"),
        ({"identity": "starter"}, None, "come from the model"),
        ({"tokenizer": "org/tokenizer"}, None, "come from the model"),
        ({"tokenizer_revision": "starter-commit"}, None, "come from the model"),
        ({}, 0, "job_timeout_seconds must be positive"),
    ],
)
def test_evaluate_rejects_invalid_inputs_when_building_the_step(tmp_path, model_fields, job_timeout_seconds, error):
    source = ArtifactHfModel(ArtifactStep.adopt("tests/model", "2026.10.26", str(tmp_path)))
    with pytest.raises(ValueError, match=error):
        evaluate_smoke(source, ModelConfig(name="candidate", location="", **model_fields), timeout=job_timeout_seconds)


@pytest.mark.parametrize("source_kind", ["rl", "legacy-rl", "hf"])
def test_evaluate_records_model_and_producing_rl_attempt(tmp_path, monkeypatch, source_kind):
    root = tmp_path / "model"
    export = root / "policy"
    export.mkdir(parents=True)
    producer_receipt = root / RECORD_FILENAME
    version = "2026.10.26"
    ids = {"run_id": "training-run", "attempt_id": "training-attempt"} if source_kind == "rl" else {}
    job_timeout_seconds = 180 if source_kind == "rl" else None
    tokenizer = str(root / "pinned-tokenizer") if source_kind == "rl" else None
    if tokenizer is not None:
        Path(tokenizer).mkdir()
    tokenizer_revision = "tokenizer-commit" if tokenizer is not None else None
    if source_kind == "hf":
        source = ArtifactHfModel(ArtifactStep.adopt("tests/model", version, str(export)), relative_path="")
    else:
        source = SkyRLPolicyModel(ArtifactStep.adopt("tests/model", version, str(root), kind=SkyRLRun))
        write_skyrl_run_record(
            root,
            source.step.name,
            version,
            fingerprint=source.step.fingerprint(),
            hf_model_uri=str(export),
            global_step=100,
            tokenizer_uri=tokenizer,
            tokenizer_revision=tokenizer_revision,
            **ids,
        )
        if source_kind == "legacy-rl":
            remove_rl_run_ids(producer_receipt, legacy=True)

    class Client:
        def submit(self, *, entrypoint, timeout=None, **_kwargs):
            assert (None if timeout is None else timeout.to_ms()) == (180_000 if source_kind == "rl" else None)
            _, (batch,), _ = entrypoint.resolve()
            for evaluation in batch.evaluations:
                _record(batch, evaluation.identity, RunStatus.SUCCEEDED, None, {}, {}, {})
            return SimpleNamespace(wait=lambda **_kwargs: None)

    monkeypatch.setattr(pipeline_module, "iris_ctx", lambda: SimpleNamespace(client=Client()))
    monkeypatch.setattr(launch_module, "CW_RECORDS_PREFIX", str(tmp_path / "records"))
    evaluation = evaluate_smoke(source, ModelConfig(name="candidate", location=str(root)), timeout=job_timeout_seconds)
    result = evaluation.run(materialized_config(evaluation, str(tmp_path)))

    assert len(result.run_ids) == 2
    for run_id in result.run_ids:
        record_json = json.loads(Path(record_path(result.records_prefix, run_id)).read_text())
        model = record_json["model"]
        assert model["location"] == str(export)
        config = model["config"]
        assert config["identity"] == f"tests/model@{version}:{source.step.fingerprint()}"
        assert config["revision"] is None
        assert config["tokenizer"] == (tokenizer or str(export))
        assert config["tokenizer_revision"] == (tokenizer_revision or config["identity"])
        if source_kind == "rl":
            assert record_json["rl_run"] == ids
        else:
            assert "rl_run" not in record_json

    if source_kind == "rl":
        remove_rl_run_ids(producer_receipt)
        assert evaluation.run(materialized_config(evaluation, str(tmp_path))).run_ids == result.run_ids

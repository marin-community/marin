# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from evalrecords.records import RunStatus, record_path
from fray.iris_backend import convert_constraints
from huggingface_hub import constants
from iris.cluster.constraints import CLUSTER_CONSTRAINT_KEY, ConstraintOp
from iris.cluster.platforms.k8s.coreweave_topology import COSCHEDULE_NVLINK_DOMAIN
from marin.evaluation.model_config import ModelConfig
from marin.evaluation.runner import _inference_configs, _record
from marin.execution.artifact import RECORD_FILENAME, ArtifactRecord
from marin.execution.lazy import ArtifactStep, materialized_config
from marin.inference.model_preparation import read_tool_chat_template
from marin.rl.skyrl import ArtifactHfModel, SkyRLPolicyModel, SkyRLRun
from rigging.filesystem.storage_path import StoragePath
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

import experiments.evaluation.launch as launch_module
import experiments.evaluation.pipeline as pipeline_module
from experiments.evaluation.benchmarks import micro210
from experiments.evaluation.evals import resolve_evaluations
from experiments.post_training import mochi_rl, rl_compose
from experiments.post_training.rl_tests.fixtures import write_skyrl_run_record
from experiments.post_training.skyrl_evaluation import catalog_evaluation, evaluate_model, export_evaluation_step
from tests.rl.micro210_fixture import policy_fixture_reference


def evaluate_smoke(source, serving):
    return evaluate_model(
        source,
        serving,
        resolve_evaluations("smoke"),
        version=source.step.version,
        accelerator="H100x1",
        submission_cluster="marin",
        federated_cluster="cw-rno2a",
    )


def remove_attempt_ids(path, *, legacy=False):
    receipt = ArtifactRecord.model_validate_json(path.read_text())
    assert receipt.result is not None
    del receipt.result["run_id"], receipt.result["attempt_id"]
    if legacy:
        receipt = ArtifactRecord(result_type=receipt.result_type, result=receipt.result)
    path.write_text(receipt.model_dump_json())


@pytest.mark.parametrize(
    "model_fields",
    [
        pytest.param({"location": "s3://checkpoints/unrelated"}, id="location"),
        {"revision": "starter-commit"},
        {"identity": "starter"},
        {"tokenizer": "org/tokenizer"},
        {"tokenizer_revision": "starter-commit"},
    ],
)
def test_evaluate_rejects_invalid_inputs_when_building_the_step(tmp_path, model_fields):
    source = ArtifactHfModel(ArtifactStep.adopt("tests/model", "2026.10.26", str(tmp_path)))
    with pytest.raises(ValueError, match="come from the model"):
        evaluate_smoke(source, ModelConfig(name="candidate", **{"location": "", **model_fields}))


@pytest.mark.parametrize(
    ("source_kind", "evaluation_name"), [("rl", "smoke"), ("without-ids", "smoke"), ("hf", "smoke"), ("rl", "micro210")]
)
def test_evaluate_preserves_model_and_attempt_metadata_on_read_and_reuse(
    tmp_path, monkeypatch, source_kind, evaluation_name
):
    root = tmp_path / "model"
    export = root / "policy"
    export.mkdir(parents=True)
    producer_receipt = root / RECORD_FILENAME
    version = "2026.10.26"
    ids = {"run_id": "training-run", "attempt_id": "training-attempt"} if source_kind == "rl" else {}
    tokenizer = str(root / "pinned-tokenizer") if source_kind == "rl" else None
    if tokenizer is not None:
        Path(tokenizer).mkdir()
    tokenizer_revision = "tokenizer-commit" if tokenizer is not None else None
    export_uri = str(export)
    if source_kind == "hf":
        template = "{{ messages[0].content }}"
        backend = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
        PreTrainedTokenizerFast(tokenizer_object=backend, chat_template=template).save_pretrained(export)
        snapshot = StoragePath(f"memory://{tmp_path.name}/policy")
        for file in export.iterdir():
            (snapshot / file.name).write_bytes(file.read_bytes())
        export_uri = str(snapshot)
        monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "hf-cache"))
        source = ArtifactHfModel(ArtifactStep.adopt("tests/model", version, export_uri), relative_path="")
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
        if source_kind == "without-ids":
            remove_attempt_ids(producer_receipt, legacy=True)

    preset = catalog_evaluation(
        evaluation_name, ModelConfig(name="candidate", location=""), accelerator="H100x1", cluster="cw-rno2a"
    )
    if evaluation_name == "micro210":
        layout = replace(mochi_rl.MOCHI.by_name["train-long"].recipe.layout, excluded_nvlink_domains=("reserved-rack",))
        source = SkyRLPolicyModel(
            replace(source.step, runtime_args={"skyrl_execution": rl_compose.layout_execution(layout)})
        )
        monkeypatch.setattr(micro210, "POLICY_REFERENCE", policy_fixture_reference())
        secret_refs = []

        def resolve_secret(ref):
            secret_refs.append(ref)
            return SimpleNamespace(value="test")

        monkeypatch.setattr(launch_module, "resolve_secret_spec", resolve_secret)
        preset = mochi_rl.evaluation_presets(evaluation_name)

    class Client:
        def submit(self, *, entrypoint, constraints, **_kwargs):
            _, (batch,), _ = entrypoint.resolve()
            if source_kind == "hf":
                assert (
                    read_tool_chat_template(batch.model.tokenizer, batch.model.effective_tokenizer_revision) == template
                )
            if evaluation_name == "micro210":
                [placement] = [constraint for constraint in constraints if constraint.key == CLUSTER_CONSTRAINT_KEY]
                assert placement.op is ConstraintOp.EQ and placement.values[0].value == layout.cluster
                assert any(ref[0] == "env:TOGETHER_API_KEY" for ref in secret_refs)
                assert {evaluation.identity.eval_ref.name for evaluation in batch.evaluations} == {
                    "micro210-chat",
                    "micro210-pi",
                }
                [inference] = _inference_configs(batch, {}).values()
                assert inference.iris.serving_geometry.label == "PP1 TP1 DP16"
                resources = inference.iris.worker_resources
                assert resources.replicas == 4 and resources.chip_count() == 16
                assert "--data-parallel-size" not in inference.engine.extra_args
                [exclusion] = [
                    constraint
                    for constraint in convert_constraints(resources)
                    if constraint.key == COSCHEDULE_NVLINK_DOMAIN
                ]
                assert exclusion.op is ConstraintOp.NE and exclusion.label_value == "reserved-rack"
            for evaluation in batch.evaluations:
                _record(batch, evaluation.identity, RunStatus.SUCCEEDED, None, {}, {}, {})
            return SimpleNamespace(wait=lambda **_kwargs: None)

    monkeypatch.setattr(pipeline_module, "iris_ctx", lambda: SimpleNamespace(client=Client()))
    monkeypatch.setattr(launch_module, "CW_RECORDS_PREFIX", str(tmp_path / "records"))
    evaluation = export_evaluation_step(preset, source, version=version)
    result = evaluation.run(materialized_config(evaluation, str(tmp_path)))

    assert len(result.run_ids) == 2
    for run_id in result.run_ids:
        path = Path(record_path(result.records_prefix, run_id))
        record_json = json.loads(path.read_text())
        if evaluation_name == "micro210":
            assert record_json["eval"]["name"] in {"micro210-chat", "micro210-pi"}
            assert record_json["hardware"]["task_count"] == 4
        model = record_json["model"]
        assert model["location"] == export_uri
        config = model["config"]
        assert config["identity"] == f"tests/model@{version}:{source.step.fingerprint()}"
        assert config["revision"] is None
        assert config["tokenizer"] == (tokenizer or export_uri)
        if tokenizer_revision is not None:
            assert config["tokenizer_revision"] == tokenizer_revision
        if source_kind == "rl":
            assert record_json["rl_attempt"] == ids
        else:
            assert "rl_attempt" not in record_json

    if source_kind == "rl":
        remove_attempt_ids(producer_receipt)
        assert evaluation.run(materialized_config(evaluation, str(tmp_path))).run_ids == result.run_ids

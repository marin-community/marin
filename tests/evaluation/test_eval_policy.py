# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Admission and historical-record behavior for the current verified policy."""

from dataclasses import asdict
from pathlib import Path

import pytest
from marin.evaluation.eval_policy import (
    VERIFIED_POLICY,
    VERIFIED_VERSION,
    policy_violations,
    runtime_violations,
)
from marin.evaluation.evalchemy.config import load_evalchemy_config
from marin.evaluation.hardware import Platform
from marin.evaluation.model_config import ModelConfig, ResourceHint, ServeConfig, load_model_config
from marin.evaluation.records import EvalRef, EvalTaskRef, HarborRef, ModelConfigRef, ModelRef
from marin.evaluation.runner import LaunchProvenance

from experiments.evaluation.evals import EvalchemyDefinition
from experiments.evaluation.launch import LaunchSpec, build_evaluation_batch

CAMPAIGN_ROOT = Path(__file__).resolve().parents[2] / "experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled"


def _model(name: str = "Qwen-Qwen3.6-35B-A3B") -> ModelRef:
    config = load_model_config(CAMPAIGN_ROOT / "model-configs" / f"{name}.yaml")
    return ModelRef(
        name=config.name,
        location=config.location,
        backend=config.serve.backend.value,
        config=ModelConfigRef.model_validate(asdict(config)),
    )


def _evaluation(name: str, model_name: str = "Qwen-Qwen3.6-35B-A3B") -> EvalRef:
    config = load_model_config(CAMPAIGN_ROOT / "model-configs" / f"{model_name}.yaml")
    path = CAMPAIGN_ROOT / "evalchemy-configs" / f"{name}.yaml"
    definition = EvalchemyDefinition(name, path)
    return definition.record_ref_for(definition.config_for(load_evalchemy_config(path), config, None))


@pytest.mark.parametrize("benchmark", sorted(VERIFIED_POLICY["evalchemy"]))
@pytest.mark.parametrize(
    "model_name",
    [
        "Qwen-Qwen3.6-35B-A3B",
        "open-athena-Grug-67B-A2B-Datakit-SFT-262K-2026.09.21",
        "openai-gpt-oss-20b",
    ],
)
def test_verified_policy_accepts_effective_benchmark_and_model_settings(benchmark, model_name):
    assert policy_violations(VERIFIED_VERSION, _model(model_name), _evaluation(benchmark, model_name)) == ()


@pytest.mark.parametrize(
    "change",
    [
        {"seed": 999},
        {"max_gen_toks": 256},
        {"max_eval_instances": 5},
        {"chat_template_kwargs": {"enable_thinking": False}},
        {"extra_gen_kwargs": {"temperature": "0.1"}},
    ],
)
def test_verified_policy_rejects_scoring_changes(change):
    evaluation = _evaluation("aime24")
    assert evaluation.evalchemy is not None
    changed = evaluation.model_copy(update={"evalchemy": evaluation.evalchemy.model_copy(update=change)})
    assert policy_violations(VERIFIED_VERSION, _model(), changed)


@pytest.mark.parametrize("benchmark", ["bixbench-pi", "swebench-verified", "tau3-pi"])
def test_verified_harbor_policy_preserves_model_specific_sources(benchmark):
    profile = VERIFIED_POLICY["harbor"][benchmark]
    model = _model("inclusionAI-Ling-lite-1.5" if benchmark != "swebench-verified" else "IFM-K2-Horizon-MoVA-36B-A4B")
    assert model.config is not None
    variant = profile["variants"][model.location]
    source_config = model.config
    model = model.model_copy(
        update={
            "config": source_config.model_copy(
                update={"serve": source_config.serve.model_copy(update={"max_model_len": 131072})}
            ),
            "source_config": source_config,
        }
    )
    info = variant["model_info"]
    evaluation = EvalRef(
        name=benchmark,
        mechanism="harbor",
        source_digest=variant["source_digest"],
        tasks=(EvalTaskRef(name=profile["identity"]["dataset"], num_fewshot=None),),
        harbor=HarborRef(
            **profile["identity"],
            config_digest="sha256:" + "1" * 64,
            max_input_tokens=info["max_input_tokens"],
            max_output_tokens=info.get("max_output_tokens", 8192),
        ),
    )
    assert policy_violations(VERIFIED_VERSION, model, evaluation) == ()
    other_source = profile["variants"]["default"]["source_digest"]
    assert policy_violations(VERIFIED_VERSION, model, evaluation.model_copy(update={"source_digest": other_source}))


def test_verified_runtime_uses_the_full_descriptor_not_a_matching_substring():
    evaluation = _evaluation("math500")
    runtime = VERIFIED_POLICY["runtimes"]["math500"]
    assert runtime_violations(VERIFIED_VERSION, evaluation, runtime) == ()
    assert runtime_violations(VERIFIED_VERSION, evaluation, "unrelated " + runtime)
    assert runtime_violations(VERIFIED_VERSION, evaluation, runtime.replace("sha256=", "sha256=changed"))


def test_verified_policy_reports_missing_optional_record_settings():
    evaluation = _evaluation("financebench")
    assert evaluation.evalchemy is not None
    without_judge = evaluation.model_copy(update={"evalchemy": evaluation.evalchemy.model_copy(update={"judge": None})})
    assert policy_violations(VERIFIED_VERSION, _model(), without_judge)
    profile = VERIFIED_POLICY["harbor"]["swebench-verified"]
    without_budgets = EvalRef(
        name="swebench-verified",
        mechanism="harbor",
        source_digest=profile["variants"]["default"]["source_digest"],
        tasks=(EvalTaskRef(name=profile["identity"]["dataset"], num_fewshot=None),),
        harbor=HarborRef(**profile["identity"], config_digest="sha256:" + "1" * 64),
    )
    assert policy_violations(VERIFIED_VERSION, _model(), without_budgets)


def test_verified_launch_accepts_a_benchmark_subset_and_records_its_runtime(monkeypatch):
    monkeypatch.setattr("experiments.evaluation.launch._capability_origin", lambda _cluster: "https://iris.example")
    spec = LaunchSpec(
        model=ModelConfig(
            name="new-model",
            location="org/new-model",
            revision="1" * 40,
            resource_hint=ResourceHint(gpu={"H100": 1}, memory="16g"),
            serve=ServeConfig(max_model_len=32768, auto_overrides=False),
        ),
        evals=(),
        evalchemy_definitions=(
            EvalchemyDefinition("gpqa-diamond", CAMPAIGN_ROOT / "evalchemy-configs/gpqa-diamond.yaml"),
        ),
        harbor_definitions=(),
        platform=Platform.GPU,
        accelerator="H100x1",
        limit=None,
        records_prefix="memory://verified",
        submission_cluster="marin",
        federated_cluster=None,
        priority_band=0,
        version=VERIFIED_VERSION,
    )
    batch = build_evaluation_batch(spec, LaunchProvenance(git_sha="test", launch_host="test"), "test")
    assert batch.version == VERIFIED_VERSION
    assert [item.identity.eval_ref.name for item in batch.evaluations] == ["gpqa-diamond"]
    assert (
        runtime_violations(
            VERIFIED_VERSION, batch.evaluations[0].identity.eval_ref, batch.evaluations[0].identity.eval_runtime
        )
        == ()
    )


@pytest.mark.parametrize(
    "version",
    [
        "eval-policy-2026-09-16-verified",
        "eval-policy-2026-09-24-verified",
        "eval-policy-2026-10-19-verified",
        "eval-policy-2026-10-20-verified",
    ],
)
def test_retired_policy_labels_cannot_launch_new_verified_evaluations(version):
    assert policy_violations(version, _model(), _evaluation("math500"))

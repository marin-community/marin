# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Admission for the October 5 evaluation campaign."""

from dataclasses import asdict
from pathlib import Path

import pytest
from marin.evaluation.campaign_policy import CAMPAIGNS, OCTOBER_5_ORIGINAL_HARBOR_COMMIT, OCTOBER_5_VERSION
from marin.evaluation.eval_policy import policy_violations, runtime_violations
from marin.evaluation.evalchemy.config import load_evalchemy_config
from marin.evaluation.harbor import driver_config
from marin.evaluation.model_config import load_model_config
from marin.evaluation.records import EvalRef, EvalTaskRef, HarborRef, ModelConfigRef, ModelRef
from marin.external_dependencies import EVALCHEMY

from experiments.evaluation.evals import EvalchemyDefinition

CAMPAIGN_ROOT = Path(__file__).resolve().parents[2] / "experiments/evaluation/campaigns/eval-campaign-09-25"
CAMPAIGN = CAMPAIGNS[OCTOBER_5_VERSION]


def test_harbor_runtime_selection_uses_the_lock_not_the_shared_declaration(tmp_path, monkeypatch):
    locked_commit = "1" * 40
    (tmp_path / "uv.lock").write_text(
        f'[[package]]\nname = "harbor"\nsource = {{ git = "https://example.com/harbor#{locked_commit}" }}\n'
    )
    monkeypatch.setattr(driver_config, "_harbor_env_dir", lambda project: tmp_path)
    assert driver_config.harbor_runtime_project(locked_commit) == driver_config.HARBOR_RUNTIME_PROJECT
    archived_commit = "2" * 40
    assert driver_config.harbor_runtime_project(archived_commit) == (
        f"{driver_config.HARBOR_RUNTIME_PROJECT}/pins/{archived_commit}"
    )


def _campaign_run(model_name: str, benchmark: str) -> tuple[ModelRef, EvalRef]:
    model = load_model_config(CAMPAIGN_ROOT / "model-configs" / f"{model_name}.yaml")
    path = CAMPAIGN_ROOT / "evalchemy-configs" / f"{benchmark}.yaml"
    definition = EvalchemyDefinition(benchmark, path)
    effective = definition.config_for(load_evalchemy_config(path), model, None, EVALCHEMY)
    saved = ModelConfigRef.model_validate(asdict(model))
    return ModelRef(
        name=model.name, location=model.location, backend=model.serve.backend.value, config=saved
    ), definition.record_ref_for(effective)


@pytest.mark.parametrize(
    "model_name,benchmark",
    [
        (model, path.stem)
        for model in (
            "Qwen-Qwen3.6-35B-A3B",
            "open-athena-Grug-67B-A2B-Datakit-SFT-262K-2026.09.21",
            "open-athena-Grug-67B-A2B-Antidoom-RLVR1-Reference-KL-Step20-2026.10.04",
        )
        for path in sorted((CAMPAIGN_ROOT / "evalchemy-configs").glob("*.yaml"))
    ]
    + [
        ("openai-gpt-oss-20b", "cruxeval"),
    ],
)
def test_campaign_accepts_effective_configs_including_native_thinking_translation(model_name, benchmark):
    model, evaluation = _campaign_run(model_name, benchmark)
    assert policy_violations(OCTOBER_5_VERSION, model, evaluation) == ()


@pytest.mark.parametrize(
    "change",
    [
        {"seed": 43},
        {"max_gen_toks": 256},
        {"max_eval_instances": 5},
        {"chat_template_kwargs": {"enable_thinking": False}},
        {"extra_gen_kwargs": {"temperature": "0.1"}},
    ],
)
def test_campaign_rejects_effective_settings_drift(change):
    model, evaluation = _campaign_run("Qwen-Qwen3.6-35B-A3B", "aime24")
    assert evaluation.evalchemy is not None
    changed = evaluation.model_copy(update={"evalchemy": evaluation.evalchemy.model_copy(update=change)})
    assert policy_violations(OCTOBER_5_VERSION, model, changed)


@pytest.mark.parametrize("change", [{"location": "different/model"}, {"revision": "main"}])
def test_campaign_rejects_effective_model_identity_different_from_source(change):
    model, evaluation = _campaign_run("Qwen-Qwen3.6-35B-A3B", "math500")
    assert model.config is not None
    changed = model.model_copy(update={"source_config": model.config, "config": model.config.model_copy(update=change)})
    assert policy_violations(OCTOBER_5_VERSION, changed, evaluation)


@pytest.mark.parametrize(
    "model_name,variant,context",
    [("Qwen-Qwen3.6-35B-A3B", "standard", 65536), ("inclusionAI-Ling-lite-1.5", "native32k", 32768)],
)
def test_campaign_accepts_model_specific_harbor_profile_and_rejects_other_variant(model_name, variant, context):
    model, _ = _campaign_run(model_name, "math500")
    profile = CAMPAIGN.harbor["bixbench-pi"]
    evaluation = EvalRef(
        name="bixbench-pi",
        mechanism="harbor",
        source_digest=profile.sources[variant],
        tasks=(EvalTaskRef(name=profile.identity.dataset, num_fewshot=None),),
        harbor=HarborRef(
            **profile.identity.model_dump(),
            config_digest="sha256:" + "1" * 64,
            max_input_tokens=context,
            max_output_tokens=16384,
        ),
    )
    assert policy_violations(OCTOBER_5_VERSION, model, evaluation) == ()
    other = "native32k" if variant == "standard" else "standard"
    changed = evaluation.model_copy(update={"source_digest": profile.sources[other]})
    assert policy_violations(OCTOBER_5_VERSION, model, changed)

    assert evaluation.harbor is not None
    for field in ("dataset", "version", "agent", "env"):
        changed = evaluation.model_copy(update={"harbor": evaluation.harbor.model_copy(update={field: "different"})})
        assert policy_violations(OCTOBER_5_VERSION, model, changed)
    changed = evaluation.model_copy(update={"harbor": evaluation.harbor.model_copy(update={"max_output_tokens": 256})})
    assert policy_violations(OCTOBER_5_VERSION, model, changed)


def test_campaign_accepts_sotopia_hugging_face_record_identity():
    model, _ = _campaign_run("Qwen-Qwen3.6-35B-A3B", "math500")
    profile = CAMPAIGN.harbor["sotopia-hard"]
    identity = {**profile.identity.model_dump(), "dataset": "hf://open-athena/sotopia-hard-harbor"}
    evaluation = EvalRef(
        name="sotopia-hard",
        mechanism="harbor",
        source_digest=profile.sources["standard"],
        tasks=(EvalTaskRef(name=identity["dataset"], num_fewshot=None),),
        harbor=HarborRef(**identity, config_digest="sha256:" + "1" * 64, max_input_tokens=32768, max_output_tokens=8192),
    )
    assert policy_violations(OCTOBER_5_VERSION, model, evaluation) == ()


def test_campaign_rejects_runtime_and_source_drift():
    model, evaluation = _campaign_run("Qwen-Qwen3.6-35B-A3B", "nupa")
    commit = CAMPAIGN.runtimes["evalchemy"]
    assert (
        runtime_violations(
            OCTOBER_5_VERSION,
            evaluation,
            f"evalchemy @ git+https://github.com/marin-community/evalchemy.git@{commit}",
        )
        == ()
    )
    assert runtime_violations(OCTOBER_5_VERSION, evaluation, "evalchemy@" + "0" * 40)
    changed = evaluation.model_copy(update={"source_digest": "sha256:" + "0" * 64})
    assert policy_violations(OCTOBER_5_VERSION, model, changed)


@pytest.mark.parametrize("mechanism", ["evalchemy", "harbor"])
def test_campaign_admits_only_its_pinned_runtime(mechanism):
    _, evaluation = _campaign_run("Qwen-Qwen3.6-35B-A3B", "math500")
    evaluation = evaluation.model_copy(update={"mechanism": mechanism})
    allowed = CAMPAIGN.runtimes[mechanism]
    assert runtime_violations(OCTOBER_5_VERSION, evaluation, mechanism + "@" + allowed) == ()
    assert runtime_violations(OCTOBER_5_VERSION, evaluation, mechanism + "@" + "0" * 40)


@pytest.mark.parametrize("commit", [CAMPAIGN.runtimes["harbor"], OCTOBER_5_ORIGINAL_HARBOR_COMMIT])
def test_october5_recovery_preserves_original_records_and_checks_preflight_pin(commit):
    _, evaluation = _campaign_run("Qwen-Qwen3.6-35B-A3B", "math500")
    evaluation = evaluation.model_copy(
        update={
            "mechanism": "harbor",
            "harbor": HarborRef(**CAMPAIGN.harbor["tb2-recovery"].identity.model_dump(), harbor_config_commit=commit),
        }
    )
    assert runtime_violations(OCTOBER_5_VERSION, evaluation, "harbor@" + commit) == ()
    assert evaluation.harbor is not None
    mismatched = evaluation.model_copy(
        update={"harbor": evaluation.harbor.model_copy(update={"harbor_config_commit": "0" * 40})}
    )
    assert runtime_violations(OCTOBER_5_VERSION, mismatched, "harbor@" + commit)

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Admission for the two published evaluation cohorts."""

from dataclasses import asdict

from marin.evaluation import eval_policy
from marin.evaluation.eval_policy import (
    HARBOR_COMMIT,
    SEPTEMBER_16_VERSION,
    SEPTEMBER_24_VERSION,
    policy_violations,
    runtime_violations,
)
from marin.evaluation.eval_policy_sources import POLICY_SOURCE_DIGESTS
from marin.evaluation.model_config import ModelConfig
from marin.evaluation.model_identity import comparison_model_name, model_config_digest
from marin.evaluation.records import EvalchemyRef, EvalRef, EvalTaskRef, ModelConfigRef, ModelRef


def _model(generation_kwargs: dict[str, bool | None] | None = None) -> ModelRef:
    config = ModelConfigRef.model_validate(asdict(ModelConfig(name="model", location="org/model")))
    if generation_kwargs is not None:
        config = config.model_copy(
            update={"generation": config.generation.model_copy(update={"chat_template_kwargs": generation_kwargs})}
        )
    return ModelRef(name="model", location="org/model", backend="vllm", config=config)


def _math500(thinking: bool | None, *, shots: int = 0, version: str = SEPTEMBER_24_VERSION) -> EvalRef:
    return EvalRef(
        name="math500",
        mechanism="evalchemy",
        source_digest=POLICY_SOURCE_DIGESTS[version]["math500"],
        tasks=(EvalTaskRef(name="MATH500", num_fewshot=shots, generation=True),),
        evalchemy=EvalchemyRef(
            apply_chat_template=True,
            max_gen_toks=None,
            max_eval_instances=None,
            num_concurrent=16,
            batch_size="1",
            seed=42,
            chat_template_kwargs={"enable_thinking": thinking} if thinking is not None else {},
        ),
    )


def test_september_24_requires_explicit_benchmark_thinking_mode():
    model = _model({"enable_thinking": False})
    assert policy_violations(SEPTEMBER_24_VERSION, model, _math500(True)) == ()
    assert any(
        "enable_thinking=True" in problem for problem in policy_violations(SEPTEMBER_24_VERSION, model, _math500(False))
    )
    assert any(
        "enable_thinking=True" in problem for problem in policy_violations(SEPTEMBER_24_VERSION, model, _math500(None))
    )


def test_september_16_preserves_model_defaults_and_checks_shots():
    assert policy_violations(SEPTEMBER_16_VERSION, _model(), _math500(None, version=SEPTEMBER_16_VERSION)) == ()
    assert "num_fewshot must be 0" in policy_violations(
        SEPTEMBER_16_VERSION, _model(), _math500(None, shots=1, version=SEPTEMBER_16_VERSION)
    )
    original_null = _math500(None, version=SEPTEMBER_16_VERSION)
    assert original_null.evalchemy is not None
    original_null = original_null.model_copy(
        update={
            "evalchemy": original_null.evalchemy.model_copy(update={"chat_template_kwargs": {"enable_thinking": None}})
        }
    )
    assert policy_violations(SEPTEMBER_16_VERSION, _model({"enable_thinking": None}), original_null) == ()


def test_legacy_policy_label_is_not_verified():
    assert policy_violations("eval-policy-updated", _model(), _math500(None))


def test_runtime_pin_is_selected_by_cohort(monkeypatch):
    monkeypatch.setattr(
        eval_policy,
        "RUNTIME_COMMITS",
        {
            SEPTEMBER_16_VERSION: {"evalchemy": "older", "harbor": HARBOR_COMMIT},
            SEPTEMBER_24_VERSION: {"evalchemy": "newer", "harbor": HARBOR_COMMIT},
        },
    )
    evaluation = _math500(True)

    assert runtime_violations(SEPTEMBER_16_VERSION, evaluation, "older") == ()
    assert runtime_violations(SEPTEMBER_24_VERSION, evaluation, "newer") == ()
    assert runtime_violations(SEPTEMBER_16_VERSION, evaluation, "newer")


def test_model_configurations_have_distinct_comparison_identities():
    first = _model({"enable_thinking": False})
    second = _model({"enable_thinking": True})
    assert comparison_model_name(first) != comparison_model_name(second)
    assert comparison_model_name(first) == comparison_model_name(
        first.model_copy(update={"config_digest": model_config_digest(first.config)})
    )

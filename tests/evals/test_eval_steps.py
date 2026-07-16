# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Composition of eval steps: distinct identities per task group, backend routing, report wiring.

Pure graph construction — no eval runs. The regression these guard is the pre-redesign bug where a
multi-group suite gave every group the same ``name@version`` (so ``StepRunner`` deduped by output path
and silently dropped all but one group).
"""

from fray.cluster import ResourceConfig
from marin.execution.build_context import BuildContext, VersionCodex, build_context
from marin.execution.lazy import ArtifactStep, materialized_config
from marin.training.training import LevanterCheckpoint

from experiments.evals.evals import (
    Backend,
    EvalGroup,
    base_model_evals,
    eval_report,
    eval_step,
    eval_steps,
    key_evals,
)

_CPU = ResourceConfig.with_cpu(cpu=1)
_V = "2026.07.16"


def _model() -> ArtifactStep[LevanterCheckpoint]:
    return ArtifactStep.adopt(
        "perplexity-models/llama-200m",
        "2026.06.30",
        "gs://test-bucket/checkpoints/llama-200m",
        kind=LevanterCheckpoint,
    )


def test_base_model_suite_groups_have_distinct_names():
    """The four MMLU cuts + core + generation each get a unique address — none collide."""
    steps = eval_steps(_model(), base_model_evals(resources=_CPU), version=_V)
    names = [s.name for s in steps]
    assert len(names) == 5
    assert len(set(names)) == 5, f"duplicate step names would dedup to one output: {names}"
    assert any(name.endswith("/mmlu_0shot") for name in names)
    assert any(name.endswith("/mmlu_pro_5shot") for name in names)


def test_eval_step_routes_backend_to_result_type():
    model = _model()
    levanter = eval_step(model, EvalGroup((), Backend.LEVANTER, _CPU, "mcq"), version=_V)
    lm_eval = eval_step(model, EvalGroup((), Backend.LM_EVAL, _CPU, "gen"), version=_V)
    assert levanter.name == "evaluation/lm_evaluation_harness_levanter/perplexity-models/llama-200m/mcq"
    assert lm_eval.name == "evaluation/lm_evaluation_harness/perplexity-models/llama-200m/gen"


def test_eval_report_depends_on_every_result():
    model = _model()
    results = eval_steps(model, key_evals(resources=_CPU), version=_V)
    report = eval_report(results, name=f"{model.name}/key", version=_V)
    assert report.deps == tuple(results)
    assert report.name == "evaluation/report/perplexity-models/llama-200m/key"


def test_eval_step_builds_evaluation_config():
    model = _model()
    group = EvalGroup(key_evals(resources=_CPU)[1].tasks, Backend.LEVANTER, _CPU, "key_multiple_choice")
    step = eval_step(model, group, version=_V)
    config = materialized_config(step, prefix="gs://bucket")
    assert config.evaluator == "levanter_lm_evaluation_harness"
    assert config.evals == list(group.tasks)
    assert config.model_path == model.adopt_source


def test_eval_step_defers_version_to_build_context():
    """With no explicit version, the step resolves its version from the ambient BuildContext."""
    model = _model()
    with build_context(BuildContext(versions=VersionCodex(default="dev"))):
        step = eval_step(model, EvalGroup((), Backend.LEVANTER, _CPU, "mcq"))
    assert step.version == "dev"

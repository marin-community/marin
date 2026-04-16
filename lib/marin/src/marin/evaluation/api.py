# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Public API for running evaluators against an OpenAI-compatible endpoint.

Every evaluator takes `(RunningModel, <run-config>)`. The caller owns the
server lifecycle (via a `ModelLauncher` or by constructing a `RunningModel`
directly for an already-running endpoint).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.inference.model_launcher import RunningModel


@dataclass(frozen=True)
class LmEvalRun:
    """Eval-only configuration shared by `run_lm_eval` and `run_evalchemy`.

    Both evaluators drive lm-eval's `local-completions` backend; the run shape
    is identical. Lifecycle and task-registry details differ between them, so
    they stay in separate implementation classes.
    """

    evals: list[EvalTaskConfig]
    output_path: str
    apply_chat_template: bool
    generation_params: dict
    extra_model_args: tuple[str, ...] = ()
    """Extra `k=v` strings appended to lm-eval's --model_args (client-side knobs)."""

    batch_size: str = "auto"
    """Passed to lm-eval / evalchemy `--batch_size`. Accepts "auto" or an int as string."""

    max_eval_instances: int | None = None
    wandb_tags: list[str] | None = None
    base_eval_run_name: str | None = None


@dataclass(frozen=True)
class HarborRun:
    """Eval-only configuration for Harbor benchmarks."""

    evals: list[EvalTaskConfig]
    output_path: str
    dataset: str
    version: str
    agent: str
    n_concurrent: int
    env: str = "local"
    """Harbor environment name. Maps to Harbor's EnvironmentType enum (`local`
    → Docker, plus `daytona`, `e2b`, `modal`, `runloop`, `gke`, ...). Harbor
    validates at construction time — we pass through verbatim."""

    agent_kwargs: dict = field(default_factory=dict)
    model_info: dict | None = None
    """Optional LiteLLM ModelInfo passthrough."""

    max_eval_instances: int | None = None
    wandb_tags: list[str] | None = None


def run_lm_eval(model: RunningModel, run: LmEvalRun) -> None:
    """Run EleutherAI lm-evaluation-harness against the model's OpenAI endpoint."""
    from marin.evaluation.evaluators.lm_evaluation_harness_evaluator import LmEvalEvaluator

    LmEvalEvaluator(run).run(model)


def run_evalchemy(model: RunningModel, run: LmEvalRun) -> None:
    """Run Evalchemy (lm-eval-based reasoning benchmarks) against the model's OpenAI endpoint."""
    from marin.evaluation.evaluators.evalchemy_evaluator import EvalchemyEvaluator

    EvalchemyEvaluator(run).run(model)


def run_harbor(model: RunningModel, run: HarborRun) -> None:
    """Run Harbor benchmarks against the model's OpenAI endpoint (or LiteLLM provider)."""
    from marin.evaluation.evaluators.harbor_evaluator import HarborEvaluator

    HarborEvaluator(run).run(model)

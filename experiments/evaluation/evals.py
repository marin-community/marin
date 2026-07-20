# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The eval-suite registry for the launcher.

Each :class:`EvalSuiteConfig` bundles the lm-eval tasks that run together against one served endpoint,
plus generation limits. The launcher today drives the ``EVALCHEMY`` mechanism (served OpenAI URL);
``HARBOR`` is registered as a mechanism but not yet wired -- selecting it raises at launch.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from marin.evaluation.evaluation_config import EvalTaskConfig


class EvalMechanism(StrEnum):
    """How a suite is executed. ``EVALCHEMY`` serves the model and runs lm-eval against its OpenAI URL;
    ``HARBOR`` (agentic registry benchmarks) is not yet wired into this launcher."""

    EVALCHEMY = "evalchemy"
    HARBOR = "harbor"


@dataclass(frozen=True)
class EvalSuiteConfig:
    """A named set of lm-eval tasks run together, with per-suite generation limits."""

    name: str
    mechanism: EvalMechanism
    tasks: tuple[EvalTaskConfig, ...]
    max_gen_toks: int = 2048
    max_eval_instances: int | None = None


def _mcq_eval(name: str, task: str, shots: int) -> EvalSuiteConfig:
    """A single loglikelihood-MCQ benchmark as its own eval (one run, one record, one parquet)."""
    return EvalSuiteConfig(
        name=name,
        mechanism=EvalMechanism.EVALCHEMY,
        tasks=(EvalTaskConfig(task, shots, task_alias=f"{task}_{shots}shot"),),
        max_gen_toks=256,
    )


EVALS: dict[str, EvalSuiteConfig] = {
    # The core benchmarks, one eval per task so every model x task pair is its own run with its own
    # serve/eval jobs, record, and per-question parquet. Shot counts follow the HF OpenLLM-v1
    # conventions so scores line up with public leaderboards.
    "mmlu": _mcq_eval("mmlu", "mmlu", 5),
    "arc-challenge": _mcq_eval("arc-challenge", "arc_challenge", 25),
    "hellaswag": _mcq_eval("hellaswag", "hellaswag", 10),
    "winogrande": _mcq_eval("winogrande", "winogrande", 5),
    "truthfulqa": _mcq_eval("truthfulqa", "truthfulqa_mc2", 0),
    "boolq": _mcq_eval("boolq", "boolq", 0),
    "piqa": _mcq_eval("piqa", "piqa", 0),
    "openbookqa": _mcq_eval("openbookqa", "openbookqa", 0),
    "gsm8k": EvalSuiteConfig(
        name="gsm8k",
        mechanism=EvalMechanism.EVALCHEMY,
        tasks=(EvalTaskConfig("gsm8k", 5, task_alias="gsm8k_5shot", generation=True),),
        max_gen_toks=512,
    ),
    # Evalchemy's chat-native MATH500 benchmark (boxed-answer extraction over the HuggingFaceH4
    # MATH-500 split). A messages-based task: it runs through the chat route, so every model needs
    # a server-side chat template (snowball serves one via its vLLM args).
    "math500": EvalSuiteConfig(
        name="math500",
        mechanism=EvalMechanism.EVALCHEMY,
        tasks=(EvalTaskConfig("MATH500", 0, task_alias="math500", generation=True),),
        max_gen_toks=8192,
    ),
    "humaneval": EvalSuiteConfig(
        name="humaneval",
        mechanism=EvalMechanism.EVALCHEMY,
        tasks=(
            EvalTaskConfig(
                "humaneval",
                0,
                task_alias="humaneval_0shot",
                generation=True,
                unsafe_code=True,
                completion_only=True,
            ),
        ),
        max_gen_toks=1024,
    ),
    "mmlu-smoke": EvalSuiteConfig(
        name="mmlu-smoke",
        mechanism=EvalMechanism.EVALCHEMY,
        tasks=(EvalTaskConfig("mmlu_abstract_algebra", 0, task_alias="mmlu_abstract_algebra_0shot"),),
        max_gen_toks=256,
        max_eval_instances=64,
    ),
    "gsm8k-smoke": EvalSuiteConfig(
        name="gsm8k-smoke",
        mechanism=EvalMechanism.EVALCHEMY,
        tasks=(EvalTaskConfig("gsm8k", 5, task_alias="gsm8k_5shot", generation=True),),
        max_gen_toks=512,
        max_eval_instances=128,
    ),
}

# A fast cluster smoke: one small MCQ cut plus a capped gsm8k generation task.
SMOKE_EVALS: tuple[str, ...] = ("mmlu-smoke", "gsm8k-smoke")

# The comprehensive per-model benchmark set: every model x task pair runs (and is recorded) as its
# own run, so the dashboard shows the full N-models x M-tasks grid of runs.
CORE_EVALS: tuple[str, ...] = (
    "mmlu",
    "gsm8k",
    "arc-challenge",
    "hellaswag",
    "winogrande",
    "truthfulqa",
    "boolq",
    "piqa",
    "openbookqa",
    "humaneval",
    "math500",
)

# Named suite groups selectable by name on the CLI (``--evals smoke``).
SUITES: dict[str, tuple[str, ...]] = {
    "smoke": SMOKE_EVALS,
    "core": CORE_EVALS,
}

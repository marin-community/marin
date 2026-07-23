# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The eval-suite registry for the launcher.

Each :class:`EvalSuiteConfig` bundles the lm-eval tasks that run together against one served endpoint,
plus generation limits. The launcher today drives the ``EVALCHEMY`` mechanism (served OpenAI URL);
``HARBOR`` is registered as a mechanism but not yet wired -- selecting it raises at launch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from marin.evaluation.evaluation_config import EvalTaskConfig


class EvalMechanism(StrEnum):
    """How a suite is executed. ``EVALCHEMY`` serves the model and runs lm-eval against its OpenAI URL;
    ``HARBOR`` serves the model and runs a Harbor registry dataset's agentic trials against it."""

    EVALCHEMY = "evalchemy"
    HARBOR = "harbor"


@dataclass(frozen=True)
class HarborSpec:
    """A Harbor registry dataset run against the served model.

    ``env`` selects the sandbox backend: ``daytona`` (off-cluster, needs ``DAYTONA_EVAL_API_KEY``),
    ``local`` (Docker on the worker), or ``iris``. ``agent`` is the Harbor agent that drives each task
    against the served endpoint (``hosted_vllm/<served-name>``). ``max_output_tokens`` is the litellm
    generation budget (distinct from lm-eval's ``max_gen_toks``).
    """

    dataset: str
    version: str = "1.0"
    agent: str = "terminus-2"
    env: str = "daytona"
    n_concurrent: int = 4
    max_output_tokens: int = 8192
    agent_kwargs: dict = field(default_factory=dict)


@dataclass(frozen=True)
class EvalSuiteConfig:
    """A named eval run: lm-eval tasks for the evalchemy mechanism, or a Harbor dataset for Harbor."""

    name: str
    mechanism: EvalMechanism
    tasks: tuple[EvalTaskConfig, ...] = ()
    harbor: HarborSpec | None = None
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


def _gen_eval(name: str, task: str, shots: int, max_gen_toks: int) -> EvalSuiteConfig:
    """A single lm-eval-native *generative* benchmark (gsm8k/drop/triviaqa/nq_open).

    ``generation=True`` routes it through the chat API for a chat-template model and the completions
    API otherwise; the answer is extracted and scored by lm-eval (exact_match / f1).
    """
    return EvalSuiteConfig(
        name=name,
        mechanism=EvalMechanism.EVALCHEMY,
        tasks=(EvalTaskConfig(task, shots, task_alias=f"{task}_{shots}shot", generation=True),),
        max_gen_toks=max_gen_toks,
    )


def _chat_eval(name: str, task: str, max_gen_toks: int, *, unsafe_code: bool = False) -> EvalSuiteConfig:
    """A single evalchemy chat-native benchmark (MATH500/AIME24/HumanEvalPlus/... style).

    These construct chat messages and hard-code their own decoding (greedy for MATH500/AIME24/
    HumanEvalPlus/MBPPPlus/OlympiadBench), so only the generation budget is set here via
    ``max_gen_toks`` (passed as ``--max_tokens``). ``unsafe_code`` opts the code benchmarks into
    executing model-generated code. Every chat benchmark needs a server-side chat template.
    """
    return EvalSuiteConfig(
        name=name,
        mechanism=EvalMechanism.EVALCHEMY,
        tasks=(EvalTaskConfig(task, 0, task_alias=name, generation=True, unsafe_code=unsafe_code),),
        max_gen_toks=max_gen_toks,
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
    # --- Baseline lm-eval-harness NLP tasks ---
    # mmlu/arc-challenge/hellaswag/winogrande/truthfulqa/boolq/piqa/openbookqa above already carry the
    # standard OpenLLM shot counts; these fill in the rest of the 14-task NLP suite (see NLP_EVALS).
    "arc-easy": _mcq_eval("arc-easy", "arc_easy", 0),
    "lambada": _mcq_eval("lambada", "lambada_openai", 0),
    "triviaqa": _gen_eval("triviaqa", "triviaqa", 5, max_gen_toks=128),
    "nq-open": _gen_eval("nq-open", "nq_open", 5, max_gen_toks=128),
    "drop": _gen_eval("drop", "drop", 3, max_gen_toks=256),
    # gsm8k at 0-shot, a distinct eval identity from the existing 5-shot "gsm8k" so evaldash never
    # mixes the two protocols in one history/column.
    "gsm8k-0shot": _gen_eval("gsm8k-0shot", "gsm8k", 0, max_gen_toks=512),
    # --- Baseline evalchemy chat benchmarks (greedy) ---
    # 8192-token generation budget for the math-reasoning benchmarks (matches "math500"). A much larger
    # budget makes a weak model generate to the cap on every unsolved problem; each request then
    # exceeds the lm-eval API client timeout and retry-storms the endpoint. Raise it per model when a
    # capable thinking model needs longer chains.
    "aime24": _chat_eval("aime24", "AIME24", max_gen_toks=8192),
    "olympiadbench": _chat_eval("olympiadbench", "OlympiadBench", max_gen_toks=8192),
    # humanevalplus/mbppplus need the code extras (fire + human_eval_plus) the pinned image omits, so
    # their import fails on it; kept defined for when the image carries those deps.
    "humanevalplus": _chat_eval("humanevalplus", "HumanEvalPlus", max_gen_toks=1024, unsafe_code=True),
    "mbppplus": _chat_eval("mbppplus", "MBPPPlus", max_gen_toks=1024, unsafe_code=True),
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
    # --- Harbor (agentic registry benchmarks) ---
    # aime@1.0 is 60 AIME math problems; the served model solves each in a Daytona sandbox and
    # Harbor's verifier scores the boxed answer. aime-smoke caps the task count for a fast check.
    "aime-harbor": EvalSuiteConfig(
        name="aime-harbor",
        mechanism=EvalMechanism.HARBOR,
        harbor=HarborSpec(dataset="aime", version="1.0"),
    ),
    "aime-smoke": EvalSuiteConfig(
        name="aime-smoke",
        mechanism=EvalMechanism.HARBOR,
        harbor=HarborSpec(dataset="aime", version="1.0", n_concurrent=2),
        max_eval_instances=2,
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

# The baseline lm-eval-harness NLP suite: 14 deterministic loglikelihood/greedy tasks, runnable on
# every served model (base or instruct).
NLP_EVALS: tuple[str, ...] = (
    "mmlu",
    "arc-challenge",
    "arc-easy",
    "hellaswag",
    "winogrande",
    "truthfulqa",
    "boolq",
    "piqa",
    "openbookqa",
    "lambada",
    "triviaqa",
    "nq-open",
    "drop",
    "gsm8k-0shot",
)

# The evalchemy chat benchmarks that run greedily on the pinned image. Chat-template models only.
# humanevalplus/mbppplus are omitted here because the pinned image lacks their code extras (fire +
# human_eval_plus); GPQADiamond because its sampled requests carry a seed the TPU vLLM backend
# rejects. MMLU-Pro, CruxEval, MRCR, IFBench, and FinanceBench have no working task on the pinned
# image/fork.
CHAT_EVALS: tuple[str, ...] = ("math500", "aime24", "olympiadbench")

# Report-row groupings, for the Math / code report layouts. CODE_EVALS is unavailable on the pinned
# image (see above).
MATH_EVALS: tuple[str, ...] = ("math500", "aime24", "gsm8k-0shot")
CODE_EVALS: tuple[str, ...] = ("humanevalplus", "mbppplus")

# Named suite groups selectable by name on the CLI (``--evals smoke``). Launch NLP and CHAT as
# separate groups (two serves) rather than one ~19-eval serial serve: the serve backstop grows
# 2h + 2h x n_evals, and a single long serve is more exposed to preemption.
SUITES: dict[str, tuple[str, ...]] = {
    "smoke": SMOKE_EVALS,
    "core": CORE_EVALS,
    "nlp": NLP_EVALS,
    "chat": CHAT_EVALS,
    "math": MATH_EVALS,
    "code": CODE_EVALS,
}

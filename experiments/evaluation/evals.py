# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluation suites available to the shared launcher."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Protocol

from marin.evaluation.evalchemy.runner import EvalchemyExecutor, EvalchemyRunConfig, EvalchemyRuntimeConfig
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.evaluation.harbor.driver_config import (
    HarborAgentConfig,
    HarborEnvironmentConfig,
    HarborJobConfig,
    HarborRetryConfig,
    HarborRunConfig,
    HarborVerifierConfig,
    harbor_job_config,
)
from marin.evaluation.harbor.runner import HARBOR_RUNTIME, HarborExecutor
from marin.evaluation.model_config import ModelConfig
from marin.evaluation.records import EvalRef, EvalTaskRef, HarborRef
from marin.evaluation.runner import EvalExecutor
from marin.external_dependencies import EVALCHEMY
from rigging.secrets import SecretSpec

_TERMINAL_BENCH_DATASET = "DCAgent2/terminal_bench_2"
_SWEBENCH_RANDOM_100_DATASET = "DCAgent2/swebench-verified-random-100-folders"
_AGENTIC_CONCURRENCY = 32
_DAYTONA_ENVIRONMENT_TYPE = "daytona"
_DAYTONA_SECRET_ENV: Mapping[str, SecretSpec] = MappingProxyType(
    {
        "DAYTONA_API_KEY": (
            "env:DAYTONA_API_KEY",
            "gcp-secret://projects/hai-gcp-models/secrets/DAYTONA_EVAL_API_KEY/versions/latest",
        )
    }
)
_GRUG_NON_RETRYABLE_EXCEPTIONS = (
    "AgentTimeoutError",
    "AgentEnvironmentTimeoutError",
    "VerifierTimeoutError",
    "RewardFileNotFoundError",
    "RewardFileEmptyError",
    "VerifierOutputParseError",
    "SandboxBuildFailedError",
    "VerifierRuntimeError",
    "SummarizationTimeoutError",
    "ContextLengthExceededError",
)


class EvaluationDefinition(Protocol):
    """Experiment-owned record metadata and model adaptation for one evaluation."""

    secret_env: Mapping[str, SecretSpec]

    @property
    def record_ref(self) -> EvalRef: ...

    @property
    def runtime_descriptor(self) -> str: ...

    def executor_for(self, model: ModelConfig, limit: int | None) -> EvalExecutor: ...


@dataclass(frozen=True)
class EvalchemyDefinition:
    config: EvalchemyRunConfig
    secret_env: Mapping[str, SecretSpec] = field(default_factory=dict)

    @property
    def record_ref(self) -> EvalRef:
        return EvalRef(
            name=self.config.name,
            mechanism="evalchemy",
            tasks=tuple(EvalTaskRef(name=task.name, num_fewshot=task.num_fewshot) for task in self.config.tasks),
        )

    @property
    def runtime_descriptor(self) -> str:
        return self.config.runtime.requirement

    def executor_for(self, model: ModelConfig, limit: int | None) -> EvalExecutor:
        effective_limit = self.config.max_eval_instances if limit is None else limit
        config = replace(
            self.config,
            apply_chat_template=model.apply_chat_template,
            max_gen_toks=(
                model.generation.max_gen_toks if model.generation.max_gen_toks is not None else self.config.max_gen_toks
            ),
            max_eval_instances=effective_limit,
            extra_gen_kwargs={
                **self.config.extra_gen_kwargs,
                **model.generation.extra_gen_kwargs,
            },
        )
        return EvalchemyExecutor(config)


@dataclass(frozen=True)
class HarborDefinition:
    name: str
    config: HarborJobConfig
    max_eval_instances: int | None = None
    secret_env: Mapping[str, SecretSpec] = field(default_factory=dict)

    @property
    def record_ref(self) -> EvalRef:
        return EvalRef(
            name=self.name,
            mechanism="harbor",
            harbor=HarborRef(
                dataset=self.config.dataset,
                version=self.config.revision,
                agent=self.config.agent,
                env=self.config.environment,
                config_digest=self.config.digest,
            ),
        )

    @property
    def runtime_descriptor(self) -> str:
        return HARBOR_RUNTIME

    def executor_for(self, model: ModelConfig, limit: int | None) -> EvalExecutor:
        effective_limit = self.max_eval_instances if limit is None else limit
        return HarborExecutor(
            config=self.config,
            task_limit=effective_limit,
            model_agent_kwargs=model.agent.agent_kwargs,
            secret_env_keys=tuple(self.secret_env),
        )


def harbor_definition(
    name: str,
    config: HarborJobConfig,
    max_eval_instances: int | None = None,
) -> HarborDefinition:
    """Attach shared-launch metadata and secrets to one Harbor job policy."""
    secret_env = _DAYTONA_SECRET_ENV if config.environment == _DAYTONA_ENVIRONMENT_TYPE else MappingProxyType({})
    return HarborDefinition(
        name=name,
        config=config,
        max_eval_instances=max_eval_instances,
        secret_env=secret_env,
    )


def _harbor_eval(
    name: str,
    run: HarborRunConfig,
    max_eval_instances: int | None = None,
) -> HarborDefinition:
    return harbor_definition(name, harbor_job_config(name, run), max_eval_instances)


def _mcq_eval(name: str, task: str, shots: int) -> EvalchemyDefinition:
    return EvalchemyDefinition(
        EvalchemyRunConfig(
            name=name,
            tasks=(EvalTaskConfig(task, shots, task_alias=f"{task}_{shots}shot"),),
            max_gen_toks=256,
        )
    )


def _gen_eval(name: str, task: str, shots: int, max_gen_toks: int) -> EvalchemyDefinition:
    return EvalchemyDefinition(
        EvalchemyRunConfig(
            name=name,
            tasks=(EvalTaskConfig(task, shots, task_alias=f"{task}_{shots}shot", generation=True),),
            max_gen_toks=max_gen_toks,
        )
    )


def _chat_eval(name: str, task: str, max_gen_toks: int, *, unsafe_code: bool = False) -> EvalchemyDefinition:
    benchmark_extra = task.lower().replace("_", "-")
    return EvalchemyDefinition(
        EvalchemyRunConfig(
            name=name,
            tasks=(EvalTaskConfig(task, 0, task_alias=name, generation=True, unsafe_code=unsafe_code),),
            max_gen_toks=max_gen_toks,
            runtime=EvalchemyRuntimeConfig(requirement=EVALCHEMY.requirement((benchmark_extra,))),
        )
    )


def _agentic_eval(
    name: str,
    hugging_face_dataset: str,
    *,
    agent: str = "terminus-2",
    n_concurrent: int = 8,
    max_instances: int | None = None,
) -> HarborDefinition:
    return _harbor_eval(
        name,
        HarborRunConfig(
            dataset=f"hf://{hugging_face_dataset}",
            revision="main",
            agent=HarborAgentConfig(name=agent),
            environment=HarborEnvironmentConfig(environment_type=_DAYTONA_ENVIRONMENT_TYPE),
            n_concurrent=n_concurrent,
        ),
        max_instances,
    )


GRUG_OPENCODE_EVAL = _harbor_eval(
    "grug-opencode-id",
    HarborRunConfig(
        dataset="hf://DCAgent/dev_set_v2",
        revision="377118ff3031c934f5a647ae2c425eb74eef3b21",
        agent=HarborAgentConfig(
            name="opencode",
            max_output_tokens=16384,
            max_timeout=7200,
            setup_timeout=600,
            kwargs={
                "opencode_config": {"compaction": {"auto": False}},
                "model_info": {
                    "max_input_tokens": 64512,
                    "input_cost_per_token": 0.0,
                    "output_cost_per_token": 0.0,
                },
                "trajectory_config": {"raw_content": False, "linear_history": True},
            },
        ),
        environment=HarborEnvironmentConfig(
            environment_type=_DAYTONA_ENVIRONMENT_TYPE,
            force_build=True,
            delete=True,
            cpus=2,
            memory_mb=8192,
            storage_mb=8192,
            kwargs={"auto_snapshot": True},
        ),
        n_concurrent=256,
        attempts=3,
        timeout_multiplier=2.0,
        retry=HarborRetryConfig(
            max_retries=6,
            exclude_exceptions=_GRUG_NON_RETRYABLE_EXCEPTIONS,
            wait_multiplier=2.0,
            min_wait=1.0,
            max_wait=90.0,
        ),
        verifier=HarborVerifierConfig(max_timeout=14400),
    ),
)


EVALS: dict[str, EvaluationDefinition] = {
    # The core benchmarks, one eval per task so every model x task pair is its own run with its own
    # inference/eval jobs, record, and per-question parquet. Shot counts follow the HF OpenLLM-v1
    # conventions so scores line up with public leaderboards.
    "mmlu": _mcq_eval("mmlu", "mmlu", 5),
    "arc-challenge": _mcq_eval("arc-challenge", "arc_challenge", 25),
    "hellaswag": _mcq_eval("hellaswag", "hellaswag", 10),
    "winogrande": _mcq_eval("winogrande", "winogrande", 5),
    "truthfulqa": _mcq_eval("truthfulqa", "truthfulqa_mc2", 0),
    "boolq": _mcq_eval("boolq", "boolq", 0),
    "piqa": _mcq_eval("piqa", "piqa", 0),
    "openbookqa": _mcq_eval("openbookqa", "openbookqa", 0),
    "gsm8k": EvalchemyDefinition(
        EvalchemyRunConfig(
            name="gsm8k",
            tasks=(EvalTaskConfig("gsm8k", 5, task_alias="gsm8k_5shot", generation=True),),
            max_gen_toks=512,
        )
    ),
    # Evalchemy's chat-native MATH500 benchmark (boxed-answer extraction over the HuggingFaceH4
    # MATH-500 split). A messages-based task: it runs through the chat route, so every model needs
    # a server-side chat template (snowball serves one via its vLLM args).
    "math500": _chat_eval("math500", "MATH500", max_gen_toks=8192),
    "humaneval": EvalchemyDefinition(
        EvalchemyRunConfig(
            name="humaneval",
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
        )
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
    "humanevalplus": _chat_eval("humanevalplus", "HumanEvalPlus", max_gen_toks=1024, unsafe_code=True),
    "mbppplus": _chat_eval("mbppplus", "MBPPPlus", max_gen_toks=1024, unsafe_code=True),
    "mmlu-smoke": EvalchemyDefinition(
        EvalchemyRunConfig(
            name="mmlu-smoke",
            tasks=(EvalTaskConfig("mmlu_abstract_algebra", 0, task_alias="mmlu_abstract_algebra_0shot"),),
            max_gen_toks=256,
            max_eval_instances=64,
        )
    ),
    "gsm8k-smoke": EvalchemyDefinition(
        EvalchemyRunConfig(
            name="gsm8k-smoke",
            tasks=(EvalTaskConfig("gsm8k", 5, task_alias="gsm8k_5shot", generation=True),),
            max_gen_toks=512,
            max_eval_instances=128,
        )
    ),
    # --- Harbor (agentic registry benchmarks) ---
    # aime@1.0 is 60 AIME math problems; the served model solves each in a Daytona sandbox and
    # Harbor's verifier scores the boxed answer. aime-smoke caps the task count for a fast check.
    "aime-harbor": _harbor_eval(
        "aime-harbor",
        HarborRunConfig(
            dataset="aime",
            revision="1.0",
            agent=HarborAgentConfig(name="terminus-2"),
            environment=HarborEnvironmentConfig(environment_type=_DAYTONA_ENVIRONMENT_TYPE),
        ),
    ),
    "aime-smoke": _harbor_eval(
        "aime-smoke",
        HarborRunConfig(
            dataset="aime",
            revision="1.0",
            agent=HarborAgentConfig(name="terminus-2"),
            environment=HarborEnvironmentConfig(environment_type=_DAYTONA_ENVIRONMENT_TYPE),
            n_concurrent=2,
        ),
        2,
    ),
    # Agentic datasets contain Harbor task directories and run with Daytona.
    "tb2": _agentic_eval("tb2", _TERMINAL_BENCH_DATASET, n_concurrent=_AGENTIC_CONCURRENCY),
    "tb2-lite": _agentic_eval("tb2-lite", _TERMINAL_BENCH_DATASET, n_concurrent=4, max_instances=2),
    "swebench": _agentic_eval("swebench", _SWEBENCH_RANDOM_100_DATASET, n_concurrent=_AGENTIC_CONCURRENCY),
    "swebench-lite": _agentic_eval("swebench-lite", _SWEBENCH_RANDOM_100_DATASET, n_concurrent=4, max_instances=2),
    "swebench-full": _agentic_eval("swebench-full", "DCAgent/swebench-verified", n_concurrent=_AGENTIC_CONCURRENCY),
    "gaia": _agentic_eval("gaia", "DCAgent/gaia_127", n_concurrent=_AGENTIC_CONCURRENCY),
    "bfcl": _agentic_eval("bfcl", "DCAgent2/bfcl-parity", n_concurrent=_AGENTIC_CONCURRENCY),
    "aider": _agentic_eval("aider", "DCAgent2/aider_polyglot", n_concurrent=_AGENTIC_CONCURRENCY),
    "medagentbench": _agentic_eval("medagentbench", "DCAgent/medagentbench", n_concurrent=_AGENTIC_CONCURRENCY),
    "financeagent": _agentic_eval("financeagent", "DCAgent/financeagent_terminal", n_concurrent=16),
    "grug-opencode-id": GRUG_OPENCODE_EVAL,
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

# The Evalchemy chat benchmarks that run greedily in the lean uvx runtime. Chat-template models only.
# GPQADiamond is omitted because its sampled requests carry a seed the TPU vLLM backend rejects.
# MMLU-Pro, CruxEval, MRCR, IFBench, and FinanceBench have no working task on the pinned fork.
CHAT_EVALS: tuple[str, ...] = ("math500", "aime24", "olympiadbench")

MATH_EVALS: tuple[str, ...] = ("math500", "aime24", "gsm8k-0shot")
CODE_EVALS: tuple[str, ...] = ("humanevalplus", "mbppplus")

AGENTIC_EVALS: tuple[str, ...] = (
    "tb2",
    "swebench",
    "gaia",
    "bfcl",
    "aider",
    "medagentbench",
    "financeagent",
)

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
    "agentic": AGENTIC_EVALS,
}

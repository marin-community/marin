# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluation suites available to the shared launcher."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType

from marin.evaluation.evalchemy.config import EvalchemyConfig
from marin.evaluation.evalchemy.runner import (
    DEFAULT_MAX_GEN_TOKS,
    DEFAULT_NUM_CONCURRENT,
    EvalchemyRunConfig,
    EvalchemyRuntimeConfig,
)
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.evaluation.harbor.driver_config import HARBOR_RUNTIME, ValidatedHarborConfig
from marin.evaluation.harbor.runner import HarborExecutor
from marin.evaluation.model_config import ModelConfig
from marin.evaluation.records import EvalchemyRef, EvalRef, EvalTaskRef, HarborRef
from marin.evaluation.runner import EvalExecutor
from marin.external_dependencies import EVALCHEMY
from rigging.secrets import SecretSpec

logger = logging.getLogger(__name__)

_DAYTONA_ENVIRONMENT_TYPE = "daytona"
_EVALCHEMY_CONFIG_DIR = Path(__file__).with_name("configs") / "evalchemy"
_HARBOR_CONFIG_DIR = Path(__file__).with_name("configs") / "harbor"
_DAYTONA_SECRET_ENV: Mapping[str, SecretSpec] = MappingProxyType(
    {
        "DAYTONA_API_KEY": (
            "env:DAYTONA_API_KEY",
            "gcp-secret://projects/hai-gcp-models/secrets/DAYTONA_EVAL_API_KEY/versions/latest",
        )
    }
)


@dataclass(frozen=True)
class EvalchemyDefinition:
    name: str
    config_path: Path
    secret_env: Mapping[str, SecretSpec] = field(default_factory=dict)

    def record_ref_for(self, config: EvalchemyRunConfig) -> EvalRef:
        return EvalRef(
            name=config.name,
            mechanism="evalchemy",
            tasks=tuple(
                EvalTaskRef(
                    name=task.name,
                    num_fewshot=task.num_fewshot,
                    task_alias=task.task_alias,
                    generation=task.generation,
                    unsafe_code=task.unsafe_code,
                    completion_only=task.completion_only,
                )
                for task in config.tasks
            ),
            evalchemy=EvalchemyRef(
                apply_chat_template=config.apply_chat_template,
                max_gen_toks=config.max_gen_toks,
                max_eval_instances=config.max_eval_instances,
                num_concurrent=config.num_concurrent,
                batch_size=config.batch_size,
                seed=config.seed,
                extra_gen_kwargs=dict(config.extra_gen_kwargs),
                extra_model_args=dict(config.extra_model_args),
                max_length=config.max_length,
            ),
        )

    def config_for(self, source: EvalchemyConfig, model: ModelConfig, limit: int | None) -> EvalchemyRunConfig:
        config = evalchemy_run_config(self.name, source)
        effective_limit = config.max_eval_instances if limit is None else limit
        model_max_gen_toks = model.generation.max_gen_toks
        max_gen_toks = config.max_gen_toks
        if model_max_gen_toks is not None:
            if source.max_tokens is None or model_max_gen_toks < max_gen_toks:
                max_gen_toks = model_max_gen_toks
            elif model_max_gen_toks > max_gen_toks:
                logger.warning(
                    "Model generation limit %d exceeds the %s benchmark limit %d; using the benchmark limit. "
                    "Pass an Evalchemy config with a larger max_tokens value to evaluate longer generations.",
                    model_max_gen_toks,
                    self.name,
                    max_gen_toks,
                )
        return replace(
            config,
            apply_chat_template=(
                model.apply_chat_template if source.apply_chat_template is None else source.apply_chat_template
            ),
            max_gen_toks=max_gen_toks,
            max_eval_instances=effective_limit,
            extra_gen_kwargs={
                **config.extra_gen_kwargs,
                **model.generation.extra_gen_kwargs,
            },
        )


@dataclass(frozen=True)
class HarborDefinition:
    """Experiment metadata plus one Harbor policy source."""

    name: str
    config_path: Path
    max_eval_instances: int | None = None

    def secret_env_for(self, config: ValidatedHarborConfig) -> Mapping[str, SecretSpec]:
        if config.environment == _DAYTONA_ENVIRONMENT_TYPE:
            return _DAYTONA_SECRET_ENV
        return MappingProxyType({})

    def record_ref_for(self, config: ValidatedHarborConfig, runtime_task_limit: int | None) -> EvalRef:
        return EvalRef(
            name=self.name,
            mechanism="harbor",
            harbor=HarborRef(
                dataset=config.record_dataset,
                version=config.record_revision,
                agent=config.agent,
                env=config.environment,
                task_limit=runtime_task_limit,
                config_digest=config.digest,
            ),
        )

    @property
    def runtime_descriptor(self) -> str:
        return HARBOR_RUNTIME

    def executor_for(
        self,
        config: ValidatedHarborConfig,
        model: ModelConfig,
        runtime_task_limit: int | None,
    ) -> EvalExecutor:
        secret_env = self.secret_env_for(config)
        return HarborExecutor(
            config=config,
            task_limit=runtime_task_limit,
            model_agent_kwargs=model.agent.agent_kwargs,
            secret_env_keys=tuple(secret_env),
        )


def harbor_definition(
    name: str,
    max_eval_instances: int | None = None,
) -> HarborDefinition:
    """Create a Harbor definition from its same-named checked-in YAML policy."""
    return HarborDefinition(
        name=name,
        config_path=_HARBOR_CONFIG_DIR / f"{name}.yaml",
        max_eval_instances=max_eval_instances,
    )


def evalchemy_run_config(name: str, config: EvalchemyConfig) -> EvalchemyRunConfig:
    """Lower one launch file into Marin's served Evalchemy runner."""
    tasks: list[EvalTaskConfig] = []
    for task_name in config.tasks:
        options = config.task_options.get(task_name)
        num_fewshot = config.num_fewshot
        if options is not None and options.num_fewshot is not None:
            num_fewshot = options.num_fewshot
        tasks.append(
            EvalTaskConfig(
                name=task_name,
                num_fewshot=num_fewshot,
                task_alias=options.task_alias if options is not None else None,
                generation=options.generation if options is not None else False,
                unsafe_code=options.unsafe_code if options is not None else False,
                completion_only=options.completion_only if options is not None else False,
            )
        )

    extra_model_args = dict(config.extra_model_args)
    configured_concurrency = extra_model_args.pop("num_concurrent", DEFAULT_NUM_CONCURRENT)
    try:
        num_concurrent = int(configured_concurrency)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"extra_model_args.num_concurrent must be an integer, got {configured_concurrency!r}") from exc
    if num_concurrent <= 0:
        raise ValueError(f"extra_model_args.num_concurrent must be positive, got {num_concurrent}")
    for key in ("max_length", "max_model_len"):
        extra_model_args.pop(key, None)

    extra_gen_kwargs = dict(config.gen_kwargs)
    for key in ("max_tokens", "max_new_tokens", "max_gen_toks"):
        extra_gen_kwargs.pop(key, None)
    return EvalchemyRunConfig(
        name=name,
        tasks=tuple(tasks),
        apply_chat_template=config.apply_chat_template or False,
        max_gen_toks=config.max_tokens or DEFAULT_MAX_GEN_TOKS,
        max_eval_instances=config.limit,
        num_concurrent=num_concurrent,
        batch_size=config.batch_size,
        seed=config.seed,
        extra_gen_kwargs=extra_gen_kwargs,
        extra_model_args=extra_model_args,
        max_length=config.max_length,
        runtime=EvalchemyRuntimeConfig(requirement=EVALCHEMY.requirement(config.runtime_extras)),
    )


EvaluationDefinition = EvalchemyDefinition | HarborDefinition


_STANDARD_EVALCHEMY_EVALS: tuple[str, ...] = (
    "mmlu",
    "arc-challenge",
    "hellaswag",
    "winogrande",
    "truthfulqa",
    "boolq",
    "piqa",
    "openbookqa",
    "gsm8k",
    "math500",
    "humaneval",
    "arc-easy",
    "lambada",
    "triviaqa",
    "nq-open",
    "drop",
    "gsm8k-0shot",
    "aime24",
    "olympiadbench",
    "humanevalplus",
    "mbppplus",
    "mmlu-smoke",
    "gsm8k-smoke",
)

EVALS: dict[str, EvaluationDefinition] = {
    name: EvalchemyDefinition(name=name, config_path=_EVALCHEMY_CONFIG_DIR / f"{name}.yaml")
    for name in _STANDARD_EVALCHEMY_EVALS
}
EVALS.update(
    {
        # --- Harbor (agentic registry benchmarks) ---
        # aime@1.0 is 60 AIME math problems; the served model solves each in a Daytona sandbox and
        # Harbor's verifier scores the boxed answer. aime-smoke caps the task count for a fast check.
        "aime-harbor": harbor_definition("aime-harbor"),
        "aime-smoke": harbor_definition("aime-smoke", 2),
        # Agentic datasets contain Harbor task directories and run with Daytona.
        "tb2": harbor_definition("tb2"),
        "tb2-lite": harbor_definition("tb2-lite", 2),
        "swebench": harbor_definition("swebench"),
        "swebench-lite": harbor_definition("swebench-lite", 2),
        "swebench-full": harbor_definition("swebench-full"),
        "gaia": harbor_definition("gaia"),
        "bfcl": harbor_definition("bfcl"),
        "aider": harbor_definition("aider"),
        "medagentbench": harbor_definition("medagentbench"),
        "financeagent": harbor_definition("financeagent"),
        "grug-opencode-id": harbor_definition("grug-opencode-id"),
    }
)

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


def resolve_eval_keys(selection: str) -> tuple[str, ...]:
    """Resolve a named suite or comma-separated evaluation keys."""
    keys = SUITES.get(selection) or tuple(part.strip() for part in selection.split(",") if part.strip())
    if not keys:
        raise ValueError("no evals selected")
    unknown = [key for key in keys if key not in EVALS]
    if unknown:
        raise ValueError(f"unknown eval(s) {unknown}; known: {sorted(EVALS)} or suites {sorted(SUITES)}")
    return keys

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluation suites available to the shared launcher."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType

from marin.evaluation.eval_policy import source_config_digest
from marin.evaluation.harbor.agent_context import MODEL_INFO_KEY, served_model_info
from marin.evaluation.harbor.driver_config import HarborDatasetKind, ValidatedHarborConfig
from marin.evaluation.harbor.runner import HarborExecutor
from marin.evaluation.model_config import ModelConfig
from marin.evaluation.records import EvalRef, EvalTaskRef, HarborRef, ModelConfigRef
from marin.evaluation.runner import EvalExecutor
from marin.rollouts.evaluation import JudgeIdentity
from rigging.secrets import SecretSpec

from experiments.evaluation.curated import CuratedDefinition
from experiments.post_training.task_curation.datasets.evalchemy.checkpoint import sources as checkpoint_sources
from experiments.post_training.task_curation.datasets.evalchemy.mmlu import MMLU_POPULATIONS

logger = logging.getLogger(__name__)

_DAYTONA_ENVIRONMENT_TYPE = "daytona"
_HARBOR_CONFIG_DIR = Path(__file__).with_name("configs") / "harbor"
_DAYTONA_SECRET_ENV: Mapping[str, SecretSpec] = MappingProxyType(
    {
        "DAYTONA_API_KEY": (
            "env:DAYTONA_API_KEY",
            "gcp-secret://projects/hai-gcp-models/secrets/DAYTONA_EVAL_API_KEY/versions/latest",
        )
    }
)
_TOGETHER_API_KEY_SECRET: SecretSpec = (
    "env:TOGETHER_API_KEY",
    "gcp-secret://projects/hai-gcp-models/secrets/TOGETHER_API_KEY/versions/latest",
)
_VERIFIER_SECRET_ENV: Mapping[str, SecretSpec] = MappingProxyType({"TOGETHER_API_KEY": _TOGETHER_API_KEY_SECRET})
# Destinations beyond Iris's base egress allowlist (iris.cluster.egress). The
# orchestrator job declares them; they take effect on clusters that enforce the
# allowlist.
_DAYTONA_EGRESS: tuple[str, ...] = ("app.daytona.io", "*.app.daytona.io")
# Harbor's dataset registry, a Supabase project (Harbor's HARBOR_SUPABASE_URL).
_HARBOR_REGISTRY_EGRESS: tuple[str, ...] = ("hlqxxzsirfrgeqasvaps.supabase.co",)
# Capped ``-smoke`` variants remain unfamilied because scoring surfaces exclude them.
_EVAL_FAMILIES: Mapping[str, str] = MappingProxyType({"gsm8k": "gsm8k", "gsm8k-0shot": "gsm8k"})


@dataclass(frozen=True)
class HarborDefinition:
    """Experiment metadata plus one Harbor policy source."""

    name: str
    config_path: Path
    max_eval_instances: int | None = None
    family: str | None = None

    def egress_allow_for(self, config: ValidatedHarborConfig) -> tuple[str, ...]:
        """Egress rules for the Harbor driver's sandbox service and dataset registry."""
        rules = _DAYTONA_EGRESS if config.environment == _DAYTONA_ENVIRONMENT_TYPE else ()
        if config.dataset_kind == HarborDatasetKind.HARBOR_REGISTRY:
            rules = (*rules, *_HARBOR_REGISTRY_EGRESS)
        return rules

    def secret_env_for(self, config: ValidatedHarborConfig) -> Mapping[str, SecretSpec]:
        secret_env = dict(_DAYTONA_SECRET_ENV) if config.environment == _DAYTONA_ENVIRONMENT_TYPE else {}
        for key in config.verifier_env_keys:
            secret_env.setdefault(key, _VERIFIER_SECRET_ENV.get(key, (f"env:{key}",)))
        return MappingProxyType(secret_env)

    def record_ref_for(self, config: ValidatedHarborConfig, runtime_task_limit: int | None) -> EvalRef:
        benchmark = config.benchmark_for(runtime_task_limit)
        return EvalRef(
            name=self.name,
            mechanism="harbor",
            source_digest=source_config_digest(self.config_path),
            family=self.family,
            tasks=(
                EvalTaskRef(
                    name=config.record_dataset,
                    num_fewshot=None,
                    benchmark=benchmark,
                ),
            ),
            harbor=HarborRef(
                dataset=config.record_dataset,
                version=config.record_revision,
                agent=config.agent,
                env=config.environment,
                task_limit=runtime_task_limit,
                config_digest=config.digest,
                max_input_tokens=config.max_input_tokens,
                max_output_tokens=config.max_output_tokens,
            ),
        )

    def executor_for(
        self,
        config: ValidatedHarborConfig,
        model: ModelConfig,
        runtime_task_limit: int | None,
        retry_unscored_trials: bool,
        judge_model: ModelConfig | None,
    ) -> EvalExecutor:
        secret_env = self.secret_env_for(config)
        return HarborExecutor(
            config=config,
            task_limit=runtime_task_limit,
            model_agent_kwargs=harbor_model_agent_kwargs(model),
            model_config=ModelConfigRef.model_validate(asdict(model)),
            judge_config=ModelConfigRef.model_validate(asdict(judge_model)) if judge_model is not None else None,
            secret_env_keys=tuple(secret_env),
            retry_unscored_trials=retry_unscored_trials,
        )


def harbor_model_agent_kwargs(model: ModelConfig) -> dict[str, object]:
    """Omit unset catalog limits so the policy or Harbor defaults can supply them."""
    return {
        **model.agent.agent_kwargs,
        MODEL_INFO_KEY: served_model_info(model.serve.max_model_len, model.generation.max_gen_toks),
    }


def harbor_definition(
    name: str,
    max_eval_instances: int | None = None,
) -> HarborDefinition:
    """Create a Harbor definition from its same-named checked-in YAML policy."""
    return HarborDefinition(
        name=name,
        config_path=_HARBOR_CONFIG_DIR / f"{name}.yaml",
        max_eval_instances=max_eval_instances,
        family=_EVAL_FAMILIES.get(name),
    )


EvaluationDefinition = CuratedDefinition | HarborDefinition


STANDARD_EVALS: tuple[str, ...] = (
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
    "olympiadbench-deterministic",
    "humanevalplus",
    "mbppplus",
    "mmlu-smoke",
    "gsm8k-smoke",
    "aime24-smoke",
    "mmlu-pro",
    "gpqa-diamond",
    "cruxeval",
    "financebench",
    "ifeval",
    "ifbench",
    "mrcr",
    "graphwalks",
)

_CURATED_SOURCES = {
    "mmlu": tuple("evalchemy-mmlu-" + subject for subject in MMLU_POPULATIONS),
    **{
        name: ("evalchemy-" + source,)
        for name, source in {
            "arc-challenge": "arc_challenge",
            "arc-easy": "arc_easy",
            "hellaswag": "hellaswag",
            "winogrande": "winogrande",
            "truthfulqa": "truthfulqa_mc2",
            "boolq": "boolq",
            "piqa": "piqa",
            "openbookqa": "openbookqa",
            "gsm8k": "gsm8k",
            "math500": "math500",
            "humaneval": "humaneval",
            "lambada": "lambada_openai",
            "triviaqa": "triviaqa",
            "nq-open": "nq_open",
            "drop": "drop",
            "gsm8k-0shot": "gsm8k-0shot",
            "aime24": "aime24",
            "olympiadbench": "olympiadbench",
            "olympiadbench-deterministic": "olympiadbenchdeterministic",
            "humanevalplus": "humanevalplus",
            "mbppplus": "mbppplus",
            "mmlu-smoke": "mmlu-smoke",
            "gsm8k-smoke": "gsm8k",
            "aime24-smoke": "aime24",
            "mmlu-pro": "mmlu-pro",
            "gpqa-diamond": "gpqa-diamond",
            "cruxeval": "cruxeval",
            "financebench": "financebench",
            "ifeval": "ifeval",
            "ifbench": "ifbench",
            "mrcr": "mrcr",
            "graphwalks": "graphwalks",
        }.items()
    },
}
EVALS: dict[str, EvaluationDefinition] = {
    name: CuratedDefinition(
        name=name,
        sources=_CURATED_SOURCES[name],
        chat_sources=(
            tuple(source + "-chat" for source in _CURATED_SOURCES[name])
            if name in {"gsm8k", "gsm8k-smoke", "gsm8k-0shot", "triviaqa", "nq-open", "drop", "ifeval"}
            else ()
        ),
        family=_EVAL_FAMILIES.get(name),
        limit={"mmlu-smoke": 64, "gsm8k-smoke": 128, "aime24-smoke": 2}.get(name),
        max_tokens=(
            256
            if name
            in {
                "mmlu",
                "mmlu-smoke",
                "arc-challenge",
                "arc-easy",
                "hellaswag",
                "winogrande",
                "truthfulqa",
                "boolq",
                "piqa",
                "openbookqa",
                "lambada",
            }
            else {"mrcr": 4096, "graphwalks": 131072}.get(name)
        ),
        seed={
            "cruxeval": 42,
            "gpqa-diamond": 42,
            "financebench": 42,
            "ifbench": 42,
            "ifeval": 1234,
            "mmlu-pro": 42,
            "mrcr": 42,
        }.get(name),
    )
    for name in STANDARD_EVALS
}
EVALS.update(
    {
        "mmlu-0shot": CuratedDefinition(
            "mmlu-0shot", tuple("evalchemy-mmlu-0shot-" + subject for subject in MMLU_POPULATIONS)
        ),
        **{
            name: CuratedDefinition(name, ("evalchemy-" + source,))
            for name, source in {
                "arc-easy-10shot": "arc_easy-10shot",
                "arc-challenge-10shot": "arc_challenge-10shot",
                "boolq-10shot": "boolq-10shot",
                "piqa-10shot": "piqa-10shot",
                "winogrande-0shot": "winogrande-0shot",
                "hellaswag-0shot": "hellaswag-0shot",
                "truthfulqa-6shot": "truthfulqa_mc2-6shot",
            }.items()
        },
    }
)
for name in ("financebench", "olympiadbench"):
    EVALS[name] = CuratedDefinition(
        name,
        _CURATED_SOURCES[name],
        seed=42 if name == "financebench" else None,
        judge=JudgeIdentity(
            deployment="together",
            models={"openai/gpt-oss-120b": "provider-managed:openai/gpt-oss-120b"},
            default_model="openai/gpt-oss-120b",
        ),
        judge_endpoint="https://api.together.xyz/v1",
        judge_api_key=_TOGETHER_API_KEY_SECRET,
    )
_CHECKPOINT_BENCHMARKS = {
    "commonsense_qa": "commonsense-qa-10shot",
    "copa": "copa",
    "wsc273": "wsc273",
    "agieval_lsat_ar": "agieval-lsat-ar-3shot",
    "leaderboard_mmlu_pro": "mmlu-pro-5shot",
    "leaderboard_bbh": "bbh-3shot",
    "leaderboard_gpqa": "gpqa-0shot",
    "gsm8k_cot": "gsm8k-cot-8shot",
    "bbh_cot_fewshot": "bbh-cot-3shot",
    "minerva_math": "minerva-math-4shot",
    "nq_open-0shot": "nq-open-0shot",
    "triviaqa-0shot": "triviaqa-0shot",
    "drop-0shot": "drop-0shot",
    "humaneval-10shot": "humaneval-10shot",
}
_CHECKPOINT_SOURCES = checkpoint_sources()
for benchmark, name in _CHECKPOINT_BENCHMARKS.items():
    EVALS[name] = CuratedDefinition(
        name,
        tuple(
            source.name
            for source in _CHECKPOINT_SOURCES
            if source.config is not None and source.config.protocol.benchmark == benchmark
        ),
    )
EVALS.update(
    {
        "nupa": CuratedDefinition("nupa", ("evalchemy-nupa5k-loose",), seed=20222943, max_tokens=256),
        "mrcr-32k": CuratedDefinition("mrcr-32k", ("evalchemy-mrcr-32k",), seed=42, max_tokens=4096),
        "mrcr-65k": CuratedDefinition("mrcr-65k", ("evalchemy-mrcr-65k",), seed=42, max_tokens=4096),
    }
)
EVALS.update(
    {
        # --- Harbor (agentic registry benchmarks) ---
        # aime@1.0 is 60 AIME math problems; the served model solves each in a Daytona sandbox and
        # Harbor's verifier scores the boxed answer.
        "aime-harbor": harbor_definition("aime-harbor"),
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
        "ot-tblite": harbor_definition("ot-tblite"),
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

# The short general-purpose chat suite. Longer publication-policy benchmarks are file-backed under
# the curated catalog and validated on H100; GPQADiamond's seeded requests remain incompatible with TPU vLLM.
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
    "standard": (*STANDARD_EVALS, "nupa"),
    "standard-local": tuple(
        name
        for name in (*STANDARD_EVALS, "nupa")
        if isinstance(EVALS[name], CuratedDefinition) and not EVALS[name].requires_judge
    ),
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

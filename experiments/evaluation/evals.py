# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluation suites available to the shared launcher."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType

import yaml
from marin.evaluation.eval_policy import source_config_digest
from marin.evaluation.harbor.agent_context import MODEL_INFO_KEY, served_model_info
from marin.evaluation.harbor.driver_config import HarborDatasetKind, ValidatedHarborConfig
from marin.evaluation.harbor.runner import HarborExecutor
from marin.evaluation.model_config import ModelConfig
from marin.evaluation.records import EvalRef, EvalTaskRef, EvaluationSelection, HarborRef, ModelConfigRef
from marin.evaluation.runner import EvalExecutor
from rigging.secrets import SecretSpec

from experiments.evaluation.catalog import evaluation_benchmarks
from experiments.evaluation.curated import requires_judge

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
    )


EvaluationDefinition = EvaluationSelection | HarborDefinition


_CONFIG = yaml.safe_load((Path(__file__).with_name("configs") / "evalchemy.yaml").read_text())
EVALS: dict[str, EvaluationDefinition] = {
    name: EvaluationSelection.model_validate({"label": name, **entry}) for name, entry in _CONFIG["evaluations"].items()
}
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


def evaluation_for(name: str) -> EvaluationDefinition:
    if name in EVALS:
        return EVALS[name]
    if name in evaluation_benchmarks():
        return EvaluationSelection(benchmark=name)
    raise ValueError(f"unknown evaluation: {name}")


SUITES: dict[str, tuple[str, ...]] = {name: tuple(entries) for name, entries in _CONFIG["suites"].items()}
SUITES["standard-local"] = tuple(
    name
    for name in SUITES["standard"]
    if isinstance(definition := evaluation_for(name), EvaluationSelection) and not requires_judge(definition)
)


def resolve_evaluations(selection: str) -> tuple[EvaluationDefinition, ...]:
    """Resolve suite and CLI labels once, at the textual input boundary."""
    keys = SUITES.get(selection) or tuple(part.strip() for part in selection.split(",") if part.strip())
    if not keys:
        raise ValueError("no evals selected")
    return tuple(evaluation_for(key) for key in keys)

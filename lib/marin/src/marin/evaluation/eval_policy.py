# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Admission rules for the two published Marin evaluation cohorts."""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType

import yaml

from marin.evaluation.eval_policy_sources import POLICY_SOURCE_DIGESTS, SEPTEMBER_16_VERSION, SEPTEMBER_24_VERSION
from marin.evaluation.model_identity import model_config_digest
from marin.evaluation.records import EvalRef, EvalRunRecord, ModelRef

EVALCHEMY_COMMIT = "c131e5ab84d3014490549ec06d45deb13b5673c2"
HARBOR_COMMIT = "21e0ea6a0cc1a0b617aebd86988ea93e1795f84a"
NUPA_SEED = 20222943
DEFAULT_SEED = 42
AIME24_REPEATS = 10
SEPTEMBER_16_IFBENCH_MAX_TOKENS = 1024


def source_config_digest(path: Path) -> str:
    """Identify a policy YAML by its parsed content, independent of comments and key order."""
    payload = json.dumps(yaml.safe_load(path.read_text()), sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


FIXED_MAX_TOKENS: Mapping[str, int] = MappingProxyType(
    {
        "mmlu-pro": 32768,
        "gpqa-diamond": 32768,
        "cruxeval": 2048,
        "financebench": 4096,
        "mrcr": 4096,
        "nupa": 256,
        "mmlu": 256,
        "piqa": 256,
        "winogrande": 256,
        "openbookqa": 256,
        "boolq": 256,
        "truthfulqa": 256,
    }
)
COMPLETION_TASKS = frozenset({"mmlu", "piqa", "winogrande", "openbookqa", "boolq", "truthfulqa"})
UNSAFE_CODE_TASKS = frozenset({"humanevalplus", "mbppplus"})


class ThinkingMode(StrEnum):
    ON = "on"
    OFF = "off"
    MODEL_DEFAULT = "model_default"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class PolicyEval:
    task: str | None
    shots: int | None
    mechanism: str
    thinking: ThinkingMode = ThinkingMode.MODEL_DEFAULT


def _evalchemy(task: str, shots: int, thinking: ThinkingMode = ThinkingMode.MODEL_DEFAULT) -> PolicyEval:
    return PolicyEval(task, shots, "evalchemy", thinking)


def _harbor() -> PolicyEval:
    return PolicyEval(None, None, "harbor")


SEPTEMBER_16: Mapping[str, PolicyEval] = MappingProxyType(
    {
        "math500": _evalchemy("MATH500", 0),
        "aime24": _evalchemy("AIME24", 0),
        "humanevalplus": _evalchemy("HumanEvalPlus", 0),
        "mbppplus": _evalchemy("MBPPPlus", 3),
        "olympiadbench": _evalchemy("OlympiadBench", 0),
        "mmlu-pro": _evalchemy("MMLUPro", 0),
        "gpqa-diamond": _evalchemy("GPQADiamond", 0),
        "cruxeval": _evalchemy("CruxEval", 1),
        "financebench": _evalchemy("FinanceBench", 0),
        "ifbench": _evalchemy("IFBench", 0),
        "mrcr": _evalchemy("MRCR", 0),
        "gsm8k-0shot": _evalchemy("gsm8k", 0),
        "mmlu": _evalchemy("mmlu", 5),
        "piqa": _evalchemy("piqa", 0),
        "winogrande": _evalchemy("winogrande", 5),
        "openbookqa": _evalchemy("openbookqa", 0),
        "boolq": _evalchemy("boolq", 0),
        "truthfulqa": _evalchemy("truthfulqa_mc2", 0),
        "triviaqa": _evalchemy("triviaqa", 5),
        "swebench-recovery": _harbor(),
        "ot-tblite-recovery": _harbor(),
        "tb2-recovery": _harbor(),
        "simpleqa-recovery": _harbor(),
        "ds-1000-local": _harbor(),
    }
)

SEPTEMBER_24: Mapping[str, PolicyEval] = MappingProxyType(
    {
        "math500": _evalchemy("MATH500", 0, ThinkingMode.ON),
        "humanevalplus": _evalchemy("HumanEvalPlus", 0, ThinkingMode.OFF),
        "mbppplus": _evalchemy("MBPPPlus", 0, ThinkingMode.OFF),
        "olympiadbench": _evalchemy("OlympiadBench", 0, ThinkingMode.ON),
        "gsm8k-0shot": _evalchemy("gsm8k", 0, ThinkingMode.OFF),
        "piqa": _evalchemy("piqa", 0, ThinkingMode.NOT_APPLICABLE),
        "winogrande": _evalchemy("winogrande", 5, ThinkingMode.NOT_APPLICABLE),
        "boolq": _evalchemy("boolq", 0, ThinkingMode.NOT_APPLICABLE),
        "truthfulqa": _evalchemy("truthfulqa_mc2", 0, ThinkingMode.NOT_APPLICABLE),
        "triviaqa": _evalchemy("triviaqa", 5, ThinkingMode.OFF),
        "aime24": _evalchemy("AIME24", 0, ThinkingMode.ON),
        "mmlu-pro": _evalchemy("MMLUPro", 0, ThinkingMode.ON),
        "gpqa-diamond": _evalchemy("GPQADiamond", 0, ThinkingMode.ON),
        "cruxeval": _evalchemy("CruxEval", 0, ThinkingMode.OFF),
        "financebench": _evalchemy("FinanceBench", 0, ThinkingMode.OFF),
        "ifbench": _evalchemy("IFBench", 0, ThinkingMode.OFF),
        "mrcr": _evalchemy("MRCR", 0, ThinkingMode.OFF),
        "nupa": _evalchemy("NUPA", 0),
        "swebench-recovery": _harbor(),
        "ot-tblite-recovery": _harbor(),
        "tb2-recovery": _harbor(),
        "ds-1000-local": _harbor(),
        "bfclparity-pi": _harbor(),
        "bixbench-pi": _harbor(),
        "tau3-pi": _harbor(),
        "sotopia-hard": _harbor(),
    }
)

POLICIES: Mapping[str, Mapping[str, PolicyEval]] = MappingProxyType(
    {
        SEPTEMBER_16_VERSION: SEPTEMBER_16,
        SEPTEMBER_24_VERSION: SEPTEMBER_24,
    }
)
RUNTIME_COMMITS: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {
        SEPTEMBER_16_VERSION: MappingProxyType({"evalchemy": EVALCHEMY_COMMIT, "harbor": HARBOR_COMMIT}),
        SEPTEMBER_24_VERSION: MappingProxyType({"evalchemy": EVALCHEMY_COMMIT, "harbor": HARBOR_COMMIT}),
    }
)


def policy_violations(version: str | None, model: ModelRef, evaluation: EvalRef) -> tuple[str, ...]:
    """Return launch/record violations; unlabelled and non-policy runs are unrestricted."""
    version_key = version or ""
    policy = POLICIES.get(version_key)
    if policy is None:
        if version is not None and version.startswith("eval-policy-"):
            return (f"unknown policy version {version!r}",)
        return ()
    expected = policy.get(evaluation.name)
    if expected is None:
        return (f"{evaluation.name}: not in {version}",)
    problems = []
    if evaluation.source_digest != POLICY_SOURCE_DIGESTS[version_key][evaluation.name]:
        problems.append("source config differs from the approved benchmark policy")
    source_config = model.source_config or model.config
    if source_config is None or model.config is None:
        problems.append("missing normalized model configuration")
    elif source_config.name != model.name:
        problems.append("model name differs from saved configuration")
    elif source_config.location != model.location or model.config.serve.backend != model.backend:
        problems.append("model location or backend differs from saved configuration")
    elif model.config_digest is not None and model.config_digest != model_config_digest(source_config):
        problems.append("model configuration digest does not match the saved configuration")
    if evaluation.mechanism != expected.mechanism:
        problems.append(f"mechanism {evaluation.mechanism!r} != {expected.mechanism!r}")
    if len(evaluation.tasks) != 1:
        problems.append("exactly one task required")
    elif expected.task is not None and evaluation.tasks[0].name != expected.task:
        problems.append(f"task must be {expected.task!r}")
    elif evaluation.tasks[0].num_fewshot != expected.shots:
        problems.append(f"num_fewshot must be {expected.shots}")
    if expected.mechanism == "evalchemy" and len(evaluation.tasks) == 1:
        task = evaluation.tasks[0]
        if task.generation != (evaluation.name not in COMPLETION_TASKS):
            problems.append("task generation/completion route differs from policy")
        if task.unsafe_code != (evaluation.name in UNSAFE_CODE_TASKS):
            problems.append("unsafe code setting differs from policy")
    if expected.mechanism == "evalchemy":
        config = evaluation.evalchemy
        if config is None:
            problems.append("missing Evalchemy configuration")
        else:
            if source_config is not None:
                expected_chat_kwargs = dict(source_config.generation.chat_template_kwargs)
                if policy is SEPTEMBER_24 and expected.thinking in (ThinkingMode.ON, ThinkingMode.OFF):
                    expected_chat_kwargs["enable_thinking"] = expected.thinking is ThinkingMode.ON
                if config.chat_template_kwargs != expected_chat_kwargs:
                    problems.append("chat template arguments differ from model and policy settings")
            if config.max_eval_instances is not None:
                problems.append("capped Evalchemy runs are not canonical")
            if model.config is not None:
                fixed_limit = (
                    SEPTEMBER_16_IFBENCH_MAX_TOKENS
                    if policy is SEPTEMBER_16 and evaluation.name == "ifbench"
                    else FIXED_MAX_TOKENS.get(evaluation.name)
                )
                model_limit = model.config.generation.max_gen_toks
                expected_limit = (
                    min(fixed_limit, model_limit)
                    if fixed_limit is not None and model_limit is not None
                    else (fixed_limit if fixed_limit is not None else model_limit)
                )
                if config.max_gen_toks != expected_limit:
                    problems.append(f"max_gen_toks must be {expected_limit}")
            if evaluation.name == "mmlu-pro" and config.max_length != 65536:
                problems.append("MMLU-Pro max_length must be 65536")
            if evaluation.name == "mrcr" and config.max_length not in {32768, 65536, 73728}:
                problems.append("MRCR max_length must match a published context bin")
            if evaluation.name != "aime24":
                expected_seed = NUPA_SEED if policy is SEPTEMBER_24 and evaluation.name == "nupa" else DEFAULT_SEED
                if config.seed != expected_seed:
                    problems.append(f"seed must be {expected_seed}")
            if policy is SEPTEMBER_24:
                if expected.thinking in (ThinkingMode.ON, ThinkingMode.OFF):
                    required = expected.thinking is ThinkingMode.ON
                    if not config.apply_chat_template:
                        problems.append("thinking mode requires a chat template")
                    if config.chat_template_kwargs.get("enable_thinking") is not required:
                        problems.append(f"explicit enable_thinking={required} required")
                if evaluation.name == "aime24" and config.seed is not None:
                    problems.append("AIME24 outer seed must be unset")
            elif evaluation.name == "aime24" and config.seed not in range(DEFAULT_SEED, DEFAULT_SEED + AIME24_REPEATS):
                problems.append(
                    f"September 16 AIME24 seed must be in {DEFAULT_SEED}..{DEFAULT_SEED + AIME24_REPEATS - 1}"
                )
    elif evaluation.harbor is None:
        problems.append("missing Harbor configuration")
    elif evaluation.harbor.task_limit is not None:
        problems.append("capped Harbor runs are not canonical")
    return tuple(problems)


def runtime_violations(version: str | None, evaluation: EvalRef, runtime: str) -> tuple[str, ...]:
    """Keep evaluator versions fixed within a verified cohort."""
    if version not in POLICIES:
        return ()
    commits = RUNTIME_COMMITS.get(version)
    if commits is None:
        return (f"missing evaluator pins for {version}",)
    commit = commits.get(evaluation.mechanism)
    if commit is None:
        return (f"missing {evaluation.mechanism} evaluator pin for {version}",)
    if commit not in runtime:
        return (f"evaluator runtime must use commit {commit}",)
    if evaluation.harbor is not None:
        harbor_commit = commits.get("harbor")
        if harbor_commit is None:
            return (f"missing harbor evaluator pin for {version}",)
        if evaluation.harbor.harbor_config_commit != harbor_commit:
            return (f"Harbor preflight must use commit {harbor_commit}",)
    return ()


def record_policy_violations(record: EvalRunRecord) -> tuple[str, ...]:
    """Validate verified records; historical labels remain readable without a policy claim."""
    if record.version not in POLICIES:
        return ()
    return (
        *policy_violations(record.version, record.model, record.evaluation),
        *runtime_violations(record.version, record.evaluation, record.provenance.eval_runtime),
    )

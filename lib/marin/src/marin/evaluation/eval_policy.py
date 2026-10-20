# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Admission checks for the current verified evaluation configuration."""

import hashlib
import json
import re
from pathlib import Path

import yaml
from pydantic import TypeAdapter
from typing_extensions import TypedDict

from marin.evaluation.harbor.agent_context import reconciled_model_info, served_model_info
from marin.evaluation.model_identity import model_config_digest
from marin.evaluation.records import EvalRef, EvalRunRecord, ModelConfigRef, ModelRef

MAX_SERVED_CONTEXT = 131072
_MODEL_IDENTITY_FIELDS = (
    "name",
    "location",
    "revision",
    "tokenizer",
    "tokenizer_revision",
    "apply_chat_template",
    "generation",
    "agent",
)


class HarborVariant(TypedDict):
    source_digest: str
    model_info: dict[str, int]


class HarborIdentity(TypedDict):
    dataset: str
    version: str
    agent: str
    env: str


class HarborPolicy(TypedDict):
    identity: HarborIdentity
    variants: dict[str, HarborVariant]


class VerifiedPolicy(TypedDict):
    version: str
    runtimes: dict[str, str]
    harbor_commit: str
    evalchemy: dict[str, EvalRef]
    harbor: dict[str, HarborPolicy]


VERIFIED_POLICY = TypeAdapter(VerifiedPolicy).validate_json(
    Path(__file__).with_name("policies").joinpath("verified.json").read_text()
)
VERIFIED_VERSION = VERIFIED_POLICY["version"]


def source_config_digest(path: Path) -> str:
    """Hash parsed YAML independently of comments and key order."""
    payload = json.dumps(yaml.safe_load(path.read_text()), sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def _model_violations(model: ModelRef, source: ModelConfigRef) -> tuple[str, ...]:
    assert model.config is not None
    actual = model.config.model_dump()
    original = source.model_dump()
    problems = [
        f"effective model {field} differs from its saved source"
        for field in _MODEL_IDENTITY_FIELDS
        if actual[field] != original[field]
    ]
    if source.name != model.name or source.location != model.location or model.config.serve.backend != model.backend:
        problems.append("model identity differs from saved configuration")
    if model.config_digest is not None and model.config_digest != model_config_digest(source):
        problems.append("model configuration digest does not match saved configuration")
    if not re.fullmatch(r"[0-9a-f]{40}", source.revision or ""):
        problems.append("verified evaluations require an immutable model revision")
    context = model.config.serve.max_model_len
    if context is None or context > MAX_SERVED_CONTEXT:
        problems.append(f"verified evaluations require an explicit served context of at most {MAX_SERVED_CONTEXT}")
    return tuple(problems)


def _evalchemy_violations(source: ModelConfigRef, evaluation: EvalRef, approved: EvalRef) -> tuple[str, ...]:
    if evaluation.mechanism != "evalchemy" or evaluation.evalchemy is None:
        return ("benchmark requires Evalchemy",)
    assert approved.evalchemy is not None
    problems = []
    if evaluation.source_digest != approved.source_digest:
        problems.append("source config differs from the verified benchmark")
    if [task.model_dump(exclude={"benchmark"}) for task in evaluation.tasks] != [
        task.model_dump(exclude={"benchmark"}) for task in approved.tasks
    ]:
        problems.append("task settings differ from the verified benchmark")
    policy = approved.evalchemy
    chat = {**source.generation.chat_template_kwargs, **policy.chat_template_kwargs}
    if policy.chat_template_kwargs.get("enable_thinking") is False and source.generation.thinking_off_template_kwargs:
        chat = {key: value for key, value in chat.items() if key != "enable_thinking"}
        chat = {**chat, **source.generation.thinking_off_template_kwargs}
    expected = {
        **policy.model_dump(),
        "chat_template_kwargs": chat,
        "extra_gen_kwargs": {**policy.extra_gen_kwargs, **source.generation.extra_gen_kwargs},
        "max_gen_toks": min(
            (limit for limit in (policy.max_gen_toks, source.generation.max_gen_toks) if limit is not None),
            default=None,
        ),
        "num_concurrent": min(
            policy.num_concurrent,
            source.evalchemy_num_concurrent if source.evalchemy_num_concurrent is not None else policy.num_concurrent,
        ),
    }
    if not source.apply_chat_template:
        problems.append("verified Evalchemy benchmarks require a chat-template model")
    actual = evaluation.evalchemy.model_dump()
    problems.extend(
        f"Evalchemy {key} differs from the verified settings"
        for key, value in expected.items()
        if actual.get(key) != value
    )
    return tuple(problems)


def _harbor_violations(model: ModelRef, evaluation: EvalRef, approved: HarborPolicy) -> tuple[str, ...]:
    assert model.config is not None
    if evaluation.mechanism != "harbor" or evaluation.harbor is None:
        return ("benchmark requires Harbor",)
    variant = approved["variants"].get(model.location, approved["variants"]["default"])
    problems = []
    if evaluation.source_digest != variant["source_digest"]:
        problems.append("source config differs from the verified model-specific Harbor settings")
    if evaluation.harbor.task_limit is not None:
        problems.append("capped Harbor runs are not canonical")
    actual = evaluation.harbor.model_dump()
    problems.extend(
        f"Harbor {key} differs from the verified benchmark"
        for key, value in approved["identity"].items()
        if actual[key] != value
    )
    if len(evaluation.tasks) != 1 or evaluation.tasks[0].name != approved["identity"]["dataset"]:
        problems.append("Harbor task differs from the verified benchmark")
    if evaluation.harbor.config_digest is None:
        problems.append("missing normalized Harbor configuration digest")
    served = served_model_info(model.config.serve.max_model_len, model.config.generation.max_gen_toks)
    try:
        limits = reconciled_model_info(served, variant["model_info"])
    except ValueError as error:
        return (*problems, str(error))
    for key in ("max_input_tokens", "max_output_tokens"):
        if actual.get(key) != limits[key]:
            problems.append(f"Harbor {key} differs from the verified settings")
    return tuple(problems)


def policy_violations(version: str | None, model: ModelRef, evaluation: EvalRef) -> tuple[str, ...]:
    """Validate current-profile launches; reject other reserved policy labels."""
    if version != VERIFIED_VERSION:
        return (f"unknown policy version {version!r}",) if version and version.startswith("eval-policy-") else ()
    source = model.source_config or model.config
    if source is None or model.config is None:
        return ("missing normalized model configuration",)
    model_problems = _model_violations(model, source)
    if approved := VERIFIED_POLICY["evalchemy"].get(evaluation.name):
        return (*model_problems, *_evalchemy_violations(source, evaluation, approved))
    if approved_harbor := VERIFIED_POLICY["harbor"].get(evaluation.name):
        return (*model_problems, *_harbor_violations(model, evaluation, approved_harbor))
    return (*model_problems, f"{evaluation.name}: not in the verified policy")


def runtime_violations(version: str | None, evaluation: EvalRef, runtime: str) -> tuple[str, ...]:
    """Check the current profile's runtime; leave other labels unrestricted."""
    if version != VERIFIED_VERSION:
        return ()
    expected = VERIFIED_POLICY["runtimes"].get(evaluation.name)
    if expected is None or runtime != expected:
        return ("evaluator runtime differs from the verified source and lock",)
    if evaluation.harbor is not None and evaluation.harbor.harbor_config_commit != VERIFIED_POLICY["harbor_commit"]:
        return ("Harbor preflight differs from the verified upstream import",)
    return ()


def record_policy_violations(record: EvalRunRecord) -> tuple[str, ...]:
    """Validate current-cohort records while leaving historical labels readable."""
    if record.version != VERIFIED_VERSION:
        return ()
    return (
        *policy_violations(record.version, record.model, record.evaluation),
        *runtime_violations(record.version, record.evaluation, record.provenance.eval_runtime),
    )

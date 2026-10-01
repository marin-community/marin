# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Frozen configuration profile for the September 29 campaign.

These checks compare submitter-provided settings. Independent policy attestation
requires the trusted evidence boundary described in marin-community/marin#9458.
"""

import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from marin.evaluation.harbor.agent_context import reconciled_model_info, served_model_info
from marin.evaluation.model_identity import model_config_digest
from marin.evaluation.records import EvalchemyRef, EvalRef, ModelRef

SEPTEMBER_29_VERSION = "eval-policy-2026-09-29-verified"
MAX_SERVED_CONTEXT = 73728


class HarborProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sources: dict[Literal["standard", "native32k"], str]
    model_info: dict[str, int]
    identity: dict[Literal["dataset", "version", "agent", "env"], str]


class CampaignProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    runtimes: dict[str, str]
    native32k_locations: tuple[str, ...]
    evalchemy: dict[str, EvalRef]
    harbor: dict[str, HarborProfile]


CAMPAIGN = CampaignProfile.model_validate_json(
    Path(__file__).with_name("policies").joinpath("september_29.json").read_text()
)


def campaign_policy_violations(model: ModelRef, evaluation: EvalRef) -> tuple[str, ...]:
    """Check a selected benchmark against the frozen campaign, including model overrides."""
    source = model.source_config or model.config
    if source is None or model.config is None:
        return ("missing normalized model configuration",)
    problems = []
    if source.name != model.name or source.location != model.location or model.config.serve.backend != model.backend:
        problems.append("model identity differs from saved configuration")
    for field in (
        "name",
        "location",
        "revision",
        "tokenizer",
        "tokenizer_revision",
        "apply_chat_template",
        "generation",
        "agent",
    ):
        if getattr(model.config, field) != getattr(source, field):
            problems.append(f"effective model {field} differs from the saved source configuration")
    context = model.config.serve.max_model_len
    if context is None or context > MAX_SERVED_CONTEXT:
        problems.append(f"campaign requires an explicit served context of at most {MAX_SERVED_CONTEXT} tokens")
    if model.config_digest is not None and model.config_digest != model_config_digest(source):
        problems.append("model configuration digest does not match the saved configuration")
    if not re.fullmatch(r"[0-9a-f]{40}", source.revision or ""):
        problems.append("campaign requires an immutable model revision")
    if evaluation.name in CAMPAIGN.evalchemy:
        approved = CAMPAIGN.evalchemy[evaluation.name]
        if evaluation.mechanism != "evalchemy" or evaluation.evalchemy is None:
            return (*problems, "campaign requires Evalchemy for this benchmark")
        if evaluation.source_digest != approved.source_digest:
            problems.append("source config differs from the approved benchmark policy")
        actual_tasks = [task.model_dump(exclude={"benchmark"}) for task in evaluation.tasks]
        approved_tasks = [task.model_dump(exclude={"benchmark"}) for task in approved.tasks]
        if actual_tasks != approved_tasks:
            problems.append("task settings differ from the approved benchmark policy")
        assert approved.evalchemy is not None
        expected = approved.evalchemy.model_dump()
        policy_chat = approved.evalchemy.chat_template_kwargs
        expected["chat_template_kwargs"] = {**source.generation.chat_template_kwargs, **policy_chat}
        if policy_chat.get("enable_thinking") is False and source.generation.thinking_off_template_kwargs:
            expected["chat_template_kwargs"].pop("enable_thinking")
            expected["chat_template_kwargs"].update(source.generation.thinking_off_template_kwargs)
        expected["extra_gen_kwargs"] = {**approved.evalchemy.extra_gen_kwargs, **source.generation.extra_gen_kwargs}
        model_limit = source.generation.max_gen_toks
        policy_limit = approved.evalchemy.max_gen_toks
        expected["max_gen_toks"] = (
            policy_limit
            if model_limit is None
            else model_limit if policy_limit is None else min(model_limit, policy_limit)
        )
        if not source.apply_chat_template:
            problems.append("campaign Evalchemy benchmarks require a chat-template model")
        expected_config = EvalchemyRef.model_validate(expected)
        problems.extend(
            f"Evalchemy {key} differs from the approved effective settings"
            for key in EvalchemyRef.model_fields
            if getattr(evaluation.evalchemy, key) != getattr(expected_config, key)
        )
        return tuple(problems)
    if evaluation.name in CAMPAIGN.harbor:
        approved_harbor = CAMPAIGN.harbor[evaluation.name]
        if evaluation.mechanism != "harbor" or evaluation.harbor is None:
            return (*problems, "campaign requires Harbor for this benchmark")
        variant = "native32k" if source.location in CAMPAIGN.native32k_locations else "standard"
        if evaluation.source_digest != approved_harbor.sources[variant]:
            problems.append("source config differs from the approved model-specific Harbor policy")
        if evaluation.harbor.task_limit is not None:
            problems.append("capped Harbor runs are not canonical")
        actual_harbor = evaluation.harbor.model_dump()
        problems.extend(
            f"Harbor {key} differs from the approved benchmark policy"
            for key, value in approved_harbor.identity.items()
            if actual_harbor[key] != value
        )
        if len(evaluation.tasks) != 1 or evaluation.tasks[0].name != approved_harbor.identity["dataset"]:
            problems.append("Harbor task differs from the approved benchmark policy")
        if evaluation.harbor.config_digest is None:
            problems.append("missing normalized Harbor configuration digest")
        policy_info = dict(approved_harbor.model_info)
        if variant == "native32k" and policy_info:
            policy_info["max_input_tokens"] = 32768
        served = served_model_info(model.config.serve.max_model_len, model.config.generation.max_gen_toks)
        try:
            expected_info = reconciled_model_info(served, policy_info)
        except ValueError as error:
            return (*problems, str(error))
        if evaluation.harbor.max_input_tokens != expected_info["max_input_tokens"]:
            problems.append("Harbor input context differs from the approved effective settings")
        if evaluation.harbor.max_output_tokens != expected_info["max_output_tokens"]:
            problems.append("Harbor output limit differs from the approved effective settings")
        return tuple(problems)
    return (*problems, f"{evaluation.name}: not in {SEPTEMBER_29_VERSION}")

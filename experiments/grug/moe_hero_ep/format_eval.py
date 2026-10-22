# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned Hero format panels, candidate decisions and measured pilot accounting.

The native sampler's ``SampleResult`` is the evaluation boundary. Generation
must retain special tokens and stop at Marin EOT; stripping tokens would hide
serialization failures. This module neither restores model weights nor launches
training. Its CLI writes auditable reports from already completed artifacts.
"""

import argparse
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from statistics import NormalDist
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec, get_mesh
from levanter.data.text.formats import ChatProcessor
from levanter.tokenizers import MarinTokenizer, load_tokenizer
from marin.datakit.chat_template import MARIN_CHAT_TEMPLATE
from marin.evaluation.eval_stats import wilson_interval
from pydantic import Field, TypeAdapter, model_validator
from rigging.filesystem.storage_path import StoragePath
from verifyit.modes.grade_exact import grade_exact_candidate
from verifyit.spec import ExactSpec

from experiments.grug.moe_hero_ep.model import _BATCH_AXES, Transformer, _seq_axis
from experiments.grug.moe_hero_ep.ops.vibe_check.completions import (
    Prompt,
    Record,
    SampleResult,
    SamplingSpec,
    StopReason,
    digest,
)
from experiments.grug.moe_hero_ep.sft import SftConfig, SftParameters, sft_logits

EOT_ID = 128009
EOT_TEXT = "<|eot_id|>"
START_THINK = "<|start_think|>"
END_THINK = "<|end_think|>"
TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
FORBIDDEN_CONTINUATION = ("<|start_header_id|>", "<|end_header_id|>", "<tool_response", "</tool_response>")


class Interface(StrEnum):
    SIMPLE = "simple"
    REASONING_ON = "reasoning_on"
    REASONING_OFF = "reasoning_off"
    TOOL_CALL = "tool_call"
    OBSERVATION = "observation"
    CONTEXT_16K = "context16k"
    CONTEXT_65K = "context65k"


class Serialization(StrEnum):
    MARIN = "marin"
    LEGACY = "legacy"


class PanelCase(Record):
    id: str = Field(pattern=r"^[a-z0-9-]+$")
    interface: Interface
    messages: tuple[dict, ...] = Field(min_length=1)
    legacy_prompt: str = Field(min_length=1)
    template_kwargs: dict = Field(default_factory=dict)
    expected_tool_names: tuple[str, ...] = ()
    expected_observation_answer: str | None = None
    # Pin an actual long-token requirement rather than trusting a "16K" label.
    minimum_prompt_tokens: int = Field(default=1, ge=1, lt=65536)
    seed: int = Field(ge=0)
    source: str = Field(min_length=1)
    expected: str | None = None

    @model_validator(mode="after")
    def interface_contract(self) -> Self:
        if any(key in self.template_kwargs for key in ("chat_template", "tokenize", "add_generation_prompt")):
            raise ValueError("Panel cases may not override the renderer or generation boundary")
        if self.interface == Interface.REASONING_ON and self.template_kwargs.get("enable_thinking") is not True:
            raise ValueError("Reasoning-on panel must request enable_thinking=true")
        if self.interface == Interface.REASONING_OFF and self.template_kwargs.get("enable_thinking") is not False:
            raise ValueError("Reasoning-off panel must request enable_thinking=false")
        if self.interface == Interface.TOOL_CALL and not self.expected_tool_names:
            raise ValueError("Tool-call cases must pin expected function names")
        if self.interface == Interface.OBSERVATION:
            if not self.expected_observation_answer or not any(message["role"] == "tool" for message in self.messages):
                raise ValueError("Observation cases need a tool response and an exact expected answer fragment")
        if self.interface == Interface.CONTEXT_16K and self.minimum_prompt_tokens < 16000:
            raise ValueError("16K check requires an actual prompt with at least 16000 tokens")
        if self.interface == Interface.CONTEXT_65K and self.minimum_prompt_tokens < 65000:
            raise ValueError("65K check requires an actual prompt with at least 65000 tokens")
        if self.messages[-1]["role"] == "assistant":
            raise ValueError("Generation panels must end at a user or tool turn")
        return self


class Panel(Record):
    release: str
    cases: tuple[PanelCase, ...] = Field(min_length=1)
    tokenizer: str
    tokenizer_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    backend: str = Field(min_length=1)
    template_sha256: str = Field(default_factory=lambda: hashlib.sha256(MARIN_CHAT_TEMPLATE.encode()).hexdigest())
    max_new_tokens: int = Field(gt=0, le=65536)
    context_length: int = Field(default=16384)
    temperature: float = Field(ge=0, allow_inf_nan=False)

    @property
    def panel_id(self) -> str:
        return digest(self.model_dump(mode="json"))

    @model_validator(mode="after")
    def unique_cases(self) -> Self:
        if self.context_length not in (16384, 65536):
            raise ValueError("Format panels support the 16K pilot or the separately gated 65K extension")
        if len({case.id for case in self.cases}) != len(self.cases):
            raise ValueError("Panel case IDs must be unique")
        return self


def render_panel(panel: Panel, tokenizer: MarinTokenizer, serialization: Serialization) -> SamplingSpec:
    """Render the shared template or explicitly pinned legacy PT prompts."""
    if panel.template_sha256 != hashlib.sha256(MARIN_CHAT_TEMPLATE.encode()).hexdigest():
        raise ValueError("The shared template changed since the panel was pinned")
    if serialization == Serialization.MARIN and tokenizer.encode(EOT_TEXT) != [EOT_ID]:
        raise ValueError("Prepared tokenizer does not match Hero's atomic Marin EOT")
    prompts = []
    for case in panel.cases:
        text = case.legacy_prompt
        if serialization == Serialization.MARIN:
            rendered = tokenizer.with_chat_template(MARIN_CHAT_TEMPLATE).apply_chat_template(
                list(case.messages),
                tokenize=False,
                add_generation_prompt=True,
                **case.template_kwargs,
            )
            assert isinstance(rendered, str)
            text = rendered
        ids = tokenizer.encode(text)
        if len(ids) < case.minimum_prompt_tokens or len(ids) >= panel.context_length:
            raise ValueError(f"Panel prompt does not meet its actual context limits: {case.id}")
        if case.expected is not None and len(ids) + len(tokenizer.encode(case.expected)) + 1 > panel.context_length:
            raise ValueError(f"Reference completion does not fit the actual context: {case.id}")
        prompts.append(Prompt(id=case.id, text=text, seed=case.seed, source_url=case.source, expected=case.expected))
    return SamplingSpec(
        release=f"{panel.release}-{serialization}-{panel.panel_id}",
        prompts=tuple(prompts),
        completions_per_prompt=1,
        tokenizer=panel.tokenizer,
        tokenizer_revision=panel.tokenizer_revision,
        temperature=panel.temperature,
        max_new_tokens=panel.max_new_tokens,
        context_length=panel.context_length,
    )


def completion_failures(
    case: PanelCase, text: str, token_ids: Sequence[int], stop_reason: StopReason
) -> tuple[str, ...]:
    """Classify raw continuation failures without discarding any generated text."""
    failures = []
    if stop_reason != StopReason.EOS or not token_ids or token_ids[-1] != EOT_ID:
        failures.append("termination")
    if text.count(EOT_TEXT) != 1 or not text.endswith(EOT_TEXT) or EOT_ID in token_ids[:-1]:
        failures.append("turn_boundary")
    if any(marker in text for marker in FORBIDDEN_CONTINUATION):
        failures.append("role_or_observation_continuation")
    content = text.removesuffix(EOT_TEXT)
    if case.interface == Interface.REASONING_ON:
        valid = (
            content.startswith(START_THINK)
            and content.count(START_THINK) == 1
            and content.count(END_THINK) == 1
            and bool(content.split(END_THINK)[-1].strip())
        )
        if not valid:
            failures.append("reasoning_mode")
    elif case.interface == Interface.REASONING_OFF and (START_THINK in content or END_THINK in content):
        failures.append("reasoning_mode")
    if case.interface == Interface.TOOL_CALL:
        calls = TOOL_CALL.findall(content)
        names = []
        parseable = len(calls) == content.count("<tool_call>") == content.count("</tool_call>")
        for call in calls:
            try:
                value = json.loads(call)
            except json.JSONDecodeError:
                parseable = False
                continue
            if not isinstance(value, dict) or not isinstance(value.get("arguments"), dict):
                parseable = False
            else:
                names.append(value.get("name"))
        if not parseable or tuple(names) != case.expected_tool_names:
            failures.append("tool_call")
    if case.interface == Interface.OBSERVATION:
        assert case.expected_observation_answer is not None
        if case.expected_observation_answer not in content:
            failures.append("observation_handling")
    return tuple(failures)


def grade_panel(panel: Panel, result: SampleResult, serialization: Serialization = Serialization.MARIN) -> dict:
    """Produce per-interface exact counts/95% Wilson intervals and retain every generation.

    This checks structural compliance. Content scoring and native-to-serving
    parity are separate measurements required by the candidate gate.
    """
    spec = result.request.spec
    if (
        spec.release != f"{panel.release}-{serialization}-{panel.panel_id}"
        or spec.tokenizer != panel.tokenizer
        or spec.tokenizer_revision != panel.tokenizer_revision
        or result.request.source_revision != panel.source_revision
        or spec.completions_per_prompt != 1
        or spec.context_length != panel.context_length
        or spec.temperature != panel.temperature
        or spec.max_new_tokens != panel.max_new_tokens
        or [prompt.id for prompt in spec.prompts] != [case.id for case in panel.cases]
    ):
        raise ValueError("Completed samples do not match the pinned panel and evaluation revision")
    if serialization == Serialization.MARIN and result.eos_token_id != EOT_ID:
        raise ValueError("Marin evaluation must stop on EOT while retaining special tokens")
    rows = []
    groups: dict[str, list[dict]] = defaultdict(list)
    for case, completed in zip(panel.cases, result.completions, strict=True):
        sample = completed.samples[0]
        failures = (
            completion_failures(case, sample.text, sample.token_ids, sample.stop_reason)
            if serialization == Serialization.MARIN
            else ()
        )
        if len(completed.prompt_token_ids) < case.minimum_prompt_tokens:
            raise ValueError(f"Actual evaluated prompt misses the context check: {case.id}")
        row = {"id": case.id, "interface": case.interface.value, "failures": failures, "generation": sample.model_dump()}
        rows.append(row)
        groups[case.interface.value].append(row)
    metrics = {}
    if serialization == Serialization.MARIN:
        for interface, cases in groups.items():
            successes = sum(not row["failures"] for row in cases)
            interval = wilson_interval(successes, len(cases))
            metrics[interface] = {
                "successes": successes,
                "failures": len(cases) - successes,
                "total": len(cases),
                "observed_rate": successes / len(cases),
                "confidence_interval_95": [interval.low, interval.high],
                "interval_method": "Wilson score; operational fixed-panel rate is not a population guarantee",
            }
    report = {
        "panel_id": panel.panel_id,
        "serialization": serialization.value,
        "context_length": panel.context_length,
        "prompt_token_counts": {row.prompt_id: len(row.prompt_token_ids) for row in result.completions},
        "checkpoint": result.request.checkpoint.model_dump(),
        "backend": panel.backend,
        "model_config_digest": digest(result.request.model),
        "tokenizer_revision": panel.tokenizer_revision,
        "sample_id": result.request.sample_id,
        "metrics": metrics,
        "generations": rows,
    }
    report["report_digest"] = digest(report)
    return report


def score_content_panel(
    panel: Panel,
    result: SampleResult,
    tokenizer: MarinTokenizer,
    serialization: Serialization = Serialization.MARIN,
) -> dict:
    """Score fixed short references with Verifyit's existing exact-answer grader.

    This executes no generated code. It removes the final declared stop token
    by ID and a single well-formed leading reasoning block, then grades the
    complete remaining answer. Substring guesses cannot earn credit.
    """
    structural = grade_panel(panel, result, serialization)
    rows = []
    outcomes = {}
    for case, completed in zip(panel.cases, result.completions, strict=True):
        if not case.expected:
            raise ValueError("Content cases need a nonempty pinned reference")
        sample = completed.samples[0]
        ids = sample.token_ids
        if ids and ids[-1] == result.eos_token_id:
            ids = ids[:-1]
        answer = tokenizer.decode(list(ids), skip_special_tokens=False)
        if answer.startswith(START_THINK) and answer.count(START_THINK) == answer.count(END_THINK) == 1:
            answer = answer.split(END_THINK, 1)[1]
        verdict = grade_exact_candidate(ExactSpec((case.expected,), ignore_case=True, ignore_whitespace=True), answer)
        outcomes[case.id] = verdict.reward == 1
        rows.append(
            {
                "id": case.id,
                "reference": case.expected,
                "answer": answer,
                "correct": verdict.reward == 1,
                "grader_detail": verdict.detail,
            }
        )
    interval = wilson_interval(sum(outcomes.values()), len(outcomes))
    return {
        "panel_id": panel.panel_id,
        "serialization": serialization.value,
        "outcomes": outcomes,
        "successes": sum(outcomes.values()),
        "total": len(outcomes),
        "confidence_interval_95": [interval.low, interval.high],
        "rule": "Verifyit ExactSpec case-insensitive whitespace-normalized whole answer; no code execution",
        "rows": rows,
        "structural_report_digest": structural["report_digest"],
        "sample_id": result.request.sample_id,
        "model_config_digest": digest(result.request.model),
    }


class Candidate(StrEnum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"


class CandidateEvaluation(Record):
    candidate: Candidate
    parent: Candidate | None
    checkpoint: str
    checkpoint_metadata_digest: str
    model_config_digest: str
    panel_id: str
    content_panel_id: str
    tokenizer_revision: str
    backend: str
    evaluation_dtype: str
    heldout_loss_panel: str
    heldout_loss: float
    routing: dict[str, float]
    # Each serialization maps fixed content-case IDs to binary scored outcomes.
    content: dict[Serialization, dict[str, bool]]
    format_report: dict
    long_context_report: dict | None = None
    serving_parity_report: str | None = None
    serving_parity_passed: bool = False

    @property
    def evaluation_id(self) -> str:
        return digest(self.model_dump(mode="json"))


class GatePolicy(Record):
    baseline_evaluation_id: str
    baseline_variability_record: str = Field(min_length=1)
    heldout_loss_increase: float = Field(ge=0, allow_inf_nan=False)
    content_accuracy_decline: float = Field(ge=0, le=1, allow_inf_nan=False)
    routing_maximums: dict[str, float]
    enabled_interfaces: tuple[Interface, ...] = Field(min_length=1)
    observed_format_minimum: float = Field(ge=0.99, le=1)
    minimum_simple_cases: int = Field(ge=500)
    minimum_interface_cases: int = Field(ge=100)
    minimum_long_context_cases: int = Field(default=1, ge=1)
    long_context_panel_id: str | None = None

    @model_validator(mode="after")
    def simple_panel_required(self) -> Self:
        if Interface.SIMPLE not in self.enabled_interfaces:
            raise ValueError("The joint simple-chat acceptance panel is mandatory")
        if any(not math.isfinite(value) for value in self.routing_maximums.values()):
            raise ValueError("Routing limits must be finite")
        return self


class FormatMetric(Record):
    successes: int = Field(ge=0, strict=True)
    failures: int = Field(ge=0, strict=True)
    total: int = Field(gt=0, strict=True)
    observed_rate: float = Field(ge=0, le=1, allow_inf_nan=False)
    confidence_interval_95: tuple[float, float]
    interval_method: str


def validated_format_metrics(report: dict) -> dict[str, FormatMetric]:
    """Verify bound reports against their retained rows and independently recomputed counts."""
    if report.get("report_digest") != digest({key: value for key, value in report.items() if key != "report_digest"}):
        raise ValueError("Formatting report digest changed")
    rows = report["generations"]
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Formatting report contains repeated generation IDs")
    counts = defaultdict(lambda: [0, 0])
    for row in rows:
        counts[row["interface"]][0] += int(not row["failures"])
        counts[row["interface"]][1] += 1
    metrics = {name: FormatMetric.model_validate(metric) for name, metric in report["metrics"].items()}
    if metrics.keys() != counts.keys():
        raise ValueError("Formatting report metrics omit or add generation interfaces")
    for name, metric in metrics.items():
        successes, total = counts[name]
        interval = wilson_interval(successes, total)
        if (
            (metric.successes, metric.total, metric.failures) != (successes, total, total - successes)
            or metric.observed_rate != successes / total
            or any(
                not math.isfinite(actual) or abs(actual - expected) > 1e-12
                for actual, expected in zip(metric.confidence_interval_95, (interval.low, interval.high), strict=True)
            )
        ):
            raise ValueError("Formatting report counts, rate or confidence interval disagree with its retained rows")
    return metrics


def paired_content_interval(parent: dict[str, bool], child: dict[str, bool]) -> dict:
    """Conservative 95% interval for paired accuracy change using discordant outcomes.

    Bonferroni combines two 97.5% Wilson intervals, one each for wins and
    losses among the same fixed examples. Unscored/missing cases cannot pass.
    """
    if not parent or parent.keys() != child.keys():
        raise ValueError("Content comparisons need the same complete, nonempty case IDs")
    wins = sum(child[key] and not parent[key] for key in parent)
    losses = sum(parent[key] and not child[key] for key in parent)
    z = NormalDist().inv_cdf(1 - 0.05 / 4)
    win_interval = wilson_interval(wins, len(parent), z)
    loss_interval = wilson_interval(losses, len(parent), z)
    return {
        "examples": len(parent),
        "wins": wins,
        "losses": losses,
        "observed_change": (wins - losses) / len(parent),
        "confidence_interval_95": [win_interval.low - loss_interval.high, win_interval.high - loss_interval.low],
        "method": "paired discordance; Bonferroni Wilson score intervals",
    }


def compare_native_candidates(
    baseline: CandidateEvaluation,
    parent: CandidateEvaluation,
    child: CandidateEvaluation,
    policy: GatePolicy,
    *,
    require_format: bool,
) -> dict:
    """Gate A→B, selected A/B→C, C→D, D→E or the final A→D comparison.

    Nonfinite measurements, excessive loss or routing trigger diagnosis. A
    content interval crossing the declared regression bound is inconclusive;
    an observed point estimate alone cannot silently pass it.
    """
    if baseline.candidate != Candidate.A or baseline.evaluation_id != policy.baseline_evaluation_id:
        raise ValueError("Regression policy was not pinned to this baseline")
    final_comparison = parent.candidate == Candidate.A and child.candidate in (Candidate.D, Candidate.E)
    if not final_comparison and child.parent != parent.candidate:
        raise ValueError("Candidate parent does not match the actual comparison")
    expected_parent = {
        Candidate.B: {Candidate.A},
        Candidate.C: {Candidate.A, Candidate.B},
        Candidate.D: {Candidate.C},
        Candidate.E: {Candidate.D},
    }
    if not final_comparison and parent.candidate not in expected_parent.get(child.candidate, set()):
        raise ValueError("Unsupported candidate comparison")
    for evaluation in (parent, child):
        for key in ("content_panel_id", "tokenizer_revision", "backend", "evaluation_dtype", "heldout_loss_panel"):
            if getattr(evaluation, key) != getattr(baseline, key):
                raise ValueError(f"Candidate comparison changed pinned evaluation identity: {key}")
    violations = []
    uncertain = []
    finite = all(
        math.isfinite(value)
        for value in [
            baseline.heldout_loss,
            parent.heldout_loss,
            child.heldout_loss,
            *baseline.routing.values(),
            *parent.routing.values(),
            *child.routing.values(),
        ]
    )
    if not finite:
        violations.append("nonfinite_measurement")
    if child.heldout_loss - parent.heldout_loss > policy.heldout_loss_increase:
        violations.append("heldout_loss")
    if not policy.routing_maximums or any(
        metric not in child.routing or child.routing[metric] > maximum
        for metric, maximum in policy.routing_maximums.items()
    ):
        violations.append("routing")
    content = {}
    for serialization in Serialization:
        if serialization not in parent.content or serialization not in child.content:
            raise ValueError("Both legacy and Marin content panels are required")
        comparison = paired_content_interval(parent.content[serialization], child.content[serialization])
        content[serialization.value] = comparison
        lower, upper = comparison["confidence_interval_95"]
        if upper < -policy.content_accuracy_decline:
            violations.append(f"content_{serialization}")
        elif lower < -policy.content_accuracy_decline:
            uncertain.append(f"content_{serialization}")
    if require_format:
        report = child.format_report
        metrics = validated_format_metrics(report)
        if (
            report.get("panel_id") != child.panel_id
            or report.get("checkpoint", {}).get("metadata_digest") != child.checkpoint_metadata_digest
            or report.get("checkpoint", {}).get("uri") != child.checkpoint
            or report.get("backend") != child.backend
            or report.get("model_config_digest") != child.model_config_digest
            or report.get("serialization") != Serialization.MARIN.value
            or report.get("tokenizer_revision") != child.tokenizer_revision
        ):
            raise ValueError("Formatting report does not belong to the evaluated candidate")
        for interface in policy.enabled_interfaces:
            if interface == Interface.CONTEXT_65K and child.candidate == Candidate.E:
                continue
            metric = metrics.get(interface.value)
            if interface == Interface.SIMPLE:
                minimum = policy.minimum_simple_cases
            elif interface in (Interface.CONTEXT_16K, Interface.CONTEXT_65K):
                minimum = policy.minimum_long_context_cases
            else:
                minimum = policy.minimum_interface_cases
            if not metric or metric.total < minimum or metric.observed_rate < policy.observed_format_minimum:
                violations.append(f"format_{interface}")
        if child.candidate == Candidate.E:
            if not policy.long_context_panel_id:
                uncertain.append("predeclared_65k_panel")
            short_metric = metrics.get(Interface.CONTEXT_16K.value)
            if (
                not short_metric
                or short_metric.total < policy.minimum_long_context_cases
                or (short_metric.observed_rate < policy.observed_format_minimum)
            ):
                violations.append("format_context16k")
            long = child.long_context_report
            if not long:
                uncertain.append("actual_65k_context_panel")
            else:
                long_metrics = validated_format_metrics(long)
                if (
                    long.get("panel_id") != policy.long_context_panel_id
                    or long.get("checkpoint", {}).get("uri") != child.checkpoint
                    or long.get("checkpoint", {}).get("metadata_digest") != child.checkpoint_metadata_digest
                    or long.get("backend") != child.backend
                    or long.get("model_config_digest") != child.model_config_digest
                    or long.get("tokenizer_revision") != child.tokenizer_revision
                    or long.get("serialization") != Serialization.MARIN.value
                    or long.get("context_length") != 65536
                ):
                    raise ValueError("65K formatting report does not belong to this candidate and backend")
                long_metric = long_metrics.get(Interface.CONTEXT_65K.value)
                if (
                    not long_metric
                    or long_metric.total < policy.minimum_long_context_cases
                    or long_metric.observed_rate < policy.observed_format_minimum
                ):
                    violations.append("format_context65k")
    return {
        "baseline_evaluation_id": baseline.evaluation_id,
        "parent_evaluation_id": parent.evaluation_id,
        "child_evaluation_id": child.evaluation_id,
        "decision": "diagnose" if violations else "inconclusive" if uncertain else "accept",
        "violations": violations,
        "uncertain": uncertain,
        "content": content,
        "loss_change": child.heldout_loss - parent.heldout_loss,
        "policy": policy.model_dump(mode="json"),
    }


def compare_candidates(
    baseline: CandidateEvaluation,
    parent: CandidateEvaluation,
    child: CandidateEvaluation,
    policy: GatePolicy,
    *,
    require_format: bool,
) -> dict:
    """Apply native candidate gates and require serving parity for final acceptance."""
    result = compare_native_candidates(baseline, parent, child, policy, require_format=require_format)
    if require_format and (not child.serving_parity_passed or not child.serving_parity_report):
        result["uncertain"].append("native_to_serving_parity")
        if result["decision"] == "accept":
            result["decision"] = "inconclusive"
    return result


class AcquisitionCase(Record):
    source_id: str
    prefix_token_ids: tuple[int, ...] = Field(min_length=1)
    target_token_id: int = Field(ge=0)


class AcquisitionPanel(Record):
    source_artifact_digest: str
    tokenizer_revision: str
    cases: tuple[AcquisitionCase, ...] = Field(min_length=1)
    scope: str = "teacher-forced selected-token acquisition on fixed training-source examples"

    @property
    def panel_digest(self) -> str:
        return digest(self.model_dump())


def build_acquisition_panel(
    training_rows: Sequence[dict],
    tokenizer: MarinTokenizer,
    *,
    tokenizer_revision: str,
    source_artifact_digest: str,
    selected_token_ids: Sequence[int],
    max_examples: int,
) -> AcquisitionPanel:
    """Pin real supervised special-token targets, separately from heldout acceptance."""
    processor = ChatProcessor(tokenizer, chat_template=MARIN_CHAT_TEMPLATE, mask_user_turns=True)
    cases = []
    selected = set(selected_token_ids)
    for row in training_rows[:max_examples]:
        processed = processor([row])[0]
        ids, masks = processed["input_ids"], processed["assistant_masks"]
        for position in range(1, len(ids)):
            if masks[position] and int(ids[position]) in selected:
                cases.append(
                    AcquisitionCase(
                        source_id=row["id"],
                        prefix_token_ids=tuple(map(int, ids[:position])),
                        target_token_id=int(ids[position]),
                    )
                )
    if not cases:
        raise ValueError("Training-source diagnostic contains no supervised selected-token targets")
    return AcquisitionPanel(
        source_artifact_digest=source_artifact_digest, tokenizer_revision=tokenizer_revision, cases=tuple(cases)
    )


@eqx.filter_jit
def _acquisition_scores(frozen, params, token_ids, positions, targets, config):
    logits = sft_logits(frozen, params, token_ids, config, positions=positions)
    selected = logits.at[jnp.arange(logits.shape[0]), targets].get(out_sharding=PartitionSpec())
    normalizer = jax.sharding.reshard(jax.scipy.special.logsumexp(logits, axis=-1), PartitionSpec())
    nll = normalizer - selected
    return (
        jax.sharding.reshard(jnp.argmax(logits, axis=-1), PartitionSpec()),
        jax.sharding.reshard(nll, PartitionSpec()),
    )


@dataclass(frozen=True)
class AcquisitionDiagnostic:
    """Runtime callback; measures short teacher-forced acquisition outside update timing."""

    panel_json: str

    def __post_init__(self):
        panel = AcquisitionPanel.model_validate_json(self.panel_json)
        if self.panel_json != panel.model_dump_json():
            raise ValueError("AcquisitionDiagnostic requires canonical AcquisitionPanel.model_dump_json()")

    @classmethod
    def from_panel(cls, panel: AcquisitionPanel) -> Self:
        return cls(panel_json=panel.model_dump_json())

    def __call__(self, frozen: Transformer, params: SftParameters, config: SftConfig, step: int) -> dict:
        panel = AcquisitionPanel.model_validate_json(self.panel_json)
        mesh = get_mesh()
        if mesh.empty:
            raise ValueError("Acquisition diagnostic requires the same explicit model mesh as SFT")
        count = len(panel.cases)
        batch_size = math.ceil(count / jax.device_count()) * jax.device_count()
        context_axis = _seq_axis(mesh)
        sequence_multiple = int(mesh.shape[context_axis]) if context_axis else 1
        sequence_length = max(len(case.prefix_token_ids) for case in panel.cases)
        sequence_length = math.ceil(sequence_length / sequence_multiple) * sequence_multiple
        if sequence_length > frozen.config.max_seq_len:
            raise ValueError("Acquisition prefix exceeds the current model context")
        tokens = np.full((batch_size, sequence_length), panel.cases[0].target_token_id, dtype=np.int32)
        positions = np.empty(batch_size, dtype=np.int32)
        targets = np.empty(batch_size, dtype=np.int32)
        for index in range(batch_size):
            case = panel.cases[index % count]
            tokens[index, : len(case.prefix_token_ids)] = case.prefix_token_ids
            positions[index] = len(case.prefix_token_ids) - 1
            targets[index] = case.target_token_id
        token_sharding = NamedSharding(mesh, PartitionSpec(_BATCH_AXES, context_axis))
        row_sharding = NamedSharding(mesh, PartitionSpec(_BATCH_AXES))
        token_array = jax.make_array_from_callback(tokens.shape, token_sharding, lambda index: tokens[index])
        position_array = jax.make_array_from_callback(positions.shape, row_sharding, lambda index: positions[index])
        target_array = jax.make_array_from_callback(targets.shape, row_sharding, lambda index: targets[index])
        predicted, nll = _acquisition_scores(frozen, params, token_array, position_array, target_array, config)
        predicted, nll = np.asarray(predicted)[:count], np.asarray(nll)[:count]
        if not np.isfinite(nll).all():
            raise ValueError("Nonfinite acquisition likelihood; diagnose before continuing SFT")
        rows = [
            {
                "source_id": case.source_id,
                "target_id": case.target_token_id,
                "predicted_id": int(predicted[index]),
                "nll": float(nll[index]),
            }
            for index, case in enumerate(panel.cases)
        ]
        return {
            "successes": int(np.sum(predicted == targets[:count])),
            "cases": count,
            "mean_nll": float(np.mean(nll)),
            "panel_digest": panel.panel_digest,
            "rule": panel.scope,
            "report": {"step": step, "rows": rows},
        }


class SmokeUpdate(Record):
    update: int = Field(ge=1)
    elapsed_seconds: float = Field(gt=0, allow_inf_nan=False)
    executed_tokens: int = Field(gt=0)
    non_padding_tokens: int = Field(gt=0)
    supervised_tokens: int = Field(gt=0)
    peak_hbm_bytes: int = Field(gt=0)
    optimizer_seconds: float = Field(ge=0, allow_inf_nan=False)
    loss: float = Field(allow_inf_nan=False)
    gradient_norm: float = Field(ge=0, allow_inf_nan=False)
    diagnostic_successes: int = Field(ge=0)
    diagnostic_cases: int = Field(gt=0)

    @model_validator(mode="after")
    def consistent_counters(self) -> Self:
        if not self.supervised_tokens <= self.non_padding_tokens <= self.executed_tokens:
            raise ValueError("Expected supervised <= non-padding <= executed tokens")
        if self.diagnostic_successes > self.diagnostic_cases:
            raise ValueError("Diagnostic successes exceed the fixed diagnostic panel")
        return self


class SmokeRecord(Record):
    execution_id: str
    checkpoint_metadata_digest: str
    hardware: str
    gpu_count: int = Field(gt=0)
    configuration_digest: str
    measurement_identity_digest: str
    sft_config: SftConfig
    terminal_checkpoint: str
    terminal_checkpoint_metadata_digest: str
    input_batches_digest: str
    sequence_length: int = Field(ge=16384, le=16384)
    global_batch_executed_tokens: int = Field(gt=0)
    microbatch_size: int = Field(gt=0)
    gradient_accumulation: int = Field(gt=0)
    optimizer: str
    trainable_parameter_bytes: int = Field(gt=0)
    optimizer_state_bytes: int = Field(gt=0)
    checkpoint_seconds: float = Field(ge=0, allow_inf_nan=False)
    compile_seconds: float = Field(ge=0, allow_inf_nan=False)
    warmup_updates: int = Field(ge=0)
    updates: tuple[SmokeUpdate, ...] = Field(min_length=1)
    gradient_structure_report: str
    gradient_structure_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compact_optimizer_report: str
    compact_optimizer_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compact_gradient_outputs: bool
    preserves_activation_gradients: bool


def validate_smoke_checkpoint(record: SmokeRecord) -> dict:
    """Check the measured smoke's actual permanent compact state without array reads."""
    metadata = json.loads((StoragePath(record.terminal_checkpoint) / "metadata.json").read_text())
    if metadata.get("is_temporary") is not False or digest(metadata) != record.terminal_checkpoint_metadata_digest:
        raise ValueError("Measured smoke terminal checkpoint changed or is incomplete")
    manifest = metadata["hero_sft"]
    if (
        manifest["base"]["metadata_digest"] != record.checkpoint_metadata_digest
        or TypeAdapter(SftConfig).validate_python(manifest["sft"]) != record.sft_config
        or metadata["step"] != len(record.updates)
        or metadata["hero_sft_counters"]
        != {
            "sft_step": len(record.updates),
            "processed_tokens": sum(update.non_padding_tokens for update in record.updates),
            "supervised_tokens": sum(update.supervised_tokens for update in record.updates),
        }
    ):
        raise ValueError("Measured smoke terminal checkpoint disagrees with its base, schedule, or actual counters")
    gradient_raw = StoragePath(record.gradient_structure_report).read_bytes()
    if hashlib.sha256(gradient_raw).hexdigest() != record.gradient_structure_report_sha256:
        raise ValueError("Gradient structure report bytes changed")
    gradient = json.loads(gradient_raw)
    if (
        gradient.get("schema_version") != 1
        or gradient.get("kind") != "hero_sft_gradient_structure"
        or gradient.get("execution_id") != record.execution_id
        or gradient.get("checkpoint_metadata_digest") != record.checkpoint_metadata_digest
        or gradient.get("measurement_identity_digest") != record.measurement_identity_digest
    ):
        raise ValueError("Gradient structure report does not match the measured smoke")
    outputs = {row["path"]: row for row in gradient["gradients"]}
    if len(outputs) != len(gradient["gradients"]) or not outputs:
        raise ValueError("Gradient structure contains duplicate paths or no outputs")
    parameter_bytes = 0
    for path, row in outputs.items():
        shape = row["shape"]
        if row["dtype"] != "float32" or not shape or any(type(dim) is not int or dim <= 0 for dim in shape):
            raise ValueError("Compact gradient outputs must have positive shapes and FP32 dtype")
        parameter_bytes += math.prod(shape) * np.dtype(np.float32).itemsize
        if path in (".input_delta", ".output_delta"):
            continue
        if not re.fullmatch(
            r"\.adapters\.stacked_blocks\.stacked\.(?:attn\.lora_[qkvo]|shared\[\d+\]\.lora_(?:gate|up|down))\.[ab]",
            path,
        ):
            raise ValueError(f"Noncompact gradient output: {path}")
        other = outputs.get(path[:-1] + ("b" if path.endswith("a") else "a"))
        if other is None or len(shape) != 3 or len(other["shape"]) != 3:
            raise ValueError("LoRA gradient outputs must contain paired stacked A/B matrices")
        a, b = (shape, other["shape"]) if path.endswith("a") else (other["shape"], shape)
        if a[0] != b[0] or a[-1] != record.sft_config.rank or b[-2] != record.sft_config.rank:
            raise ValueError("LoRA gradient output shapes differ from the configured rank")
    input_shape = outputs.get(".input_delta", {}).get("shape", [])
    output_shape = outputs.get(".output_delta", {}).get("shape", [])
    if (
        len(input_shape) != 2
        or output_shape != list(reversed(input_shape))
        or input_shape[0] != len(record.sft_config.token_ids)
        or not any(path.startswith(".adapters.") for path in outputs)
        or parameter_bytes != record.trainable_parameter_bytes
        or not record.compact_gradient_outputs
    ):
        raise ValueError("Gradient outputs disagree with the compact parameter bytes or selected vocabulary")
    optimizer_raw = StoragePath(record.compact_optimizer_report).read_bytes()
    if hashlib.sha256(optimizer_raw).hexdigest() != record.compact_optimizer_report_sha256:
        raise ValueError("Smoke compact optimizer report bytes changed")
    optimizer = json.loads(optimizer_raw)
    if (
        optimizer.get("schema_version") != 1
        or optimizer.get("compact_optimizer_only") is not True
        or optimizer.get("errors") != []
        or optimizer.get("optimizer_step") != 0
        or optimizer.get("execution_id") != record.execution_id
        or optimizer.get("checkpoint_metadata_digest") != record.checkpoint_metadata_digest
        or optimizer.get("measurement_identity_digest") != record.measurement_identity_digest
        or optimizer.get("trainable_parameter_bytes") != record.trainable_parameter_bytes
        or optimizer.get("optimizer_state_bytes") != record.optimizer_state_bytes
        or optimizer.get("expected_optimizer_state_bytes") != record.optimizer_state_bytes
    ):
        raise ValueError("Smoke lacks passing structural evidence of fresh compact Adam state")
    return metadata


def assemble_smoke(
    telemetry_directory: Path,
    *,
    warmup_updates: int,
    microbatch_size: int,
    gradient_accumulation: int,
    optimizer: str,
) -> SmokeRecord:
    """Build a budget-eligible record from actual persisted SFT runtime telemetry.

    Missing HBM, checkpoint timing, compact gradient outputs or measured input-delta
    gradients blocks assembly. Caller-supplied booleans cannot manufacture those
    measurements. The timing/gradient files remain independently inspectable.
    """
    paths = sorted(telemetry_directory.glob("update-*.json"))
    if not paths:
        raise ValueError("No measured SFT updates exist in this telemetry directory")
    raw = [json.loads(path.read_text()) for path in paths]
    first = raw[0]
    identity_fields = (
        "execution_id",
        "checkpoint_metadata_digest",
        "configuration_digest",
        "measurement_identity_digest",
        "sft_config",
        "input_batches_digest",
        "hardware",
        "gpu_count",
        "sequence_length",
        "trainable_parameter_bytes",
        "optimizer_state_bytes",
    )
    for row in raw:
        if row.get("kind") != "hero_sft_raw_smoke_update" or row.get("schema_version") != 1:
            raise ValueError("Unknown raw smoke telemetry schema")
        if any(row.get(field) != first.get(field) for field in identity_fields):
            raise ValueError("Smoke telemetry contains mixed execution, model, data or optimizer identities")
        if row.get("missing_smoke_measurements"):
            raise ValueError(f"Runtime telemetry reports missing measurements: {row['missing_smoke_measurements']}")
    panel_digests = {row["acquisition_diagnostic"]["panel_digest"] for row in raw}
    if len(panel_digests) != 1:
        raise ValueError("Smoke changed the fixed acquisition diagnostic panel")
    gradient_path = telemetry_directory / "gradient-structure-report.json"
    gradient_raw = gradient_path.read_bytes()
    optimizer_path = telemetry_directory / "compact-optimizer-report.json"
    optimizer_sha256 = hashlib.sha256(optimizer_path.read_bytes()).hexdigest()
    gradient_norms = [row["input_delta_gradient_norm"] for row in raw]
    if any(not math.isfinite(value) for value in gradient_norms) or not any(value > 0 for value in gradient_norms):
        raise ValueError("No measured input-delta gradient proves activation backpropagation through the frozen base")
    checkpoints = [json.loads(path.read_text()) for path in sorted(telemetry_directory.glob("checkpoint-*.json"))]
    if not checkpoints:
        raise ValueError("No measured compact checkpoint write timing exists")
    if any(row["execution_id"] != first["execution_id"] for row in checkpoints):
        raise ValueError("Smoke checkpoint timings belong to another execution")
    terminal = max(checkpoints, key=lambda row: row["update"])
    updates = tuple(SmokeUpdate.model_validate({key: row[key] for key in SmokeUpdate.model_fields}) for row in raw)
    record = SmokeRecord(
        **{key: first[key] for key in identity_fields},
        terminal_checkpoint=terminal["checkpoint"],
        terminal_checkpoint_metadata_digest=terminal["metadata_digest"],
        global_batch_executed_tokens=first["executed_tokens"],
        microbatch_size=microbatch_size,
        gradient_accumulation=gradient_accumulation,
        optimizer=optimizer,
        checkpoint_seconds=sum(row["checkpoint_seconds"] for row in checkpoints),
        compile_seconds=first["compile_seconds"],
        warmup_updates=warmup_updates,
        updates=updates,
        gradient_structure_report=str(gradient_path),
        gradient_structure_report_sha256=hashlib.sha256(gradient_raw).hexdigest(),
        compact_optimizer_report=str(optimizer_path),
        compact_optimizer_report_sha256=optimizer_sha256,
        compact_gradient_outputs=True,
        preserves_activation_gradients=True,
    )
    summarize_smoke(record)
    validate_smoke_checkpoint(record)
    return record


def summarize_smoke(record: SmokeRecord) -> dict:
    """Separate measured steady-state rates from compilation and checkpoint cost."""
    if record.warmup_updates != record.sft_config.warmup_steps:
        raise ValueError("Measured steady-state warmup differs from the actual smoke optimizer schedule")
    if [update.update for update in record.updates] != list(range(1, len(record.updates) + 1)):
        raise ValueError("Smoke must contain every completed update from a fresh optimizer")
    if any(update.executed_tokens != record.global_batch_executed_tokens for update in record.updates):
        raise ValueError("Smoke batch counters do not match its declared global batch")
    steady = record.updates[record.warmup_updates :]
    if not steady:
        raise ValueError("Smoke has no measured steady-state updates after warmup")
    return {
        "execution_id": record.execution_id,
        "record_digest": digest(record.model_dump()),
        "completed_updates": len(record.updates),
        "steady_state_updates": len(steady),
        "median_update_seconds": statistics.median(update.elapsed_seconds for update in steady),
        "median_non_padding_tokens_per_update": statistics.median(update.non_padding_tokens for update in steady),
        "median_non_padding_tokens_per_second": statistics.median(
            update.non_padding_tokens / update.elapsed_seconds for update in steady
        ),
        "median_executed_tokens_per_second": statistics.median(
            update.executed_tokens / update.elapsed_seconds for update in steady
        ),
        "supervised_fraction": (
            sum(update.supervised_tokens for update in steady) / sum(update.non_padding_tokens for update in steady)
        ),
        "padding_fraction": (
            1 - sum(update.non_padding_tokens for update in steady) / sum(update.executed_tokens for update in steady)
        ),
        "median_optimizer_seconds": statistics.median(update.optimizer_seconds for update in steady),
        "peak_hbm_bytes": max(update.peak_hbm_bytes for update in record.updates),
        "trainable_parameter_bytes": record.trainable_parameter_bytes,
        "optimizer_state_bytes": record.optimizer_state_bytes,
        "checkpoint_seconds": record.checkpoint_seconds,
        "compile_seconds": record.compile_seconds,
        "non_padding_tokens": sum(update.non_padding_tokens for update in record.updates),
        "supervised_tokens": sum(update.supervised_tokens for update in record.updates),
        "executed_tokens": sum(update.executed_tokens for update in record.updates),
        "diagnostic_successes": [update.diagnostic_successes for update in record.updates],
        "losses": [update.loss for update in record.updates],
        "gradient_norms": [update.gradient_norm for update in record.updates],
    }


class PilotBudget(Record):
    format_panel_id: str = Field(min_length=1)
    inspection_non_padding_tokens: int = Field(gt=0)
    cap_non_padding_tokens: int = Field(gt=0)
    minimum_learning_updates: int = Field(gt=0)
    early_inspection_update: int = Field(gt=0)
    warmup_updates: int = Field(ge=0)
    cooldown_updates: int = Field(gt=0)
    adapter_peak_lr: float = Field(gt=0, allow_inf_nan=False)
    vocab_peak_lr: float = Field(gt=0, allow_inf_nan=False)
    smoke_learning_review: str = Field(min_length=1)
    reuse_smoke_optimizer: bool


def plan_pilot(record: SmokeRecord, budget: PilotBudget) -> dict:
    """Require real smoke counters and enough updates before authorizing a budget.

    Estimates use measured non-padding counts and conservatively ceil the final
    complete update. The runtime must stop on actual accumulated token counters;
    padding and masked prompt tokens are not interchangeable with supervision.
    """
    measured = summarize_smoke(record)
    if not record.compact_gradient_outputs or not record.preserves_activation_gradients:
        raise ValueError("Smoke lacks compact gradient outputs or measured activation backpropagation")
    if any(update.gradient_norm == 0 for update in record.updates):
        raise ValueError("Smoke contains zero trainable gradients; diagnose before budgeting")
    if budget.cap_non_padding_tokens < budget.inspection_non_padding_tokens:
        raise ValueError("Pilot cap precedes its token inspection")
    consumed_tokens = measured["non_padding_tokens"] if budget.reuse_smoke_optimizer else 0
    consumed_updates = measured["completed_updates"] if budget.reuse_smoke_optimizer else 0
    per_update = measured["median_non_padding_tokens_per_update"]
    inspection_updates = consumed_updates + max(
        0, math.ceil((budget.inspection_non_padding_tokens - consumed_tokens) / per_update)
    )
    cap_updates = consumed_updates + math.ceil((budget.cap_non_padding_tokens - consumed_tokens) / per_update)
    if inspection_updates < budget.minimum_learning_updates or cap_updates < budget.minimum_learning_updates:
        raise ValueError("Measured batch gives too few learning updates; reduce batch or revise the budget")
    if budget.early_inspection_update > inspection_updates or budget.early_inspection_update < 1:
        raise ValueError("Early update inspection must precede the token inspection")
    if budget.warmup_updates + budget.cooldown_updates != cap_updates:
        raise ValueError("Warmup and linear cooldown must cover the exact finite pilot update budget")
    if consumed_tokens >= budget.cap_non_padding_tokens:
        raise ValueError("Reused smoke already exhausted the pilot cap")
    return {
        "smoke": measured,
        "budget": budget.model_dump(),
        "expected_inspection_updates": inspection_updates,
        "expected_cap_updates": cap_updates,
        "already_consumed_non_padding_tokens": consumed_tokens,
        "already_completed_updates": consumed_updates,
        "expected_remaining_seconds": (
            (budget.cap_non_padding_tokens - consumed_tokens) / measured["median_non_padding_tokens_per_second"]
        ),
        "expected_first_inspection_update": min(budget.early_inspection_update, inspection_updates),
        "stop_rule": "Inspect by actual accumulated non-padding tokens or earlier update; stop when gates pass",
        "duration_excludes": ["compilation", "restore", "evaluation", "checkpoint IO"],
    }


def matched_throughput(lora: SmokeRecord, full_sft: SmokeRecord) -> dict:
    """Claim speedup only for matched weights, hardware, batches and accumulation."""
    for key in (
        "checkpoint_metadata_digest",
        "hardware",
        "gpu_count",
        "input_batches_digest",
        "sequence_length",
        "global_batch_executed_tokens",
        "microbatch_size",
        "gradient_accumulation",
    ):
        if getattr(lora, key) != getattr(full_sft, key):
            raise ValueError(f"Throughput control differs in {key}; no measured speedup claim is valid")
    lora_summary, full_summary = summarize_smoke(lora), summarize_smoke(full_sft)
    return {
        "non_padding_speedup": (
            lora_summary["median_non_padding_tokens_per_second"] / full_summary["median_non_padding_tokens_per_second"]
        ),
        "lora_optimizer": lora.optimizer,
        "full_sft_optimizer": full_sft.optimizer,
        "lora": lora_summary,
        "full_sft": full_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    grade = commands.add_parser("grade")
    grade.add_argument("panel", type=Path)
    grade.add_argument("result", type=Path)
    grade.add_argument("--serialization", type=Serialization, choices=list(Serialization), default=Serialization.MARIN)
    content = commands.add_parser("content")
    content.add_argument("panel", type=Path)
    content.add_argument("result", type=Path)
    content.add_argument("--tokenizer-path")
    content.add_argument("--serialization", type=Serialization, choices=list(Serialization), default=Serialization.MARIN)
    compare = commands.add_parser("compare")
    for name in ("baseline", "parent", "child", "policy"):
        compare.add_argument(name, type=Path)
    compare.add_argument("--require-format", action="store_true")
    assemble = commands.add_parser("assemble-smoke")
    assemble.add_argument("telemetry_directory", type=Path)
    assemble.add_argument("--warmup-updates", type=int, required=True)
    assemble.add_argument("--microbatch-size", type=int, required=True)
    assemble.add_argument("--gradient-accumulation", type=int, required=True)
    assemble.add_argument("--optimizer", required=True)
    smoke = commands.add_parser("smoke")
    smoke.add_argument("record", type=Path)
    pilot = commands.add_parser("pilot")
    pilot.add_argument("record", type=Path)
    pilot.add_argument("budget", type=Path)
    matched = commands.add_parser("matched-throughput")
    matched.add_argument("lora", type=Path)
    matched.add_argument("full_sft", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "grade":
        report = grade_panel(
            Panel.model_validate_json(args.panel.read_text()),
            SampleResult.model_validate_json(args.result.read_text()),
            args.serialization,
        )
    elif args.command == "content":
        panel = Panel.model_validate_json(args.panel.read_text())
        tokenizer = load_tokenizer(args.tokenizer_path or f"{panel.tokenizer}@{panel.tokenizer_revision}")
        report = score_content_panel(
            panel, SampleResult.model_validate_json(args.result.read_text()), tokenizer, args.serialization
        )
    elif args.command == "compare":
        report = compare_candidates(
            CandidateEvaluation.model_validate_json(args.baseline.read_text()),
            CandidateEvaluation.model_validate_json(args.parent.read_text()),
            CandidateEvaluation.model_validate_json(args.child.read_text()),
            GatePolicy.model_validate_json(args.policy.read_text()),
            require_format=args.require_format,
        )
    elif args.command == "assemble-smoke":
        report = assemble_smoke(
            args.telemetry_directory,
            warmup_updates=args.warmup_updates,
            microbatch_size=args.microbatch_size,
            gradient_accumulation=args.gradient_accumulation,
            optimizer=args.optimizer,
        ).model_dump()
    elif args.command == "smoke":
        report = summarize_smoke(SmokeRecord.model_validate_json(args.record.read_text()))
    elif args.command == "pilot":
        report = plan_pilot(
            SmokeRecord.model_validate_json(args.record.read_text()),
            PilotBudget.model_validate_json(args.budget.read_text()),
        )
    else:
        report = matched_throughput(
            SmokeRecord.model_validate_json(args.lora.read_text()),
            SmokeRecord.model_validate_json(args.full_sft.read_text()),
        )
    if args.output.exists():
        raise FileExistsError("Choose a new report path; evaluation evidence is immutable")
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

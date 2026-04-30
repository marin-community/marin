# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Alignment evaluation: run inference on a target model and judge compliance.

This module closes the alignment loop. After DPO training, it answers:
"did the model actually internalize the behavioral spec?"

Two-step pipeline:
  1. run_eval_inference — load eval prompts (no spec guidance), generate
     responses via vLLM, write sharded JSONL results.
  2. run_eval_judge — score each response against the spec statement using
     an LM judge, write per-statement compliance summary.

Supports two prompt input formats:
  - MARIN: flat sharded JSONL.GZ from generate_prompts_from_spec()
  - BLOOM: <statement>/eval_prompts.json tree (existing Bloom eval dataset)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

from levanter.data.utils import batched
from rigging.filesystem import marin_prefix, url_to_fs

from marin.alignment.batched_vllm_serve import BatchedVllmServeSession, write_vllm_metrics_artifact
from marin.alignment.generate_prompts import load_sharded_jsonl_gz, load_spec, write_sharded_jsonl_gz
from marin.alignment.inference_config import InferenceConfig, OpenAIConfig, VLLMConfig
from marin.alignment.judge import (
    JudgeRequest,
    judge_responses_api_batch,
    judge_responses_local_batch,
    write_json,
)
from marin.alignment.live_progress import LiveProgressReporter, vllm_stage_metrics_provider
from marin.alignment.types import ComplianceResult, Statement

logger = logging.getLogger(__name__)


class PromptFormat(StrEnum):
    """Input format for eval prompts."""

    MARIN = "marin"
    BLOOM = "bloom"


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------


def _load_bloom_prompts(path: str, statement_ids: list[str] | None = None) -> list[dict[str, Any]]:
    """Load prompts from Bloom-format <statement>/eval_prompts.json tree."""
    fs, base_path = url_to_fs(path)
    records: list[dict[str, Any]] = []
    entries = sorted(fs.ls(base_path, detail=False))
    for entry in entries:
        # Each entry is a statement directory containing eval_prompts.json
        prompts_file = f"{entry}/eval_prompts.json"
        if not fs.exists(prompts_file):
            continue
        with fs.open(prompts_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        behavior_id = data.get("behavior_name", entry.rstrip("/").rsplit("/", 1)[-1])
        if statement_ids is not None and behavior_id not in statement_ids:
            continue
        for prompt in data.get("prompts", []):
            prompt["behavior_id"] = behavior_id
            records.append(prompt)
    return records


def load_eval_prompts(
    path: str,
    fmt: PromptFormat,
    statement_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Load eval prompts in either Marin or Bloom format.

    Returns a flat list of prompt dicts, each containing at minimum:
    behavior_id, system_prompt, user_message, rubric, config_id.
    """
    if fmt == PromptFormat.BLOOM:
        records = _load_bloom_prompts(path, statement_ids)
    else:
        records = load_sharded_jsonl_gz(path)
        if statement_ids is not None:
            records = [r for r in records if r.get("behavior_id") in statement_ids]
    logger.info("Loaded %d eval prompts (%s format)", len(records), fmt.value)
    return records


# ---------------------------------------------------------------------------
# Inference config resolution (same pattern as generate_responses.py)
# ---------------------------------------------------------------------------


def _resolve_inference_config(model_config: dict[str, Any] | InferenceConfig) -> InferenceConfig:
    if isinstance(model_config, InferenceConfig):
        return model_config
    cfg = dict(model_config)
    backend = cfg.pop("backend")
    if backend == "openai":
        return OpenAIConfig(**cfg)
    if backend == "vllm":
        return VLLMConfig(**cfg)
    raise ValueError(f"Unknown inference backend: {backend}")


def _materialize_mirror_path(path: str, label: str) -> str:
    """Copy a mirror:// path into the local Marin prefix and return its concrete path."""
    if not path.startswith("mirror://"):
        return path

    fs, fs_path = url_to_fs(path)
    mirror_path = fs_path.strip("/")
    if "://" in mirror_path:
        raise ValueError(f"{label} must be a mirror-relative path, got {path!r}")

    entries = fs.find(mirror_path, detail=True)
    if not entries:
        fs.info(mirror_path)
        logger.info("Materialized mirrored %s %s into %s", label, path, marin_prefix())
        return f"{marin_prefix().rstrip('/')}/{mirror_path}"

    copied = 0
    for entry_path, entry in sorted(entries.items()):
        if entry.get("type") == "directory":
            continue
        fs.info(entry_path)
        copied += 1

    local_path = f"{marin_prefix().rstrip('/')}/{mirror_path}"
    logger.info("Materialized mirrored %s %s into %s (%d files)", label, path, local_path, copied)
    return local_path


def _materialize_vllm_mirror_paths(config: VLLMConfig) -> VLLMConfig:
    model = _materialize_mirror_path(config.model, "model")
    tokenizer = _materialize_mirror_path(config.tokenizer, "tokenizer") if config.tokenizer is not None else None
    if model == config.model and tokenizer == config.tokenizer:
        return config
    return replace(config, model=model, tokenizer=tokenizer)


# ---------------------------------------------------------------------------
# Eval inference
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalInferenceConfig:
    """Configuration for running inference on a target model against eval prompts."""

    prompts_path: str
    spec_path: str
    output_path: str
    model_config: dict[str, Any] | InferenceConfig
    prompt_format: PromptFormat = PromptFormat.MARIN
    temperature: float = 0.7
    max_tokens: int = 1500
    n: int = 1
    batch_size: int = 8
    statement_ids: list[str] | None = None
    dependency_path: str | None = None
    """Executor-only dependency used to order steps. Ignored at runtime."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt_format", PromptFormat(self.prompt_format))

    def resolve_inference_config(self) -> InferenceConfig:
        return _resolve_inference_config(self.model_config)


def _build_eval_messages(prompt: dict[str, Any]) -> list[dict[str, str]]:
    """Build inference messages without spec guidance — tests internalized behavior."""
    messages: list[dict[str, str]] = []
    system_prompt = prompt.get("system_prompt", "").strip()
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt.get("user_message", "")})
    return messages


def _make_eval_result_record(
    prompt: dict[str, Any],
    model_name: str,
    response_text: str,
    sample_idx: int = 0,
) -> dict[str, Any]:
    return {
        "prompt_id": f"{prompt.get('behavior_id', '')}/{prompt.get('config_id', '')}",
        "behavior_id": prompt.get("behavior_id", ""),
        "system_prompt": prompt.get("system_prompt", ""),
        "user_message": prompt.get("user_message", ""),
        "rubric": prompt.get("rubric", ""),
        "config_id": prompt.get("config_id", ""),
        "model": model_name,
        "sample_idx": sample_idx,
        "response_text": response_text,
    }


def run_eval_inference(config: EvalInferenceConfig) -> None:
    """Run inference on a target model against eval prompts and save results."""
    prompts = load_eval_prompts(config.prompts_path, config.prompt_format, config.statement_ids)
    if not prompts:
        raise ValueError(f"No eval prompts found at {config.prompts_path} ({config.prompt_format.value} format)")

    inference_config = config.resolve_inference_config()
    if not isinstance(inference_config, VLLMConfig):
        raise ValueError("Eval inference currently supports only VLLMConfig (local vLLM).")
    inference_config = _materialize_vllm_mirror_paths(inference_config)

    logger.info(
        "Running eval inference: %d prompts, n=%d, model=%s, temp=%.2f, max_tokens=%d",
        len(prompts),
        config.n,
        inference_config.model,
        config.temperature,
        config.max_tokens,
    )

    with BatchedVllmServeSession(inference_config) as session:
        reporter = LiveProgressReporter(
            stage_name="Eval Inference",
            total_items=len(prompts),
            batch_size=config.batch_size,
            metrics_provider=vllm_stage_metrics_provider(session, stage_name="eval_inference"),
        )
        results: list[dict[str, Any]] = []
        for prompt_batch in batched(prompts, config.batch_size):
            message_batches = [_build_eval_messages(prompt) for prompt in prompt_batch]
            outputs = session.generate_from_messages(
                message_batches,
                stage_name="eval_inference",
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                n=config.n,
            )
            for prompt, output in zip(prompt_batch, outputs, strict=True):
                for sample_idx, response_text in enumerate(output):
                    results.append(_make_eval_result_record(prompt, inference_config.model, response_text, sample_idx))
            reporter.maybe_log(len(results))
    metrics_snapshot = session.metrics_snapshot()

    write_sharded_jsonl_gz(results, config.output_path, shard_size=5000)
    write_vllm_metrics_artifact(
        f"{config.output_path}/artifacts/vllm_metrics.json",
        logical_stage="eval_inference",
        sessions=[("eval_inference", metrics_snapshot)],
    )
    logger.info("Wrote %d eval inference results to %s", len(results), config.output_path)


# ---------------------------------------------------------------------------
# Eval judge
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalJudgeConfig:
    """Configuration for judging eval inference outputs."""

    eval_responses_path: str
    spec_path: str
    output_path: str
    judge_model: InferenceConfig | str = "gpt-4.1"
    workers: int = 64
    batch_size: int = 8
    judge_max_tokens: int = 1000
    max_failure_rate: float = 0.2


def _build_eval_judge_requests(
    eval_results: list[dict[str, Any]],
    statements: dict[str, Statement],
) -> tuple[list[JudgeRequest], list[dict[str, Any]]]:
    """Build JudgeRequest list from eval result records.

    Returns (requests, skipped) where skipped records have no matching statement.
    """
    requests: list[JudgeRequest] = []
    skipped: list[dict[str, Any]] = []
    for result in eval_results:
        behavior_id = result.get("behavior_id", "")
        statement = statements.get(behavior_id)
        if statement is None:
            skipped.append(result)
            continue
        response_text = result.get("response_text", "")
        if not response_text:
            skipped.append(result)
            continue
        requests.append(
            JudgeRequest(
                prompt_id=result.get("prompt_id", ""),
                bucket="eval",
                response_index=0,
                statement=statement,
                user_message=result.get("user_message", ""),
                rubric=result.get("rubric") or None,
                response_text=response_text,
            )
        )
    return requests, skipped


def _compute_compliance_summary(
    results: list[dict[str, Any]],
    model_name: str,
) -> dict[str, Any]:
    """Compute per-statement and overall compliance statistics.

    Judgments with score=None (parse failures) are excluded from all
    aggregates — they represent "unknown", not "zero".
    """
    per_statement: dict[str, dict[str, Any]] = {}
    all_scores: list[float] = []
    all_compliant: list[bool] = []

    for result in results:
        judgment = result.get("judgment")
        if judgment is None:
            continue
        score = judgment.get("score")
        compliant = judgment.get("compliant")
        if score is None or compliant is None:
            continue
        behavior_id = result.get("behavior_id", "unknown")
        all_scores.append(score)
        all_compliant.append(compliant)

        if behavior_id not in per_statement:
            per_statement[behavior_id] = {"scores": [], "compliant": []}
        per_statement[behavior_id]["scores"].append(score)
        per_statement[behavior_id]["compliant"].append(compliant)

    per_statement_summary = {}
    for stmt_id, data in sorted(per_statement.items()):
        scores = data["scores"]
        compliant_list = data["compliant"]
        per_statement_summary[stmt_id] = {
            "mean_score": sum(scores) / len(scores) if scores else 0.0,
            "compliance_rate": sum(compliant_list) / len(compliant_list) if compliant_list else 0.0,
            "count": len(scores),
        }

    return {
        "model": model_name,
        "total_evaluated": len(all_scores),
        "overall_mean_score": sum(all_scores) / len(all_scores) if all_scores else 0.0,
        "overall_compliance_rate": sum(all_compliant) / len(all_compliant) if all_compliant else 0.0,
        "per_statement": per_statement_summary,
    }


def _judge_one_artifact(
    *,
    eval_responses_path: str,
    output_path: str,
    statements: dict[str, Statement],
    judge_model: InferenceConfig | str,
    batch_size: int,
    judge_max_tokens: int,
    workers: int,
    max_failure_rate: float,
    session: BatchedVllmServeSession | None,
) -> None:
    """Judge one eval-responses artifact. Caller owns session lifecycle.

    When ``session`` is provided, judging is routed through the shared
    ``BatchedVllmServeSession`` (local vLLM path). When ``None``, the
    OpenAI-compatible API path is used via ``judge_responses_api_batch``.
    Writes sharded JSONL, ``summary.json``, and (for the local path)
    ``artifacts/vllm_metrics.json`` to ``output_path``.
    """
    eval_results = load_sharded_jsonl_gz(eval_responses_path)
    logger.info("Loaded %d eval results for judging", len(eval_results))

    requests, skipped = _build_eval_judge_requests(eval_results, statements)
    if skipped:
        logger.warning("Skipped %d results (missing statement or empty response)", len(skipped))
    if not requests:
        raise ValueError("No valid eval results to judge.")

    if session is not None:
        reporter = LiveProgressReporter(
            stage_name="Eval Judge",
            total_items=len(requests),
            batch_size=batch_size,
            metrics_provider=vllm_stage_metrics_provider(session, stage_name="eval_judge"),
        )
        all_compliance_results: list[ComplianceResult] = []
        for batch_start in range(0, len(requests), batch_size):
            batch = requests[batch_start : batch_start + batch_size]
            batch_results = judge_responses_local_batch(
                batch,
                session=session,
                max_tokens=judge_max_tokens,
            )
            all_compliance_results.extend(batch_results)
            reporter.maybe_log(len(all_compliance_results))
        session_metrics = session.metrics_snapshot()
        write_vllm_metrics_artifact(
            f"{output_path}/artifacts/vllm_metrics.json",
            logical_stage="eval_judge",
            sessions=[("eval_judge", session_metrics)],
        )
    else:
        reporter = LiveProgressReporter(
            stage_name="Eval Judge",
            total_items=len(requests),
            batch_size=batch_size,
        )
        all_compliance_results = []
        for batch_start in range(0, len(requests), batch_size):
            batch = requests[batch_start : batch_start + batch_size]
            batch_results = judge_responses_api_batch(
                batch,
                judge_model=judge_model,
                max_tokens=judge_max_tokens,
                workers=workers,
            )
            all_compliance_results.extend(batch_results)
            reporter.maybe_log(len(all_compliance_results))

    # Build judged result records
    judged_results: list[dict[str, Any]] = []
    failure_count = 0
    for request, compliance_result in zip(requests, all_compliance_results, strict=True):
        if compliance_result.score is None:
            failure_count += 1
        judged_results.append(
            {
                "prompt_id": request.prompt_id,
                "behavior_id": request.statement.id,
                "user_message": request.user_message,
                "response_text": request.response_text,
                "rubric": request.rubric,
                "judgment": {
                    "score": compliance_result.score,
                    "compliant": compliance_result.compliant,
                    "confidence": compliance_result.confidence,
                    "explanation": compliance_result.explanation,
                    "highlights": list(compliance_result.highlights),
                },
            }
        )

    # Determine model name for summary
    model_name = "unknown"
    for result in eval_results:
        if result.get("model"):
            model_name = result["model"]
            break

    write_sharded_jsonl_gz(judged_results, output_path, shard_size=5000)
    summary = _compute_compliance_summary(judged_results, model_name)
    summary["failure_count"] = failure_count
    summary["skipped_count"] = len(skipped)
    write_json(f"{output_path}/summary.json", summary)

    logger.info(
        "Eval complete: %d judged, overall_mean=%.2f, compliance_rate=%.1f%%",
        summary["total_evaluated"],
        summary["overall_mean_score"],
        summary["overall_compliance_rate"] * 100,
    )

    if requests and failure_count / len(requests) > max_failure_rate:
        raise RuntimeError(
            f"Judge failure rate {failure_count}/{len(requests)} exceeds threshold {max_failure_rate:.0%}"
        )


def run_eval_judge(config: EvalJudgeConfig) -> None:
    """Judge eval inference outputs and produce compliance summary."""
    statements = load_spec(config.spec_path)
    if isinstance(config.judge_model, VLLMConfig):
        with BatchedVllmServeSession(config.judge_model) as session:
            _judge_one_artifact(
                eval_responses_path=config.eval_responses_path,
                output_path=config.output_path,
                statements=statements,
                judge_model=config.judge_model,
                batch_size=config.batch_size,
                judge_max_tokens=config.judge_max_tokens,
                workers=config.workers,
                max_failure_rate=config.max_failure_rate,
                session=session,
            )
    else:
        _judge_one_artifact(
            eval_responses_path=config.eval_responses_path,
            output_path=config.output_path,
            statements=statements,
            judge_model=config.judge_model,
            batch_size=config.batch_size,
            judge_max_tokens=config.judge_max_tokens,
            workers=config.workers,
            max_failure_rate=config.max_failure_rate,
            session=None,
        )


# ---------------------------------------------------------------------------
# Batched eval judge — reuses one vLLM session across many artifacts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalJudgeTarget:
    """One inference artifact to judge inside a batched run."""

    label: str
    eval_responses_path: str
    output_path: str


@dataclass(frozen=True)
class BatchEvalJudgeConfig:
    """Judge multiple eval artifacts with one shared vLLM session.

    Resumable: any target whose ``{output_path}/summary.json`` already exists
    is skipped on restart, so preemption only costs rejudging the in-flight
    target plus any unstarted ones.
    """

    targets: list[EvalJudgeTarget]
    spec_path: str
    judge_model: InferenceConfig | str = "gpt-4.1"
    workers: int = 64
    batch_size: int = 8
    judge_max_tokens: int = 1000
    max_failure_rate: float = 0.2


def _target_already_done(output_path: str) -> bool:
    """Return True if this target has a completed summary.json."""
    summary_path = f"{output_path}/summary.json"
    fs, fs_path = url_to_fs(summary_path)
    return fs.exists(fs_path)


def run_batch_eval_judge(config: BatchEvalJudgeConfig) -> None:
    """Judge many eval inference artifacts using one shared vLLM session.

    Loads the judge model exactly once. Each target writes its own sharded
    JSONL + ``summary.json`` to its own output directory. On restart after
    preemption, targets whose ``summary.json`` already exists are skipped, so
    a restart only rejudges the in-flight target plus any unstarted ones.
    Fails fast inside a target if its parse failure rate exceeds threshold.
    """
    if not config.targets:
        raise ValueError("run_batch_eval_judge requires at least one target.")

    # Partition into done vs pending before loading the model so a fully
    # completed run is free on re-entry.
    total = len(config.targets)
    pending: list[tuple[int, EvalJudgeTarget]] = []
    for idx, target in enumerate(config.targets):
        if _target_already_done(target.output_path):
            logger.info(
                "[batch-judge %d/%d] SKIP %s (summary.json exists at %s)",
                idx + 1,
                total,
                target.label,
                target.output_path,
            )
        else:
            pending.append((idx, target))

    if not pending:
        logger.info("All %d targets already have summary.json; nothing to do.", total)
        return

    logger.info(
        "Batch judge plan: %d total, %d done, %d pending",
        total,
        total - len(pending),
        len(pending),
    )

    statements = load_spec(config.spec_path)
    is_local = isinstance(config.judge_model, VLLMConfig)

    if is_local:
        with BatchedVllmServeSession(config.judge_model) as session:
            for pending_idx, (orig_idx, target) in enumerate(pending):
                logger.info(
                    "[batch-judge pending %d/%d (total %d/%d)] judging %s -> %s",
                    pending_idx + 1,
                    len(pending),
                    orig_idx + 1,
                    total,
                    target.label,
                    target.output_path,
                )
                _judge_one_artifact(
                    eval_responses_path=target.eval_responses_path,
                    output_path=target.output_path,
                    statements=statements,
                    judge_model=config.judge_model,
                    batch_size=config.batch_size,
                    judge_max_tokens=config.judge_max_tokens,
                    workers=config.workers,
                    max_failure_rate=config.max_failure_rate,
                    session=session,
                )
    else:
        for pending_idx, (orig_idx, target) in enumerate(pending):
            logger.info(
                "[batch-judge pending %d/%d (total %d/%d)] judging %s -> %s",
                pending_idx + 1,
                len(pending),
                orig_idx + 1,
                total,
                target.label,
                target.output_path,
            )
            _judge_one_artifact(
                eval_responses_path=target.eval_responses_path,
                output_path=target.output_path,
                statements=statements,
                judge_model=config.judge_model,
                batch_size=config.batch_size,
                judge_max_tokens=config.judge_max_tokens,
                workers=config.workers,
                max_failure_rate=config.max_failure_rate,
                session=None,
            )

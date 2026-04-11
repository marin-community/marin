# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 3: Judge responses, persist all scores, then build preference pairs.

The judge stage now writes a full judgments artifact containing every scored
chosen/rejected candidate, including the parsed score and raw judge output.
Preference-pair filtering happens in a separate step so thresholds can change
without rerunning the expensive judging stage.
"""

from __future__ import annotations

import collections
import concurrent.futures
import contextlib
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from rigging.filesystem import url_to_fs

from marin.alignment.batched_vllm_serve import BatchedVllmServeSession, write_vllm_metrics_artifact
from marin.alignment.generate_prompts import load_sharded_jsonl_gz, load_spec, write_sharded_jsonl_gz
from marin.alignment.inference_config import InferenceConfig, VLLMConfig
from marin.alignment.llm_client import llm_chat_single
from marin.alignment.live_progress import LiveProgressReporter, vllm_stage_metrics_provider
from marin.alignment.prompts.judge import build_compliance_judge_prompt, build_judge_system_prompt
from marin.alignment.types import ComplianceResult, Statement

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class JudgeConfig:
    """Configuration for the judgment stage."""

    chosen_responses_path: str
    rejected_responses_path: str
    spec_path: str
    output_path: str

    judge_model: InferenceConfig | str = "gpt-4.1"
    workers: int = 64
    batch_size: int = 8
    judge_max_tokens: int = 1000
    max_failure_rate: float = 0.2


@dataclass(frozen=True)
class PreferencePairFilterConfig:
    """Configuration for turning scored judgments into DPO preference pairs."""

    judgments_path: str
    output_path: str
    min_chosen_score: float = 7.0
    min_gap: float = 2.0


@dataclass(frozen=True)
class JudgePairConfig:
    """Compatibility wrapper for combined judge + pair building."""

    chosen_responses_path: str
    rejected_responses_path: str
    spec_path: str
    output_path: str

    judge_model: InferenceConfig | str = "gpt-4.1"
    min_chosen_score: float = 7.0
    min_gap: float = 2.0
    workers: int = 64
    batch_size: int = 8
    judge_max_tokens: int = 1000
    max_failure_rate: float = 0.2


@dataclass(frozen=True)
class JudgeRequest:
    prompt_id: str
    bucket: str
    response_index: int
    statement: Statement
    user_message: str
    rubric: str | None
    response_text: str


def _load_responses(path: str) -> dict[str, dict[str, Any]]:
    """Load responses keyed by prompt_id. Supports both local and GCS paths."""
    records = load_sharded_jsonl_gz(path)
    return {record.get("prompt_id", ""): record for record in records}


def parse_judge_response(content: str) -> dict[str, Any]:
    """Parse the JSON judgment from the judge response."""
    json_match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(1))

    json_match = re.search(r"\{[^{}]*\"score\"[^{}]*\}", content, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(0))

    return json.loads(content)


def build_judge_messages(
    statement: Statement,
    user_message: str,
    model_response: str,
    rubric: str | None,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": build_judge_system_prompt()},
        {
            "role": "user",
            "content": build_compliance_judge_prompt(
                statement=statement,
                user_input=user_message,
                model_response=model_response,
                question_rubric=rubric,
            ),
        },
    ]


def parse_compliance_result(content: str) -> ComplianceResult:
    try:
        parsed = parse_judge_response(content)
        return ComplianceResult.from_judge_output(parsed, raw_response=content)
    except (json.JSONDecodeError, KeyError) as exc:
        logger.warning("Failed to parse judge response: %s", exc)
        return ComplianceResult(
            score=None,
            compliant=None,
            confidence=0.0,
            explanation=f"Parse failure: {exc}",
            raw_response=content,
        )


def judge_response(
    statement: Statement,
    user_message: str,
    model_response: str,
    rubric: str | None,
    judge_model: InferenceConfig | str,
    max_tokens: int,
) -> ComplianceResult:
    response = llm_chat_single(
        config=judge_model,
        messages=build_judge_messages(statement, user_message, model_response, rubric),
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return parse_compliance_result(response.content)


def judge_responses_local_batch(
    requests: list[JudgeRequest],
    *,
    session: BatchedVllmServeSession,
    max_tokens: int,
) -> list[ComplianceResult]:
    if not requests:
        return []

    logger.info("Scoring %d responses via local batched judge", len(requests))
    message_batches = [
        build_judge_messages(
            request.statement,
            request.user_message,
            request.response_text,
            request.rubric,
        )
        for request in requests
    ]
    outputs = session.generate_from_messages(
        message_batches,
        stage_name="judge",
        temperature=0.0,
        max_tokens=max_tokens,
        n=1,
    )
    results: list[ComplianceResult] = []
    for output in outputs:
        raw_response = output[0] if output else ""
        results.append(parse_compliance_result(raw_response))
    return results


def judge_responses_api_batch(
    requests: list[JudgeRequest],
    *,
    judge_model: InferenceConfig | str,
    max_tokens: int,
    workers: int,
) -> list[ComplianceResult]:
    if not requests:
        return []

    logger.info("Scoring %d responses via API judge", len(requests))
    results: list[ComplianceResult | None] = [None] * len(requests)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {
            pool.submit(
                judge_response,
                request.statement,
                request.user_message,
                request.response_text,
                request.rubric,
                judge_model,
                max_tokens,
            ): idx
            for idx, request in enumerate(requests)
        }
        for future in concurrent.futures.as_completed(future_map):
            idx = future_map[future]
            results[idx] = future.result()
    return [result for result in results if result is not None]


def _response_candidates(record: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for position, response in enumerate(record.get("responses", [])):
        response_text = response.get("content", "")
        if not response_text:
            continue
        response_index = response.get("index", position)
        if not isinstance(response_index, int):
            response_index = position
        candidates.append({"response_index": response_index, "response_text": response_text})
    return candidates


def _statement_record(statement: Statement) -> dict[str, Any]:
    return {
        "id": statement.id,
        "text": statement.text,
        "type": statement.type.value,
        "authority_level": statement.authority_level.value,
        "section": statement.section,
        "subsection": statement.subsection,
    }


def compliance_result_record(result: ComplianceResult) -> dict[str, Any]:
    return {
        "score": result.score,
        "compliant": result.compliant,
        "confidence": result.confidence,
        "explanation": result.explanation,
        "highlights": list(result.highlights),
        "raw_response": result.raw_response,
    }


def _judged_candidate_record(
    *,
    response_index: int,
    response_text: str,
    result: ComplianceResult,
) -> dict[str, Any]:
    return {
        "response_index": response_index,
        "response_text": response_text,
        "judgment": compliance_result_record(result),
    }


def _empty_judgment_record(
    *,
    prompt_id: str,
    behavior_id: str,
    system_prompt: str,
    user_message: str,
    rubric: str | None,
    statement: Statement | None,
    status: str,
    errors: list[str],
) -> dict[str, Any]:
    return {
        "prompt_id": prompt_id,
        "behavior_id": behavior_id,
        "system_prompt": system_prompt,
        "user_message": user_message,
        "rubric": rubric,
        "statement": _statement_record(statement) if statement is not None else None,
        "status": status,
        "errors": errors,
        "chosen_candidates": [],
        "rejected_candidates": [],
        "best_chosen": None,
        "worst_rejected": None,
        "gap": None,
    }


def _judgment_record(
    *,
    prompt_id: str,
    behavior_id: str,
    system_prompt: str,
    user_message: str,
    rubric: str | None,
    statement: Statement,
    chosen_candidates: list[dict[str, Any]],
    rejected_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    # Candidates with score=None failed to parse; excluding them from
    # best/worst selection prevents max/min from comparing None to ints.
    scored_chosen = [c for c in chosen_candidates if c["judgment"]["score"] is not None]
    scored_rejected = [c for c in rejected_candidates if c["judgment"]["score"] is not None]
    best_chosen = max(scored_chosen, key=lambda item: item["judgment"]["score"]) if scored_chosen else None
    worst_rejected = min(scored_rejected, key=lambda item: item["judgment"]["score"]) if scored_rejected else None
    gap = None
    if best_chosen is not None and worst_rejected is not None:
        gap = best_chosen["judgment"]["score"] - worst_rejected["judgment"]["score"]
    return {
        "prompt_id": prompt_id,
        "behavior_id": behavior_id,
        "system_prompt": system_prompt,
        "user_message": user_message,
        "rubric": rubric,
        "statement": _statement_record(statement),
        "status": "ok",
        "errors": [],
        "chosen_candidates": chosen_candidates,
        "rejected_candidates": rejected_candidates,
        "best_chosen": best_chosen,
        "worst_rejected": worst_rejected,
        "gap": gap,
    }


def write_json(path: str, payload: dict[str, Any]) -> None:
    fs, fs_path = url_to_fs(path)
    parent = fs_path.rsplit("/", 1)[0] if "/" in fs_path else ""
    if parent:
        fs.makedirs(parent, exist_ok=True)
    with fs.open(fs_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _judgments_summary(
    *,
    chosen_count: int,
    rejected_count: int,
    common_count: int,
    judgments: list[dict[str, Any]],
    failure_messages: list[str],
) -> dict[str, Any]:
    status_counts = collections.Counter(record["status"] for record in judgments)
    return {
        "chosen_record_count": chosen_count,
        "rejected_record_count": rejected_count,
        "common_prompt_count": common_count,
        "judgment_record_count": len(judgments),
        "status_counts": dict(sorted(status_counts.items())),
        "failure_count": len(failure_messages),
        "failures": failure_messages,
    }


def judge_responses(config: JudgeConfig) -> None:
    """Score all chosen/rejected candidates and persist the full judgments artifact."""
    statements = load_spec(config.spec_path)
    chosen = _load_responses(config.chosen_responses_path)
    rejected = _load_responses(config.rejected_responses_path)
    logger.info("Loaded %d chosen, %d rejected responses", len(chosen), len(rejected))

    common_ids = sorted(set(chosen.keys()) & set(rejected.keys()))
    logger.info("Processing %d prompt pairs for judgment logging", len(common_ids))

    judgments: list[dict[str, Any]] = []
    failure_messages: list[str] = []
    is_local = isinstance(config.judge_model, VLLMConfig)
    session_manager: contextlib.AbstractContextManager[BatchedVllmServeSession | None]
    if is_local:
        session_manager = BatchedVllmServeSession(config.judge_model)
    else:
        session_manager = contextlib.nullcontext(None)

    session_metrics: dict[str, object] | None = None
    with session_manager as session:
        reporter = LiveProgressReporter(
            stage_name="Judge",
            total_items=len(common_ids),
            batch_size=max(1, config.batch_size),
            metrics_provider=vllm_stage_metrics_provider(session, stage_name="judge") if session is not None else None,
        )
        processed_prompt_pairs = 0
        for start in range(0, len(common_ids), config.batch_size):
            batch_ids = common_ids[start : start + config.batch_size]
            requests: list[JudgeRequest] = []
            prompt_contexts: dict[str, dict[str, Any]] = {}

            for prompt_id in batch_ids:
                chosen_record = chosen[prompt_id]
                rejected_record = rejected[prompt_id]
                behavior_id = chosen_record.get("behavior_id", "")
                system_prompt = chosen_record.get("system_prompt", "")
                user_message = chosen_record.get("user_message", "")
                rubric = chosen_record.get("rubric")
                statement = statements.get(behavior_id)
                if statement is None:
                    logger.warning("No statement found for behavior_id '%s'", behavior_id)
                    judgments.append(
                        _empty_judgment_record(
                            prompt_id=prompt_id,
                            behavior_id=behavior_id,
                            system_prompt=system_prompt,
                            user_message=user_message,
                            rubric=rubric,
                            statement=None,
                            status="missing_statement",
                            errors=[f"No statement found for behavior_id '{behavior_id}'"],
                        )
                    )
                    continue

                chosen_candidates = _response_candidates(chosen_record)
                rejected_candidates = _response_candidates(rejected_record)
                if not chosen_candidates:
                    judgments.append(
                        _empty_judgment_record(
                            prompt_id=prompt_id,
                            behavior_id=behavior_id,
                            system_prompt=system_prompt,
                            user_message=user_message,
                            rubric=rubric,
                            statement=statement,
                            status="missing_chosen_responses",
                            errors=["No chosen responses found"],
                        )
                    )
                    continue
                if not rejected_candidates:
                    judgments.append(
                        _empty_judgment_record(
                            prompt_id=prompt_id,
                            behavior_id=behavior_id,
                            system_prompt=system_prompt,
                            user_message=user_message,
                            rubric=rubric,
                            statement=statement,
                            status="missing_rejected_responses",
                            errors=["No rejected responses found"],
                        )
                    )
                    continue

                prompt_contexts[prompt_id] = {
                    "behavior_id": behavior_id,
                    "system_prompt": system_prompt,
                    "user_message": user_message,
                    "rubric": rubric,
                    "statement": statement,
                }
                for candidate in chosen_candidates:
                    requests.append(
                        JudgeRequest(
                            prompt_id=prompt_id,
                            bucket="chosen",
                            response_index=candidate["response_index"],
                            statement=statement,
                            user_message=user_message,
                            rubric=rubric,
                            response_text=candidate["response_text"],
                        )
                    )
                for candidate in rejected_candidates:
                    requests.append(
                        JudgeRequest(
                            prompt_id=prompt_id,
                            bucket="rejected",
                            response_index=candidate["response_index"],
                            statement=statement,
                            user_message=user_message,
                            rubric=rubric,
                            response_text=candidate["response_text"],
                        )
                    )

            if not requests:
                processed_prompt_pairs += len(batch_ids)
                reporter.maybe_log(processed_prompt_pairs, details=[f"failures={len(failure_messages)}"])
                continue

            try:
                if is_local:
                    assert session is not None
                    results = judge_responses_local_batch(
                        requests,
                        session=session,
                        max_tokens=config.judge_max_tokens,
                    )
                else:
                    results = judge_responses_api_batch(
                        requests,
                        judge_model=config.judge_model,
                        max_tokens=config.judge_max_tokens,
                        workers=config.workers,
                    )
            except Exception as exc:
                logger.error("Judging failed for prompt batch %s: %s", batch_ids, exc)
                for prompt_id, context in prompt_contexts.items():
                    failure_messages.append(f"{prompt_id}: {exc}")
                    judgments.append(
                        _empty_judgment_record(
                            prompt_id=prompt_id,
                            behavior_id=context["behavior_id"],
                            system_prompt=context["system_prompt"],
                            user_message=context["user_message"],
                            rubric=context["rubric"],
                            statement=context["statement"],
                            status="judge_batch_error",
                            errors=[str(exc)],
                        )
                    )
                processed_prompt_pairs += len(batch_ids)
                reporter.maybe_log(processed_prompt_pairs, details=[f"failures={len(failure_messages)}"])
                continue

            grouped_candidates: dict[str, dict[str, list[dict[str, Any]]]] = {}
            for request, result in zip(requests, results, strict=True):
                grouped_candidates.setdefault(request.prompt_id, {}).setdefault(request.bucket, []).append(
                    _judged_candidate_record(
                        response_index=request.response_index,
                        response_text=request.response_text,
                        result=result,
                    )
                )

            for prompt_id, context in prompt_contexts.items():
                chosen_candidates = grouped_candidates.get(prompt_id, {}).get("chosen", [])
                rejected_candidates = grouped_candidates.get(prompt_id, {}).get("rejected", [])
                if not chosen_candidates or not rejected_candidates:
                    missing_bucket = "chosen" if not chosen_candidates else "rejected"
                    failure_messages.append(f"{prompt_id}: missing judged {missing_bucket} candidates")
                    judgments.append(
                        _empty_judgment_record(
                            prompt_id=prompt_id,
                            behavior_id=context["behavior_id"],
                            system_prompt=context["system_prompt"],
                            user_message=context["user_message"],
                            rubric=context["rubric"],
                            statement=context["statement"],
                            status="missing_judged_candidates",
                            errors=[f"Missing judged {missing_bucket} candidates"],
                        )
                    )
                    continue

                judgments.append(
                    _judgment_record(
                        prompt_id=prompt_id,
                        behavior_id=context["behavior_id"],
                        system_prompt=context["system_prompt"],
                        user_message=context["user_message"],
                        rubric=context["rubric"],
                        statement=context["statement"],
                        chosen_candidates=chosen_candidates,
                        rejected_candidates=rejected_candidates,
                    )
                )
            processed_prompt_pairs += len(batch_ids)
            reporter.maybe_log(processed_prompt_pairs, details=[f"failures={len(failure_messages)}"])
        if is_local and session is not None:
            session_metrics = session.metrics_snapshot()

    write_sharded_jsonl_gz(judgments, config.output_path, shard_size=5000)
    summary = _judgments_summary(
        chosen_count=len(chosen),
        rejected_count=len(rejected),
        common_count=len(common_ids),
        judgments=judgments,
        failure_messages=failure_messages,
    )
    write_json(f"{config.output_path}/summary.json", summary)
    if session_metrics is not None:
        write_vllm_metrics_artifact(
            f"{config.output_path}/artifacts/vllm_metrics.json",
            logical_stage="judgments",
            sessions=[("judge", session_metrics)],
        )
    logger.info("Wrote %d judgment records to %s", len(judgments), config.output_path)

    if failure_messages:
        failure_rate = len(failure_messages) / len(common_ids) if common_ids else 0.0
        if failure_rate > config.max_failure_rate:
            raise RuntimeError(
                f"Judge failure rate {failure_rate:.0%} exceeds threshold "
                f"{config.max_failure_rate:.0%}: {len(failure_messages)}/{len(common_ids)} prompts failed"
            )


def build_preference_pairs(config: PreferencePairFilterConfig) -> None:
    """Filter persisted judgments into marin-format preference pairs."""
    judgments = load_sharded_jsonl_gz(config.judgments_path)
    logger.info("Loaded %d judgment records", len(judgments))

    decision_records: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    decision_counts = collections.Counter()

    for judgment in judgments:
        prompt_id = judgment.get("prompt_id", "")
        status = judgment.get("status", "unknown")
        best_chosen = judgment.get("best_chosen")
        worst_rejected = judgment.get("worst_rejected")
        gap = judgment.get("gap")
        chosen_score = best_chosen.get("judgment", {}).get("score") if isinstance(best_chosen, dict) else None
        rejected_score = worst_rejected.get("judgment", {}).get("score") if isinstance(worst_rejected, dict) else None

        if status != "ok":
            reason = f"status:{status}"
            decision_counts[reason] += 1
            decision_records.append(
                {
                    "prompt_id": prompt_id,
                    "behavior_id": judgment.get("behavior_id", ""),
                    "status": status,
                    "best_chosen_score": chosen_score,
                    "worst_rejected_score": rejected_score,
                    "gap": gap,
                    "passed": False,
                    "reason": reason,
                    "errors": judgment.get("errors", []),
                }
            )
            continue

        if (
            best_chosen is None
            or worst_rejected is None
            or chosen_score is None
            or rejected_score is None
            or gap is None
        ):
            reason = "missing_best_or_worst"
            decision_counts[reason] += 1
            decision_records.append(
                {
                    "prompt_id": prompt_id,
                    "behavior_id": judgment.get("behavior_id", ""),
                    "status": status,
                    "best_chosen_score": chosen_score,
                    "worst_rejected_score": rejected_score,
                    "gap": gap,
                    "passed": False,
                    "reason": reason,
                    "errors": judgment.get("errors", []),
                }
            )
            continue

        if chosen_score < config.min_chosen_score:
            reason = "low_chosen_score"
        elif gap < config.min_gap:
            reason = "low_gap"
        else:
            reason = "passed"

        passed = reason == "passed"
        decision_counts[reason] += 1
        decision_records.append(
            {
                "prompt_id": prompt_id,
                "behavior_id": judgment.get("behavior_id", ""),
                "status": status,
                "best_chosen_score": chosen_score,
                "worst_rejected_score": rejected_score,
                "gap": gap,
                "passed": passed,
                "reason": reason,
                "errors": judgment.get("errors", []),
            }
        )
        if not passed:
            continue

        chosen_messages: list[dict[str, str]] = []
        rejected_messages: list[dict[str, str]] = []
        system_prompt = judgment.get("system_prompt", "")
        user_message = judgment.get("user_message", "")
        if system_prompt:
            chosen_messages.append({"role": "system", "content": system_prompt})
            rejected_messages.append({"role": "system", "content": system_prompt})
        chosen_messages.append({"role": "user", "content": user_message})
        chosen_messages.append({"role": "assistant", "content": best_chosen["response_text"]})
        rejected_messages.append({"role": "user", "content": user_message})
        rejected_messages.append({"role": "assistant", "content": worst_rejected["response_text"]})
        pairs.append({"chosen": chosen_messages, "rejected": rejected_messages})

    write_sharded_jsonl_gz(pairs, config.output_path, shard_size=5000)
    write_sharded_jsonl_gz(decision_records, f"{config.output_path}/artifacts/filter_decisions", shard_size=5000)
    write_json(
        f"{config.output_path}/artifacts/filter_summary.json",
        {
            "judgment_record_count": len(judgments),
            "pair_count": len(pairs),
            "decision_counts": dict(sorted(decision_counts.items())),
            "min_chosen_score": config.min_chosen_score,
            "min_gap": config.min_gap,
        },
    )
    logger.info("Built %d preference pairs from %s", len(pairs), config.judgments_path)


def judge_and_build_pairs(config: JudgePairConfig) -> None:
    """Compatibility wrapper that writes judgments first, then filtered pairs."""
    judgments_path = f"{config.output_path}/artifacts/judgments"
    judge_responses(
        JudgeConfig(
            chosen_responses_path=config.chosen_responses_path,
            rejected_responses_path=config.rejected_responses_path,
            spec_path=config.spec_path,
            output_path=judgments_path,
            judge_model=config.judge_model,
            workers=config.workers,
            batch_size=config.batch_size,
            judge_max_tokens=config.judge_max_tokens,
            max_failure_rate=config.max_failure_rate,
        )
    )
    build_preference_pairs(
        PreferencePairFilterConfig(
            judgments_path=judgments_path,
            output_path=config.output_path,
            min_chosen_score=config.min_chosen_score,
            min_gap=config.min_gap,
        )
    )

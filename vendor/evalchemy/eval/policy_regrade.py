# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regrade frozen policy answers in the vendored Evalchemy environment."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from dataclasses import asdict, replace

import aiohttp
from eval.chat_benchmarks.OlympiadBench.eval_instruct import _flatten_reference_answers
from eval.contracts.eval_spec import EvalSpec, Grade, GraderSpec, JudgePlan, SourceRef
from eval.contracts.policy import TrialVerdict
from eval.execution.backend import retryable_transport, transport_retry_delay
from eval.graders.answer_equivalence import JudgeLabel
from eval.graders.judged import equivalence_judge, equivalence_label
from eval.graders.math_specs import math_equivalence
from eval.runtime.pool import RetryPolicy, retry
from finestore.eval import EvaluationStore, Grading, sample_from_archive_row
from finestore.reader import ReadView

CONFIG_ENV = "MARIN_POLICY_REGRADE"
RAW_SAMPLES = "policy-10197/raw-samples.json"
VERDICTS = "policy-10197/judge-verdicts.json"
TOKEN_BUDGETS = (128, 512, 2048)
JUDGE_CONCURRENCY = 16


def source_grade(row: dict, benchmark: str) -> tuple[EvalSpec, Grade | JudgePlan]:
    """Apply Evalchemy's native extraction and deterministic grading to a saved response."""
    sample = sample_from_archive_row(row)
    doc = json.loads(sample.doc)
    if benchmark == "financebench":
        reference = {"question": doc["question"], "answers": [str(doc["answer"])]}
        grader = GraderSpec("equivalence_judge", reference)
    elif benchmark == "olympiadbench":
        reference = {"question": doc["problem"], "answers": _flatten_reference_answers(doc["answer"]), "judge": True}
        grader = GraderSpec("math_equivalence", reference)
    else:
        raise ValueError(f"unsupported judged policy benchmark: {benchmark}")
    spec = EvalSpec(sample.task, SourceRef(benchmark), sample.doc_id, 0, (), grader, doc=doc)
    responses = [sample.output or ""]
    return spec, equivalence_judge(spec, responses) if benchmark == "financebench" else math_equivalence(spec, responses)


async def judge_plan(session: aiohttp.ClientSession, endpoint: str, model: str, plan: JudgePlan) -> str:
    """Use the policy's constrained labels, timeout, retries, and empty-response escalation."""
    if len(plan.requests) != 1:
        raise ValueError("answer equivalence requires exactly one judge request")
    prompt = plan.requests[0].prompt
    for tokens in TOKEN_BUDGETS:
        async def call() -> str:
            async with session.post(
                endpoint.rstrip("/") + "/chat/completions",
                headers={"Authorization": "Bearer local-policy-judge"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": tokens,
                    "structured_outputs": {"choice": [label.value for label in JudgeLabel]},
                },
            ) as response:
                response.raise_for_status()
                data = await response.json()
                return data["choices"][0]["message"]["content"] or ""

        raw = await retry(
            call, RetryPolicy(budget=1800, attempt_timeout=300, max_attempts=6),
            retryable_transport, transport_retry_delay,
        )
        if raw.strip():
            JudgeLabel(raw.strip().lower())
            return raw
    raise ValueError("MiniMax returned empty judgments at every policy token budget")


async def grade_rows(rows: list[dict], benchmark: str, endpoint: str, model: str) -> list[TrialVerdict]:
    semaphore = asyncio.Semaphore(JUDGE_CONCURRENCY)
    async with aiohttp.ClientSession() as session:
        async def grade_row(row: dict) -> TrialVerdict:
            async with semaphore:
                spec, grade = source_grade(row, benchmark)
                if isinstance(grade, JudgePlan):
                    raw = await judge_plan(session, endpoint, model, grade)
                    grade = equivalence_label(replace(spec, grader=grade.grader), [raw])
                return TrialVerdict(
                    task=row["task"], doc_id=row["doc_id"],
                    trial_id=row.get("trial_id") or "", filter=row.get("filter") or "",
                    score=float(grade.metrics["accuracy"]), audit=dict(grade.audit),
                    extracted=str(grade.filtered_resps[-1]) if grade.filtered_resps else "",
                )
        return await asyncio.gather(*(grade_row(row) for row in rows))


def main() -> None:
    config = json.loads(os.environ[CONFIG_ENV])
    root, benchmark = config["results_path"], config["benchmark"]
    reader = ReadView(root)
    table = reader.scan("samples")
    if table is None or not table.num_rows:
        raise ValueError("policy regrade requires saved samples")
    current = table.to_pylist(maps_as_pydicts="strict")
    frozen = reader.read_blob(RAW_SAMPLES)
    if frozen is None:
        frozen = json.dumps(current, sort_keys=True).encode()
        with EvaluationStore.open(root, writer_id="policy-raw-samples") as store:
            store.add_artifact(RAW_SAMPLES, frozen)
            store.seal()
    rows = json.loads(frozen)
    source_fields = ("task", "doc_id", "trial_id", "filter", "doc", "output")
    saved_sources = sorted(json.dumps({key: row.get(key) for key in source_fields}, sort_keys=True) for row in rows)
    current_sources = sorted(json.dumps({key: row.get(key) for key in source_fields}, sort_keys=True) for row in current)
    if saved_sources != current_sources:
        raise ValueError("saved policy judgments have different source answers")
    # Failed transport rows remain infrastructure failures; only answers with a recorded score
    # enter the replay. No request to the candidate endpoint is made by this process.
    scored = [row for row in rows if row.get("grading") and row["grading"].get("score") is not None
              and not any("[EVALCHEMY_INFRASTRUCTURE_ERROR]" in (row.get(key) or "")
                          for key in ("output", "extracted"))]
    if not scored:
        raise ValueError("policy regrade has no scoreable source answers")
    identity = {"benchmark": benchmark, "judge": config["judge_config"],
                "source_sha256": hashlib.sha256(frozen).hexdigest()}
    existing = ReadView(root).read_blob(VERDICTS)
    if existing is None:
        verdicts = asyncio.run(grade_rows(scored, benchmark, config["endpoint"], config["model"]))
    else:
        saved = json.loads(existing)
        if any(saved[key] != value for key, value in identity.items()):
            raise ValueError("saved policy judgments have different source or judge provenance")
        verdicts = [TrialVerdict(**row) for row in saved["trials"]]
    with EvaluationStore.open(root, writer_id="policy-minimax-regrade") as store:
        for row, verdict in zip(scored, verdicts, strict=True):
            sample = sample_from_archive_row(row)
            metrics = {**sample.metrics, "policy_raw_accuracy": sample.grading.score, "accuracy": verdict.score}
            updated = sample.model_copy(update={
                "metrics": metrics, "correct": bool(verdict.score), "extracted": verdict.extracted,
                "grading": Grading(method="judge:" + config["model"], metric="accuracy", filter=row.get("filter"),
                                   score=verdict.score, passed=bool(verdict.score), detail=json.dumps(verdict.audit)),
            })
            store.add_sample(updated, trial_id=verdict.trial_id, extraction_filter=verdict.filter)
        store.add_artifact(
            VERDICTS,
            json.dumps({**identity, "trials": [asdict(verdict) for verdict in verdicts]}, sort_keys=True).encode(),
        )
        store.seal()


if __name__ == "__main__":
    main()

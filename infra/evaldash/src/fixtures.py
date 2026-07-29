# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deterministic local fixtures for the eval dashboard.

Writes a small, self-contained set of ``record.json`` files and their per-sample parquet exports
under a target directory, in the exact on-disk layout the ingestor and sample reader expect. Point
the dashboard at it with ``EVALDASH_STORE=local RECORDS_PREFIXES=<dir>`` to develop against realistic
data with no database or cluster.

The dataset is intentionally varied: several models across version cohorts, all three sample kinds
(multiple-choice, generation, agentic), and success/eval-failure/infra-failure/ungraded cases, so the
dashboard's views all have something to render. It is deterministic (fixed timestamps and values) so
regenerating it produces byte-stable records.

Run: ``python infra/evaldash/src/fixtures.py <dest>`` (with ``lib/marin/src`` on ``PYTHONPATH``).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta

from fsspec.core import url_to_fs
from marin.evaluation.records import (
    RECORD_FILE,
    EvalRef,
    EvalRunRecord,
    EvalTaskRef,
    HarborRef,
    HardwareRef,
    ModelRef,
    Provenance,
    RunStatus,
    RunTiming,
    write_record,
)
from marin.evaluation.samples import (
    SAMPLES_PREFIX,
    SAMPLES_SUFFIX,
    Choice,
    EvalSample,
    Grading,
    Message,
    SampleKind,
    write_sample_parquet,
)

# The dashboard's EvalProfileChart highlights this reference model; keep it in the fixtures so that
# view renders its emphasis. See dashboard/src/components/charts/EvalProfileChart.vue.
REFERENCE_MODEL = "snowball"

USER = "dev@marin"
GIT_SHA = "0f1e2d3c4b5a69788796a5b4c3d2e1f00a1b2c3d"


def _hardware() -> HardwareRef:
    return HardwareRef(platform="tpu", accelerator="v6e-8", region_or_cluster="us-central2")


def _provenance() -> Provenance:
    return Provenance(git_sha=GIT_SHA, eval_runtime="evalchemy==0.1.0", launch_host="dev-host")


def _lm_eval_ref(eval_name: str, num_fewshot: int) -> EvalRef:
    return EvalRef(name=eval_name, mechanism="evalchemy", tasks=(EvalTaskRef(name=eval_name, num_fewshot=num_fewshot),))


def _harbor_ref(dataset: str) -> EvalRef:
    return EvalRef(
        name=dataset,
        mechanism="harbor",
        harbor=HarborRef(dataset=dataset, version="1.0", agent=dataset, env="daytona"),
    )


def _record(
    *,
    run_id: str,
    group_id: str,
    model: str,
    version: str | None,
    created_at: str,
    evaluation: EvalRef,
    status: RunStatus,
    results_path: str,
    metrics: dict[str, dict[str, float]],
    description: str | None,
    error: str | None = None,
    log_tails: dict[str, tuple[str, ...]] | None = None,
    runtime_minutes: float = 8.0,
) -> EvalRunRecord:
    # created_at is the terminal write time, so the run's window ends there and starts runtime_minutes
    # earlier -- enough to give the dashboard a real duration to show.
    finished = datetime.fromisoformat(created_at)
    started = finished - timedelta(minutes=runtime_minutes)
    return EvalRunRecord(
        run_id=run_id,
        group_id=group_id,
        created_at=created_at,
        user=USER,
        version=version,
        description=description,
        model=ModelRef(name=model, location=f"gs://marin-models/{model}", backend="vllm"),
        evaluation=evaluation,
        hardware=_hardware(),
        status=status,
        error=error,
        results_path=results_path,
        metrics=metrics,
        jobs={"serve": f"jobs/{group_id}/serve", "eval": f"jobs/{run_id}/eval"},
        log_tails=log_tails or {},
        provenance=_provenance(),
        timing=RunTiming(started_at=started.isoformat(), finished_at=finished.isoformat()),
    )


# --------------------------------------------------------------------------------------------------
# Sample builders
# --------------------------------------------------------------------------------------------------

_MCQ_CHOICES = ("A", "B", "C", "D")


def _mcq_sample(task: str, doc_id: str, *, model_choice: int, target_choice: int, ungraded: bool = False) -> EvalSample:
    """A loglikelihood multiple-choice sample; ``ungraded`` leaves it unscored (correct is None)."""
    logls = [-3.1, -2.2, -1.4, -0.9]
    # Put the model's argmax at model_choice by making that the highest loglikelihood.
    logls[model_choice] = -0.4
    choices = [
        Choice(label=label, text=f"Option {label} for {task} {doc_id}", loglikelihood=ll, is_greedy=(i == model_choice))
        for i, (label, ll) in enumerate(zip(_MCQ_CHOICES, logls, strict=True))
    ]
    if ungraded:
        return EvalSample(
            task=task,
            doc_id=doc_id,
            kind=SampleKind.MULTIPLE_CHOICE,
            prompt_text=f"[{task}] Which option is correct for question {doc_id}?",
            choices=choices,
            model_choice=model_choice,
            target_choice=target_choice,
            grading=Grading(method="lm-eval:acc", metric=None, score=None, passed=None),
            metrics={},
            correct=None,
            doc=json.dumps({"question": f"q{doc_id}", "answer": _MCQ_CHOICES[target_choice]}),
        )
    correct = model_choice == target_choice
    score = 1.0 if correct else 0.0
    return EvalSample(
        task=task,
        doc_id=doc_id,
        kind=SampleKind.MULTIPLE_CHOICE,
        prompt_text=f"[{task}] Which option is correct for question {doc_id}?",
        choices=choices,
        model_choice=model_choice,
        target_choice=target_choice,
        grading=Grading(method="lm-eval:acc", metric="acc,none", filter=None, score=score, passed=correct),
        metrics={"acc,none": score},
        correct=correct,
        doc=json.dumps({"question": f"q{doc_id}", "answer": _MCQ_CHOICES[target_choice]}),
    )


def _generation_sample(task: str, doc_id: str, *, extracted: str, target: str) -> EvalSample:
    correct = extracted.strip() == target.strip()
    score = 1.0 if correct else 0.0
    return EvalSample(
        task=task,
        doc_id=doc_id,
        kind=SampleKind.GENERATION,
        prompt_messages=[
            Message(role="user", content=f"[{task}] Solve problem {doc_id}. Show your work and give the final answer."),
        ],
        output=f"Let me work through problem {doc_id}.\n\nThe answer is {extracted}.",
        extracted=extracted,
        target_text=target,
        grading=Grading(
            method="lm-eval:exact_match",
            metric="exact_match,flexible-extract",
            filter="flexible-extract",
            score=score,
            passed=correct,
        ),
        metrics={"exact_match,flexible-extract": score},
        correct=correct,
        doc=json.dumps({"problem": f"problem {doc_id}", "answer": target}),
    )


def _write_trajectory(fs, results_path: str, task: str, doc_id: str, *, reward: float, errored: bool) -> str:
    """Write an ATIF trajectory JSON sibling and return its URI (under results_path)."""
    steps = [
        {
            "step_id": 0,
            "source": "user",
            "message": f"[{task}] Task {doc_id}: compute the requested value in the sandbox.",
        },
        {
            "step_id": 1,
            "source": "agent",
            "model_name": "snowball",
            "message": "I'll compute it step by step and run the check.",
            "tool_calls": [{"tool_call_id": "c1", "function_name": "python", "arguments": {"code": "print(6*7)"}}],
            "observation": {"results": [{"source_call_id": "c1", "content": "42\n"}]},
            "metrics": {"prompt_tokens": 120.0, "completion_tokens": 44.0},
        },
        {
            "step_id": 2,
            "source": "agent",
            "model_name": "snowball",
            "message": "The final answer is 42." if not errored else "",
            "observation": {
                "results": [{"source_call_id": "c2", "content": "sandbox error: tool crashed" if errored else "ok"}]
            },
            "metrics": {"prompt_tokens": 60.0, "completion_tokens": 12.0},
        },
    ]
    trajectory = {
        "schema_version": "1.0",
        "session_id": f"{task}-{doc_id}",
        "agent": {"name": task, "version": "1.0", "model_name": "snowball"},
        "steps": steps,
        "final_metrics": {"reward": reward},
    }
    uri = f"{results_path}/trajectories/{task}_{doc_id}.json"
    fs.makedirs(f"{results_path}/trajectories", exist_ok=True)
    with fs.open(uri, "w") as handle:
        handle.write(json.dumps(trajectory, indent=2))
    return uri


def _agentic_sample(
    fs, results_path: str, task: str, doc_id: str, *, reward: float, error: str | None = None
) -> EvalSample:
    solved = error is None and reward >= 0.99
    uri = _write_trajectory(fs, results_path, task, doc_id, reward=reward, errored=error is not None)
    return EvalSample(
        task=task,
        doc_id=doc_id,
        kind=SampleKind.AGENTIC,
        trajectory_uri=uri,
        target_text="42",
        grading=Grading(
            method="harbor:verifier",
            metric="reward",
            score=reward,
            passed=solved,
            detail=json.dumps({"reward": reward, "error": error}),
        ),
        metrics={"reward": reward},
        correct=solved,
        doc=json.dumps({"task_id": doc_id, "dataset": task}),
    )


def _write_samples(fs, results_path: str, task: str, samples: list[EvalSample]) -> None:
    fs.makedirs(results_path, exist_ok=True)
    dest = f"{results_path}/{SAMPLES_PREFIX}{task}_20260720{SAMPLES_SUFFIX}"
    write_sample_parquet(fs, dest, samples)


def _write_broken_record(fs, dest: str) -> str:
    """Write one record.json missing a required field, to exercise the Debug view's parse-failure list.

    Mirrors the real schema drift seen in object storage: a record written before ``eval_runtime`` was
    a required provenance field. ``read_records`` collects it as a failure rather than silently
    dropping it, so the Debug tab has an error to show. Returns the path written."""
    run_id = "20260722-000000-legacy-mmlu-broken"
    path = f"{dest}/{run_id}/{RECORD_FILE}"
    fs.makedirs(f"{dest}/{run_id}", exist_ok=True)
    broken = {
        "run_id": run_id,
        "group_id": run_id,
        "created_at": "2026-07-22T00:00:00+00:00",
        "user": USER,
        "model": {"name": "legacy", "location": "gs://marin-models/legacy", "backend": "vllm"},
        "eval": {"name": "mmlu", "mechanism": "evalchemy", "tasks": [{"name": "mmlu", "num_fewshot": 5}]},
        "hardware": {"platform": "tpu", "accelerator": "v6e-8", "region_or_cluster": "us-central2"},
        "status": "succeeded",
        "error": None,
        "results_path": f"{dest}/{run_id}/results",
        "metrics": {"mmlu": {"acc,none": 0.5}},
        "jobs": {},
        "log_tails": {},
        "provenance": {"git_sha": "deadbeef", "launch_host": "old-host"},  # missing required eval_runtime
    }
    with fs.open(path, "w") as handle:
        handle.write(json.dumps(broken, indent=2))
    return path


# --------------------------------------------------------------------------------------------------
# Dataset assembly
# --------------------------------------------------------------------------------------------------

# (metric key, value, stderr) headline for each lm-eval task the fixtures use.
_HEADLINE = {
    "mmlu": ("acc,none", "acc_stderr,none"),
    "arc-challenge": ("acc_norm,none", "acc_norm_stderr,none"),
    "gsm8k-0shot": ("exact_match,flexible-extract", "exact_match_stderr,flexible-extract"),
    "math500": ("accuracy", None),
}


def _lm_metrics(task: str, value: float, stderr: float | None) -> dict[str, dict[str, float]]:
    metric, stderr_key = _HEADLINE[task]
    inner = {metric: value}
    if stderr_key is not None and stderr is not None:
        inner[stderr_key] = stderr
    return {task: inner}


def build_fixtures(dest: str) -> list[str]:
    """Write the full fixture dataset under ``dest``; return the record paths written."""
    fs, _root = url_to_fs(dest)
    written: list[str] = []

    def emit(record: EvalRunRecord) -> None:
        written.append(write_record(record, dest))

    def results_of(run_id: str) -> str:
        return f"{dest}/{run_id}/results"

    # --- snowball 2026.07.20: a full clean launch across all suites (the reference model) ---
    grp = "snowball-2026.07.20"
    desc = "Nightly baseline sweep"
    r = f"{grp}-mmlu"
    emit(
        _record(
            run_id=r,
            group_id=grp,
            model=REFERENCE_MODEL,
            version="2026.07.20",
            created_at="2026-07-20T02:00:00+00:00",
            evaluation=_lm_eval_ref("mmlu", 5),
            status=RunStatus.SUCCEEDED,
            results_path=results_of(r),
            metrics=_lm_metrics("mmlu", 0.741, 0.011),
            description=desc,
        )
    )
    _write_samples(
        fs,
        results_of(r),
        "mmlu",
        [
            _mcq_sample("mmlu", "0", model_choice=2, target_choice=2),
            _mcq_sample("mmlu", "1", model_choice=0, target_choice=1),
            _mcq_sample("mmlu", "2", model_choice=3, target_choice=3),
            _mcq_sample("mmlu", "3", model_choice=1, target_choice=1),
            _mcq_sample("mmlu", "4", model_choice=0, target_choice=2, ungraded=True),
        ],
    )
    r = f"{grp}-arc-challenge"
    emit(
        _record(
            run_id=r,
            group_id=grp,
            model=REFERENCE_MODEL,
            version="2026.07.20",
            created_at="2026-07-20T02:05:00+00:00",
            evaluation=_lm_eval_ref("arc-challenge", 25),
            status=RunStatus.SUCCEEDED,
            results_path=results_of(r),
            metrics=_lm_metrics("arc-challenge", 0.632, 0.014),
            description=desc,
        )
    )
    _write_samples(
        fs,
        results_of(r),
        "arc-challenge",
        [_mcq_sample("arc-challenge", str(i), model_choice=i % 4, target_choice=(i * 3) % 4) for i in range(6)],
    )
    r = f"{grp}-gsm8k-0shot"
    emit(
        _record(
            run_id=r,
            group_id=grp,
            model=REFERENCE_MODEL,
            version="2026.07.20",
            created_at="2026-07-20T02:10:00+00:00",
            evaluation=_lm_eval_ref("gsm8k-0shot", 0),
            status=RunStatus.SUCCEEDED,
            results_path=results_of(r),
            metrics=_lm_metrics("gsm8k-0shot", 0.564, 0.019),
            description=desc,
        )
    )
    _write_samples(
        fs,
        results_of(r),
        "gsm8k-0shot",
        [
            _generation_sample("gsm8k-0shot", "0", extracted="42", target="42"),
            _generation_sample("gsm8k-0shot", "1", extracted="17", target="18"),
            _generation_sample("gsm8k-0shot", "2", extracted="120", target="120"),
        ],
    )
    r = f"{grp}-aime"
    emit(
        _record(
            run_id=r,
            group_id=grp,
            model=REFERENCE_MODEL,
            version="2026.07.20",
            created_at="2026-07-20T02:20:00+00:00",
            evaluation=_harbor_ref("aime"),
            status=RunStatus.SUCCEEDED,
            results_path=results_of(r),
            metrics={"aime": {"accuracy": 0.333, "mean_reward": 0.333, "solved": 3.0, "total": 9.0}},
            description=desc,
            runtime_minutes=42.0,  # agentic sandbox rollouts run far longer than the lm-eval tasks
        )
    )
    _write_samples(
        fs,
        results_of(r),
        "aime",
        [
            _agentic_sample(fs, results_of(r), "aime", "1", reward=1.0),
            _agentic_sample(fs, results_of(r), "aime", "2", reward=0.0),
            _agentic_sample(fs, results_of(r), "aime", "3", reward=0.0, error="sandbox timed out"),
        ],
    )

    # --- tootsie-8b 2026.07.20: mixed outcomes (an eval failure and an infra failure) ---
    grp = "tootsie-8b-2026.07.20"
    desc = "Tootsie 8B checkpoint eval"
    r = f"{grp}-mmlu"
    emit(
        _record(
            run_id=r,
            group_id=grp,
            model="tootsie-8b",
            version="2026.07.20",
            created_at="2026-07-20T04:00:00+00:00",
            evaluation=_lm_eval_ref("mmlu", 5),
            status=RunStatus.SUCCEEDED,
            results_path=results_of(r),
            metrics=_lm_metrics("mmlu", 0.688, 0.012),
            description=desc,
        )
    )
    _write_samples(
        fs,
        results_of(r),
        "mmlu",
        [_mcq_sample("mmlu", str(i), model_choice=i % 4, target_choice=(i + 1) % 4) for i in range(6)],
    )
    r = f"{grp}-gsm8k-0shot"
    emit(
        _record(
            run_id=r,
            group_id=grp,
            model="tootsie-8b",
            version="2026.07.20",
            created_at="2026-07-20T04:10:00+00:00",
            evaluation=_lm_eval_ref("gsm8k-0shot", 0),
            status=RunStatus.SUCCEEDED,
            results_path=results_of(r),
            metrics=_lm_metrics("gsm8k-0shot", 0.491, 0.021),
            description=desc,
        )
    )
    _write_samples(
        fs,
        results_of(r),
        "gsm8k-0shot",
        [
            _generation_sample("gsm8k-0shot", str(i), extracted=str(i), target=str(i if i % 2 else i + 1))
            for i in range(4)
        ],
    )
    r = f"{grp}-math500"
    emit(
        _record(
            run_id=r,
            group_id=grp,
            model="tootsie-8b",
            version="2026.07.20",
            created_at="2026-07-20T04:20:00+00:00",
            evaluation=_lm_eval_ref("math500", 0),
            status=RunStatus.FAILED,
            results_path=results_of(r),
            metrics={},
            description=desc,
            error="grader raised: unparived latex in 3 of 500 problems",
            log_tails={"eval": ("Traceback (most recent call last):", "ValueError: could not parse boxed answer")},
            runtime_minutes=3.5,  # failed partway through grading
        )
    )
    r = f"{grp}-aime"
    emit(
        _record(
            run_id=r,
            group_id=grp,
            model="tootsie-8b",
            version="2026.07.20",
            created_at="2026-07-20T04:30:00+00:00",
            evaluation=_harbor_ref("aime"),
            status=RunStatus.INFRA_FAILED,
            results_path=results_of(r),
            metrics={},
            description=desc,
            error="serving endpoint never became healthy (timeout after 1800s)",
            log_tails={"serve": ("vllm: waiting for model weights", "readiness probe failed")},
            runtime_minutes=30.0,  # matches the 1800s serving-readiness timeout above
        )
    )

    # --- qwen3-8b: two version cohorts; the matrix should show only the newer (2026.07.21) ---
    for version, created, mmlu_acc in (
        ("2026.07.19", "2026-07-19T09:00:00+00:00", 0.702),
        ("2026.07.21", "2026-07-21T09:00:00+00:00", 0.719),
    ):
        grp = f"qwen3-8b-{version}"
        r = f"{grp}-mmlu"
        emit(
            _record(
                run_id=r,
                group_id=grp,
                model="qwen3-8b",
                version=version,
                created_at=created,
                evaluation=_lm_eval_ref("mmlu", 5),
                status=RunStatus.SUCCEEDED,
                results_path=results_of(r),
                metrics=_lm_metrics("mmlu", mmlu_acc, 0.011),
                description="Qwen3 8B reference",
            )
        )
        _write_samples(
            fs,
            results_of(r),
            "mmlu",
            [
                _mcq_sample(
                    "mmlu", str(i), model_choice=(i * 2) % 4, target_choice=(i * 2) % 4 if i % 3 else (i + 1) % 4
                )
                for i in range(6)
            ],
        )
        r = f"{grp}-arc-challenge"
        emit(
            _record(
                run_id=r,
                group_id=grp,
                model="qwen3-8b",
                version=version,
                created_at=created,
                evaluation=_lm_eval_ref("arc-challenge", 25),
                status=RunStatus.SUCCEEDED,
                results_path=results_of(r),
                metrics=_lm_metrics("arc-challenge", 0.58 + (version == "2026.07.21") * 0.02, 0.015),
                description="Qwen3 8B reference",
            )
        )
        _write_samples(
            fs,
            results_of(r),
            "arc-challenge",
            [_mcq_sample("arc-challenge", str(i), model_choice=i % 4, target_choice=(i + 2) % 4) for i in range(5)],
        )

    # --- llama3-8b: an unlabelled launch (version None), partial coverage ---
    grp = "llama3-8b-run"
    r = f"{grp}-mmlu"
    emit(
        _record(
            run_id=r,
            group_id=grp,
            model="llama3-8b",
            version=None,
            created_at="2026-07-18T12:00:00+00:00",
            evaluation=_lm_eval_ref("mmlu", 5),
            status=RunStatus.SUCCEEDED,
            results_path=results_of(r),
            metrics=_lm_metrics("mmlu", 0.653, 0.012),
            description=None,
        )
    )
    _write_samples(
        fs,
        results_of(r),
        "mmlu",
        [
            _mcq_sample("mmlu", str(i), model_choice=i % 4, target_choice=i % 4 if i % 2 else (i + 1) % 4)
            for i in range(5)
        ],
    )

    # One malformed record so the Debug view has a parse failure to surface (not counted in written).
    _write_broken_record(fs, dest)

    return written


def main() -> None:
    dest = sys.argv[1] if len(sys.argv) > 1 else "/tmp/evaldash-fixtures"
    written = build_fixtures(dest)
    print(f"wrote {len(written)} records under {dest}")


if __name__ == "__main__":
    main()

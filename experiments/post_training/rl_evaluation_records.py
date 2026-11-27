# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Archive SkyRL answer rows and publish the evaluation record consumed by EvalDash."""

import getpass
import hashlib
import json
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from evalrecords.records import (
    CW_RECORDS_PREFIX,
    DEFAULT_RECORDS_PREFIX,
    EvalRef,
    EvalRunRecord,
    HardwareRef,
    ModelRef,
    Provenance,
    RLAttemptRef,
    RunStatus,
    TaskCoverage,
    read_record,
    record_path,
    write_record,
)
from finestore.eval import EvalSample, EvaluationStore, Grading, SampleKind
from finestore.reader import ReadView
from finestore.rl.run_records import AttemptRecord, EvaluationTaskRow
from rigging.filesystem.storage_path import StoragePath, prefix_join


@dataclass(frozen=True)
class SkyRLSession:
    attempt: AttemptRecord
    model: ModelRef
    rows: Sequence[EvaluationTaskRow]
    rl_attempt: RLAttemptRef | None
    score: float | None


@dataclass(frozen=True)
class SkyRLRecordedEvaluation:
    evaluation: EvalRef

    def evaluated(self, source: SkyRLSession, *, version: str) -> EvalRunRecord:
        return publish_skyrl_evaluation(source, evaluation=self.evaluation, version=version)


def publish_skyrl_evaluation(
    source: SkyRLSession,
    *,
    evaluation: EvalRef,
    version: str,
) -> EvalRunRecord:
    """Seal a native session's answer archive, then publish its shared evaluation record."""
    attempt, model, rows, rl_attempt, score = source.attempt, source.model, source.rows, source.rl_attempt, source.score
    if not rows:
        raise ValueError(f"No native answer rows for {evaluation.name}")
    step = rows[0].step
    if any(
        row.attempt_id != attempt.attempt_id or row.step != step or row.evaluation != rows[0].evaluation for row in rows
    ):
        raise ValueError("SkyRL evaluation rows must belong to one attempt and session")
    source = "\n".join(row.model_dump_json() for row in sorted(rows, key=lambda row: row.task_id)) + "\n"
    digest = hashlib.sha256(source.encode()).hexdigest()
    key = json.dumps((attempt.run_id, attempt.attempt_id, evaluation.name, rows[0].evaluation, step, digest, score))
    run_id = "skyrl-" + hashlib.sha256(key.encode()).hexdigest()[:24]
    native = attempt.native
    iris = native.config["iris"]
    cluster, allocation = iris["cluster"], iris["allocation"]
    prefix = CW_RECORDS_PREFIX if cluster.startswith("cw-") else DEFAULT_RECORDS_PREFIX
    results_path = prefix_join(prefix, f"{run_id}/results")
    path = record_path(prefix, run_id)
    archive_sealed = (StoragePath(results_path) / "HEAD").exists() and ReadView(results_path).is_sealed()
    evaluation = evaluation.model_copy(
        update={"postprocessing": {**evaluation.postprocessing, "skyrl_step": str(step), "answer_rows_sha256": digest}}
    )
    if StoragePath(path).exists():
        existing = read_record(path)
        if (existing.evaluation, existing.model, existing.rl_attempt) != (evaluation, model, rl_attempt):
            raise ValueError(f"Published SkyRL evaluation inputs changed for {run_id}")
        if not archive_sealed:
            raise ValueError(f"Published SkyRL evaluation archive is unsealed for {run_id}")
        return existing

    errors = Counter(row.error_type for row in rows if row.error_type is not None)
    coverage = TaskCoverage(
        n_attempted=len(rows),
        n_scored=sum(row.score is not None and math.isfinite(row.score) for row in rows),
        errors=dict(errors),
    )
    if not archive_sealed:
        with EvaluationStore.open(results_path, writer_id="skyrl-evaluation") as store:
            store.add_source_artifact("answers.jsonl", source.encode(), content_type="application/x-ndjson")
            for row in rows:
                for answer in row.answers:
                    value = answer.verifier_score
                    if value is not None and not math.isfinite(value):
                        value = None
                    store.add_sample(
                        EvalSample(
                            task=evaluation.name,
                            doc_id=row.task_id,
                            kind=SampleKind.AGENTIC if row.strata.agentic else SampleKind.GENERATION,
                            grading=Grading(
                                method="native-verifier",
                                metric="reward",
                                score=value,
                                passed=answer.passed if value is not None else None,
                                detail=answer.model_dump_json(),
                            ),
                            metrics={"reward": value} if value is not None else {},
                            doc=json.dumps({"data_source": row.data_source, "native_uid": answer.native_uid}),
                        ),
                        trial_id=str(answer.ordinal),
                    )
            store.seal()

    result = native.result
    record = EvalRunRecord(
        run_id=run_id,
        group_id=run_id,
        # Training sessions retain the unlabeled cohort used by existing records.
        version=version if step == -1 else None,
        created_at=result.finished_at or result.started_at or datetime.now(UTC).isoformat(),
        user=getpass.getuser(),
        model=model,
        rl_attempt=rl_attempt,
        evaluation=evaluation,
        hardware=HardwareRef(
            platform="gpu",
            accelerator=allocation["gpu_variant"],
            region_or_cluster=cluster,
            task_count=allocation["num_nodes"],
        ),
        status=RunStatus.SUCCEEDED if coverage.n_scored else RunStatus.FAILED,
        error=None if coverage.n_scored else "No usable verifier scores",
        results_path=results_path,
        metrics={evaluation.name: {"reward": score}} if score is not None else {},
        coverage={evaluation.name: coverage},
        jobs={"eval": result.iris_job_id} if result.iris_job_id else {},
        log_tails={},
        provenance=Provenance(git_sha=(result.provenance or {}).get("base_commit") or "unknown", launch_host=cluster),
    )
    write_record(record, prefix)
    return record

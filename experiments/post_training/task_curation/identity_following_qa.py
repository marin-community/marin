# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""QA controls and SAMPLE review for the repaired identity-following revision (ruling 2026-11-23).

The controls grade the repaired tasks' pinned binary labels grader on stratified submissions: a
correct identity answer in the pinned language must score CORRECT, an NVIDIA attribution must score
INCORRECT, and the empty NoP must score zero without a judge call. The SAMPLE review draws a
seed-zero stratified panel and reviews it with the pinned V4 reviewer under the standard rubric.
Both receipts record the serving identity, verdicts and the unavailable share.
"""

import argparse
import json
import logging
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from fray.types import ANY_REGION
from iris.rpc import job_pb2
from marin.evaluation.hardware import AcceleratorChoice, Platform
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.models import ReviewRecord
from taskcompendium.pipeline.reviewer_spec import JudgeTier
from verifyit.modes.grade_judge import grade_judge_candidate
from verifyit.spec import spec_from_table

from experiments.post_training.task_curation.full_qc import (
    REVIEW_REQUEST_TIMEOUTS,
    REVIEW_RUBRIC,
    SAMPLE_REVIEWER_VERSION,
    ReadyChatReviewer,
    local_reviewer,
    qualify_reviewer,
)
from experiments.post_training.task_curation.full_qc_inputs import task_contract
from experiments.post_training.task_curation.full_qc_review import REVIEW_BATCH_SIZE, review_batch
from experiments.post_training.task_curation.judges import LocalJudgeEndpoint
from experiments.post_training.task_curation.review_quality import quality_result

logger = logging.getLogger(__name__)

IDENTITY_LANGUAGES = {
    "German": (
        "Ich bin Marin, ein KI-Assistent, entwickelt von Forschenden an der Stanford University und Open Athena.",
        "Ich bin Marin, ein KI-Assistent, trainiert und entwickelt von Forschenden bei NVIDIA.",
    ),
    "English": (
        "I am Marin, an AI assistant built by researchers at Stanford and Open Athena.",
        "I am Marin, an AI assistant trained and developed by researchers from NVIDIA.",
    ),
    "Spanish": (
        "Soy Marin, un asistente de IA desarrollado por investigadores de Stanford y Open Athena.",
        "Soy Marin, un asistente de IA entrenado y desarrollado por investigadores de NVIDIA.",
    ),
    "French": (
        "Je suis Marin, un assistant IA développé par des chercheurs de Stanford et d'Open Athena.",
        "Je suis Marin, un assistant IA entraîné et développé par des chercheurs de NVIDIA.",
    ),
    "Hindi": (
        "मैं Marin हूँ, एक AI सहायक जिसे Stanford और Open Athena के शोधकर्ताओं ने विकसित किया है।",
        "मैं Marin हूँ, एक AI सहायक जिसे NVIDIA के शोधकर्ताओं ने प्रशिक्षित और विकसित किया है।",
    ),
    "Italian": (
        "Sono Marin, un assistente AI sviluppato da ricercatori di Stanford e Open Athena.",
        "Sono Marin, un assistente AI addestrato e sviluppato da ricercatori di NVIDIA.",
    ),
    "Japanese": (
        "私はMarinです。StanfordとOpen Athenaの研究者によって開発されたAIアシスタントです。",
        "私はMarinです。NVIDIAの研究者によって訓練され、開発されたAIアシスタントです。",
    ),
    "Korean": (
        "저는 Marin이며, Stanford와 Open Athena의 연구자들이 개발한 AI 어시스턴트입니다.",
        "저는 Marin이며, NVIDIA의 연구자들이 훈련하고 개발한 AI 어시스턴트입니다.",
    ),
    "Portugese": (
        "Sou Marin, um assistente de IA desenvolvido por pesquisadores de Stanford e Open Athena.",
        "Sou Marin, um assistente de IA treinado e desenvolvido por pesquisadores da NVIDIA.",
    ),
    "Chinese": (
        "我是 Marin，一个由斯坦福大学和 Open Athena 的研究人员构建的 AI 助手。",  # noqa: RUF001
        "我是 Marin，一个由 NVIDIA 的研究人员训练和开发的 AI 助手。",  # noqa: RUF001
    ),
}
SAMPLE_SEED = 0
CONTROLS_PER_LANGUAGE = 3
SAMPLE_PANEL_PER_LANGUAGE = 12
MEDIUM_MODEL_CONFIG = Path(
    "experiments/post_training/task_curation/judge_model_configs/MiniMaxAI-MiniMax-M3-MXFP8-262k.yaml"
)
MEDIUM_CONTEXT_WINDOW_TOKENS = 262_144


def load_repaired_tasks(parquet_uri: str) -> list[tuple[TaskSpec, dict]]:
    """Load the repaired population as (task, runtime) pairs with their language pin."""
    with StoragePath(parquet_uri).open("rb") as handle:
        table = pq.ParquetFile(handle)
        pairs = []
        for batch in table.iter_batches(batch_size=128):
            for row in batch.to_pylist():
                task = TaskSpec.model_validate_json(row["task_spec"])
                pairs.append((task, row["runtime"]))
    return [(task, runtime) for task, runtime in pairs]


def language_pin(task: TaskSpec) -> str:
    """The response language the task's second criterion pins."""
    rubric = json.loads(task.grader.parameters["references"][0])
    marker = "The response must be in the "
    for entry in rubric:
        criteria = entry["criteria"]
        if criteria.startswith(marker) and criteria.endswith(" language."):
            return criteria[len(marker) : -len(" language.")]
    raise ValueError(f"Task {task.id} pins no recognizable response language: {rubric}")


def stratified_panel(tasks: list[tuple[TaskSpec, dict]], per_language: int, seed: int) -> list[tuple[TaskSpec, dict]]:
    """A seed-zero panel with ``per_language`` tasks from every pinned language."""
    by_language: dict[str, list[tuple[TaskSpec, dict]]] = {}
    for task, runtime in tasks:
        by_language.setdefault(language_pin(task), []).append((task, runtime))
    generator = random.Random(seed)
    panel = []
    for language in sorted(by_language):
        population = sorted(by_language[language], key=lambda pair: pair[0].id)
        if len(population) < per_language:
            raise ValueError(f"Language {language} has only {len(population)} tasks")
        panel.extend(generator.sample(population, per_language))
    return panel


def control_submissions(language: str) -> dict[str, str]:
    """The positive and NVIDIA-attributing submissions for one pinned language."""
    positive, nvidia = IDENTITY_LANGUAGES[language]
    return {"positive": positive, "nvidia": nvidia}


def judge_spec(task: TaskSpec):
    """Rebuild the task's pinned JudgeSpec from its grading parameters."""
    parameters = dict(task.grader.parameters)
    parameters["mode"] = task.grader.mode
    return spec_from_table(parameters)


def run_controls(parquet_uri: str, output: str) -> dict[str, Any]:
    """Grade stratified controls with the pinned MEDIUM grader and write the receipt."""
    tasks = load_repaired_tasks(parquet_uri)
    panel = stratified_panel(tasks, CONTROLS_PER_LANGUAGE, SAMPLE_SEED)
    endpoint = LocalJudgeEndpoint(
        tier=JudgeTier.MEDIUM,
        model_config=MEDIUM_MODEL_CONFIG,
        context_window_tokens=MEDIUM_CONTEXT_WINDOW_TOKENS,
        accelerator=_medium_accelerator(),
        priority=_batch_priority(),
        env_vars={"MARIN_TEMP_PREFIX": "s3://marin-us-east-02a"},
    )
    results: list[dict[str, Any]] = []
    with endpoint.connect() as connection:
        for task, _runtime in panel:
            language = language_pin(task)
            spec = judge_spec(task)
            submissions = {**control_submissions(language), "nop": ""}
            for kind, candidate in submissions.items():
                reward = grade_judge_candidate(spec, candidate, connection=connection)
                results.append(
                    {
                        "task_id": task.id,
                        "language": language,
                        "submission": kind,
                        "reward": reward.reward,
                        "detail": reward.detail,
                    }
                )
                logger.info("CONTROL %s %s %s -> %s", task.id, language, kind, reward.reward)
    receipt = {
        "grader_identity": endpoint.identity,
        "serving_jobs": endpoint.job_ids,
        "panel": [{"task_id": task.id, "language": language_pin(task)} for task, _ in panel],
        "results": results,
        "summary": _controls_summary(results),
    }
    _write_json(output, receipt)
    return receipt


def _controls_summary(results: list[dict[str, Any]]) -> dict[str, int]:
    expected = {"positive": 1.0, "nvidia": 0.0, "nop": 0.0}
    summary = {f"{kind}_expected_{name}": 0 for kind in expected for name in ("met", "violated")}
    for row in results:
        key = f"{row['submission']}_expected_{'met' if row['reward'] == expected[row['submission']] else 'violated'}"
        summary[key] += 1
    return summary


def run_sample(parquet_uri: str, output: str, review_cache: str) -> dict[str, Any]:
    """Review a seed-zero stratified panel with the pinned V4 reviewer and write the receipt."""
    tasks = load_repaired_tasks(parquet_uri)
    panel = stratified_panel(tasks, SAMPLE_PANEL_PER_LANGUAGE, SAMPLE_SEED)
    spec = local_reviewer(SAMPLE_REVIEWER_VERSION)
    logger.info("Reviewer: %s", json.dumps(spec.identity, sort_keys=True))
    batch_root = StoragePath(output.rsplit("/", 1)[0] + "/sample-batches/")
    batches = [panel[index : index + REVIEW_BATCH_SIZE] for index in range(0, len(panel), REVIEW_BATCH_SIZE)]
    reviews: list[ReviewRecord] = []
    receipts: list[str] = []
    with spec.endpoint.serve() as session:
        reviewer = ReadyChatReviewer(
            session,
            spec,
            query_cache_root=review_cache,
            max_concurrent=32,
            request_timeout=REVIEW_REQUEST_TIMEOUTS[SAMPLE_REVIEWER_VERSION],
        )
        qualify_reviewer(reviewer)
        for index, batch in enumerate(batches):
            batch_tasks = [task for task, _runtime in batch]
            contracts = [task_contract(task, json.loads(runtime)) for task, runtime in batch]
            batch_reviews, receipt_path = review_batch(batch_tasks, contracts, REVIEW_RUBRIC, reviewer, batch_root)
            reviews.extend(batch_reviews)
            receipts.append(receipt_path)
            logger.info("Batch %d/%d reviewed: %d records", index + 1, len(batches), len(batch_reviews))
    result = quality_result(reviews)
    total = len(reviews)
    unavailable_share = result.unavailable_count / total if total else 0.0
    receipt = {
        "panel": [{"task_id": task.id, "language": language_pin(task)} for task, _runtime in panel],
        "reviewer": spec.identity,
        "rubric": asdict(REVIEW_RUBRIC),
        "batch_receipts": receipts,
        "reviews": [review.model_dump(mode="json") for review in reviews],
        "quality_result": result.model_dump(),
        "coverage": {
            "panel_size": len(panel),
            "reviewed": result.reviewed,
            "uncertain": result.uncertain_count,
            "unavailable": result.unavailable_count,
            "unavailable_share": unavailable_share,
            "meets_unavailable_standard": unavailable_share <= 0.05 and total >= 100,
        },
        "serving_jobs": spec.endpoint.job_ids,
    }
    _write_json(output, receipt)
    return receipt


def _write_json(uri: str, payload: dict[str, Any]) -> None:
    path = StoragePath(uri)
    path.write_bytes(json.dumps(payload, indent=1, sort_keys=True).encode())
    logger.info("Receipt written to %s", uri)


def _medium_accelerator() -> AcceleratorChoice:
    return AcceleratorChoice(Platform.GPU, gpu_type="H100", gpu_count=8, region=ANY_REGION, preemptible=False)


def _batch_priority() -> int:
    return job_pb2.PRIORITY_BAND_BATCH


def main() -> None:
    """Run the QA CLI: the pinned-grader controls, the SAMPLE panel review, or both."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", required=True, help="Repaired population tasks.parquet URI")
    parser.add_argument("--output-root", required=True, help="Receipt output URI prefix")
    parser.add_argument("--review-cache", required=True, help="Review query cache URI")
    parser.add_argument("--mode", choices=("controls", "sample", "both"), default="both")
    parser.add_argument("--run", action="store_true", help="Serve and grade; without it, print the plan")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    configure_coreweave_s3()
    if args.mode in ("controls", "both"):
        receipt = run_controls(args.parquet, f"{args.output_root}/controls.json")
        logger.info("Controls summary: %s", json.dumps(receipt["summary"], sort_keys=True))
    if args.mode in ("sample", "both"):
        receipt = run_sample(args.parquet, f"{args.output_root}/sample-review.json", args.review_cache)
        logger.info("SAMPLE coverage: %s", json.dumps(receipt["coverage"], sort_keys=True))


if __name__ == "__main__":
    main()

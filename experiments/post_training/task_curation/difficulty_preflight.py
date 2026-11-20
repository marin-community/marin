# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify frozen source contracts and judge routes before an attempt batch."""

import logging
from dataclasses import dataclass
from pathlib import Path

from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.secrets import require_secret

from experiments.post_training.task_curation.difficulty.runtime import (
    TOGETHER_API_KEY_ENV,
    TOGETHER_BASE_URL,
    TOGETHER_MODEL,
    hosted_endpoint_liveness,
    queue_judge_tier,
)
from experiments.post_training.task_curation.difficulty_contracts import (
    ContractVintageError,
    describe_vintage,
    resolve_receipt_vintage,
)
from experiments.post_training.task_curation.difficulty_preparation import prepare_source
from experiments.post_training.task_curation.difficulty_samples import DifficultySource, load_queue_sources
from experiments.post_training.task_curation.sample_qc import panel_task_runtimes

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourcePreflight:
    """The receipt-stage verdict for one frozen source."""

    source_id: str
    vintage: str | None
    tasks: int
    error: str | None
    # Judge models the staged prepared tasks declare; empty without staging.
    judge_models: tuple[str, ...] = ()

    def to_row(self) -> dict:
        row = {"source_id": self.source_id, "tasks": self.tasks, "ok": self.error is None}
        if self.vintage is not None:
            row["contract_vintage"] = self.vintage
            row["contract_vintage_basis"] = describe_vintage(self.vintage)
        if self.error is not None:
            row["error"] = self.error
        return row


def preflight_source(source: DifficultySource, stage_root: Path | None = None) -> SourcePreflight:
    """Reproduce the campaign's receipt-stage checks for one source.

    With ``stage_root`` the source is also fully prepared locally (the same
    ``prepare_source`` the campaign runs), which catches task-validation
    failures — such as a runtime packet conflicting with its own task specs —
    before anything is submitted. Nothing staged is published; the root is a
    disposable scratch directory.
    """
    tasks, runtimes = panel_task_runtimes(source.panel)
    frozen = {selected.task_id: selected.task_sha256 for selected in source.sample.tasks}
    for task in tasks:
        if task.id not in frozen:
            return SourcePreflight(
                source.declaration.id, None, len(frozen), f"Prepared task {task.id} is outside the frozen selection"
            )
    try:
        vintage = resolve_receipt_vintage(frozen, tasks, runtimes)
    except ContractVintageError as error:
        return SourcePreflight(source.declaration.id, None, len(frozen), str(error))
    if stage_root is not None:
        try:
            prepared = prepare_source(source, stage_root)
        except (ContractVintageError, ValueError) as error:
            return SourcePreflight(source.declaration.id, vintage, len(frozen), str(error))
        judge_models = tuple(
            sorted({task.judge_model for task in prepared.tasks if task.judge_model is not None and task.reason is None})
        )
        return SourcePreflight(source.declaration.id, vintage, len(frozen), None, judge_models)
    return SourcePreflight(source.declaration.id, vintage, len(frozen), None)


def preflight_batch(
    sources: str,
    sources_sha256: str,
    source_ids: tuple[str, ...] | None,
    stage_root: Path | None = None,
    probe_hosted: bool = False,
) -> dict:
    """Verify endpoint liveness, frozen task contracts and the staged judge route."""
    configure_coreweave_s3()
    if probe_hosted:
        try:
            hosted_endpoint_liveness(TOGETHER_BASE_URL, require_secret(TOGETHER_API_KEY_ENV), TOGETHER_MODEL)
        except Exception as error:
            raise ValueError(f"Hosted endpoint liveness failed; refusing the submission: {error}") from error
    queue_sources = load_queue_sources(sources, sources_sha256, source_ids)
    bindings: dict = {"source_queue": sources, "source_queue_sha256": sources_sha256}
    results = [preflight_source(source, stage_root) for source in queue_sources]
    failures = [result for result in results if result.error is not None]
    receipt = {
        **bindings,
        "sources": len(results),
        "tasks": sum(result.tasks for result in results),
        "vintages": sorted({result.vintage for result in results if result.vintage is not None}),
        "staged": stage_root is not None,
        "failures": [result.to_row() for result in failures],
    }
    if stage_root is not None:
        # The campaign refuses a mixed judge tier only after Iris prepared it
        # (large-v21 spent a submission and its prepare cost there). Staging
        # knows every source's declared judge, so refuse the mixed queue here,
        # before any submit.
        judged_models = {model for result in results for model in result.judge_models}
        tier = queue_judge_tier(judged_models)
        receipt["judge_tier"] = "relay" if tier is None else tier.value
    if failures:
        detail = "\n".join(f"{result.source_id}:\n{result.error}" for result in failures)
        raise ValueError(f"Batch preflight failed for {len(failures)}/{len(results)} sources:\n{detail}")
    return receipt

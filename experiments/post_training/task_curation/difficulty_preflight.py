# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Continuation preflight: fail a submission locally, before Iris runs it.

hosted-v13 was submitted and only then died in the campaign's first
``prepare_source`` call with "Prepared task differs from the frozen sample
receipt" — a mismatch the launcher could have reproduced on this machine for
the cost of reading the frozen artifacts. The preflight performs exactly the
campaign's receipt-stage work per selected source: verify the panel against
its frozen census, resolve the receipt's contract vintage, and compare every
prepared task digest against the frozen sample receipt. It also verifies the
continuation bindings (queue digest, parent ancestry, parent traffic seal)
that the trials worker would otherwise refuse after submission. Any failure
refuses the launch with the precise diff.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.secrets import require_secret
from taskcompendium.evaluation.corpus import file_sha256

from experiments.post_training.task_curation.difficulty_campaign import (
    TOGETHER_API_KEY_ENV,
    TOGETHER_BASE_URL,
    TOGETHER_MODEL,
    hosted_endpoint_liveness,
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
            prepare_source(source, stage_root)
        except (ContractVintageError, ValueError) as error:
            return SourcePreflight(source.declaration.id, vintage, len(frozen), str(error))
    return SourcePreflight(source.declaration.id, vintage, len(frozen), None)


def preflight_continuation(
    sources: str,
    sources_sha256: str,
    source_ids: tuple[str, ...] | None,
    previous_campaign: str | None = None,
    previous_serving_sha256: str | None = None,
    stage_root: Path | None = None,
    probe_hosted: bool = False,
) -> dict:
    """Verify a submission locally: hosted endpoint liveness, continuation bindings,
    then every selected source's receipt (and staging when ``stage_root`` is set).

    Returns the preflight receipt; raises ``ValueError`` naming the failures.
    """
    configure_coreweave_s3()
    if probe_hosted:
        try:
            hosted_endpoint_liveness(TOGETHER_BASE_URL, require_secret(TOGETHER_API_KEY_ENV), TOGETHER_MODEL)
        except Exception as error:
            raise ValueError(f"Hosted endpoint liveness failed; refusing the submission: {error}") from error
    if bool(previous_campaign) != bool(previous_serving_sha256):
        raise ValueError("Continuation requires both the previous campaign and its serving receipt digest")
    queue_sources = load_queue_sources(sources, sources_sha256, source_ids)
    bindings: dict = {"source_queue": sources, "source_queue_sha256": sources_sha256}
    if previous_campaign is not None:
        root = StoragePath(previous_campaign)
        prepared = root / "prepared.json"
        if not prepared.exists():
            raise ValueError(f"Previous campaign has no prepared.json: {previous_campaign}")
        parent = json.loads(prepared.read_text())
        if parent.get("source_queue_sha256") != sources_sha256:
            raise ValueError(
                "Continuation cannot change the frozen source queue: the parent binds "
                f"{parent.get('source_queue_sha256')}"
            )
        serving = root / "serving.json"
        if not serving.exists():
            raise ValueError(f"Previous campaign has no serving.json: {previous_campaign}")
        serving_digest = file_sha256(serving)
        if serving_digest != previous_serving_sha256:
            hint = (
                " The requested digest is not a full sha256; compute it programmatically from the "
                "parent's serving.json bytes (small-v22 died on a hand-truncated digest)."
                if len(previous_serving_sha256) != 64
                else " Compute it programmatically from the parent's serving.json bytes."
            )
            raise ValueError(
                "Previous serving receipt differs from its requested digest: "
                f"requested {previous_serving_sha256}, found {serving_digest}.{hint}"
            )
        seal = root / "trials" / "model-requests-final.json"
        if not seal.exists():
            raise ValueError(
                f"Previous campaign has not sealed trials/model-requests-final.json: {previous_campaign}; "
                "the trials worker refuses an unsealed parent after submission, so refuse here"
            )
        bindings.update(
            {
                "previous_campaign": previous_campaign,
                "previous_serving_sha256": previous_serving_sha256,
                "previous_sources": len(parent.get("sources", [])),
            }
        )
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
    if failures:
        detail = "\n".join(f"{result.source_id}:\n{result.error}" for result in failures)
        raise ValueError(f"Continuation preflight failed for {len(failures)}/{len(results)} sources:\n{detail}")
    return receipt

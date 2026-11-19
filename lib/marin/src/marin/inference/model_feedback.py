# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Save model-feedback reports from the serve dashboard to Marin storage.

The record format and bucket prefixes live in ``evalrecords.model_feedback``, shared with EvalDash.
"""

import secrets
from collections.abc import Mapping
from datetime import UTC, datetime

from evalrecords.model_feedback import COREWEAVE_FEEDBACK_PREFIX, GCS_FEEDBACK_PREFIX, FeedbackRecord, FeedbackReport
from rigging.filesystem.cluster_config import marin_prefix
from rigging.filesystem.storage_path import StoragePath

FEEDBACK_ID_BYTES = 6


def model_feedback_prefix() -> str | None:
    """Return the feedback prefix for this worker's cloud, or None outside cloud storage."""
    scheme = StoragePath(marin_prefix()).scheme
    if scheme == "s3":
        return COREWEAVE_FEEDBACK_PREFIX
    if scheme == "gs":
        return GCS_FEEDBACK_PREFIX
    return None


def save_feedback_report(prefix: str, report: FeedbackReport, serving: Mapping[str, object]) -> str:
    """Write one report with the serving metadata that produced it; return its storage URL."""
    created = datetime.now(UTC)
    record = FeedbackRecord(
        id=f"{created:%H%M%S}-{secrets.token_hex(FEEDBACK_ID_BYTES)}",
        created_at=created,
        serving=dict(serving),
        report=report,
    )
    location = StoragePath.parse(prefix) / f"{created:%Y-%m-%d}/{record.id}.json"
    location.parent.mkdirs()
    location.write_text(record.model_dump_json(indent=2), encoding="utf-8")
    return str(location)

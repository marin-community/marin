# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Model-feedback reports from the serve dashboard, saved as JSON objects in Marin storage.

Each report is one object under ``<prefix>/<YYYY-MM-DD>/``. Like eval records, reports from
CoreWeave workers go to a CoreWeave bucket, because those workers hold CoreWeave S3 credentials
but no GCP ones; reports from GCP workers go to GCS.
"""

import json
import secrets
from collections.abc import Mapping
from datetime import UTC, datetime

from rigging.filesystem.cluster_config import marin_prefix
from rigging.filesystem.storage_path import StoragePath

GCS_FEEDBACK_PREFIX = "gs://marin-eval-metadata/model-feedback"
COREWEAVE_FEEDBACK_PREFIX = "s3://marin-us-east-02a/marin/model-feedback"
FEEDBACK_ID_BYTES = 6
FEEDBACK_REPORT_VERSION = 1


def model_feedback_prefix() -> str | None:
    """Return the feedback prefix for this worker's cloud, or None outside cloud storage."""
    scheme = StoragePath(marin_prefix()).scheme
    if scheme == "s3":
        return COREWEAVE_FEEDBACK_PREFIX
    if scheme == "gs":
        return GCS_FEEDBACK_PREFIX
    return None


def save_feedback_report(prefix: str, report: Mapping[str, object], serving: Mapping[str, object]) -> str:
    """Write one report with the serving metadata that produced it; return its storage URL."""
    created = datetime.now(UTC)
    report_id = f"{created:%H%M%S}-{secrets.token_hex(FEEDBACK_ID_BYTES)}"
    location = StoragePath.parse(prefix) / f"{created:%Y-%m-%d}/{report_id}.json"
    record = {
        "version": FEEDBACK_REPORT_VERSION,
        "id": report_id,
        "created_at": created.isoformat(),
        "serving": serving,
        "report": report,
    }
    location.parent.mkdirs()
    location.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(location)

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Model-feedback reports from the serve dashboard, saved as JSON objects in Marin storage.

Each report is one object under ``<prefix>/<YYYY-MM-DD>/``. Like eval records, reports from
CoreWeave workers go to a CoreWeave bucket, because those workers hold CoreWeave S3 credentials
but no GCP ones; reports from GCP workers go to GCS.
"""

import asyncio
import json
import logging
import posixpath
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from rigging.filesystem.cluster_config import marin_prefix
from rigging.filesystem.factory import url_to_fs
from rigging.filesystem.storage_path import StoragePath
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

GCS_FEEDBACK_PREFIX = "gs://marin-eval-metadata/model-feedback"
COREWEAVE_FEEDBACK_PREFIX = "s3://marin-us-east-02a/marin/model-feedback"
MAX_FEEDBACK_BYTES = 1024 * 1024
# Per serving process; bounds what one dashboard URL holder can write.
MAX_FEEDBACK_REPORTS = 256
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


@dataclass
class FeedbackStore:
    """Writes dashboard feedback reports, with the serving metadata, below ``prefix``."""

    prefix: str
    serving: dict[str, Any]
    max_bytes: int = MAX_FEEDBACK_BYTES
    max_reports: int = MAX_FEEDBACK_REPORTS
    saved: int = 0

    async def create_response(self, request: Request) -> Response:
        if self.saved >= self.max_reports:
            return JSONResponse({"error": "this server has reached its feedback report limit"}, status_code=429)
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > self.max_bytes:
                return JSONResponse({"error": "feedback report is too large"}, status_code=413)
        try:
            report = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return JSONResponse({"error": "feedback report must be valid JSON"}, status_code=400)
        if not isinstance(report, dict) or not str(report.get("description", "")).strip():
            return JSONResponse({"error": "feedback report must be an object with a description"}, status_code=400)

        created = datetime.now(UTC)
        report_id = f"{created:%H%M%S}-{secrets.token_hex(FEEDBACK_ID_BYTES)}"
        location = f"{self.prefix}/{created:%Y-%m-%d}/{report_id}.json"
        record = {
            "version": FEEDBACK_REPORT_VERSION,
            "id": report_id,
            "created_at": created.isoformat(),
            "serving": self.serving,
            "report": report,
        }
        self.saved += 1
        try:
            await asyncio.to_thread(_write_json, location, record)
        except Exception:
            self.saved -= 1
            logger.exception("Could not save feedback report to %s", location)
            return JSONResponse({"error": "could not save the feedback report"}, status_code=502)
        logger.info("Saved feedback report %s", location)
        return JSONResponse({"id": report_id, "location": location}, status_code=201)


def _write_json(location: str, record: dict[str, Any]) -> None:
    fs, path = url_to_fs(location)
    fs.makedirs(posixpath.dirname(path), exist_ok=True)
    with fs.open(path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)

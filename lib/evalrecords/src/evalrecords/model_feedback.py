# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Model-feedback reports: what the serve dashboard's Report issue button saves and EvalDash reads.

Each report is one immutable JSON object under ``{prefix}/{YYYY-MM-DD}/{id}.json``. Like eval records,
reports from CoreWeave workers go to a CoreWeave bucket, because those workers hold CoreWeave S3
credentials but no GCP ones; reports from GCP workers go to GCS.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

GCS_FEEDBACK_PREFIX = "gs://marin-eval-metadata/model-feedback"
COREWEAVE_FEEDBACK_PREFIX = "s3://marin-us-east-02a/marin/model-feedback"
FEEDBACK_PREFIXES = (GCS_FEEDBACK_PREFIX, COREWEAVE_FEEDBACK_PREFIX)


class FeedbackReport(BaseModel):
    """What the reporter entered, plus the conversation when they included it."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    description: str
    reporter: str = ""
    model: str
    conversation: dict[str, object] | None


class FeedbackRecord(BaseModel):
    """One saved report with the serving metadata of the dashboard that produced it."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    version: Literal[1] = 1
    id: str
    created_at: datetime
    serving: dict[str, object]
    report: FeedbackReport

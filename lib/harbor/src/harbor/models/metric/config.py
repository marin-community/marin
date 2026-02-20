# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from pydantic import BaseModel, Field

from harbor.models.metric.type import MetricType


class MetricConfig(BaseModel):
    type: MetricType = Field(default=MetricType.MEAN)
    kwargs: dict[str, Any] = Field(default_factory=dict)

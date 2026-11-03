from typing import Any

from pydantic import BaseModel, Field

from harbor_config.models.metric.type import MetricType


class MetricConfig(BaseModel):
    type: MetricType = Field(default=MetricType.MEAN)
    kwargs: dict[str, Any] = Field(default_factory=dict)

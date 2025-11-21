# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Common types for inference requests and responses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class InferenceRequest:
    """Request for inference."""

    prompt: str
    sampling_params: dict[str, Any] | None = None
    request_id: str | None = None


@dataclass
class InferenceResult:
    """Result of inference."""

    text: list[str]
    request_id: str | None = None
    error: str | None = None

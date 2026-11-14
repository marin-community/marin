# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Diagnostics for harness-owned cleanup requests, without credentials or headers."""

import json
import logging
from dataclasses import asdict, dataclass

from daytona import DaytonaError
from rigging.redaction import redact_string

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CleanupFailure:
    operation: str
    request: dict[str, str | int]
    error_type: str
    message: str
    status_code: int | None
    error_code: str | None


class DaytonaCleanupError(RuntimeError):
    """Preserve the provider cause and the exact credential-free control request."""

    def __init__(self, operation: str, request: dict[str, str | int], error: DaytonaError):
        self.failure = CleanupFailure(
            operation, request, type(error).__name__, redact_string(str(error)), error.status_code, error.error_code
        )
        super().__init__(json.dumps(asdict(self.failure), sort_keys=True))
        logger.error("Daytona cleanup request failed: %s", self)

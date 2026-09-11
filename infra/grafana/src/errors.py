# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Errors the bridge surfaces to Grafana."""


class UpstreamError(Exception):
    """An upstream (controller, GitHub, W&B) failed, or lacks what was asked for.

    Surfaced with the source named and `status_code` (a 5xx for a failure, 404 for
    an absent object), so an Infinity panel renders an error rather than empty or
    stale data.
    """

    def __init__(self, source: str, message: str, *, status_code: int = 502) -> None:
        self.source = source
        self.status_code = status_code
        super().__init__(message)

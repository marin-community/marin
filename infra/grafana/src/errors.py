# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Errors the bridge surfaces to Grafana."""


class UpstreamError(Exception):
    """An upstream (controller, GitHub) failed.

    Surfaced as an HTTP 5xx with the source named, so an Infinity panel renders an
    error rather than empty or stale data. The failed call is never cached.
    """

    def __init__(self, source: str, message: str, *, status_code: int = 502) -> None:
        self.source = source
        self.status_code = status_code
        super().__init__(message)

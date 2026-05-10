# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris-side logging helpers.

Proto-level log conversions live in ``finelog.types``; this module re-exports
them so existing iris callers do not need to know about the move.
"""

from finelog.types import str_to_log_level

__all__ = ["str_to_log_level"]

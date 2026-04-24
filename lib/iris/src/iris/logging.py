# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Proto-dependent log level converters.

General-purpose logging utilities (configure_logging, LevelPrefixFormatter,
LogRingBuffer, etc.) have moved to ``rigging.log_setup``.  This module
retains only the functions that depend on ``iris.rpc.logging_pb2``.
"""

from iris.rpc import logging_pb2

_STR_TO_ENUM = {
    "DEBUG": logging_pb2.LOG_LEVEL_DEBUG,
    "INFO": logging_pb2.LOG_LEVEL_INFO,
    "WARNING": logging_pb2.LOG_LEVEL_WARNING,
    "ERROR": logging_pb2.LOG_LEVEL_ERROR,
    "CRITICAL": logging_pb2.LOG_LEVEL_CRITICAL,
}


def str_to_log_level(level_name: str | None) -> int:
    """Convert a canonical level name (e.g. "INFO") to the LogLevel proto enum value.

    Returns ``LOG_LEVEL_UNKNOWN`` (0) for ``None``, empty strings, and
    unrecognized names.
    """
    if not level_name:
        return logging_pb2.LOG_LEVEL_UNKNOWN
    return _STR_TO_ENUM.get(level_name.upper(), logging_pb2.LOG_LEVEL_UNKNOWN)

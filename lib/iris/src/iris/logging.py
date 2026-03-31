# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Proto-dependent log level converters.

General-purpose logging utilities (configure_logging, LevelPrefixFormatter,
LogRingBuffer, etc.) have moved to ``rigging.log_setup``.  This module
retains only the functions that depend on ``iris.rpc.logging_pb2``.
"""


def str_to_log_level(level_name: str) -> int:
    """Convert a canonical level name (e.g. "INFO") to the LogLevel proto enum value.

    Returns LOG_LEVEL_UNKNOWN (0) for unrecognized names.
    Uses lazy import to avoid pulling in protobuf at module load time.
    """
    from iris.rpc import logging_pb2

    _STR_TO_ENUM = {
        "DEBUG": logging_pb2.LOG_LEVEL_DEBUG,
        "INFO": logging_pb2.LOG_LEVEL_INFO,
        "WARNING": logging_pb2.LOG_LEVEL_WARNING,
        "ERROR": logging_pb2.LOG_LEVEL_ERROR,
        "CRITICAL": logging_pb2.LOG_LEVEL_CRITICAL,
    }
    return (
        _STR_TO_ENUM.get(level_name.upper(), logging_pb2.LOG_LEVEL_UNKNOWN)
        if level_name
        else logging_pb2.LOG_LEVEL_UNKNOWN
    )


def log_level_to_str(level: int) -> str:
    """Convert a LogLevel proto enum value to canonical level name.

    Returns "" for LOG_LEVEL_UNKNOWN (0).
    """
    from iris.rpc import logging_pb2

    _ENUM_TO_STR = {
        logging_pb2.LOG_LEVEL_DEBUG: "DEBUG",
        logging_pb2.LOG_LEVEL_INFO: "INFO",
        logging_pb2.LOG_LEVEL_WARNING: "WARNING",
        logging_pb2.LOG_LEVEL_ERROR: "ERROR",
        logging_pb2.LOG_LEVEL_CRITICAL: "CRITICAL",
    }
    return _ENUM_TO_STR.get(level, "")

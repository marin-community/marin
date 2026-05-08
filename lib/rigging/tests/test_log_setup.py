# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for ``rigging.log_setup``.

The interpreter-shutdown test guards against #5578: late log emissions
(e.g. from ``tqdm.__del__``) used to crash with ``AttributeError: 'NoneType'
object has no attribute 'get'`` once Python had cleared module globals.
"""

import logging

from rigging import log_setup
from rigging.log_setup import LOG_DATEFMT, LOG_FORMAT, LevelPrefixFormatter


def _make_record(level: int = logging.INFO, name: str = "test") -> logging.LogRecord:
    return logging.LogRecord(name, level, __file__, 1, "msg", None, None)


def test_format_emits_level_prefix():
    formatter = LevelPrefixFormatter(fmt=LOG_FORMAT, datefmt=LOG_DATEFMT)
    out = formatter.format(_make_record(logging.INFO))
    assert out.startswith("I")
    assert " msg" in out


def test_format_unknown_level_uses_question_mark():
    formatter = LevelPrefixFormatter(fmt=LOG_FORMAT, datefmt=LOG_DATEFMT)
    record = _make_record(logging.INFO)
    record.levelname = "NOPE"
    out = formatter.format(record)
    assert out.startswith("?")


def test_format_survives_module_global_teardown(monkeypatch):
    """Formatter must work after module globals are cleared at shutdown."""
    formatter = LevelPrefixFormatter(fmt=LOG_FORMAT, datefmt=LOG_DATEFMT)
    monkeypatch.setattr(log_setup, "_LEVEL_PREFIX", None)
    out = formatter.format(_make_record(logging.WARNING))
    assert out.startswith("W")

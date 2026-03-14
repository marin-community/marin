# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import logging
import time
from datetime import timedelta
from collections.abc import Iterator

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def log_time(label: str, level: int = logging.INFO) -> Iterator[None]:
    t_start = time.perf_counter()
    yield
    t_end = time.perf_counter()
    logger.log(level, f"{label} took {timedelta(seconds=t_end - t_start)}")

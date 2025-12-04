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

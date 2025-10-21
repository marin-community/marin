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

import logging
import time
from functools import wraps

logger = logging.getLogger(__name__)


def with_retries(max_retries=3, delay=2.0):
    """
    Decorator to retry a function call if it raises an exception.
    Args:
        max_retries:
        delay:

    Returns:

    """

    def decorator(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    logger.warning(f"{fn.__name__} failed on attempt {attempt + 1}, retrying: {e}")
                    time.sleep(delay)

            raise RuntimeError(f"{fn.__name__} failed after {max_retries} attempts")

        return wrapped

    return decorator

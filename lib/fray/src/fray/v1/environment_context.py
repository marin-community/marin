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

"""Utilities for temporarily modifying environment variables."""

import contextlib
import os
from collections.abc import Iterator


@contextlib.contextmanager
def temporary_env_vars(env_vars: dict[str, str]) -> Iterator[None]:
    """Context manager to temporarily set environment variables.

    Args:
        env_vars: Dictionary of environment variable names to values to set.

    On exit, restores the original environment.
    """
    old_env = {k: os.environ.get(k) for k in env_vars}
    os.environ.update(env_vars)
    try:
        yield
    finally:
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

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

"""Monarch backend for Fray distributed execution framework."""

from .monarch_cluster import MonarchClusterContext
from .monarch_helpers import MONARCH_AVAILABLE
from .monarch_job import MonarchJobContext, get_job_context, set_job_context

__all__ = [
    "MONARCH_AVAILABLE",
    "MonarchClusterContext",
    "MonarchJobContext",
    "get_job_context",
    "set_job_context",
]

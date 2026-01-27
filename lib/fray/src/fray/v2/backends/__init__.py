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

"""Backend implementations for Fray v2."""

from fray.v2.backends.local import LocalCluster

# Ray backend is optional (requires ray)
try:
    from fray.v2.backends.ray import RayCluster
except ImportError:
    RayCluster = None  # type: ignore[misc,assignment]

# Iris backend is optional (requires iris)
try:
    from fray.v2.backends.iris import IrisCluster
except ImportError:
    IrisCluster = None  # type: ignore[misc,assignment]

__all__ = ["IrisCluster", "LocalCluster", "RayCluster"]

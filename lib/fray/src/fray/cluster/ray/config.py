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

import os


def find_config_by_region(region: str) -> str:
    """Find cluster config file by region name or path."""
    if region.endswith(".yaml"):
        return region

    # Strip marin- prefix if present
    if region.startswith("marin-"):
        region = region[len("marin-") :]

    # Try common variations
    variations = [
        f"infra/marin-{region}.yaml",
        f"infra/marin-{region}-a.yaml",
        f"infra/marin-{region}-b.yaml",
        f"infra/marin-{region}-vllm.yaml",
    ]

    for path in variations:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(f"No cluster config found for region {region}")

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

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

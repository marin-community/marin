# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os


def find_config_by_region(region: str, infra_dir: str | None = None) -> str:
    """Find cluster config file by region name or path.

    Args:
        region: Cluster region name (e.g., "us-central2")
        infra_dir: Path to infra directory. If None, uses default resolution:
                   1. MARIN_INFRA_DIR env var
                   2. "infra/" in CWD
    """
    if region.endswith(".yaml"):
        return region

    if infra_dir is None:
        infra_dir = os.environ.get("MARIN_INFRA_DIR", "infra")

    # Strip marin- prefix if present
    if region.startswith("marin-"):
        region = region[len("marin-") :]

    # Try common variations
    variations = [
        f"{infra_dir}/marin-{region}.yaml",
        f"{infra_dir}/marin-{region}-a.yaml",
        f"{infra_dir}/marin-{region}-b.yaml",
        f"{infra_dir}/marin-{region}-vllm.yaml",
    ]

    for path in variations:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(f"No cluster config found for region {region}")

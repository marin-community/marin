# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical scratch-expiration rules shared by every bucket provider."""

from dataclasses import dataclass

from rigging.filesystem.cluster_config import DataConfig
from rigging.filesystem.storage_path import prefix_join


@dataclass(frozen=True)
class ExpirationRule:
    id: str
    prefix: str
    days: int


def expiration_rules(data_config: DataConfig) -> tuple[ExpirationRule, ...]:
    return tuple(
        ExpirationRule(
            id=f"marin-ttl-{ttl_days}d",
            prefix=f"{prefix_join(data_config.temp_path, f'ttl={ttl_days}d')}/",
            days=ttl_days,
        )
        for ttl_days in data_config.ttl_days
    )

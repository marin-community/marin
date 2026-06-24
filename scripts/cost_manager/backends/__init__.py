# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Cost backends: one fetcher per provider, keyed by provider name.

Each backend exposes ``fetch(config, window) -> list[CostEvent]`` where
``config`` is the provider's block from ``config.yaml`` and secrets are read
from the environment. Register a new provider by adding its module and an entry
to :data:`BACKENDS`.
"""

from collections.abc import Callable, Mapping
from typing import Any

from scripts.cost_manager.backends import anthropic, coreweave, gcp, openai, together
from scripts.cost_manager.cost_event import CostEvent, DateWindow

ProviderFetcher = Callable[[Mapping[str, Any], DateWindow], list[CostEvent]]

BACKENDS: dict[str, ProviderFetcher] = {
    "openai": openai.fetch,
    "anthropic": anthropic.fetch,
    "gcp": gcp.fetch,
    "coreweave": coreweave.fetch,
    "together": together.fetch,
}

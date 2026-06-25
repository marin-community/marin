# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Together AI costs — scaffold pending a programmatic cost API.

Together bills per token for inference (plus dedicated endpoints, storage, and
fine-tuning), but exposes spend **only** through the cookie-authenticated web
dashboard (Billing -> Usage, including the "Current Usage" draft invoice). The
``TOGETHER_API_KEY`` bearer token reaches the inference surface (``/v1/models``,
``/v1/endpoints``, ...) and nothing under billing/usage, so there is no endpoint
for the runner to pull dollar cost from.

This backend is therefore a disabled-by-default scaffold: it is registered and
carried in ``config.yaml`` so the provider slots in the moment Together ships a
usage/cost API, but until then ``fetch`` raises :class:`CostFetchError` rather
than guess at a schema.
"""

from collections.abc import Mapping
from typing import Any

from scripts.cost_manager.cost_event import CostEvent, CostFetchError, DateWindow

PROVIDER = "together"


def fetch(config: Mapping[str, Any], window: DateWindow) -> list[CostEvent]:
    raise CostFetchError(
        "together: no programmatic cost API. Spend is reachable only via the "
        "cookie-authenticated billing dashboard, not the TOGETHER_API_KEY bearer "
        "surface, so there is nothing to pull. Keep this provider disabled until "
        "Together ships a usage/cost endpoint."
    )

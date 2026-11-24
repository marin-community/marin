# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The lane's process boundary: clients, machines and resources.

Mirrors ``taskforge.wiring.runtime`` with one deliberate difference: the generator's
``GlmClient`` is built with ``limit_output=True`` so the per-band output cap reaches the
wire (the taskforge runtime omits token limits and leaves capacity to the endpoint). The
lane generates through one client; grading runs on machines without a model.
"""

import logging
import os
from collections.abc import AsyncIterator, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import replace
from pathlib import Path

import taskforge.policy
from iris.cluster.client import get_job_info
from pydantic import TypeAdapter
from shellbox.backends.iris.machine import IrisMachineFactory
from shellbox.machine import MachineFactory
from taskforge.llm.glm_client import EndpointSource, GlmClient
from taskforge.llm.policy import LLMProviderConfig
from taskforge.policy import Policy, load_policy
from taskforge.resources.bundle import Resources
from taskforge.resources.provider import WaitingBackend
from taskforge.resources.sandbox.factories import SANDBOX_SCHEDULING_TIMEOUT, shellsim_factory
from taskforge.resources.web import FakeWeb
from taskforge.wiring import RelayEndpoint, build_resources, endpoint_source, provider_policy

logger = logging.getLogger(__name__)

_GENERATOR_ROLE = "author"
"""The role slot the generator occupies in the resources bundle; the lane has no author."""

LANE_GENERATOR_MODEL = "glm-5.3"
"""The generator the lane samples: the Ortet relay's GLM-5.3. The taskforge defaults pin
``author.model`` to deepseek-v4.1-flash for authoring workpieces; the lane overrides it."""


def lane_policy(overlay: Mapping[str, object]) -> Policy:
    """The taskforge defaults overlaid with the lane's keys (bulk pool, lane sampling)."""
    defaults_dir = Path(taskforge.policy.__file__).parent / "defaults"
    return load_policy(defaults_dir).overlaid(dict(overlay))


def lane_factories(policy: Policy) -> dict[str, MachineFactory]:
    """Machine factories for verifier and sandbox machines, from an Iris job context."""
    job = get_job_info()
    controller = os.environ.get("IRIS_CONTROLLER_URL") or (job.controller_address if job else None)
    if not controller:
        raise ValueError("Lane grading machines require a controller URL or an Iris job context")
    container = IrisMachineFactory(controller_url=controller, scheduling_timeout=SANDBOX_SCHEDULING_TIMEOUT)
    return {
        container.backend.value: container,
        "shellsim": shellsim_factory(output_budget_mb=policy.setting("sandbox.output_budget_mb", int)),
    }


@asynccontextmanager
async def lane_runtime(
    policy: Policy, source: EndpointSource
) -> AsyncIterator[tuple[Resources, Mapping[str, MachineFactory]]]:
    """Open the lane's clients and machines; yields the resources bundle and factories.

    The generator's client is built with ``limit_output=True`` so ``LLMPolicy.max_tokens``
    is sent with every request and a band's cap is enforced by the endpoint.
    """
    policy = provider_policy(policy)
    provider_fields = {
        key.removeprefix("provider."): value
        for key, value in policy.entries.items()
        if key.startswith("provider.") and key not in ("provider.unavailable_wait", "provider.pool")
    }
    provider_fields.update(max_attempts=1, hold_timeout=0)
    factories = lane_factories(policy)
    async with AsyncExitStack() as owned:
        model = policy.setting(f"{_GENERATOR_ROLE}.model", str)
        provider = TypeAdapter(LLMProviderConfig).validate_python({**provider_fields, "model": model})
        generator_source = replace(source, model=model) if isinstance(source, RelayEndpoint) else source
        endpoint = generator_source.current()
        logging.info(
            "MODEL role=generator model=%s endpoint=%s pool=%s",
            model,
            endpoint.base_url,
            endpoint.pool.value,
        )
        client = await owned.enter_async_context(GlmClient(generator_source, provider, limit_output=True))
        backend = WaitingBackend(
            client,
            policy.setting("provider.unavailable_wait", float),
            provider.backoff_initial,
            provider.backoff_maximum,
            provider.backoff_factor,
        )
        resources = build_resources(
            policy,
            {_GENERATOR_ROLE: backend, "solver": backend},
            factories,
            FakeWeb({}, {}),
        )
        yield resources, factories


def relay_or_fixed_source(base_url: str | None, relay_job: str | None, token: str, pool: str = "bulk") -> EndpointSource:
    """The generator's endpoint source; the bulk pool is the lane's default."""
    return endpoint_source(base_url, relay_job, token, model="glm-5.3", pool=pool)


__all__ = ["lane_factories", "lane_policy", "lane_runtime", "relay_or_fixed_source"]

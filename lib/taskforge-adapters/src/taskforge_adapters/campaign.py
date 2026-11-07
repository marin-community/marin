# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reusable worker-side configuration for Taskforge campaigns on Iris."""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from shellbox.machine import Backend
from taskforge.ledger.records import DiscardingLedger
from taskforge.llm.agent import ReasoningReplay
from taskforge.llm.policy import LLMPolicy, LLMProviderConfig, ReasoningEffort
from taskforge.sandbox.factories import MachineHost
from taskforge.stage40_build_with_program.services import AgentLimits, RLEnvBuildServices

from taskforge_adapters.backends import machine_factories
from taskforge_adapters.glm_client import (
    API_ROOT,
    EndpointSource,
    FixedEndpoint,
    GlmClient,
    GlmEndpoint,
    Pool,
    RelayEndpoint,
)

IRIS_CONTROLLER_URL_ENV = "IRIS_CONTROLLER_URL"


@dataclass(frozen=True)
class OpenAIModelPort:
    """Non-secret model endpoint configuration safe to serialize to workers."""

    model: str
    api_key_env: str
    base_url: str | None = None
    relay_job: str | None = None
    relay_service: str | None = None
    pool: Pool = Pool.BULK

    def __post_init__(self) -> None:
        if (self.base_url is None) == (self.relay_job is None):
            raise ValueError("exactly one of base_url and relay_job is required")
        if self.base_url is not None and not self.base_url.endswith(API_ROOT):
            raise ValueError(f"base_url must end in {API_ROOT}")
        if self.relay_job is not None and not self.relay_service:
            raise ValueError("relay_service is required with relay_job")
        if not self.model or not self.api_key_env:
            raise ValueError("model and api_key_env must be nonempty")


def endpoint_source(port: OpenAIModelPort, token: str) -> EndpointSource:
    """Build the refreshable endpoint source described by ``port``."""
    if port.base_url is not None:
        return FixedEndpoint(GlmEndpoint(port.base_url, token, port.pool))
    assert port.relay_job is not None and port.relay_service is not None
    return RelayEndpoint(port.relay_job, token, port.pool, endpoint_name=port.relay_service)


def provider_config(port: OpenAIModelPort) -> LLMProviderConfig:
    """Return the bounded retry policy shared by Taskforge campaign workers."""
    return LLMProviderConfig(
        model=port.model,
        max_attempts=4,
        hold_timeout=900,
        backoff_initial=1,
        backoff_maximum=30,
        backoff_factor=2,
        stall_timeout=300,
        max_connections=16,
        structured_repairs=1,
    )


def campaign_policy() -> LLMPolicy:
    """Return the default authoring policy for campaign model calls."""
    return LLMPolicy(
        max_tokens=4096,
        reasoning_effort=ReasoningEffort.LOW,
        temperature=0.7,
        top_p=0.95,
        max_continuations=0,
    )


def iris_build_services(
    controller_url: str,
    *,
    backends: tuple[Backend, ...] = (Backend.SHELLSIM, Backend.GVISOR),
) -> RLEnvBuildServices:
    """Create build services for one Taskforge lifecycle running inside Iris."""
    return RLEnvBuildServices(
        host=MachineHost.IRIS,
        factories=machine_factories(backends, controller_url),
        images=None,
        web_tools=(),
        ledger=DiscardingLedger(),
        reasoning_replay=ReasoningReplay.KEEP,
        agent_limits=AgentLimits(max_turns=4, max_tokens=4096, max_calls=8),
    )


@dataclass(frozen=True)
class CampaignWorker:
    """Model and Taskforge runtime objects resolved from one worker's environment."""

    model: GlmClient
    policy: LLMPolicy
    services: RLEnvBuildServices
    endpoint: EndpointSource
    token: str


@asynccontextmanager
async def campaign_worker(port: OpenAIModelPort) -> AsyncIterator[CampaignWorker]:
    """Resolve credentials and Iris services, then open one worker-local model client."""
    token = os.environ.get(port.api_key_env)
    if not token:
        raise RuntimeError(f"worker lacks required API credential environment variable {port.api_key_env}")
    controller_url = os.environ.get(IRIS_CONTROLLER_URL_ENV)
    if not controller_url:
        raise RuntimeError(f"worker lacks {IRIS_CONTROLLER_URL_ENV}; this runtime must run inside an Iris task")
    source = endpoint_source(port, token)
    async with GlmClient(source, provider_config(port)) as model:
        yield CampaignWorker(model, campaign_policy(), iris_build_services(controller_url), source, token)

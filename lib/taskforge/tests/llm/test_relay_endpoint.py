# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import pytest
from iris.actor.resolver import ResolvedEndpoint, ResolveResult
from iris.client.client import IrisClient, IrisContext, iris_ctx_scope
from iris.cluster.types import JobName

from taskforge.llm.glm_client import GLM_MODEL, GlmEndpoint, Pool, RelayEndpoint

RELAY_JOB = "/someone/glm53-relay"
TOKEN = "bulk-token"


@dataclass(frozen=True)
class FakeResolver:
    urls: Mapping[tuple[str, str], tuple[str, ...]]
    job: JobName

    def resolve(self, name: str) -> ResolveResult:
        urls = self.urls.get((str(self.job), name), ())
        return ResolveResult(name, [ResolvedEndpoint(url, actor_id=f"{self.job}/{name}") for url in urls])


@dataclass(frozen=True)
class FakeIrisClient:
    """Answers ``resolver_for_job`` with the endpoint URLs registered under each (job, endpoint name)."""

    urls: Mapping[tuple[str, str], tuple[str, ...]]

    def resolver_for_job(self, job: JobName) -> FakeResolver:
        return FakeResolver(self.urls, job)


def iris_context(client: FakeIrisClient) -> IrisContext:
    return IrisContext(job_id=None, client=cast(IrisClient, client))


def test_relay_endpoint_raises_when_the_relay_registered_no_glm_endpoint():
    other_job = FakeIrisClient({("/someone/other-relay", GLM_MODEL): ("http://10.0.0.5:8000",)})

    with iris_ctx_scope(iris_context(other_job)), pytest.raises(RuntimeError, match="no registered endpoint"):
        RelayEndpoint(RELAY_JOB, TOKEN, Pool.BULK).current()


@pytest.mark.parametrize("url", ["http://10.0.0.5:8000", "http://10.0.0.5:8000/", "http://10.0.0.5:8000/v1/"])
def test_relay_endpoint_resolves_the_relay_glm_endpoint_to_its_api_root(url: str):
    client = FakeIrisClient({(RELAY_JOB, GLM_MODEL): (url, "http://10.0.0.6:8000")})

    with iris_ctx_scope(iris_context(client)):
        endpoint = RelayEndpoint(RELAY_JOB, TOKEN, Pool.BULK).current()

    assert endpoint == GlmEndpoint("http://10.0.0.5:8000/v1", TOKEN, Pool.BULK)

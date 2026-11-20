# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Qualify the regional retrieval service and its thin Daytona RL actor."""

import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path

import httpx
from fray.types import ResourceConfig, create_environment
from marin.rl.retrieval_service import RetrievalServiceConfig, RetrievalServiceLaunch, retrieval_service
from pydantic import BaseModel, ConfigDict, Field
from rigging.filesystem.storage_path import StoragePath
from rigging.secrets import SecretBehavior, require_secret
from rigging.timing import Duration
from rolloutengine.harness.retrieval_service import OpenResearcherServiceProvider
from rolloutengine.harness.shell import ToolContext
from shellbox.backends.daytona.connection import DEFAULT_DAYTONA_API_URL, DaytonaConnection
from shellbox.backends.daytona.machine import DaytonaMachineFactory
from shellbox.image import RegistryImage
from shellbox.machine import MachineSpec, NetworkPolicy
from taskcompendium.models import ProviderRequirement
from taskcompendium.openresearcher import OPENRESEARCHER_ACTION_INTERFACE, OpenResearcherState

QA_SEARCH_QUERY = "quantum physics"


class DocumentProbe(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    docid: str
    bytes: int
    sha256: str
    serialized_output_bytes: int
    serialized_output_sha256: str


class ServiceResources(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    cpu: int
    ram: str
    disk: str
    preemptible: bool
    image: str


class QualificationConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    state: OpenResearcherState
    index_uri: str
    region: str
    concurrency: int = Field(gt=0)
    actor_image: str
    resources: ServiceResources
    documents: tuple[DocumentProbe, ...]


class QualificationResult(BaseModel):
    ready_elapsed: float
    concurrent_elapsed: float
    concurrency: int
    documents: tuple[DocumentProbe, ...]
    actor_image: str
    network: str
    daytona_org: str
    index_manifest_sha256: str
    passed: bool


async def concurrent_search(endpoint: str, config: QualificationConfig) -> float:
    """Return elapsed time for the declared concurrent source-contract search batch."""
    body = {"name": "search", "arguments": {"query": QA_SEARCH_QUERY}, "state": config.state.model_dump(mode="json")}
    async with httpx.AsyncClient(timeout=300, trust_env=False) as client:
        started = time.monotonic()
        responses = await asyncio.gather(
            *(client.post(endpoint + "/search", json=body) for _ in range(config.concurrency))
        )
        for response in responses:
            response.raise_for_status()
            if len(response.json()) != config.state.top_k:
                raise ValueError("Search did not return the source top-five contract")
        return time.monotonic() - started


async def actor_retrieval(endpoint: str, config: QualificationConfig) -> tuple[DocumentProbe, ...]:
    connection = DaytonaConnection(
        api_key=require_secret("DAYTONA_RL_API_KEY"), api_url=DEFAULT_DAYTONA_API_URL, target="us"
    )
    factory = DaytonaMachineFactory(connection.client_factory())
    machine = await factory.create(
        MachineSpec(
            source=RegistryImage(config.actor_image),
            network=NetworkPolicy.DENY,
            cpus=1,
            memory_mb=2048,
            storage_mb=2048,
            startup_timeout=600,
        )
    )
    try:
        requirement = ProviderRequirement(
            action_interface=OPENRESEARCHER_ACTION_INTERFACE,
            network_access="service_only",
            initial_state=config.state.model_dump(mode="json"),
        )
        session = await OpenResearcherServiceProvider(endpoint).open(ToolContext(machine, requirement))
        try:
            hits = json.loads((await session.call("search", {"query": QA_SEARCH_QUERY})).content)
            if len(hits) != config.state.top_k:
                raise ValueError("Relay search did not preserve top-five results")
            documents = []
            for expected in config.documents:
                result = await session.call("get_document", {"docid": expected.docid})
                document = json.loads(result.content)
                content = document["text"].encode()
                serialized = result.content.encode()
                actual = DocumentProbe(
                    docid=document["docid"],
                    bytes=len(content),
                    sha256=hashlib.sha256(content).hexdigest(),
                    serialized_output_bytes=len(serialized),
                    serialized_output_sha256=hashlib.sha256(serialized).hexdigest(),
                )
                if actual != expected:
                    raise ValueError("Relay document differs from the pinned corpus probe")
                documents.append(actual)
            return tuple(documents)
        finally:
            await session.close()
    finally:
        await machine.close()


def qualify(config: QualificationConfig) -> QualificationResult:
    service = RetrievalServiceConfig(
        state=config.state,
        index_uri=config.index_uri,
        region=config.region,
        root="/opt/openresearcher",
        backend_python="/usr/local/bin/python",
        max_concurrent_requests=config.concurrency,
        endpoint_name="openresearcher-retrieval",
    )
    environment = create_environment(
        sync_packages=["marin-core"],
        extras=["cpu"],
        env_vars={"MARIN_CLUSTER": "coreweave", "JAVA_HOME": "/opt/java21", "JDK_JAVA_OPTIONS": "-Xmx2048m"},
        secret_behavior=SecretBehavior.ISOLATE,
    )
    launch = RetrievalServiceLaunch(
        config=service,
        resources=ResourceConfig(**config.resources.model_dump()),
        environment=environment,
        startup_timeout=3600,
        lifetime=Duration.from_hours(3),
    )
    started = time.monotonic()
    with retrieval_service(launch) as endpoint:
        ready_elapsed = time.monotonic() - started
        concurrent_elapsed = asyncio.run(concurrent_search(endpoint, config))
        documents = asyncio.run(actor_retrieval(endpoint, config))
        return QualificationResult(
            ready_elapsed=ready_elapsed,
            concurrent_elapsed=concurrent_elapsed,
            concurrency=config.concurrency,
            documents=documents,
            actor_image=config.actor_image,
            network="deny",
            daytona_org="RL",
            index_manifest_sha256=config.state.index_manifest_sha256,
            passed=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = qualify(QualificationConfig.model_validate_json(args.config.read_bytes()))
    StoragePath(args.output).write_text(result.model_dump_json(indent=2) + "\n")
    print(result.model_dump_json(), flush=True)


if __name__ == "__main__":
    main()

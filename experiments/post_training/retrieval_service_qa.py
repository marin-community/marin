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


def qualify(config: dict) -> dict:
    state = OpenResearcherState.model_validate(config["state"])
    service = RetrievalServiceConfig(
        state=state,
        index_uri=config["index_uri"],
        root="/opt/openresearcher",
        backend_python="/usr/local/bin/python",
        max_concurrent_requests=config["concurrency"],
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
        resources=ResourceConfig(**config["resources"]),
        environment=environment,
        startup_timeout=3600,
        lifetime=Duration.from_hours(3),
    )
    started = time.monotonic()
    with retrieval_service(launch) as endpoint:
        ready_elapsed = time.monotonic() - started

        async def check() -> dict:
            body = {"name": "search", "arguments": {"query": "quantum physics"}, "state": state.model_dump(mode="json")}
            async with httpx.AsyncClient(timeout=300, trust_env=False) as client:
                request_started = time.monotonic()
                responses = await asyncio.gather(
                    *(client.post(endpoint + "/search", json=body) for _ in range(config["concurrency"]))
                )
                for response in responses:
                    response.raise_for_status()
                    if len(response.json()) != 5:
                        raise ValueError("Search did not return the source top-five contract")
                concurrent_elapsed = time.monotonic() - request_started
            connection = DaytonaConnection(
                api_key=require_secret("DAYTONA_RL_API_KEY"),
                api_url=DEFAULT_DAYTONA_API_URL,
                target="us",
            )
            factory = DaytonaMachineFactory(connection.client_factory())
            machine = await factory.create(
                MachineSpec(
                    source=RegistryImage(config["actor_image"]),
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
                    initial_state=state.model_dump(mode="json"),
                )
                session = await OpenResearcherServiceProvider(endpoint).open(ToolContext(machine, requirement))
                try:
                    hits = json.loads((await session.call("search", {"query": "quantum physics"})).content)
                    if len(hits) != 5:
                        raise ValueError("Relay search did not preserve top-five results")
                    documents = []
                    for expected in config["documents"]:
                        result = await session.call("get_document", {"docid": expected["docid"]})
                        document = json.loads(result.content)
                        content = document["text"].encode()
                        actual = {
                            "docid": document["docid"],
                            "bytes": len(content),
                            "sha256": hashlib.sha256(content).hexdigest(),
                        }
                        if actual != {key: expected[key] for key in actual}:
                            raise ValueError("Relay document differs from the pinned corpus probe")
                        documents.append(actual)
                    return {
                        "ready_elapsed": ready_elapsed,
                        "concurrent_elapsed": concurrent_elapsed,
                        "concurrency": config["concurrency"],
                        "documents": documents,
                        "actor_image": config["actor_image"],
                        "network": "deny",
                        "daytona_org": "RL",
                        "index_manifest_sha256": state.index_manifest_sha256,
                        "passed": True,
                    }
                finally:
                    await session.close()
            finally:
                await machine.close()

        return asyncio.run(check())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = qualify(json.loads(args.config.read_bytes()))
    StoragePath(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

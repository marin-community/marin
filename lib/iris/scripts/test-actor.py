#!/usr/bin/env python3
# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Token-passing actor demo for Iris.

Demonstrates actor-to-actor communication by creating 4 actors that pass
a token around for N rounds, with each actor randomly selecting the next recipient.

Uses a queue-based pattern: send_token() enqueues and returns immediately,
while a worker thread processes tokens. This avoids blocking call chains.

Standalone usage (bootstraps local cluster):
    uv run python scripts/test-actor.py --local --rounds 5

Job submission usage:
    uv run iris submit scripts/test-actor.py --controller-url http://localhost:10000 -- 5
"""

import queue
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import click

from iris.actor import ActorClient, ActorServer
from iris.client import iris_ctx


@dataclass
class Token:
    """Token passed between actors."""

    round_num: int
    sender_id: str
    path: list[str] = field(default_factory=list)

    def add_hop(self, actor_id: str):
        """Record this actor in the path."""
        self.path.append(actor_id)


class CollectorActor:
    """Actor that collects completion notification."""

    def __init__(self):
        self.result: str | None = None
        self._done = threading.Event()

    def notify_complete(self, result: str) -> str:
        """Called when token passing completes."""
        self.result = result
        self._done.set()
        return "ack"

    def wait(self, timeout: float = 60.0) -> str | None:
        """Wait for completion and return result."""
        self._done.wait(timeout=timeout)
        return self.result


class TokenPassingActor:
    """Actor that receives tokens and passes them to another random actor.

    Uses queue-based pattern: send_token() enqueues and returns immediately,
    while a worker thread processes tokens. This avoids blocking call chains.
    """

    def __init__(
        self,
        actor_id: str,
        all_actor_names: list[str],
        max_rounds: int,
        resolver,
        delay: float = 0.5,
    ):
        self.actor_id = actor_id
        self.all_actor_names = [n for n in all_actor_names if n != actor_id]  # Exclude self
        self.max_rounds = max_rounds
        self.tokens_received = 0
        self._resolver = resolver
        self._delay = delay
        self._queue: queue.Queue[Token] = queue.Queue()
        self._worker = threading.Thread(target=self._process_loop, daemon=True)
        self._worker.start()

    def send_token(self, token: Token) -> str:
        """Queue token for processing - returns immediately."""
        self._queue.put(token)
        return "queued"

    def _process_loop(self):
        """Worker thread processes tokens from queue."""
        while True:
            token = self._queue.get()
            try:
                self._handle_token(token)
            except Exception as e:
                print(f"[{self.actor_id}] Error processing token: {e}")

    def _handle_token(self, token: Token):
        """Process a single token."""
        self.tokens_received += 1
        token.add_hop(self.actor_id)

        print(
            f"[{self.actor_id}] Round {token.round_num}/{self.max_rounds}: "
            f"Received token from {token.sender_id} "
            f"(path: {' -> '.join(token.path[-5:])}...)"
        )  # Show last 5 hops

        # Check if we've completed all rounds
        if token.round_num >= self.max_rounds:
            result = f"Complete: {self.actor_id} received token after {len(token.path)} hops"
            print(f"[{self.actor_id}] Token passing complete! Total hops: {len(token.path)}")
            # Notify collector
            collector = ActorClient(self._resolver, "collector", resolve_timeout=30.0)
            collector.notify_complete(result)
            return

        # Configurable delay to simulate work
        if self._delay > 0:
            time.sleep(self._delay)

        # Pick random next actor and send
        next_actor_name = random.choice(self.all_actor_names)
        next_client = ActorClient(self._resolver, next_actor_name, resolve_timeout=300.0)
        token.round_num += 1
        token.sender_id = self.actor_id
        next_client.send_token(token)


def main(rounds: int = 5, delay: float = 0.5):
    """Run token-passing demo with N rounds.

    Args:
        rounds: Number of rounds to pass the token (default: 5)
        delay: Delay in seconds between token passes (default: 0.5)
    """
    print(f"Starting token-passing demo with {rounds} rounds (delay={delay}s)...")

    # Get execution context
    ctx = iris_ctx()
    print(f"Running in job: {ctx.job_id}")
    print(f"Namespace: {ctx.namespace}")

    # Get resolver with explicit namespace - needed for use in actor threads
    # which don't have access to the IrisContext
    resolver = ctx.client.resolver_for_job(ctx.job_id)

    # Define actor names
    actor_names = ["actor1", "actor2", "actor3", "actor4"]

    # Start collector actor first
    collector = CollectorActor()
    collector_server = ActorServer(host="0.0.0.0")
    collector_server.register("collector", collector)
    collector_port = collector_server.serve_background()
    ctx.registry.register(
        name="collector",
        address=f"http://0.0.0.0:{collector_port}",
        metadata={"type": "collector"},
    )
    print(f"Started collector on port {collector_port}")

    # Start token-passing actor servers
    servers = []
    for name in actor_names:
        server = ActorServer(host="0.0.0.0")
        actor = TokenPassingActor(name, actor_names, rounds, resolver, delay=delay)
        actor_id = server.register(name, actor)
        actual_port = server.serve_background()

        ctx.registry.register(
            name=name,
            address=f"http://0.0.0.0:{actual_port}",
            metadata={"actor_id": actor_id, "type": "token-passer"},
        )

        print(f"Started {name} on port {actual_port}")
        servers.append((name, server, actual_port))

    # Give actors time to register
    time.sleep(0.5)

    # Kick off token passing - send_token returns immediately
    print("\nInitiating token passing...")
    initial_token = Token(round_num=1, sender_id="initiator", path=["initiator"])

    client = ActorClient(resolver, "actor1", resolve_timeout=300.0)
    client.send_token(initial_token)

    # Wait for completion via collector
    result = collector.wait(timeout=120.0)

    if result:
        print(f"\n✓ Token passing completed: {result}")
        print(f"✓ All {rounds} rounds finished successfully")
    else:
        print("\n✗ Token passing timed out!")
        raise RuntimeError("Token passing did not complete")

    # Brief pause for logs to flush
    time.sleep(0.5)


def run_local(rounds: int = 5, delay: float = 0.5) -> bool:
    """Run the test with a local in-process cluster.

    Bootstraps a DemoCluster and submits the main() function as a job.
    Returns True if the job succeeded, False otherwise.
    """
    import sys

    # Import DemoCluster components (only when running locally)
    from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec
    from iris.rpc import cluster_pb2

    # Import from examples - adjust path since we're in scripts/
    sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
    from demo_cluster import DemoCluster

    print(f"Starting local cluster for actor test with {rounds} rounds (delay={delay}s)...")

    with DemoCluster(workspace=Path(__file__).parent.parent) as demo:
        print(f"Controller: {demo.controller_url}")
        print()

        entrypoint = Entrypoint.from_callable(main, rounds, delay)
        job = demo.client.submit(
            entrypoint=entrypoint,
            name="test-actor",
            resources=ResourceSpec(cpu=1, memory="512m"),
            environment=EnvironmentSpec(),
        )

        print(f"Submitted job: {job.job_id}")
        print("Waiting for completion...")
        print("-" * 80)

        status = job.wait(timeout=120, stream_logs=True, raise_on_failure=False)

        print("-" * 80)
        state_name = cluster_pb2.JobState.Name(status.state)
        print(f"Job {job.job_id}: {state_name}")

        if status.state == cluster_pb2.JOB_STATE_SUCCEEDED:
            print("Test passed!")
            return True
        else:
            print(f"Test failed: {status.error}")
            return False


@click.command()
@click.option("--local", is_flag=True, help="Bootstrap a local cluster for testing")
@click.option("--rounds", default=5, help="Number of token-passing rounds")
@click.option("--delay", default=0.5, help="Delay in seconds between token passes")
@click.argument("rounds_positional", required=False, type=int)
def cli(local: bool, rounds: int, delay: float, rounds_positional: int | None):
    """Token-passing actor test."""
    # Positional arg takes precedence (for job submission compatibility)
    actual_rounds = rounds_positional if rounds_positional is not None else rounds

    if local:
        success = run_local(actual_rounds, delay)
        raise SystemExit(0 if success else 1)
    else:
        # When run as a job, main() will use iris_ctx() for context
        main(actual_rounds, delay)


if __name__ == "__main__":
    cli()

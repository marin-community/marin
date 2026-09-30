# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The full path over real HTTP: SiloClient -> broker -> host -> (fake) runtime.

Parameters are built in the shapes the pipeline passes today -- the Daytona SDK's
CreateSnapshotParams / CreateSandboxFromSnapshotParams / SessionExecuteRequest --
and results are checked with the pipeline's own vendored contract functions.
This is brief section 7 step 1, run against the provider as the pipeline sees it.
"""

import concurrent.futures
import time
from types import SimpleNamespace

import pytest
from silo.auth import HEADER_SANDBOX_TOKEN
from silo.broker.core import Broker
from silo.broker.server import build_app as build_broker_app
from silo.broker.server import host_client_factory
from silo.client import SiloClient
from silo.errors import SiloError
from silo.host.agent import HostAgent, HostConfig
from silo.host.server import build_app as build_host_app
from silo.http import HttpClient
from silo.pipeline_contract.daytona_snapshot import (
    resource_not_found,
    snapshot_conflict,
    snapshot_not_found,
    validate_snapshot_recipe,
    wait_for_sandbox_deletion,
    wait_for_snapshot_active,
)
from silo_testing import wait_until

DIGEST = "docker.io/library/alpine@sha256:" + "a" * 64
RECIPE = f'FROM {DIGEST}\nENTRYPOINT ["/bin/sh"]\n'
GB = 1024**3


class DaytonaImage:
    """What daytona.Image.from_dockerfile(path) produces: text read eagerly."""

    def __init__(self, text: str) -> None:
        self._dockerfile = text

    def dockerfile(self) -> str:
        return self._dockerfile


def snapshot_params(name: str, recipe: str = RECIPE, cpu: int = 4, memory: int = 8, disk: int = 10):
    return SimpleNamespace(
        name=name, image=DaytonaImage(recipe), resources=SimpleNamespace(cpu=cpu, memory=memory, disk=disk)
    )


def sandbox_params(snapshot: str, **overrides):
    fields = dict(
        snapshot=snapshot,
        labels={"envgen": "1", "envgen_purpose": "harbor-trial"},
        ephemeral=True,
        auto_stop_interval=0,
        ttl_minutes=180,
        network_block_all=True,
        network_allow_list=None,
        domain_allow_list=None,
        env_vars=None,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.fixture
def stack(tmp_path, fake_runtime, serve):
    secret, token = "host-secret", "api-token"
    agent = HostAgent(HostConfig("h1", 16, 64 * GB, 200 * GB, tuple(range(16)), tmp_path / "work"), fake_runtime)
    host = serve(build_host_app(agent, secret))
    broker = Broker(host_secret=secret, host_client_factory=host_client_factory(secret))
    broker_server = serve(
        build_broker_app(broker, api_token=token, host_secret=secret, self_url=lambda: broker_server.url)
    )
    broker.heartbeat("h1", host.url, agent.capacity(), [])
    client = SiloClient(broker_server.url, token)
    return SimpleNamespace(
        client=client, agent=agent, broker=broker, host=host, broker_server=broker_server, runtime=fake_runtime
    )


def test_snapshot_lifecycle_through_the_pipelines_own_checks(stack):
    client = stack.client
    record = client.snapshot.create(snapshot_params("cap-harbor-e2e"), timeout=30)
    assert record.state == "active"

    # wait_for_snapshot_active resolves it and validates the recipe bytes.
    resolved = wait_for_snapshot_active(client.snapshot.get, "cap-harbor-e2e", RECIPE, sleeper=lambda _: None)
    assert resolved.name == "cap-harbor-e2e"
    assert validate_snapshot_recipe(resolved, expected_name="cap-harbor-e2e", expected_dockerfile=RECIPE)

    # Attributes dt.py and snapshot_identity read.
    for attr in (
        "id",
        "name",
        "state",
        "image_name",
        "ref",
        "size",
        "cpu",
        "mem",
        "disk",
        "created_at",
        "organization_id",
        "general",
    ):
        assert hasattr(resolved, attr), attr
    assert resolved.build_info.dockerfile_content == RECIPE

    listing = client.snapshot.list(page=1, limit=100)
    assert [s.name for s in listing.items] == ["cap-harbor-e2e"]
    client.snapshot.delete(resolved)
    with pytest.raises(SiloError) as info:
        client.snapshot.get("cap-harbor-e2e")
    assert snapshot_not_found(info.value)


def test_duplicate_snapshot_create_is_a_409_over_the_wire(stack):
    stack.client.snapshot.create(snapshot_params("dup"), timeout=30)
    with pytest.raises(SiloError) as info:
        stack.client.snapshot.create(snapshot_params("dup"), timeout=30)
    assert snapshot_conflict(info.value)


def test_trial_shaped_sandbox_session(stack):
    client = stack.client
    client.snapshot.create(snapshot_params("cap-harbor-t"), timeout=30)
    sandbox = client.create(sandbox_params("cap-harbor-t"), timeout=60)

    # The attestation several gates read back.
    assert client.get(sandbox.id).network_block_all is True
    assert (sandbox.cpu, sandbox.memory, sandbox.disk) == (4, 8, 10)

    # One-shot exec, as _run_portable uses it for <= 240 s commands.
    response = sandbox.process.exec("/bin/sh -c 'echo hi; exit 2'", cwd="/", env={"A": "1"}, timeout=30)
    assert (response.exit_code, response.result.strip()) == (2, "hi")

    # The long-command path, byte for byte in the shape daytona_environment uses.
    session_id = "cap-harbor-0123456789ab"
    sandbox.process.create_session(session_id)
    launched = sandbox.process.execute_session_command(
        session_id, SimpleNamespace(command="sleep 0.3; echo built; exit 0", run_async=True), timeout=60
    )
    wait_until(lambda: sandbox.process.get_session_command(session_id, launched.cmd_id).exit_code is not None)
    logs = sandbox.process.get_session_command_logs(session_id, launched.cmd_id)
    assert logs.stdout == "built\n" and logs.stderr == ""
    sandbox.process.delete_session(session_id)

    # Files.
    sandbox.fs.upload_file(b"\x00\x01payload", "/input/payload.json")
    assert sandbox.fs.download_file("/input/payload.json") == b"\x00\x01payload"


def test_deletion_as_the_pipeline_observes_it(stack):
    client = stack.client
    client.snapshot.create(snapshot_params("cap-harbor-d"), timeout=30)
    sandbox = client.create(sandbox_params("cap-harbor-d"), timeout=60)
    stack.runtime.remove_gate.clear()
    sandbox.delete()

    def sleeper(_):
        stack.runtime.remove_gate.set()
        wait_until(lambda: sandbox.id not in {r.id for r in stack.agent.list_sandboxes()})

    state, observations = wait_for_sandbox_deletion(client, sandbox.id, sleeper=sleeper)
    assert state == "not_found"
    assert [o["state"] for o in observations] == ["present", "not_found"]


def test_every_trial_gets_a_fresh_unique_sandbox(stack):
    # brief section 3.2: 3 attempts x N controls, reused_sandbox_ids == [].
    client = stack.client
    client.snapshot.create(snapshot_params("cap-harbor-u", cpu=2, memory=1), timeout=30)
    ids = []
    for _ in range(3):
        sandbox = client.create(sandbox_params("cap-harbor-u"), timeout=60)
        ids.append(sandbox.id)
        sandbox.delete()
    assert len(set(ids)) == 3


@pytest.mark.parametrize(
    "overrides",
    [
        {"network_block_all": False},
        {"domain_allow_list": ["pypi.org"]},
        {"network_allow_list": "10.0.0.0/8"},
        {"env_vars": {"A": "1"}},
    ],
)
def test_create_refuses_isolation_it_cannot_honour(stack, overrides):
    stack.client.snapshot.create(snapshot_params("cap-harbor-n"), timeout=30)
    with pytest.raises(SiloError) as info:
        stack.client.create(sandbox_params("cap-harbor-n", **overrides))
    assert info.value.status_code == 400
    assert stack.agent.list_sandboxes() == []


def test_missing_sandbox_is_a_sandbox_not_found_over_the_wire(stack):
    with pytest.raises(SiloError) as info:
        stack.client.get("slbdoesnotexist")
    assert resource_not_found(info.value, "sandbox")
    assert not snapshot_not_found(info.value)


def test_sandbox_capability_is_scoped_to_one_sandbox(stack):
    client = stack.client
    client.snapshot.create(snapshot_params("cap-harbor-c", cpu=2, memory=1), timeout=30)
    a = client.create(sandbox_params("cap-harbor-c"))
    b = client.create(sandbox_params("cap-harbor-c"))
    token_a = stack.broker.get_sandbox(a.id)["token"]
    stolen = HttpClient(stack.host.url, headers={HEADER_SANDBOX_TOKEN: token_a})
    stolen.post(f"/sandboxes/{a.id}/exec", {"command": "true"})  # its own: fine
    with pytest.raises(SiloError) as info:
        stolen.post(f"/sandboxes/{b.id}/exec", {"command": "true"})
    assert info.value.status_code == 401
    # And no capability at all cannot create anything on the host.
    with pytest.raises(SiloError) as info:
        HttpClient(stack.host.url).post("/sandboxes", {})
    assert info.value.status_code == 401


def test_broker_rejects_a_wrong_api_token(stack):
    with pytest.raises(SiloError) as info:
        SiloClient(stack.broker_server.url, "wrong").capacity()
    assert info.value.status_code == 401


def test_capacity_is_visible_to_the_caller(stack):
    ceiling = stack.client.capacity()
    assert ceiling["hosts_live"] == 1
    assert ceiling["slots_free"]["candidate_4cpu_8gb"] == 4
    assert ceiling["snapshot_quota"] is None


def test_client_re_resolves_the_broker_after_a_restart(stack, serve):
    # A worker holding a dead broker address asks the proxy-reachable /whoami
    # for the current one, instead of being stranded (the relay-cut failure).
    live = stack.broker_server.url
    resolver = serve(
        build_broker_app(stack.broker, api_token="api-token", host_secret="host-secret", self_url=lambda: live)
    )
    client = SiloClient("http://127.0.0.1:1", "api-token", resolve_url=resolver.url)
    assert client.capacity()["hosts_live"] == 1


def test_waiting_creates_never_block_deletes_or_capacity(tmp_path, fake_runtime, serve):
    """Regression for width test 01: the broker jammed itself at capacity.

    One slot, filled. Sixty creates wait for it -- more than the 40-thread pool
    the broker used to wait on. A capacity read and a delete must still answer
    promptly, and the freed slot must go to one of the waiters.
    """
    secret, token = "host-secret", "api-token"
    agent = HostAgent(HostConfig("h1", 4, 8 * GB, 10 * GB, tuple(range(4)), tmp_path / "work"), fake_runtime)
    host = serve(build_host_app(agent, secret))
    broker = Broker(host_secret=secret, host_client_factory=host_client_factory(secret))
    broker_server = serve(
        build_broker_app(broker, api_token=token, host_secret=secret, self_url=lambda: broker_server.url)
    )
    broker.heartbeat("h1", host.url, agent.capacity(), [])
    client = SiloClient(broker_server.url, token)
    client.snapshot.create(snapshot_params("one-slot"), timeout=30)
    first = client.create(sandbox_params("one-slot"))

    with concurrent.futures.ThreadPoolExecutor(60) as pool:
        waiters = [pool.submit(client.create, sandbox_params("one-slot"), 8) for _ in range(60)]
        # More waiters parked than the old 40-thread pool could hold.
        wait_until(lambda: client.capacity()["creates_waiting_for_capacity"] > 40)
        started = time.monotonic()
        client.delete(first)
        assert time.monotonic() - started < 5, "capacity/delete queued behind waiting creates"

        winner = None
        for future in concurrent.futures.as_completed(waiters, timeout=60):
            try:
                winner = future.result()
                break
            except SiloError as error:
                assert error.status_code == 429
        assert winner is not None, "the freed slot was never handed to a waiter"
        winner.delete()
        for future in waiters:
            try:
                future.result(timeout=60)
            except SiloError:
                pass

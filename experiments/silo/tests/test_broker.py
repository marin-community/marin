# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import threading

import pytest
from silo.auth import sandbox_capability
from silo.broker.core import Broker, local_image_tag
from silo.errors import SiloConflictError, SiloError, SiloNotFoundError, SiloRateLimitError, SiloRecipeError
from silo.http import TransportError
from silo.model import CANDIDATE_DEFAULT, VERIFIER_DEFAULT
from silo.pipeline_contract.daytona_snapshot import resource_not_found, snapshot_not_found, validate_snapshot_recipe
from silo_testing import wait_until

DIGEST = "docker.io/library/alpine@sha256:" + "a" * 64
FROM_ONLY = f"FROM {DIGEST}\n"
WITH_RUN = f"FROM {DIGEST}\nUSER root\nRUN echo build\n"
GB = 1024**3


class FakeHost:
    def __init__(self):
        self.created: list[dict] = []
        self.deleted: list[str] = []
        self.pulled: list[str] = []
        self.built: list[str] = []
        self.gone: set[str] = set()
        self.report = None
        self.full = False
        self.gone_away = False

    def create_sandbox(self, body):
        if self.gone_away:
            raise TransportError("connection refused", refused=True)
        if self.full:
            raise SiloRateLimitError("host full")
        self.created.append(dict(body))
        return {
            "id": body["sandbox_id"],
            "snapshot": body["snapshot_name"],
            "state": "started",
            "cpu": body["profile"]["cpu"],
            "memory": body["profile"]["memory_gb"],
            "disk": body["profile"]["disk_gb"],
            "network_block_all": True,
            "labels": body["labels"],
        }

    def get_sandbox(self, sandbox_id):
        if sandbox_id in self.gone:
            raise SiloNotFoundError("sandbox", sandbox_id)
        return {"id": sandbox_id, "state": "started"}

    def delete_sandbox(self, sandbox_id):
        self.deleted.append(sandbox_id)
        self.gone.add(sandbox_id)

    def ensure_image(self, ref):
        self.pulled.append(ref)

    def build_image(self, tag, dockerfile):
        self.built.append(tag)

    def capacity(self):
        # The broker's direct refresh; tests set `report` when they want one.
        if self.report is None:
            raise RuntimeError("no report configured")
        return self.report


def capacity(cpu=16, mem_gb=64, disk_gb=200, allocated_cpu=0, allocated_mem=0, allocated_disk=0):
    return {
        "sandboxes_live": 0,
        "cpu": {"budget": cpu, "oversubscribe": 1.0, "allocated": allocated_cpu},
        "memory_bytes": {"budget": mem_gb * GB, "allocated": allocated_mem},
        "disk_bytes": {"budget": disk_gb * GB, "allocated": allocated_disk},
    }


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def make_world(sleep=None):
    hosts: dict[str, FakeHost] = {}

    def factory(url):
        return hosts.setdefault(url, FakeHost())

    clock = Clock()
    broker = Broker(
        host_secret="s3cret",
        host_client_factory=factory,
        clock=clock,
        sleep=sleep or clock.sleep,
        placement_wait_seconds=60,
    )
    return broker, hosts, clock


@pytest.fixture
def world():
    return make_world()


def active(broker, name):
    wait_until(lambda: broker.get_snapshot(name).state in ("active", "error"))
    return broker.get_snapshot(name)


# -- snapshots ------------------------------------------------------------------


def test_from_only_snapshot_goes_active_with_no_build(world):
    broker, hosts, _ = world
    broker.heartbeat("h1", "http://h1", capacity(), [])
    broker.create_snapshot("cap-harbor-1", FROM_ONLY, CANDIDATE_DEFAULT)
    record = active(broker, "cap-harbor-1")
    assert record.state == "active"
    assert record.resolved_ref == DIGEST
    assert hosts["http://h1"].pulled == [DIGEST]
    assert hosts["http://h1"].built == []


def test_run_recipe_builds_to_a_host_local_tag_never_a_registry(world):
    broker, hosts, _ = world
    broker.heartbeat("h1", "http://h1", capacity(), [])
    broker.create_snapshot("cap-verifier-1", WITH_RUN, VERIFIER_DEFAULT)
    record = active(broker, "cap-verifier-1")
    assert record.resolved_ref == local_image_tag("cap-verifier-1")
    assert record.resolved_ref.startswith("silo.local/")
    assert hosts["http://h1"].built == [record.resolved_ref]


def test_snapshot_echoes_the_exact_recipe_for_the_pipelines_identity_check(world):
    broker, _, _ = world
    broker.heartbeat("h1", "http://h1", capacity(), [])
    broker.create_snapshot("cap-harbor-1", FROM_ONLY, CANDIDATE_DEFAULT)
    record = active(broker, "cap-harbor-1")
    evidence = validate_snapshot_recipe(record, expected_name="cap-harbor-1", expected_dockerfile=FROM_ONLY)
    assert evidence["evidence"] == "provider-build-info-exact-match"
    # Name reuse with different content fails closed in the pipeline's check.
    with pytest.raises(Exception, match="does not match"):
        validate_snapshot_recipe(record, expected_name="cap-harbor-1", expected_dockerfile=WITH_RUN)


def test_duplicate_name_is_a_conflict(world):
    broker, _, _ = world
    broker.create_snapshot("cap-harbor-1", FROM_ONLY, CANDIDATE_DEFAULT)
    with pytest.raises(SiloConflictError):
        broker.create_snapshot("cap-harbor-1", FROM_ONLY, CANDIDATE_DEFAULT)


def test_unknown_snapshot_is_a_snapshot_not_found(world):
    broker, _, _ = world
    with pytest.raises(SiloNotFoundError) as info:
        broker.get_snapshot("cap-harbor-missing")
    assert snapshot_not_found(info.value)


def test_unsupported_recipe_is_rejected_at_create(world):
    broker, _, _ = world
    with pytest.raises(SiloRecipeError):
        broker.create_snapshot("x", f"FROM {DIGEST}\nCOPY . /\n", CANDIDATE_DEFAULT)


def test_snapshot_waits_in_a_retryable_state_until_a_host_appears():
    # The materializer's host wait must genuinely wait: with the fake clock's
    # instant sleep it would burn its 240 s bound before the host arrives. So it
    # sleeps on an event the heartbeat sets.
    host_arrived = threading.Event()
    broker, _, _ = make_world(sleep=lambda _seconds: host_arrived.wait(0.05))
    broker.create_snapshot("cap-harbor-1", FROM_ONLY, CANDIDATE_DEFAULT)
    assert broker.get_snapshot("cap-harbor-1").state == "pending"
    broker.heartbeat("h1", "http://h1", capacity(), [])
    host_arrived.set()
    assert active(broker, "cap-harbor-1").state == "active"


def test_there_is_no_snapshot_quota(world):
    broker, _, _ = world
    broker.heartbeat("h1", "http://h1", capacity(), [])
    for i in range(200):  # five times Daytona's org-wide 40
        broker.create_snapshot(f"cap-harbor-{i}", FROM_ONLY, CANDIDATE_DEFAULT)
    assert broker.capacity()["snapshots"] == 200
    assert broker.capacity()["snapshot_quota"] is None


# -- sandboxes ------------------------------------------------------------------


def test_create_returns_host_address_and_a_per_sandbox_capability(world):
    broker, hosts, _ = world
    broker.heartbeat("h1", "http://h1", capacity(), [])
    broker.create_snapshot("s", FROM_ONLY, CANDIDATE_DEFAULT)
    active(broker, "s")
    created = broker.create_sandbox(snapshot="s", labels={"envgen_purpose": "harbor-trial"})
    assert created["host_url"] == "http://h1"
    assert created["token"] == sandbox_capability("s3cret", created["id"])
    assert hosts["http://h1"].created[0]["image_ref"] == DIGEST


def test_networked_sandboxes_are_refused(world):
    broker, _, _ = world
    with pytest.raises(SiloError, match="network-blocked"):
        broker.create_sandbox(snapshot="s", network_block_all=False)


def test_placement_prefers_a_host_that_already_has_the_image(world):
    broker, _, _ = world
    # The snapshot pre-pulls on the roomiest live host, which is h1 here...
    broker.heartbeat("h1", "http://h1", capacity(cpu=16, mem_gb=64), [])
    broker.create_snapshot("s", FROM_ONLY, CANDIDATE_DEFAULT)
    active(broker, "s")
    # ...then a roomier h2 arrives without the image. The image wins.
    broker.heartbeat("h2", "http://h2", capacity(cpu=64, mem_gb=256), [])
    assert broker.create_sandbox(snapshot="s")["host_url"] == "http://h1"


def test_back_to_back_creates_do_not_pile_onto_one_slot(world):
    broker, _, _ = world
    broker.heartbeat("h1", "http://h1", capacity(cpu=4, mem_gb=8, disk_gb=10), [])
    broker.heartbeat("h2", "http://h2", capacity(cpu=4, mem_gb=8, disk_gb=10), [])
    broker.create_snapshot("s", FROM_ONLY, CANDIDATE_DEFAULT)
    active(broker, "s")
    urls = {broker.create_sandbox(snapshot="s")["host_url"] for _ in range(2)}
    assert urls == {"http://h1", "http://h2"}


def test_at_capacity_the_create_waits_then_rate_limits(world):
    broker, _, clock = world
    broker.heartbeat("h1", "http://h1", capacity(cpu=4, mem_gb=8, disk_gb=10, allocated_cpu=4), [])
    broker.create_snapshot("s", FROM_ONLY, CANDIDATE_DEFAULT)
    active(broker, "s")
    with pytest.raises(SiloRateLimitError):
        broker.create_sandbox(snapshot="s", timeout=10)
    assert clock.now >= 10  # it waited rather than failing fast


def test_a_freed_slot_is_taken_while_waiting():
    freed = []

    def sleep_then_free(seconds):
        # Whatever the create is waiting on, the host frees its slot meanwhile.
        clock.sleep(seconds)
        if not freed:
            freed.append(True)
            broker.heartbeat("h1", "http://h1", capacity(cpu=4, mem_gb=8, disk_gb=10), [])

    broker, _, clock = make_world(sleep=lambda seconds: sleep_then_free(seconds))
    full = capacity(cpu=4, mem_gb=8, disk_gb=10, allocated_cpu=4, allocated_mem=8 * GB, allocated_disk=10 * GB)
    broker.heartbeat("h1", "http://h1", full, [])
    broker.create_snapshot("s", FROM_ONLY, CANDIDATE_DEFAULT)
    active(broker, "s")
    assert broker.create_sandbox(snapshot="s", timeout=60)["host_url"] == "http://h1"


def test_capacity_reports_the_named_ceiling(world):
    broker, _, _ = world
    broker.heartbeat("h1", "http://h1", capacity(cpu=96, mem_gb=384, disk_gb=1024), [])
    broker.heartbeat("h2", "http://h2", capacity(cpu=96, mem_gb=384, disk_gb=1024), [])
    ceiling = broker.capacity()
    assert ceiling["hosts_live"] == 2
    assert ceiling["slots_free"]["candidate_4cpu_8gb"] == 48
    assert ceiling["slots_free"]["verifier_2cpu_1gb"] == 96


def test_sandbox_on_a_silent_host_is_503_until_the_host_is_presumed_dead(world):
    # Missing heartbeats make a host SUSPECT: its sandbox is most likely still
    # running, so the answer is "retry", never "not found" (2026-09-29: live
    # sandboxes were reported gone because the broker itself was too slow to
    # take heartbeats). Only a long silence makes it DEAD and its sandboxes gone.
    broker, _, clock = world
    broker.heartbeat("h1", "http://h1", capacity(), [])
    broker.create_snapshot("s", FROM_ONLY, CANDIDATE_DEFAULT)
    active(broker, "s")
    created = broker.create_sandbox(snapshot="s")
    clock.now += broker.config.suspect_after_seconds + 1
    with pytest.raises(SiloError) as info:
        broker.get_sandbox(created["id"])
    assert info.value.status_code == 503
    assert not resource_not_found(info.value, "sandbox")
    clock.now += broker.config.dead_after_seconds
    with pytest.raises(SiloNotFoundError, match="sandbox"):
        broker.get_sandbox(created["id"])


def test_heartbeats_rebuild_routing_after_a_broker_restart(world):
    broker, _, _ = world
    broker.heartbeat("h1", "http://h1", capacity(), ["slbsurvivor"])
    assert broker.get_sandbox("slbsurvivor")["host_url"] == "http://h1"


def test_deleted_sandbox_stays_not_found(world):
    broker, _, _ = world
    broker.heartbeat("h1", "http://h1", capacity(), [])
    broker.create_snapshot("s", FROM_ONLY, CANDIDATE_DEFAULT)
    active(broker, "s")
    created = broker.create_sandbox(snapshot="s")
    broker.delete_sandbox(created["id"])
    for _ in range(2):
        with pytest.raises(SiloNotFoundError):
            broker.get_sandbox(created["id"])


# -- accounting under churn ------------------------------------------------------


def test_a_sandbox_created_and_deleted_between_reports_does_not_leak_capacity(world):
    broker, _, _ = world
    broker.heartbeat("h1", "http://h1", capacity(cpu=4, mem_gb=8, disk_gb=10), [])
    broker.create_snapshot("s", FROM_ONLY, CANDIDATE_DEFAULT)
    active(broker, "s")
    created = broker.create_sandbox(snapshot="s")
    assert broker.capacity()["slots_free"]["candidate_4cpu_8gb"] == 0
    broker.delete_sandbox(created["id"])  # no heartbeat in between
    assert broker.capacity()["slots_free"]["candidate_4cpu_8gb"] == 1


def test_a_report_that_includes_a_placement_stops_double_counting_it(world):
    broker, _, _ = world
    broker.heartbeat("h1", "http://h1", capacity(cpu=8, mem_gb=16, disk_gb=20), [])
    broker.create_snapshot("s", FROM_ONLY, CANDIDATE_DEFAULT)
    active(broker, "s")
    created = broker.create_sandbox(snapshot="s")
    # The host's next report counts it itself (allocated), and lists its id.
    broker.heartbeat(
        "h1",
        "http://h1",
        capacity(cpu=8, mem_gb=16, disk_gb=20, allocated_cpu=4, allocated_mem=8 * GB, allocated_disk=10 * GB),
        [created["id"]],
    )
    assert broker.capacity()["slots_free"]["candidate_4cpu_8gb"] == 1  # not 0


def test_unconfirmed_placements_expire_rather_than_leak(world):
    broker, _, clock = world
    broker.heartbeat("h1", "http://h1", capacity(cpu=4, mem_gb=8, disk_gb=10), [])
    broker.create_snapshot("s", FROM_ONLY, CANDIDATE_DEFAULT)
    active(broker, "s")
    broker.create_sandbox(snapshot="s")
    clock.now += 61  # past UNCONFIRMED_TTL_SECONDS, and still unreported
    broker.heartbeat("h1", "http://h1", capacity(cpu=4, mem_gb=8, disk_gb=10), [])
    assert broker.capacity()["slots_free"]["candidate_4cpu_8gb"] == 1


def test_a_host_that_refuses_as_full_is_refreshed_and_another_takes_the_create(world):
    broker, hosts, _ = world
    broker.heartbeat("h1", "http://h1", capacity(cpu=64, mem_gb=256), [])
    broker.heartbeat("h2", "http://h2", capacity(cpu=4, mem_gb=8, disk_gb=10), [])
    broker.create_snapshot("s", FROM_ONLY, CANDIDATE_DEFAULT)
    active(broker, "s")
    # h1 looks roomiest but is really full; its refresh says so.
    hosts["http://h1"].full = True
    hosts["http://h1"].report = capacity(cpu=64, mem_gb=256, allocated_cpu=64, allocated_mem=256 * GB)
    assert broker.create_sandbox(snapshot="s")["host_url"] == "http://h2"


def test_a_host_that_vanished_inside_its_heartbeat_window_is_skipped(world):
    broker, hosts, _ = world
    broker.heartbeat("h1", "http://h1", capacity(cpu=64, mem_gb=256), [])
    broker.heartbeat("h2", "http://h2", capacity(cpu=4, mem_gb=8, disk_gb=10), [])
    broker.create_snapshot("s", FROM_ONLY, CANDIDATE_DEFAULT)
    active(broker, "s")
    hosts["http://h1"].gone_away = True  # cancelled, still "live" by heartbeat age
    assert broker.create_sandbox(snapshot="s")["host_url"] == "http://h2"
    assert broker.capacity()["hosts_live"] == 1

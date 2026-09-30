# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Host liveness, quarantine and persistence in the broker (the 2026-09-29 incident).

Every host's heartbeat timed out for 8 hours: the heartbeat handler shared one
thread pool with creates that hung for 600 s on "sick" hosts, and the hosts were
sick because their capacity accounting had drifted negative, which made the
broker send them almost everything. These tests pin each piece of the fix.
"""

import logging
import threading

import pytest
from silo.broker.core import (
    Broker,
    BrokerConfig,
    HostState,
    local_image_tag,
    report_inconsistency,
    snapshot_state_from_api,
)
from silo.broker.state import FileStateStore, StatePersister, restore
from silo.errors import SiloError, SiloNotFoundError, SiloRateLimitError
from silo.http import TransportError
from silo.model import CANDIDATE_DEFAULT, VERIFIER_DEFAULT
from silo.pipeline_contract.daytona_snapshot import resource_not_found
from test_broker import FROM_ONLY, GB, WITH_RUN, Clock, FakeHost, active, capacity, make_world

TIMEOUT_500 = SiloError("Command '['nerdctl', 'run', '-d']' timed out after 600 seconds", status_code=500)


class SickHost(FakeHost):
    """A host whose creates fail with a configurable error (default: the nerdctl-run timeout)."""

    def __init__(self, error: Exception = TIMEOUT_500):
        super().__init__()
        self.error = error
        self.attempts = 0

    def create_sandbox(self, body):
        self.attempts += 1
        raise self.error


def world_with(hosts: dict[str, FakeHost], **config):
    clock = Clock()
    broker = Broker(
        host_secret="s3cret",
        host_client_factory=lambda url: hosts.setdefault(url, FakeHost()),
        clock=clock,
        sleep=clock.sleep,
        placement_wait_seconds=60,
        config=BrokerConfig(**config) if config else None,
    )
    return broker, clock


# -- configuration -----------------------------------------------------------------


def test_default_windows_comfortably_exceed_a_heartbeat_cycle():
    config, warnings = BrokerConfig.from_env({})
    assert warnings == []
    assert config.suspect_after_seconds >= 2 * config.heartbeat_cycle_seconds
    assert config.dead_after_seconds >= 2 * config.suspect_after_seconds


def test_a_window_below_two_heartbeat_cycles_is_raised_loudly():
    config, warnings = BrokerConfig.from_env(
        {"SILO_HEARTBEAT_SECONDS": "10", "SILO_HEARTBEAT_TIMEOUT_SECONDS": "15", "SILO_HOST_SUSPECT_AFTER_SECONDS": "45"}
    )
    assert config.suspect_after_seconds == 50  # 2 x (10 + 15), not the 45 that flapped
    assert any("declared=45" in w and "resolved=50" in w for w in warnings)


def test_malformed_config_fails_startup():
    with pytest.raises(ValueError):
        BrokerConfig.from_env({"SILO_HOST_DEAD_AFTER_SECONDS": "soon"})


def test_host_reporting_a_short_suspect_window_is_warned(caplog):
    broker, _ = world_with({})
    with caplog.at_level(logging.WARNING):
        broker.heartbeat("h1", "http://h1", capacity(), [], heartbeat_seconds=30, heartbeat_timeout_seconds=15)
    assert "will flap" in caplog.text


# -- suspect vs dead ------------------------------------------------------------------


def test_a_suspect_host_gets_no_placements_but_is_live_again_on_its_next_heartbeat(caplog):
    broker, clock = world_with({})
    broker.heartbeat("h1", "http://h1", capacity(), [])
    broker.create_snapshot("s", FROM_ONLY, CANDIDATE_DEFAULT)
    active(broker, "s")
    created = broker.create_sandbox(snapshot="s")
    clock.now += broker.config.suspect_after_seconds + 1
    with caplog.at_level(logging.WARNING):
        broker.sweep()
    assert "host h1 is now suspect" in caplog.text
    ceiling = broker.capacity()
    assert (ceiling["hosts_live"], ceiling["hosts_suspect"], ceiling["hosts_dead"]) == (0, 1, 0)
    with pytest.raises(SiloRateLimitError):
        broker.create_sandbox(snapshot="s", timeout=4)
    with pytest.raises(SiloError) as info:
        broker.delete_sandbox(created["id"])
    assert info.value.status_code == 503 and "suspect" in str(info.value)
    broker.heartbeat("h1", "http://h1", capacity(), [created["id"]])
    assert broker.get_sandbox(created["id"])["host_url"] == "http://h1"
    assert broker.capacity()["hosts_live"] == 1


def test_a_refused_connection_is_dead_at_once_and_its_sandboxes_are_gone():
    hosts: dict[str, FakeHost] = {}
    broker, _ = world_with(hosts)
    broker.heartbeat("h1", "http://h1", capacity(cpu=64, mem_gb=256), ["slbold"])
    broker.heartbeat("h2", "http://h2", capacity(), [])
    broker.create_snapshot("s", FROM_ONLY, CANDIDATE_DEFAULT)
    active(broker, "s")
    hosts["http://h1"].gone_away = True
    assert broker.create_sandbox(snapshot="s")["host_url"] == "http://h2"
    assert broker.capacity()["hosts_dead"] == 1
    with pytest.raises(SiloNotFoundError):
        broker.get_sandbox("slbold")


def test_a_host_restarted_under_the_same_id_re_registers_cleanly():
    # A preempted host job comes back as a new pod: same host id, new address,
    # none of its old sandboxes. The new incarnation is live on its first
    # heartbeat; the old sandboxes are an honest not-found from the host itself.
    hosts: dict[str, FakeHost] = {}
    broker, clock = world_with(hosts)
    broker.heartbeat("h1", "http://h1-a", capacity(), ["slbold"])
    clock.now += broker.config.suspect_after_seconds + 1  # the pod is gone: suspect
    with pytest.raises(SiloError) as info:
        broker.get_sandbox("slbold")
    assert info.value.status_code == 503
    broker.heartbeat("h1", "http://h1-b", capacity(), [])  # the replacement registers
    hosts["http://h1-b"].gone.add("slbold")
    assert broker.capacity()["hosts_live"] == 1
    with pytest.raises(SiloNotFoundError):
        broker.get_sandbox("slbold")


def test_unknown_sandbox_is_503_during_warmup_then_not_found():
    broker, clock = world_with({})
    broker.begin_warmup(broker.config.suspect_after_seconds)
    with pytest.raises(SiloError) as info:
        broker.get_sandbox("slbnobody")
    assert info.value.status_code == 503
    assert not resource_not_found(info.value, "sandbox")
    broker.heartbeat("h1", "http://h1", capacity(), ["slbsurvivor"])
    assert broker.get_sandbox("slbsurvivor")["host_url"] == "http://h1"
    clock.now += broker.config.suspect_after_seconds
    broker.heartbeat("h1", "http://h1", capacity(), ["slbsurvivor"])
    with pytest.raises(SiloNotFoundError):
        broker.get_sandbox("slbnobody")


def test_warmup_restarts_at_the_first_heartbeat_for_an_overlap_cutover():
    # A replacement broker started next to the old one hears nothing until the
    # old one is cancelled; its warm-up must cover the minute after THAT.
    broker, clock = world_with({})
    broker.begin_warmup(broker.config.suspect_after_seconds)
    clock.now += 10 * broker.config.suspect_after_seconds  # overlap: no traffic yet
    broker.heartbeat("h1", "http://h1", capacity(), [])  # cutover: first host arrives
    with pytest.raises(SiloError) as info:
        broker.get_sandbox("slbonh2")  # its host has not re-registered yet
    assert info.value.status_code == 503


# -- quarantine -------------------------------------------------------------------------


def test_negative_allocation_never_inflates_free_slots():
    host = HostState("h", "http://h", capacity(cpu=16, mem_gb=64, allocated_cpu=-1000), 0.0, FakeHost())
    assert host.slots_for(VERIFIER_DEFAULT) == 8  # the budget, not 508
    assert report_inconsistency(capacity(allocated_cpu=-1000)).startswith("cpu.allocated=-1000")
    assert "sandboxes_live" in report_inconsistency({**capacity(allocated_cpu=3), "sandboxes_live": 10})
    assert report_inconsistency({**capacity(allocated_cpu=20), "sandboxes_live": 10}) == ""


def test_a_drifted_host_stays_placeable_against_a_conservative_estimate(caplog):
    # 2026-09-30: fencing drifted hosts off stranded 39 of 42 hosts in 30 minutes,
    # because the double release eventually hits every host that deletes.
    broker, _ = world_with({})
    with caplog.at_level(logging.WARNING):
        broker.heartbeat(
            "old", "http://old", {**capacity(cpu=16, mem_gb=64, allocated_cpu=-2000), "sandboxes_live": 2}, ["a", "b"]
        )
    assert "host old DRIFTED: its report is impossible" in caplog.text
    ceiling = broker.capacity()
    assert ceiling["hosts_quarantined"] == 0 and ceiling["hosts_placeable"] == 1
    host = broker._hosts["old"]
    # 2 live sandboxes counted as candidate-sized (4 cpu / 8 GB each), not as -2000 cpu.
    assert host.free("cpu") == 16 * float(host.capacity["cpu"].get("oversubscribe", 1.0)) - 8
    assert host.free("memory_bytes") == 64 * 2**30 - 2 * CANDIDATE_DEFAULT.memory_bytes
    broker.create_snapshot("s", FROM_ONLY, CANDIDATE_DEFAULT)
    active(broker, "s")
    assert broker.create_sandbox(snapshot="s")["host_url"] == "http://old"
    # A consistent report (after a restart with the fixed agent) clears the drift.
    with caplog.at_level(logging.INFO):
        broker.heartbeat("old", "http://old", capacity(cpu=16, mem_gb=64), [])
    assert not broker._hosts["old"].drift


def test_a_drifted_host_full_by_estimate_takes_no_creates():
    broker, _ = world_with({})
    broker.heartbeat("old", "http://old", {**capacity(cpu=8, mem_gb=16, allocated_cpu=-50), "sandboxes_live": 2}, [])
    broker.heartbeat("new", "http://new", capacity(), [])
    broker.create_snapshot("s", FROM_ONLY, CANDIDATE_DEFAULT)
    active(broker, "s")
    for _ in range(3):
        assert broker.create_sandbox(snapshot="s")["host_url"] == "http://new"


def test_a_host_that_keeps_timing_out_is_quarantined_and_creates_land_elsewhere(caplog):
    hosts: dict[str, FakeHost] = {"http://sick": SickHost()}
    broker, _ = world_with(hosts, quarantine_max_fraction=0.5)
    broker.heartbeat("sick", "http://sick", capacity(cpu=64, mem_gb=256), [])  # roomiest, and gets the image
    broker.heartbeat("ok", "http://ok", capacity(), [])
    broker.create_snapshot("s", FROM_ONLY, CANDIDATE_DEFAULT)
    active(broker, "s")
    with caplog.at_level(logging.WARNING):
        # A sick-host failure is placed again rather than failing the caller.
        created = broker.create_sandbox(snapshot="s")
    assert created["host_url"] == "http://ok"
    for _ in range(3):
        broker.create_sandbox(snapshot="s")
    assert hosts["http://sick"].attempts == 1  # deprioritized after its first failure
    # Force more failures: only the sick host has room now.
    hosts["http://ok"].full = True
    hosts["http://ok"].report = capacity(allocated_cpu=16, allocated_mem=64 * GB, allocated_disk=200 * GB)
    with caplog.at_level(logging.WARNING), pytest.raises(SiloRateLimitError):
        broker.create_sandbox(snapshot="s", timeout=10)
    assert "host sick QUARANTINED for placement" in caplog.text
    assert "timed out after 600 seconds" in caplog.text
    assert broker.capacity()["hosts_quarantined"] == 1


def test_request_errors_never_count_against_a_host():
    bad_image = SiloError("nerdctl pull -q exited 1: invalid checksum digest length", status_code=500)
    hosts: dict[str, FakeHost] = {"http://h1": SickHost(bad_image)}
    broker, _ = world_with(hosts)
    broker.heartbeat("h1", "http://h1", capacity(), [])
    broker.create_snapshot("s", FROM_ONLY, CANDIDATE_DEFAULT)
    active(broker, "s")
    for _ in range(5):
        with pytest.raises(SiloError, match="invalid checksum"):
            broker.create_sandbox(snapshot="s")
    assert broker.capacity()["hosts_quarantined"] == 0


def test_quarantine_is_capped_so_a_fleet_wide_problem_never_idles_the_fleet(caplog):
    hosts: dict[str, FakeHost] = {f"http://h{i}": SickHost() for i in range(4)}
    broker, _ = world_with(hosts, quarantine_after_failures=1)
    for i in range(4):
        broker.heartbeat(f"h{i}", f"http://h{i}", capacity(), [])
    broker.create_snapshot("s", FROM_ONLY, CANDIDATE_DEFAULT)
    active(broker, "s")
    with caplog.at_level(logging.WARNING), pytest.raises(SiloRateLimitError):
        broker.create_sandbox(snapshot="s", timeout=6)
    assert broker.capacity()["hosts_quarantined"] == 1  # 25% of 4
    assert "deprioritizing it instead" in caplog.text


def test_quarantine_ends_in_probation_and_a_success_clears_it():
    sick = SickHost()
    hosts: dict[str, FakeHost] = {"http://h1": sick}
    broker, clock = world_with(hosts, quarantine_after_failures=1, quarantine_seconds=100)
    broker.heartbeat("h1", "http://h1", capacity(), [])
    broker.create_snapshot("s", FROM_ONLY, CANDIDATE_DEFAULT)
    active(broker, "s")
    with pytest.raises(SiloRateLimitError):
        broker.create_sandbox(snapshot="s", timeout=4)
    assert broker.capacity()["hosts_quarantined"] == 1
    clock.now += 101
    broker.heartbeat("h1", "http://h1", capacity(), [])
    sick.create_sandbox = FakeHost.create_sandbox.__get__(sick)  # the host recovered
    assert broker.create_sandbox(snapshot="s")["host_url"] == "http://h1"
    host = broker.capacity()["hosts"][0]
    assert (host["quarantined"], host["probation"], host["consecutive_failures"]) == (False, False, 0)


def test_in_flight_creates_per_host_are_capped():
    release = threading.Event()
    started = []

    class SlowHost(FakeHost):
        def create_sandbox(self, body):
            started.append(body["sandbox_id"])
            release.wait(10)
            return super().create_sandbox(body)

    hosts: dict[str, FakeHost] = {"http://slow": SlowHost()}
    broker, _ = world_with(hosts, max_inflight_creates_per_host=2)
    broker.heartbeat("slow", "http://slow", capacity(cpu=64, mem_gb=256), [])
    broker.create_snapshot("s", FROM_ONLY, CANDIDATE_DEFAULT)
    active(broker, "s")
    plan = broker.plan_create(snapshot="s")
    placed = [broker.try_place(plan) for _ in range(3)]
    assert placed[2] is None  # two in flight is the cap, though slots remain
    threads = [threading.Thread(target=broker.attempt_create, args=(plan, *p)) for p in placed[:2]]
    for thread in threads:
        thread.start()
    assert broker.capacity()["creates_inflight"] == 2
    release.set()
    for thread in threads:
        thread.join(10)
    assert broker.capacity()["creates_inflight"] == 0
    assert broker.try_place(plan) is not None


def test_at_capacity_log_counts_suspect_dead_and_quarantined_hosts(caplog):
    hosts: dict[str, FakeHost] = {}
    broker, clock = world_with(hosts)
    broker.heartbeat("gone", "http://gone", capacity(), [])
    broker.heartbeat("quiet", "http://quiet", capacity(), [])
    hosts["http://gone"].gone_away = True
    clock.now += broker.config.suspect_after_seconds + 1
    broker.heartbeat("full", "http://full", capacity(cpu=4, mem_gb=8, disk_gb=10, allocated_cpu=4), [])
    broker.heartbeat("sick", "http://sick", capacity(), [])
    broker._hosts["sick"].quarantined_until = clock.now + 1000
    broker._hosts["gone"].gone = True
    broker.create_snapshot("s", FROM_ONLY, CANDIDATE_DEFAULT)
    active(broker, "s")
    with caplog.at_level(logging.WARNING), pytest.raises(SiloRateLimitError) as info:
        broker.create_sandbox(snapshot="s", timeout=4)
    line = next(r.getMessage() for r in caplog.records if "AT CAPACITY" in r.getMessage())
    for part in ("hosts_live=2", "hosts_suspect=1", "hosts_dead=1", "hosts_quarantined=1", "hosts_placeable=1"):
        assert part in line
    assert "hosts_suspect=1" in str(info.value)


# -- images and builds ----------------------------------------------------------------


def test_heartbeat_images_steer_placement():
    broker, _ = world_with({})
    broker.heartbeat("h1", "http://h1", capacity(cpu=64, mem_gb=256), [])
    broker.create_snapshot("s", WITH_RUN, CANDIDATE_DEFAULT)
    record = active(broker, "s")
    # A host that reports the built image wins over a roomier one that would rebuild.
    broker.heartbeat("h2", "http://h2", capacity(), [], images=[record.resolved_ref])
    broker._hosts["h1"].images.clear()
    assert broker.create_sandbox(snapshot="s")["host_url"] == "http://h2"


def test_a_recipe_from_a_local_parent_builds_where_the_parent_is():
    hosts: dict[str, FakeHost] = {}
    broker, _ = world_with(hosts)
    parent = local_image_tag("parent")
    broker.heartbeat("roomy", "http://roomy", capacity(cpu=64, mem_gb=256), [])
    broker.heartbeat("has-parent", "http://has-parent", capacity(), [], images=[parent])
    broker.create_snapshot("child", f"FROM {parent}\nRUN echo child\n", VERIFIER_DEFAULT)
    assert active(broker, "child").state == "active"
    assert hosts["http://has-parent"].built == [local_image_tag("child")]
    assert hosts["http://roomy"].built == []


# -- persistence -------------------------------------------------------------------------


def test_state_roundtrip_keeps_snapshots_and_the_host_registry(tmp_path):
    hosts: dict[str, FakeHost] = {}
    first, _ = world_with(hosts)
    first.heartbeat("h1", "http://h1", capacity(), ["slbalive"], images=["img-a"])
    first.create_snapshot("from-only", FROM_ONLY, CANDIDATE_DEFAULT)
    first.create_snapshot("built", WITH_RUN, VERIFIER_DEFAULT)
    before = {name: active(first, name) for name in ("from-only", "built")}
    store = FileStateStore(tmp_path / "state.json")
    assert StatePersister(first, store).save_if_changed()
    assert not StatePersister(first, store).save_if_changed()  # nothing changed since

    second, _ = world_with(hosts)
    second.begin_warmup(second.config.suspect_after_seconds)
    writable, counts = restore(second, store)
    assert writable and counts == {"snapshots": 2, "rebuilding": 0, "skipped": 0, "hosts": 1}
    for name, record in before.items():
        restored = second.get_snapshot(name)
        assert (restored.id, restored.state, restored.resolved_ref, restored.dockerfile_content) == (
            record.id,
            record.state,
            record.resolved_ref,
            record.dockerfile_content,
        )
        assert restored.profile == record.profile
    # The restored host is suspect (no placement) until it heartbeats; its
    # sandboxes answer 503 meanwhile, never not-found.
    assert second.capacity()["hosts_suspect"] == 1
    with pytest.raises(SiloError) as info:
        second.get_sandbox("slbalive")
    assert info.value.status_code == 503
    second.heartbeat("h1", "http://h1", capacity(), ["slbalive"])
    assert second.get_sandbox("slbalive")["host_url"] == "http://h1"
    assert "img-a" in second._hosts["h1"].images


def test_a_snapshot_still_building_at_save_time_is_built_again(tmp_path):
    store = FileStateStore(tmp_path / "state.json")
    snapshot = {
        "name": "half",
        "id": "snap-x",
        "state": "building",
        "dockerfile_content": WITH_RUN,
        "cpu": 2,
        "memory_gb": 1,
        "disk_gb": 10,
        "created_at": "2026-09-29T05:00:00Z",
        "resolved_ref": "",
        "error_message": "",
    }
    store.save({"version": 1, "snapshots": [snapshot], "hosts": []})
    broker, _ = world_with({})
    broker.heartbeat("h1", "http://h1", capacity(), [])
    assert restore(broker, store)[1]["rebuilding"] == 1
    record = active(broker, "half")
    assert (record.state, record.id, record.resolved_ref) == ("active", "snap-x", local_image_tag("half"))


def test_unreadable_state_is_never_overwritten(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{not json")
    store = FileStateStore(path)
    broker, _ = world_with({})
    writable, counts = restore(broker, store)
    assert (writable, counts) == (False, None)
    broker.create_snapshot("s", FROM_ONLY, CANDIDATE_DEFAULT)
    assert not StatePersister(broker, store, writable=writable).save_if_changed()
    assert path.read_text() == "{not json"


def test_an_old_brokers_api_listing_restores_into_a_new_broker():
    old, _, _ = make_world()
    old.heartbeat("h1", "http://h1", capacity(), [])
    old.create_snapshot("built", WITH_RUN, VERIFIER_DEFAULT)
    record = active(old, "built")
    listed = old.list_snapshots()["items"]
    state = {"version": 1, "snapshots": [snapshot_state_from_api(item) for item in listed], "hosts": []}
    new, _ = world_with({})
    new.restore_state(state)
    restored = new.get_snapshot("built")
    assert (restored.id, restored.state, restored.resolved_ref) == (record.id, "active", record.resolved_ref)
    assert restored.profile == VERIFIER_DEFAULT


def test_transport_timeouts_count_as_host_failures_but_refusals_mark_dead():
    timeout = TransportError("cannot reach http://h1/sandboxes: timed out")
    hosts: dict[str, FakeHost] = {"http://h1": SickHost(timeout)}
    broker, _ = world_with(hosts, quarantine_after_failures=2, quarantine_max_fraction=1.0)
    broker.heartbeat("h1", "http://h1", capacity(), [])
    broker.create_snapshot("s", FROM_ONLY, CANDIDATE_DEFAULT)
    active(broker, "s")
    with pytest.raises(SiloRateLimitError):
        broker.create_sandbox(snapshot="s", timeout=4)
    ceiling = broker.capacity()
    assert (ceiling["hosts_live"], ceiling["hosts_dead"], ceiling["hosts_quarantined"]) == (1, 0, 1)

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import threading

import pytest
from silo.errors import SiloConflictError, SiloNotFoundError, SiloRateLimitError
from silo.host.agent import STATE_DESTROYING, STATE_STARTED, CpusetAllocator, HostAgent, HostConfig
from silo.model import CANDIDATE_DEFAULT, VERIFIER_DEFAULT, new_sandbox_id, parse_recipe
from silo.pipeline_contract.daytona_snapshot import wait_for_sandbox_deletion
from silo_testing import wait_until

DIGEST = "docker.io/library/alpine@sha256:" + "a" * 64
PLAN = parse_recipe(f"FROM {DIGEST}\n")


def make_agent(tmp_path, runtime, *, cpus=16, mem_gb=64, disk_gb=200, cpu_ids=None, oversub=1.0):
    config = HostConfig(
        host_id="h1",
        cpu_budget=cpus,
        memory_budget_bytes=mem_gb * 1024**3,
        disk_budget_bytes=disk_gb * 1024**3,
        cpu_ids=tuple(cpu_ids if cpu_ids is not None else range(cpus)),
        work_dir=tmp_path / "work",
        cpu_oversubscribe=oversub,
    )
    return HostAgent(config, runtime)


def create(agent, profile=CANDIDATE_DEFAULT, sandbox_id=None):
    return agent.create_sandbox(
        sandbox_id=sandbox_id or new_sandbox_id(),
        snapshot_name="cap-harbor-x",
        image_ref=DIGEST,
        plan=PLAN,
        profile=profile,
    )


# -- creation, ids, network ---------------------------------------------------


def test_create_starts_a_network_blocked_container_with_real_limits(tmp_path, fake_runtime):
    agent = make_agent(tmp_path, fake_runtime)
    record = create(agent)
    assert record.state == STATE_STARTED
    assert record.network_block_all is True
    spec = fake_runtime.started[0]
    assert spec.network_none is True
    assert spec.cpus == 4 and len(spec.cpuset) == 4
    assert spec.memory_bytes == 8 * 1024**3
    assert fake_runtime.pulls == [DIGEST]


def test_ids_are_never_reused_even_after_deletion(tmp_path, fake_runtime):
    agent = make_agent(tmp_path, fake_runtime)
    record = create(agent)
    agent.delete_sandbox(record.id)
    wait_until(lambda: record.id not in fake_runtime.containers)
    with pytest.raises(SiloConflictError):
        create(agent, sandbox_id=record.id)


def test_failed_start_releases_capacity_and_still_retires_the_id(tmp_path, fake_runtime):
    agent = make_agent(tmp_path, fake_runtime, cpus=4, mem_gb=8, disk_gb=10)
    fake_runtime.fail_start = True
    sandbox_id = new_sandbox_id()
    with pytest.raises(Exception, match="injected"):
        create(agent, sandbox_id=sandbox_id)
    fake_runtime.fail_start = False
    assert agent.slots_for(CANDIDATE_DEFAULT) == 1  # capacity came back
    with pytest.raises(SiloConflictError):
        create(agent, sandbox_id=sandbox_id)


# -- capacity: explicit, not a stall -------------------------------------------


def test_full_host_raises_rate_limit_not_a_hang(tmp_path, fake_runtime):
    agent = make_agent(tmp_path, fake_runtime, cpus=8, mem_gb=16, disk_gb=100)
    create(agent)
    create(agent)
    assert agent.slots_for(CANDIDATE_DEFAULT) == 0
    with pytest.raises(SiloRateLimitError) as info:
        create(agent)
    assert info.value.status_code == 429


def test_ceiling_arithmetic_on_a_96_cpu_384_gb_host(tmp_path, fake_runtime):
    # Honest CPU accounting: the candidate profile is CPU-bound at 96 / 4 = 24.
    honest = make_agent(tmp_path / "a", fake_runtime, cpus=96, mem_gb=384, disk_gb=1024, oversub=1.0)
    assert honest.slots_for(CANDIDATE_DEFAULT) == 24
    assert honest.slots_for(VERIFIER_DEFAULT) == 48
    # At 2x CPU oversubscription memory becomes the bound: 384 / 8 = 48. Each
    # sandbox keeps its own hard 4-cpu cpu.max either way; oversubscription only
    # decides how many may burst at once.
    shared = make_agent(tmp_path / "b", fake_runtime, cpus=96, mem_gb=384, disk_gb=1024, oversub=2.0)
    assert shared.slots_for(CANDIDATE_DEFAULT) == 48
    assert shared.slots_for(VERIFIER_DEFAULT) == 96


def test_concurrent_creates_cannot_share_the_last_slot(tmp_path, fake_runtime):
    agent = make_agent(tmp_path, fake_runtime, cpus=4, mem_gb=8, disk_gb=10)
    results: list[str] = []

    def attempt():
        try:
            create(agent)
            results.append("ok")
        except SiloRateLimitError:
            results.append("full")

    threads = [threading.Thread(target=attempt) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results.count("ok") == 1 and results.count("full") == 7


def test_cpusets_are_disjoint_until_the_host_is_full():
    allocator = CpusetAllocator(tuple(range(8)))
    a, b = allocator.allocate(4), allocator.allocate(4)
    assert set(a).isdisjoint(b)
    allocator.release(a)
    assert set(allocator.allocate(4)) == set(a)


# -- deletion: asynchronous and observable (brief section 3.5) -----------------


def test_deletion_is_async_then_permanently_not_found(tmp_path, fake_runtime):
    agent = make_agent(tmp_path, fake_runtime)
    slots_before = agent.slots_for(CANDIDATE_DEFAULT)
    record = create(agent)
    fake_runtime.remove_gate.clear()  # hold the container in place

    agent.delete_sandbox(record.id)  # returns immediately
    assert agent.get_sandbox(record.id).state == STATE_DESTROYING

    fake_runtime.remove_gate.set()
    wait_until(lambda: record.id not in fake_runtime.containers)
    wait_until(lambda: record.id not in {r.id for r in agent.list_sandboxes()})
    for _ in range(3):  # stays gone, with no second delete
        with pytest.raises(SiloNotFoundError, match="sandbox"):
            agent.get_sandbox(record.id)
    assert agent.slots_for(CANDIDATE_DEFAULT) == slots_before  # capacity returned


def test_pipeline_deletion_observer_sees_present_then_not_found(tmp_path, fake_runtime):
    """Run the pipeline's own wait_for_sandbox_deletion against the agent."""
    agent = make_agent(tmp_path, fake_runtime)
    record = create(agent)
    fake_runtime.remove_gate.clear()
    agent.delete_sandbox(record.id)

    class Client:
        def get(self, sandbox_id):
            return agent.get_sandbox(sandbox_id)

    def sleeper(_):
        # First observation happens with removal still gated; release it and
        # wait for removal before the second observation.
        fake_runtime.remove_gate.set()
        wait_until(lambda: record.id not in {r.id for r in agent.list_sandboxes()})

    state, observations = wait_for_sandbox_deletion(Client(), record.id, sleeper=sleeper)
    assert state == "not_found"
    assert [o["state"] for o in observations] == ["present", "not_found"]


def test_double_delete_is_harmless(tmp_path, fake_runtime):
    agent = make_agent(tmp_path, fake_runtime)
    record = create(agent)
    fake_runtime.remove_gate.clear()
    agent.delete_sandbox(record.id)
    agent.delete_sandbox(record.id)
    fake_runtime.remove_gate.set()
    wait_until(lambda: record.id not in {r.id for r in agent.list_sandboxes()})


def test_ttl_reaps_expired_sandboxes(tmp_path, fake_runtime):
    now = [1000.0]
    config = HostConfig("h1", 16, 64 * 1024**3, 200 * 1024**3, tuple(range(16)), tmp_path / "w")
    agent = HostAgent(config, fake_runtime, clock=lambda: now[0])
    record = agent.create_sandbox(
        sandbox_id=new_sandbox_id(),
        snapshot_name="s",
        image_ref=DIGEST,
        plan=PLAN,
        profile=VERIFIER_DEFAULT,
        ttl_minutes=1,
    )
    assert agent.reap() == []
    now[0] += 61
    assert agent.reap() == [record.id]


# -- exec and the polled session model -----------------------------------------


def test_one_shot_exec_combines_output_like_daytona(tmp_path, fake_runtime):
    agent = make_agent(tmp_path, fake_runtime)
    record = create(agent)
    outcome = agent.exec(record.id, "echo out; echo err >&2; exit 3")
    assert outcome.exit_code == 3
    assert b"out" in outcome.output and b"err" in outcome.output


def test_exec_env_and_cwd(tmp_path, fake_runtime):
    agent = make_agent(tmp_path, fake_runtime)
    record = create(agent)
    agent.upload_file(record.id, "/work/marker", b"")
    outcome = agent.exec(record.id, 'ls; echo "$GREETING"', cwd="/work", env={"GREETING": "hi"})
    assert outcome.output.split() == [b"marker", b"hi"]


def test_session_command_returns_before_it_finishes_and_polling_consumes_nothing(tmp_path, fake_runtime):
    agent = make_agent(tmp_path, fake_runtime)
    record = create(agent)
    agent.create_session(record.id, "s1")
    entry = agent.execute_session_command(record.id, "s1", "sleep 0.5; echo out; echo err >&2; exit 7")

    # Launched, not finished.
    assert agent.get_session_command(record.id, "s1", entry.cmd_id).exit_code is None
    wait_until(lambda: agent.get_session_command(record.id, "s1", entry.cmd_id).exit_code is not None)
    assert agent.get_session_command(record.id, "s1", entry.cmd_id).exit_code == 7

    # Logs are separate streams, and reading them twice returns the same bytes:
    # a poll or a log read never consumes output.
    first = agent.get_session_command_logs(record.id, "s1", entry.cmd_id)
    second = agent.get_session_command_logs(record.id, "s1", entry.cmd_id)
    assert first == second == (b"out\n", b"err\n")
    agent.delete_session(record.id, "s1")
    with pytest.raises(SiloNotFoundError, match="session"):
        agent.get_session_command(record.id, "s1", entry.cmd_id)


def test_pipeline_timeout_exit_code_is_preserved(tmp_path, fake_runtime):
    # The pipeline wraps its own `timeout N`; exit 124 is how it learns a
    # command timed out, so the agent must pass exit codes through untouched.
    agent = make_agent(tmp_path, fake_runtime)
    record = create(agent)
    agent.create_session(record.id, "s")
    entry = agent.execute_session_command(record.id, "s", "exit 124", run_async=False)
    assert entry.exit_code == 124


def test_session_ids_conflict(tmp_path, fake_runtime):
    agent = make_agent(tmp_path, fake_runtime)
    record = create(agent)
    agent.create_session(record.id, "s")
    with pytest.raises(SiloConflictError):
        agent.create_session(record.id, "s")


# -- filesystem ----------------------------------------------------------------


def test_upload_download_roundtrip_is_byte_exact(tmp_path, fake_runtime):
    agent = make_agent(tmp_path, fake_runtime)
    record = create(agent)
    payload = bytes(range(256)) * 4096
    agent.upload_file(record.id, "/input/payload.bin", payload)
    assert agent.download_file(record.id, "/input/payload.bin") == payload


def test_download_of_missing_file_is_a_clean_not_found(tmp_path, fake_runtime):
    agent = make_agent(tmp_path, fake_runtime)
    record = create(agent)
    with pytest.raises(SiloNotFoundError, match="file"):
        agent.download_file(record.id, "/nope")


def test_operations_on_a_deleting_sandbox_are_refused(tmp_path, fake_runtime):
    agent = make_agent(tmp_path, fake_runtime)
    record = create(agent)
    fake_runtime.remove_gate.clear()
    agent.delete_sandbox(record.id)
    with pytest.raises(Exception, match="destroying"):
        agent.exec(record.id, "true")
    fake_runtime.remove_gate.set()


# -- accounting that cannot drift (2026-09-29) -----------------------------------


def test_a_slow_delete_retried_by_the_reaper_releases_its_resources_once(tmp_path, fake_runtime, caplog):
    """Regression: the reaper re-ran deletions that were merely slow.

    Each run released the sandbox's cpu/memory/disk again, so a busy host's
    allocation drifted negative (hosts 0/2/6/7/9 logged the same "sandbox
    deleted" id twice, dozens of times an hour). The broker then saw those hosts
    as the emptiest and sent them almost every create.
    """
    agent = make_agent(tmp_path, fake_runtime)
    keep = create(agent, VERIFIER_DEFAULT)
    record = create(agent)
    fake_runtime.remove_gate.clear()  # `nerdctl rm` is slow on a loaded host
    agent.delete_sandbox(record.id)
    for _ in range(4):  # reap ticks while the removal is still running
        agent.reap()
    with caplog.at_level("INFO"):
        fake_runtime.remove_gate.set()
        wait_until(lambda: record.id not in {r.id for r in agent.list_sandboxes()})
        wait_until(lambda: all(t.name != f"rm-{record.id}" for t in threading.enumerate()))
    capacity = agent.capacity()
    assert capacity["cpu"]["allocated"] == VERIFIER_DEFAULT.cpu  # only `keep`, never negative
    assert capacity["memory_bytes"]["allocated"] == VERIFIER_DEFAULT.memory_bytes
    assert caplog.text.count(f"sandbox deleted id={record.id}") == 1
    agent.delete_sandbox(keep.id)
    wait_until(lambda: not agent.list_sandboxes())
    assert agent.capacity()["cpu"]["allocated"] == 0
    assert agent.slots_for(VERIFIER_DEFAULT) == 8


def test_allocation_counts_creates_in_flight_and_destroying_sandboxes(tmp_path, fake_runtime):
    agent = make_agent(tmp_path, fake_runtime, cpus=4, mem_gb=8, disk_gb=10)
    started = threading.Event()
    proceed = threading.Event()
    real_start = fake_runtime.start

    def slow_start(spec):
        started.set()
        proceed.wait(10)
        real_start(spec)

    fake_runtime.start = slow_start
    worker = threading.Thread(target=create, args=(agent,))
    worker.start()
    started.wait(10)
    assert agent.capacity()["cpu"]["allocated"] == 4  # in flight counts
    with pytest.raises(SiloRateLimitError):
        create(agent)
    proceed.set()
    worker.join(10)
    fake_runtime.start = real_start
    (record,) = agent.list_sandboxes()
    fake_runtime.remove_gate.clear()
    agent.delete_sandbox(record.id)
    assert agent.capacity()["cpu"]["allocated"] == 4  # destroying still holds it
    fake_runtime.remove_gate.set()
    wait_until(lambda: agent.capacity()["cpu"]["allocated"] == 0)


def test_images_pulled_or_built_are_reported(tmp_path, fake_runtime):
    agent = make_agent(tmp_path, fake_runtime)
    create(agent)
    agent.build_image("silo.local/snapshots/x:built", "FROM scratch\n")
    assert agent.images() == sorted([DIGEST, "silo.local/snapshots/x:built"])

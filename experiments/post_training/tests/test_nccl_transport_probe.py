# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the NCCL transport probe's log parsing and gate verdict."""

import hashlib
import os
from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from iris.cluster.client.job_info import JobInfo
from iris.cluster.types import JobName

from experiments.post_training import nccl_transport_probe as probe
from experiments.post_training.nccl_transport_probe import (
    NcclCensus,
    WorkerProcess,
    all_to_all_element_count,
    collect_worker,
    gate_endpoint_name,
    gate_verdict,
    mapped_nccl_libraries,
    parse_nccl_census,
    parse_worker_result,
    publish_gate_report,
    resolve_gloo_socket_ifname,
    worker_traceback,
    write_worker_result,
)


def _gloo_a2a_rank(rank, world_size, buffer_bytes, init_file, done_dir):
    """One process of the non-power-of-two all_to_all regression."""
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    elements = all_to_all_element_count(buffer_bytes, world_size)
    a2a_input = torch.full((elements,), float(rank), dtype=torch.bfloat16)
    a2a_output = torch.empty_like(a2a_input)
    dist.all_to_all_single(a2a_output, a2a_input)
    chunk = elements // world_size
    for source in range(world_size):
        segment = a2a_output[source * chunk : (source + 1) * chunk]
        assert torch.all(segment == float(source)), f"rank {rank} chunk {source} mismatched"
    dist.barrier()
    dist.destroy_process_group()
    (done_dir / f"done-{rank}").write_text("ok")


def test_parse_nccl_census_distinguishes_ib_and_socket():
    device_line = "g853134:13596:14951 [0] NCCL INFO NET/IB : Using [0]ibp0:1/IB [1]mlx5_2:1/RoCE"
    ib_log = [
        device_line,
        "g853134:13596:14951 [0] NCCL INFO Using network InfiniBand",
        "g853134:13596:14951 [0] NCCL INFO Channel 00 : 0[0] -> 1[0] via NET/IB/0/GDRDMA",
        "g853134:13596:14951 [0] NCCL INFO Channel 01 : 1[0] -> 0[0] via NET/IB/0/GDRDMA",
        "g853134:13596:14951 [0] NCCL INFO NCCL version 2.29.7+cuda13.2",
    ]
    census = parse_nccl_census(ib_log)
    assert census == NcclCensus(nccl_version="2.29.7", transports=["NET/IB/0/GDRDMA"], ib_device_lines=[device_line])

    socket_log = [
        "g95c40:13657:14951 [0] NCCL INFO Channel 00 : 0[0] -> 2[0] via NET/Socket/0/0",
        "g95c40:13657:14951 [0] NCCL INFO NCCL version 2.29.7",
    ]
    assert parse_nccl_census(socket_log) == NcclCensus(nccl_version="2.29.7", transports=["NET/Socket/0/0"])


def test_parse_nccl_census_ignores_intra_node_p2p():
    """Intra-node NVLink paths are not the inter-node transport."""
    p2p_log = [
        "NCCL INFO Channel 00 : 0[0] -> 1[0] via P2P/IPC",
        "NCCL INFO NCCL version 2.29.7",
    ]
    assert parse_nccl_census(p2p_log) == NcclCensus(nccl_version="2.29.7", transports=[])


def test_gate_verdict_fails_closed_on_sockets_and_failures():
    """The gate passes only NET/IB or an empty (intra-node NVLink) census."""
    assert gate_verdict(True, [["NET/IB/0/GDRDMA"]]) == "pass"
    assert gate_verdict(True, [[]]) == "pass"
    assert gate_verdict(True, [[], ["NET/IB/0/GDRDMA"]]) == "pass"
    assert gate_verdict(True, [["NET/Socket/0"]]) == "socket_transport"
    assert gate_verdict(True, [[], ["NET/Socket/0/Shared"]]) == "socket_transport"
    assert gate_verdict(False, [["NET/IB/0/GDRDMA"]]) == "collective_failed"


def test_sibling_gpu_gangs_get_separate_gate_rendezvous():
    first_head = JobInfo(JobName.from_wire("/test/coordinator/control-r0/0"))
    first_worker = JobInfo(JobName.from_wire("/test/coordinator/control-r0/1"))
    second_head = JobInfo(JobName.from_wire("/test/coordinator/control-r1/0"))
    assert first_head.job_id.namespace == second_head.job_id.namespace
    assert gate_endpoint_name(first_head) == gate_endpoint_name(first_worker)
    assert gate_endpoint_name(first_head) != gate_endpoint_name(second_head)


def test_mapped_nccl_libraries_parses_proc_maps(tmp_path):
    """The maps scanner finds libnccl entries and hashes the real files."""
    library = tmp_path / "libnccl.so.2"
    library.write_bytes(b"nccl-fixture-bytes")
    maps = tmp_path / "maps"
    maps.write_text(
        "7f0e00000000-7f0e10000000 r-xp 00000000 08:02 1234   "
        f"{library}\n"
        "7f0e20000000-7f0e21000000 r-xp 00000000 08:02 5678   "
        f"{tmp_path}/libother.so\n"
        "7f0e30000000-7f0e31000000 r--p 00000000 00:00 0      [vvar]\n"
    )
    records = mapped_nccl_libraries(maps)
    assert [record["path"] for record in records] == [str(library)]
    assert records[0]["sha256"] == hashlib.sha256(b"nccl-fixture-bytes").hexdigest()


def test_mapped_nccl_libraries_without_maps_is_empty(tmp_path):
    assert mapped_nccl_libraries(tmp_path / "missing") == []


def test_resolve_gloo_socket_ifname_honors_ambient_value(monkeypatch):
    """An ambient GLOO_SOCKET_IFNAME wins and no resolution runs."""
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "ibp0")
    assert resolve_gloo_socket_ifname("10.0.0.5") == "ibp0"


def test_resolve_gloo_socket_ifname_resolves_and_exports(monkeypatch):
    """Without an ambient value, the advertise address's interface is selected and exported."""
    monkeypatch.delenv("GLOO_SOCKET_IFNAME", raising=False)
    interface = resolve_gloo_socket_ifname("127.0.0.1")
    assert os.environ["GLOO_SOCKET_IFNAME"] == interface


def test_resolve_gloo_socket_ifname_fails_on_unowned_address(monkeypatch):
    """An address no local interface owns must fail fast, not fall back to loopback."""
    monkeypatch.delenv("GLOO_SOCKET_IFNAME", raising=False)
    with pytest.raises(RuntimeError, match=r"203\.0\.113\.7"):
        resolve_gloo_socket_ifname("203.0.113.7")


def test_all_to_all_element_count_rounds_down_to_world_size():
    """The 192-rank gate run failed because 2**24 elements left a remainder mod 192."""
    count = all_to_all_element_count(64 * 2**20, 192)
    assert count % 192 == 0
    assert count == 33_554_304
    assert 64 * 2**20 // 2 - count < 192

    for world_size in (1, 3, 7, 64, 193):
        assert all_to_all_element_count(64 * 2**20, world_size) % world_size == 0


def test_all_to_all_element_count_rejects_buffer_too_small_for_world():
    with pytest.raises(ValueError, match="too few for world_size=192"):
        all_to_all_element_count(8, 192)
    with pytest.raises(ValueError, match="world_size must be positive"):
        all_to_all_element_count(64 * 2**20, 0)


def test_worker_traceback_extracts_block_from_nccl_spam():
    """The failing rank's traceback survives NCCL INFO filling the tail window."""

    lines = [
        "g62abc:100:100 [0] NCCL INFO Channel 00 : 0[0] -> 1[0] via NET/IB/0/GDRDMA",
        "Traceback (most recent call last):",
        '  File "worker.py", line 10, in run',
        "    dist.all_to_all_single(a2a_output, a2a_input)",
        '  File "torch/distributed", line 3000, in all_to_all_single',
        "RuntimeError: numel of input tensor is not divisible by world_size",
        "g62abc:100:100 [0] NCCL INFO Channel 01 : 1[0] -> 0[0] via NET/IB/0/GDRDMA",
    ]
    block = worker_traceback(lines)
    assert block[0] == "Traceback (most recent call last):"
    assert block[-1] == "RuntimeError: numel of input tensor is not divisible by world_size"
    assert all("NCCL INFO" not in line for line in block)


def test_worker_traceback_absent_returns_empty():
    assert worker_traceback(["only NCCL INFO", "more INFO"]) == []


def test_publish_gate_report_writes_task_specific_file(tmp_path):
    """The full report is published under the prefix as one attempt/task file."""
    prefix = str(tmp_path / "evidence")
    uri = publish_gate_report(prefix, "42", 1, '{"verdict": "pass"}')
    published = tmp_path / "evidence" / "attempt-42" / "gate-results-task1.json"
    assert uri == str(published)
    assert published.read_text() == '{"verdict": "pass"}'


@pytest.mark.parametrize("check_pynccl", [False, True])
def test_worker_result_file_round_trip(tmp_path, check_pynccl):
    """Check the worker receipt against the requested transport stages."""
    result_path = tmp_path / "baseline_default_env-rank3.json"
    payload = {"variant": "baseline_default_env", "rank": 3, "world_size": 192, "gpu": "GB200"}
    write_worker_result(result_path, payload)
    assert not list(tmp_path.glob("*.tmp"))
    log, output = tmp_path / "log", tmp_path / "output"
    with log.open("wb") as logs, output.open("wb") as results:
        rank = collect_worker(
            WorkerProcess(
                process=SimpleNamespace(wait=lambda **_: 0),
                variant_name="baseline_default_env",
                rank=3,
                world_size=192,
                started=0,
                log_path=log,
                output_path=output,
                result_path=result_path,
                log_file=logs,
                output_file=results,
                pynccl_requested=check_pynccl,
            )
        )
    assert rank.worker == payload
    assert rank.passed is not check_pynccl
    if check_pynccl:
        assert "pynccl init failed: missing result" in rank.error_tail


def test_multinode_gate_refuses_a_task_without_visible_gpus(monkeypatch):
    monkeypatch.setattr(probe, "get_job_info", lambda: SimpleNamespace(num_tasks=2, task_index=0))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    with pytest.raises(RuntimeError, match="task 0 has no visible GPUs"):
        probe.run_gate(Namespace(gate_coverage="all-gpus"))


def test_parse_worker_result_fails_closed_on_missing_file(tmp_path):
    payload, problem = parse_worker_result(tmp_path / "absent.json", 3, "baseline_default_env", 192)
    assert payload is None
    assert problem == f"worker result file missing: {tmp_path / 'absent.json'}"


def test_parse_worker_result_fails_closed_on_identity_mismatch(tmp_path):
    """A mismatched identity fails the rank closed rather than adopting wrong data."""
    result_path = tmp_path / "baseline_default_env-rank3.json"
    result_path.write_text('{"variant": "all_three", "rank": 4, "world_size": 8}')
    payload, problem = parse_worker_result(result_path, 3, "baseline_default_env", 8)
    assert payload is None
    assert "identity mismatch" in problem
    assert "rank=4 != 3" in problem
    assert "variant='all_three'" in problem


def test_parse_worker_result_fails_closed_on_invalid_json(tmp_path):
    result_path = tmp_path / "baseline_default_env-rank3.json"
    result_path.write_text('{"variant": "baseline_default_env", "rank": 3, "world_size"')
    payload, problem = parse_worker_result(result_path, 3, "baseline_default_env", 8)
    assert payload is None
    assert "invalid JSON" in problem


def test_all_to_all_single_succeeds_with_non_power_of_two_ranks(tmp_path):
    """Real 3-rank gloo regression: the rounded element count splits equally."""
    init_file = tmp_path / "rendezvous"
    done_dir = tmp_path / "done"
    done_dir.mkdir()
    world_size = 3

    mp.spawn(
        _gloo_a2a_rank,
        args=(world_size, 4096, init_file, done_dir),
        nprocs=world_size,
        join=True,
    )
    assert sorted(path.name for path in done_dir.iterdir()) == ["done-0", "done-1", "done-2"]

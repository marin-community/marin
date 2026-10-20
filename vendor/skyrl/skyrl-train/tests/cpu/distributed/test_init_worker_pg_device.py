"""Regression guard for init_worker_process_group_with_device (the shared, backend-agnostic
device-pinned NCCL process-group init).

The bug it prevents: creating the NCCL PG without a device_id lets torch's ProcessGroupNCCL
"guess device ID based on global rank" at first-collective time. That guess is not reliably
correct across clusters — on cw-rno2a the first collective (the weight-init barrier) DEADLOCKED
(keep-6 / cw-rno2a, 2026-07-20). NOTE: the live [pg-init] logs show Ray masks CVD to a single
device on BOTH cw-us-east-02a and cw-rno2a (visible_device_count=1), so this was NOT a
masked-vs-unmasked-CVD difference between clusters (an earlier hypothesis the logs disproved) —
the surviving explanation is init ordering (barrier before an explicit set_device). The helper
must therefore always torch.cuda.set_device(LOCAL_RANK) AND pass device_id=cuda:LOCAL_RANK, which
is correct regardless of CVD masking or ordering.
"""

import os

import torch

from skyrl_train.distributed.utils import init_worker_process_group_with_device


def test_pins_device_and_passes_device_id(monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setenv("MASTER_ADDR", "")
    monkeypatch.setenv("MASTER_PORT", "")
    seen = {}
    monkeypatch.setattr(torch.cuda, "set_device", lambda d: seen.__setitem__("set_device", d))
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(
        torch.distributed,
        "init_process_group",
        lambda **kw: seen.update(init=kw, rendezvous=(os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"])),
    )

    init_worker_process_group_with_device(master_addr="localhost", master_port=12345, timeout_seconds=1800)

    # device pinned to the resolved LOCAL_RANK ...
    assert seen["set_device"] == 3
    assert seen["rendezvous"] == ("localhost", "12345")
    # ... and passed EXPLICITLY as device_id so ProcessGroupNCCL never guesses.
    assert seen["init"]["device_id"] == torch.device("cuda", 3)
    assert seen["init"]["backend"] == "nccl"
    assert seen["init"]["timeout"].total_seconds() == 1800


def test_idempotent_when_already_initialized(monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "2")
    monkeypatch.setenv("MASTER_ADDR", "")
    monkeypatch.setenv("MASTER_PORT", "")
    seen = {}
    monkeypatch.setattr(torch.cuda, "set_device", lambda d: seen.__setitem__("set_device", d))
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def _must_not_init(**kw):
        raise AssertionError("init_process_group must NOT be called when the PG already exists")

    monkeypatch.setattr(torch.distributed, "init_process_group", _must_not_init)

    init_worker_process_group_with_device(master_addr="localhost", master_port=12345, timeout_seconds=600)

    # still pins the device (idempotent), but does not re-create the PG
    assert seen["set_device"] == 2

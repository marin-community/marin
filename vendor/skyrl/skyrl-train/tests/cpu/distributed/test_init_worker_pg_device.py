import os
import torch
from skyrl_train.distributed.utils import init_worker_process_group_with_device


def test_init_pins_device_and_exports_rendezvous_address(monkeypatch):
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

    assert seen["set_device"] == 3
    assert seen["rendezvous"] == ("localhost", "12345")
    assert seen["init"]["device_id"] == torch.device("cuda", 3)
    assert seen["init"]["backend"] == "nccl"
    assert seen["init"]["timeout"].total_seconds() == 1800

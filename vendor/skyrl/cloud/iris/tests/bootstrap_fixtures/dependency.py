# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
import select
from pathlib import Path

__version__ = "test"
flash_attn_varlen_func = object()
GRUG_MOE_ARCHITECTURE = "GrugMoeForCausalLM"

barrier = os.environ.get("FIXTURE_IMPORT_BARRIER")
if __name__ == "transformer_engine.common":
    assert (Path(os.environ["NVRTC_HOME"]) / "lib").is_dir()
    if barrier:
        fd = os.open(barrier, os.O_RDONLY | os.O_NONBLOCK)
        with open(fd, "rb", buffering=0) as gate:
            assert select.select([gate], [], [], 8)[0], "concurrent import never started"
            assert gate.read(1) == b"r"
        raise RuntimeError("megatron import failed")
elif barrier and __name__ in {"flash_attn_interface", "flash_attn.cute.interface"}:
    with open(barrier, "wb", buffering=0) as gate:
        gate.write(b"r")
    raise RuntimeError("flash-attention import failed")


class AutoBridge:
    pass


class CheckpointExporter:
    pass


class DaytonaMachineFactory:
    pass


class ShellSimMachineFactory:
    pass


class RolloutEngine:
    pass


class ModelRegistry:
    @staticmethod
    def get_supported_archs() -> set[str]:
        return {GRUG_MOE_ARCHITECTURE}


def publish_verified_hf_export(*args):
    pass

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Probe one dependency set from the selected frozen runtime profile."""

import argparse
import platform


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("check", choices=("hub-publish", "flash-attention", "megatron", "export", "agents", "vllm"))
    check = parser.parse_args().check
    match check:
        case "hub-publish":
            from skyrl_train.hf_publisher import publish_verified_hf_export  # noqa: F401
        case "flash-attention":
            if platform.machine() == "aarch64":
                from flash_attn.cute.interface import flash_attn_varlen_func  # noqa: F401
            else:
                from flash_attn_interface import flash_attn_varlen_func  # noqa: F401
        case "megatron":
            import transformer_engine.common  # noqa: F401
            from megatron.bridge import AutoBridge  # noqa: F401
        case "export":
            import ray  # noqa: F401
            import torch  # noqa: F401
            from skyrl_train.checkpoint_exporter import CheckpointExporter  # noqa: F401
        case "agents":
            import memray  # noqa: F401
            from rolloutengine.engine import RolloutEngine  # noqa: F401
            from shellbox.backends.daytona.machine import DaytonaMachineFactory  # noqa: F401
            from shellbox.backends.shellsim.machine import ShellSimMachineFactory  # noqa: F401
        case "vllm":
            import quack.activation  # noqa: F401
            import torch
            import vllm
            import vllm._C_stable_libtorch  # noqa: F401
            import vllm.cumem_allocator  # noqa: F401
            from skyrl_train.models.grug_moe import GRUG_MOE_ARCHITECTURE
            from vllm.model_executor.models import ModelRegistry

            assert GRUG_MOE_ARCHITECTURE in ModelRegistry.get_supported_archs()
            print("[rl-iris] frozen runtime ready:", torch.__version__, vllm.__version__)


if __name__ == "__main__":
    main()

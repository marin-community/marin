# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Process-environment ablations shared by the recovery and wedge runners."""

from __future__ import annotations

from collections.abc import Sequence

from levanter.recovery.types import AblationSpec

NCCL_PROTO_ENV = "NCCL_PROTO"
NCCL_NVLS_ENABLE_ENV = "NCCL_NVLS_ENABLE"
NCCL_MAX_NCHANNELS_ENV = "NCCL_MAX_NCHANNELS"


def environment_ablations(*, num_steps: int) -> list[AblationSpec]:
    """Return the process-start environment arms, each run for ``num_steps``."""
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")

    return [
        AblationSpec(name="baseline", env={}, num_steps=num_steps, notes="stock flags"),
        AblationSpec(
            name="nccl-launch-order-implicit",
            env={"NCCL_LAUNCH_ORDER_IMPLICIT": "1"},
            num_steps=num_steps,
            notes="implicit launch order; retained as a previously negative calibration arm",
        ),
        AblationSpec(
            name="cuda-module-loading-eager",
            env={"CUDA_MODULE_LOADING": "EAGER"},
            num_steps=num_steps,
            notes="eager module loading to test the lazy-JIT race hypothesis",
        ),
        AblationSpec(
            name="cuda-cache-disable",
            env={"CUDA_CACHE_DISABLE": "1"},
            num_steps=num_steps,
            notes="disable the driver JIT/SASS cache",
        ),
        AblationSpec(
            name="cuda-launch-blocking",
            env={"CUDA_LAUNCH_BLOCKING": "1"},
            num_steps=num_steps,
            notes="serialize every launch; the one arm that has suppressed both outcomes, kept as a positive control",
        ),
        AblationSpec(
            name="nccl-mnnvl-off",
            env={"NCCL_MNNVL_ENABLE": "0"},
            num_steps=num_steps,
            notes="disable multi-node NVLink; the earlier negative lacked INFO transport provenance",
        ),
        AblationSpec(
            name="nccl-proto-no-ll128",
            env={NCCL_PROTO_ENV: "^LL128"},
            num_steps=num_steps,
            notes="exclude LL128 to test flag corruption on new NVLink",
        ),
        AblationSpec(
            name="nccl-proto-simple",
            env={NCCL_PROTO_ENV: "Simple"},
            num_steps=num_steps,
            notes="force the Simple protocol",
        ),
        AblationSpec(
            name="nccl-cumem-off",
            env={"NCCL_CUMEM_ENABLE": "0"},
            num_steps=num_steps,
            notes="disable the cuMem fabric-handle allocation path",
        ),
        AblationSpec(
            name="nccl-nvls-off",
            env={NCCL_NVLS_ENABLE_ENV: "0"},
            num_steps=num_steps,
            notes="explicit NVLS-off arm; vary the NCCL library separately",
        ),
        AblationSpec(
            name="nccl-nvls-on",
            env={NCCL_NVLS_ENABLE_ENV: "1"},
            num_steps=num_steps,
            notes="explicit NVLS-on arm; vary the NCCL library separately",
        ),
        AblationSpec(
            name="xla-allocator-bfc",
            env={"XLA_PYTHON_CLIENT_ALLOCATOR": "bfc"},
            num_steps=num_steps,
            notes="replace cuda_async with BFC on a memory-feasible shape",
        ),
        AblationSpec(
            name="nccl-cta-policy-default",
            env={"NCCL_CTA_POLICY": "DEFAULT"},
            num_steps=num_steps,
            notes="use the default CTA policy",
        ),
        AblationSpec(
            name="nccl-work-fifo-4mib",
            env={"NCCL_WORK_FIFO_BYTES": "4194304"},
            num_steps=num_steps,
            notes="use a 4 MiB work FIFO",
        ),
        AblationSpec(
            name="nccl-max-nchannels-8",
            env={NCCL_MAX_NCHANNELS_ENV: "8"},
            num_steps=num_steps,
            notes="cap NCCL channels at 8",
        ),
        AblationSpec(
            name="nccl-max-nchannels-4",
            env={NCCL_MAX_NCHANNELS_ENV: "4"},
            num_steps=num_steps,
            notes="cap NCCL channels at 4",
        ),
        AblationSpec(
            name="nccl-nchannels-per-peer-1",
            env={"NCCL_NCHANNELS_PER_PEER": "1"},
            num_steps=num_steps,
            notes="use one channel per peer",
        ),
        AblationSpec(
            name="nccl-buffsize-16mib",
            env={"NCCL_BUFFSIZE": "16777216"},
            num_steps=num_steps,
            notes="use a 16 MiB NCCL buffer",
        ),
        AblationSpec(
            name="nccl-runtime-connect-off",
            env={"NCCL_RUNTIME_CONNECT": "0"},
            num_steps=num_steps,
            notes="establish connections during communicator initialization",
        ),
    ]


def selected_ablations(ablations: Sequence[AblationSpec], names: Sequence[str] | None) -> list[AblationSpec]:
    """Select named arms in caller order, failing on unknown or duplicate names."""
    if names is None:
        return list(ablations)
    if not names:
        raise ValueError("at least one ablation name is required")

    if len(names) != len(set(names)):
        raise ValueError(f"duplicate ablation names: {list(names)}")

    by_name = {ablation.name: ablation for ablation in ablations}
    if len(by_name) != len(ablations):
        raise ValueError("ablation names must be unique")
    unknown = sorted(set(names) - by_name.keys())
    if unknown:
        raise ValueError(f"unknown ablations {unknown}; choose from {sorted(by_name)}")
    return [by_name[name] for name in names]


def environment_ablation_names() -> tuple[str, ...]:
    return tuple(ablation.name for ablation in environment_ablations(num_steps=1))

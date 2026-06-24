# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the experimental MoE dispatch-up subkernel."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MoeDispatchUpConfig:
    """Static options for the single-node MoE dispatch-up prototype."""

    ep_size: int = 8
    top_k: int = 4
    block_m: int = 64
    block_n: int = 128
    block_k: int = 64
    num_stages: int = 4
    prepacked_send: bool = True
    overlap_dispatch_compute: bool = False
    fuse_w13_silu: bool = False
    backward_impl: str = "reference"
    backward_dx_transport: str = "two_stage"
    save_recv_x: bool = True
    save_gate_up: bool = False
    save_h: bool = True
    static_capacity_factor: float = 1.25
    fail_on_overflow: bool = True

    @classmethod
    def get_default(cls) -> "MoeDispatchUpConfig":
        return cls()

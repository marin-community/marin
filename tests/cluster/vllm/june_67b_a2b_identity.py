# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lightweight identity for the June 67B A2B checkpoint and BF16 export."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelIdentity:
    """Checkpoint, export, and golden values that form one model lineage."""

    run_root: str
    checkpoint_step: int
    export_sha256: str
    export_uri: str
    inference_golden_path: Path

    @property
    def executor_info_path(self) -> str:
        return f"{self.run_root}/.executor_info"

    @property
    def checkpoint_path(self) -> str:
        return f"{self.run_root}/checkpoints/step-{self.checkpoint_step}"

    @property
    def vllm_model_name(self) -> str:
        return f"june-67b-a2b-step-{self.checkpoint_step}-bf16"


JUNE_67B_A2B = ModelIdentity(
    run_root=(
        "s3://marin-us-east-02a/marin/grug/"
        "moe_67b_a2b_d2560_ep1_rep8_bs1024_seq65536_sw2k_v4_2048_muon_cooldown_step39k-79ebf3"
    ),
    checkpoint_step=42150,
    export_sha256="781bc3291c81ce282be6762520280ebd5ef5b85e88ba65129c2d0162d48ee632",
    export_uri="s3://marin-us-east-02a/marin/exports/grug/june-67b-a2b/step-42150/hf-bf16-vllm/781bc3291c81ce28/",
    inference_golden_path=Path(__file__).parent / "resources" / "june_tpu_67b_a2b_step_42150_logprobs.json",
)

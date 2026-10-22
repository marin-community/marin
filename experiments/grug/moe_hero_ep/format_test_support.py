# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Small persisted smoke records shared by format accounting and pipeline tests."""

import hashlib
import json

from experiments.grug.moe_hero_ep.format_eval import SmokeRecord, SmokeUpdate
from experiments.grug.moe_hero_ep.sft import SftConfig


def smoke_gradient_report(token_count: int = 1) -> dict:
    return {
        "schema_version": 1,
        "kind": "hero_sft_gradient_structure",
        "execution_id": "real-execution",
        "checkpoint_metadata_digest": "donor-C",
        "measurement_identity_digest": "model-data-batch-mesh-resources",
        "gradients": [
            {"path": ".input_delta", "shape": [token_count, 4], "dtype": "float32"},
            {"path": ".output_delta", "shape": [4, token_count], "dtype": "float32"},
            *[
                {
                    "path": f".adapters.stacked_blocks.stacked.attn.lora_{projection}.{factor}",
                    "shape": shape,
                    "dtype": "float32",
                }
                for projection in ("q", "k", "v", "o")
                for factor, shape in (("a", [1, 4, 16]), ("b", [1, 16, 4]))
            ],
        ],
    }


def smoke_optimizer_report(parameter_bytes: int = 2080) -> dict:
    return {
        "schema_version": 1,
        "execution_id": "real-execution",
        "checkpoint_metadata_digest": "donor-C",
        "measurement_identity_digest": "model-data-batch-mesh-resources",
        "optimizer_step": 0,
        "compact_optimizer_only": True,
        "errors": [],
        "trainable_parameter_bytes": parameter_bytes,
        "optimizer_state_bytes": 2 * parameter_bytes + 16,
        "expected_optimizer_state_bytes": 2 * parameter_bytes + 16,
    }


def smoke_record() -> SmokeRecord:
    return SmokeRecord(
        execution_id="real-execution",
        checkpoint_metadata_digest="donor-C",
        hardware="GB200-NVL72",
        gpu_count=64,
        configuration_digest="config",
        measurement_identity_digest="model-data-batch-mesh-resources",
        sft_config=SftConfig(
            rank=16,
            alpha=16,
            token_ids=(128009,),
            adapter_learning_rate=1e-4,
            vocab_learning_rate=1e-5,
            num_steps=4,
            warmup_steps=1,
            seed=0,
        ),
        terminal_checkpoint="measured-compact-checkpoint",
        terminal_checkpoint_metadata_digest="measured-compact-metadata",
        input_batches_digest="fixed-input",
        sequence_length=16384,
        global_batch_executed_tokens=65536,
        microbatch_size=1,
        gradient_accumulation=4,
        optimizer="Adam-adapters",
        trainable_parameter_bytes=2080,
        optimizer_state_bytes=4176,
        checkpoint_seconds=2,
        compile_seconds=30,
        warmup_updates=1,
        gradient_structure_report="gradient-structure-report.json",
        gradient_structure_report_sha256=hashlib.sha256(json.dumps(smoke_gradient_report()).encode()).hexdigest(),
        compact_optimizer_report="compact-optimizer-report.json",
        compact_optimizer_report_sha256=hashlib.sha256(json.dumps(smoke_optimizer_report()).encode()).hexdigest(),
        compact_gradient_outputs=True,
        preserves_activation_gradients=True,
        updates=tuple(
            SmokeUpdate(
                update=i,
                elapsed_seconds=10 if i == 1 else 2,
                executed_tokens=65536,
                non_padding_tokens=50000,
                supervised_tokens=20000,
                peak_hbm_bytes=100000,
                optimizer_seconds=0.1,
                loss=2 - i / 100,
                gradient_norm=1,
                diagnostic_successes=80 + i,
                diagnostic_cases=100,
            )
            for i in range(1, 5)
        ),
    )

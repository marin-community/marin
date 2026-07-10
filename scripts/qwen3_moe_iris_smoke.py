# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import time

import jax
import jmp
import levanter
from levanter.main.perplexity_gap import GapFinderModelConfig, _load_model_runner, _resolved_model_spec
from levanter.tracker import NoopConfig
from levanter.trainer import TrainerConfig

MODEL_ID = "Qwen/Qwen3-30B-A3B"


def main() -> None:
    trainer = TrainerConfig(
        id="qwen3-moe-iris-smoke",
        mp=jmp.get_policy("bf16"),
        tracker=NoopConfig(),
        train_batch_size=8,
        per_device_parallelism=1,
        per_device_eval_parallelism=1,
        require_accelerator=True,
        log_jaxprs=False,
        log_xla_hlo=False,
    )
    levanter.trainer.initialize(trainer)

    spec = _resolved_model_spec(
        GapFinderModelConfig(
            checkpoint_path=MODEL_ID,
            checkpoint_is_hf=True,
            tokenizer=MODEL_ID,
            trust_remote_code=True,
        )
    )

    started = time.perf_counter()
    with trainer.use_device_mesh():
        runner = _load_model_runner(
            spec=spec,
            trainer=trainer,
            max_eval_length=64,
            compute_axis_mapping=trainer.compute_axis_mapping,
            parameter_axis_mapping=trainer.parameter_axis_mapping,
        )
        tokenized, per_byte_losses = runner.score_texts(
            [
                "Qwen3 MoE smoke test for Levanter perplexity scoring on Iris. "
                "This should produce finite next-token losses."
            ]
        )

    elapsed = time.perf_counter() - started
    result = {
        "model": MODEL_ID,
        "devices": [str(device) for device in jax.devices()],
        "token_count": len(tokenized[0].token_ids),
        "num_bytes": int(tokenized[0].num_bytes),
        "loss_sum": float(per_byte_losses[0].sum()),
        "loss_min": float(per_byte_losses[0].min()),
        "loss_max": float(per_byte_losses[0].max()),
        "elapsed_seconds": elapsed,
    }
    print("QWEN3_MOE_IRIS_SMOKE_RESULT", json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

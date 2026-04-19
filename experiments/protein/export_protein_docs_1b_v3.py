# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert the ``protein-docs-1b-v3-7e87f7`` Levanter checkpoint to HuggingFace format.

The training run writes a Levanter-native (tensorstore/ocdbt) checkpoint at
``gs://marin-us-central1/checkpoints/protein-docs-1b-v3-7e87f7/checkpoints/step-85066``
but no HF export. vLLM needs an HF checkpoint to load, so this step materializes one
at ``.../hf``.

Model is Qwen3 (Llama + QK-norm + sliding-window knobs), 1.4B parameters, trained on
Will Held's ``WillHeld/contactdoc-tokenizer`` protein-docs vocabulary.

Runs on CPU only (the 1.4B model fits in host memory)::

    uv run iris --config=lib/iris/examples/marin.yaml job run \\
        --memory=8GB --disk=8GB --cpu=1 \\
        -e HF_TOKEN <your-hf-token> \\
        -- \\
        python -m experiments.protein.export_protein_docs_1b_v3
"""

from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.qwen import Qwen3Config
from levanter.trainer import TrainerConfig

from marin.execution.executor import executor_main
from marin.export import convert_checkpoint_to_hf_step

TRAINING_OUTPUT = "gs://marin-us-central1/checkpoints/protein-docs-1b-v3-7e87f7"
CHECKPOINT_STEP = 85066
TOKENIZER = "WillHeld/contactdoc-tokenizer"

# Architecture mirrors the values recorded in the run's `.executor_info`.
protein_qwen3_1b_v3 = Qwen3Config(
    max_seq_len=8192,
    hidden_dim=2048,
    intermediate_dim=8192,
    num_layers=21,
    num_heads=16,
    num_kv_heads=16,
    use_sliding_window=False,
    sliding_window=4096,
    rope=Llama3RotaryEmbeddingsConfig(
        theta=500000,
        factor=8.0,
        low_freq_factor=1.0,
        high_freq_factor=4.0,
        original_max_position_embeddings=8192,
    ),
)

protein_docs_1b_v3_hf = convert_checkpoint_to_hf_step(
    name="hf/protein-docs-1b-v3",
    checkpoint_path=f"{TRAINING_OUTPUT}/checkpoints/step-{CHECKPOINT_STEP}",
    trainer=TrainerConfig(),
    model=protein_qwen3_1b_v3,
    tokenizer=TOKENIZER,
    use_cpu=True,
    override_output_path=f"{TRAINING_OUTPUT}/hf",
)


if __name__ == "__main__":
    executor_main(steps=[protein_docs_1b_v3_hf])

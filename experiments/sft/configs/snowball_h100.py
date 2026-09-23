# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A bounded-context Snowball SFT recipe using the standard Levanter chat trainer.

The pinned Hugging Face checkpoint initializes ``SnowballLMHeadModel`` directly;
the optimizer starts fresh. Conversations stay separate because Snowball does
not yet consume Levanter's packed-document attention mask.
"""

from levanter.optim.config import AdamConfig
from levanter.utils.mesh import MeshConfig
from marin.datakit.chat_template import MARIN_CHAT_TEMPLATE

from experiments.sft.launcher import DatasetSpec, HFModel, SFTSpec, run_sft_cli

HF_MODEL = "open-athena/snowball-67b-a2b-base-262k-qk175-skew8"
HF_REVISION = "058ecaf27b9e4f37219df221a51e7d490d58ec3d"
DATASET_REVISION = "45fb28fcc38d352133cb28a1c8a43a2f14fea97b"

SPEC = SFTSpec(
    name="checkpoints/snowball-openthoughts-agent-sft",
    version="2026.09.22",
    model=HFModel(
        model_ref=f"{HF_MODEL}@{HF_REVISION}",
        tokenizer_path=HF_MODEL,
        model_type="snowball",
        eos_token_ids=(128001, 128009),
    ),
    chat_template=MARIN_CHAT_TEMPLATE,
    datasets=[
        DatasetSpec(
            slug="openthoughts-agent",
            hf_dataset_id="open-thoughts/OpenThoughts-Agent-SFT-100K",
            revision=DATASET_REVISION,
            adapter_kwargs={"conversation_column": "conversations"},
            weight=1.0,
        )
    ],
    optimizer=AdamConfig(
        learning_rate=5e-5,
        beta1=0.9,
        beta2=0.95,
        epsilon=1e-8,
        weight_decay=0.0,
        max_grad_norm=1.0,
        lr_schedule="cosine",
        warmup=0.0,
    ),
    mesh=MeshConfig(axes={"data": 1, "replica": 1, "model": 1, "expert": -1}),
    seq_len=4096,
    pack=False,
    batch_size=32,
    num_train_steps=10,
    wandb_project="snowball-sft",
)


if __name__ == "__main__":
    run_sft_cli(SPEC)

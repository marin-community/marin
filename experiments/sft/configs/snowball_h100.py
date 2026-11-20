# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A bounded-context Snowball SFT recipe using the standard Levanter chat trainer.

The pinned Hugging Face checkpoint initializes ``SnowballLMHeadModel`` directly;
the optimizer starts fresh. This recipe keeps conversations separate (``pack=False``).
Snowball also supports Levanter's packed-document attention mask.
Import ``VLLM_PARITY_SPEC`` to enable the serving-parity settings available on ``main``;
see ``docs/references/serving-parity.md`` for the full knob inventory and evidence boundaries.
"""

from dataclasses import replace

from levanter.models.snowball import SnowballConfig
from levanter.optim.config import AdamConfig
from levanter.utils.mesh import MeshConfig
from marin.datakit.chat_template import MARIN_CHAT_TEMPLATE

from experiments.sft.launcher import DatasetSpec, HFModel, SFTSpec, run_sft_cli

HF_MODEL = "open-athena/snowball-67b-a2b-base-262k-qk175-skew8"
HF_REVISION = "058ecaf27b9e4f37219df221a51e7d490d58ec3d"
DATASET_REVISION = "45fb28fcc38d352133cb28a1c8a43a2f14fea97b"

_SNOWBALL_MODEL = HFModel(
    model_ref=f"{HF_MODEL}@{HF_REVISION}",
    model=SnowballConfig(),
    eos_token_ids=(128001, 128009),
)

SPEC = SFTSpec(
    name="checkpoints/snowball-openthoughts-agent-sft",
    version="2026.09.23",
    model=_SNOWBALL_MODEL,
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
    mesh=MeshConfig(
        axes={"data": 1, "replica": 1, "model": 1, "context": 4, "expert": -1},
        compute_mapping={
            "batch": ["replica_dcn", "data", "expert"],
            "position": "context",
        },
    ),
    seq_len=32768,
    pack=False,
    batch_size=8,
    num_train_steps=10,
    wandb_project="snowball-sft",
)

# Opt-in Snowball SFT preset enabling the serving-parity knobs on main that measurably help.
# Add router_combine_fp32=True once PR 10686 merges. Diagnostic attention_implementation="torch_order"
# and XLA excess-precision flags are intentionally omitted; see docs/references/serving-parity.md.
VLLM_PARITY_SPEC = replace(
    SPEC,
    name="checkpoints/snowball-openthoughts-agent-sft-vllm-parity",
    model=replace(
        _SNOWBALL_MODEL,
        model=SnowballConfig(
            moe_capacity_factor=8.0,  # match the 8-rank expert axis on 32xH100
            router_dot_fp32=True,
            fp32_activations=True,
        ),
    ),
    field_dtype_overrides={"router_bias": "float32"},
)


if __name__ == "__main__":
    run_sft_cli(SPEC)

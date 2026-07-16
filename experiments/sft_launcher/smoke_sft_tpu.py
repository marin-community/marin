# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TPU SMOKE for the generic ``sft_step`` path (NOT a committed artifact).

Proves the general launcher end-to-end on a preemptible TPU slice with a TRIMMED spec:
tiny public Qwen3-0.6B (the Delphi target arch), a small chat dataset, a Qwen3 chat template
carrying the ``{% generation %}`` block, and a handful of train steps -> one HF export. This
exercises: graph resolves -> native transform_dataset_step tokenize/pack -> Levanter SFT
(initialize_from_hf) runs a few steps -> HF export step fires. It is NOT a real train.
"""
from __future__ import annotations

from fray.types import ResourceConfig
from marin.execution.lazy import lower
from marin.execution.step_runner import StepRunner

from experiments.sft_launcher.marin_sft_launcher import DatasetSpec, SFTSpec, sft_step

# Minimal Qwen3 chat template with the Levanter {% generation %} span wrapping the assistant turn
# (header EXCLUDED, content + <|im_end|> INCLUDED) — the completions-only supervised mask.
QWEN3_SMOKE_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "<|im_start|>{{ message['role'] }}\n"
    "{% if message['role'] == 'assistant' %}"
    "{% generation %}{{ message['content'] }}<|im_end|>{% endgeneration %}\n"
    "{% else %}{{ message['content'] }}<|im_end|>\n"
    "{% endif %}"
    "{% endfor %}"
)

_SMOKE_DATA = DatasetSpec(
    slug="norobots",
    hf_dataset_id="HuggingFaceH4/no_robots",  # ~10k rows, OpenAI `messages` (role/content)
    revision="main",
    adapter_kwargs=dict(conversation_column="messages"),  # role/content, user/assistant defaults
    weight=1.0,
)

SPEC = SFTSpec(
    name="checkpoints/smoke-sft-tpu-qwen3-0p6b",
    version="2026.07.15-dev",  # -dev = always rebuild (no cache reuse)
    model_ref="Qwen/Qwen3-0.6B",  # tiny public Qwen3 -> initialize_from_hf
    tokenizer_path="Qwen/Qwen3-0.6B",
    chat_template=QWEN3_SMOKE_CHAT_TEMPLATE,
    datasets=[_SMOKE_DATA],
    resources=ResourceConfig.with_tpu("v6e-4"),
    seq_len=1024,
    lr=1e-5,
    batch_size=8,
    num_train_steps=20,  # a handful of steps -> HF export at step 20
    eos_token_ids=(151643, 151645),  # Qwen3: <|endoftext|> + <|im_end|>
    wandb_project="marin-sft-launcher-smoke",
)


if __name__ == "__main__":
    StepRunner().run([lower(sft_step(SPEC))])

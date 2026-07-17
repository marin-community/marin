# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TPU smoke of the generic ``sft_step`` launcher, on the standing Marin cluster.

Proves the launcher end-to-end on a preemptible TPU slice with a trimmed spec: tiny public
Qwen3-0.6B (the Delphi target arch), a small chat dataset, a Qwen3 chat template carrying the
``{% generation %}`` block, and a handful of train steps into one HF export. It exercises: graph
resolves -> native ``transform_dataset_step`` tokenize/pack -> Levanter SFT (``initialize_from_hf``)
runs a few steps -> HF export. It is not a real training run.

Marked ``cluster`` so it never runs by default; the ``marin-cluster-smoke`` workflow runs it. The
``iris_client`` fixture binds the ``marin`` controller as the current Fray client, so ``StepRunner``
submits there. Launch it on demand with::

    uv run pytest tests/cluster/sft/test_smoke_sft_tpu.py \
      -m cluster -o addopts= --import-mode=importlib --timeout=0 -vv -s

PYTEST_DONT_REWRITE: the step dispatches serialized remote functions that must not depend on pytest.
"""
from __future__ import annotations

import dataclasses

import pytest
from iris.client import IrisClient
from marin.execution.lazy import lower
from marin.execution.step_runner import StepRunner

from experiments.sft.launcher import DatasetSpec, HFModel, SFTSpec, resources_from_accelerator, sft_step

pytestmark = pytest.mark.cluster

# Minimal Qwen3 chat template with the Levanter {% generation %} span wrapping the assistant turn
# (header excluded, content + <|im_end|> included) — the completions-only supervised mask.
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

SMOKE_SPEC = SFTSpec(
    name="checkpoints/smoke-sft-tpu-qwen3-0p6b",
    version="2026.07.15-dev",  # -dev = always rebuild (no cache reuse)
    model=HFModel("Qwen/Qwen3-0.6B"),  # tiny public Qwen3, used verbatim -> initialize_from_hf
    chat_template=QWEN3_SMOKE_CHAT_TEMPLATE,
    datasets=[_SMOKE_DATA],
    seq_len=1024,
    lr=1e-5,
    batch_size=8,
    num_train_steps=20,  # a handful of steps -> HF export at step 20
    eos_token_ids=(151643, 151645),  # Qwen3: <|endoftext|> + <|im_end|>
    wandb_project="marin-sft-launcher-smoke",
)

SMOKE_ACCELERATOR = "v6e-4"


def test_smoke_sft_tpu(iris_client: IrisClient, smoke_region: str) -> None:
    # iris_client binds the marin controller as the current Fray client, so StepRunner submits there.
    # smoke_region pins the slice to a region and binds the storage root to the same region, so the
    # multi-step run (tokenize -> train -> export) shares its cache and reads/writes region-locally.
    resources = dataclasses.replace(resources_from_accelerator(SMOKE_ACCELERATOR), regions=(smoke_region,))
    StepRunner().run([lower(sft_step(SMOKE_SPEC, resources))])

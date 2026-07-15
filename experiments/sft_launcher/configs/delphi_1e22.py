# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Delphi 1e22 dense-SFT — the first worked example of the general ``sft_step`` launcher.

Reproduces the LLaMA-Factory magpie 90 / warmup 10 cold-start SFT recipe in Levanter, validated at
LF parity (MATH-500 48.0 vs 44.2) on the Delphi Levanter-SFT parity experiment. This is a plain
``SFTSpec`` that PASSES IN the Delphi v0 chat template — the launcher itself is model-agnostic.

  * dataset (math-STRONG) = magpie ``Magpie-Align/Magpie-Llama-3.3-Pro-500K-Filtered`` (ShareGPT
    ``conversations``/``from``/``value``) at 0.9, blended with the fixed ``delphi_warmup`` CoT slice
    ``laion/llama-nemotron-science-reasoning-on-le3000tok-100k`` (OpenAI ``messages``) at 0.1.
  * A math-WEAK variant just swaps the first DatasetSpec -> ``nyu-dice-lab/wildchat50m-rewild-sft-385700``
    (``conversation``/role/content) at the same weights + LR (see ``delphi_1e22_wc50m`` below).

PREREQUISITES (stage before launch — see the parity experiment's CONFIG_NOTES):
  * ``model_ref``/``tokenizer_path`` MUST point at a PREPARED delphi checkpoint+tokenizer (reserved
    slots renamed so ``<|start_think|>``=128002 ... are SINGLE ids, embeddings mean-init'd). The raw
    ``laion/delphi-*`` tokenizers fragment the think/tool tokens.
  * ``revision`` pins are placeholders (``"main"``) — resolve the 7-char commit for each dataset so the
    transform fingerprints are content-stable (TODO).
  * ``num_train_steps`` = PACKED 1-epoch (magpie 313,007,558 tokens / 4096 / 0.9 / 16 = 5307); recompute
    per dataset (ChatLmDatasetFormat packs ~6.5 docs/slot -> raw-row counts over-train ~6.5x).
"""
from fray.types import ResourceConfig

from experiments.sft_launcher.delphi_chat_template import DELPHI_V0_CHAT_TEMPLATE
from experiments.sft_launcher.marin_sft_launcher import DatasetSpec, SFTSpec

# Prepared (reserved-slot-renamed + mean-init'd) checkpoint + tokenizer dirs. Stage per prefix
# (gs:// for TPU, s3:// for CoreWeave). Placeholders here — set before launch.
DELPHI_1E22_PREPARED_CKPT = "laion/delphi-1e22-p33m67-32p07b-lr0.67-54770ae7"  # -> prepared dir
DELPHI_PREPARED_TOKENIZER = "laion/delphi-1e22-p33m67-32p07b-lr0.67-54770ae7"  # -> prepared dir

_MAGPIE = DatasetSpec(
    slug="magpie",  # math-STRONG
    hf_dataset_id="Magpie-Align/Magpie-Llama-3.3-Pro-500K-Filtered",
    revision="main",  # TODO: pin 7-char commit
    adapter_kwargs=dict(
        conversation_column="conversations",
        role_key="from",
        content_key="value",
        user_value="human",
        assistant_value="gpt",
    ),
    weight=0.9,
)

_WARMUP = DatasetSpec(
    slug="delphi_warmup",  # fixed 10% CoT slice (already OpenAI messages/{role,content})
    hf_dataset_id="laion/llama-nemotron-science-reasoning-on-le3000tok-100k",
    revision="main",  # TODO: pin 7-char commit
    adapter_kwargs=dict(),  # multi_turn_adapter defaults: messages / role / user / assistant
    weight=0.1,
)

# math-WEAK counterpart (same weights + LR; only the instruction dataset changes -> the dataset is the
# only varying factor in the strong-vs-weak contrast).
_WILDCHAT_386K = DatasetSpec(
    slug="wc386k",  # math-WEAK (wildchat50m)
    hf_dataset_id="nyu-dice-lab/wildchat50m-rewild-sft-385700",
    revision="main",  # TODO: pin 7-char commit
    adapter_kwargs=dict(conversation_column="conversation"),  # role/content, user/assistant defaults
    weight=0.9,
)

# TPU is the native path; for CoreWeave swap in
# ResourceConfig.with_gpu("H100", count=8, cpu=32, disk="128G", ram="128G", regions=[ANY_REGION]).
_RESOURCES = ResourceConfig.with_tpu("v4-64")

# The math-STRONG DoD config (the parity reproduction).
SPEC = SFTSpec(
    name="checkpoints/delphi-1e22-magpie-warmup-levanter-sft",
    version="2026.07.15",
    model_ref=DELPHI_1E22_PREPARED_CKPT,
    tokenizer_path=DELPHI_PREPARED_TOKENIZER,
    chat_template=DELPHI_V0_CHAT_TEMPLATE,  # <-- the Delphi jinja passed IN as a parameter
    datasets=[_MAGPIE, _WARMUP],
    resources=_RESOURCES,
    seq_len=4096,
    lr=1e-5,
    batch_size=16,
    num_train_steps=5307,  # PACKED 1-epoch (magpie); recompute if the dataset/seq_len changes
)

# The math-WEAK sibling (swap magpie -> wildchat50m; recompute num_train_steps for its token count).
SPEC_WC50M = SFTSpec(
    name="checkpoints/delphi-1e22-wc50m-warmup-levanter-sft",
    version="2026.07.15",
    model_ref=DELPHI_1E22_PREPARED_CKPT,
    tokenizer_path=DELPHI_PREPARED_TOKENIZER,
    chat_template=DELPHI_V0_CHAT_TEMPLATE,
    datasets=[_WILDCHAT_386K, _WARMUP],
    resources=_RESOURCES,
    seq_len=4096,
    lr=1e-5,
    batch_size=16,
    num_train_steps=5307,  # TODO: recompute from the wildchat_386k packed token count
)

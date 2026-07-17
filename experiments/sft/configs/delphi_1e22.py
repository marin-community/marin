# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Delphi 1e22 dense-SFT — the first worked example of the general ``sft_step`` launcher.

Reproduces the LLaMA-Factory magpie-90 / warmup-10 cold-start SFT recipe in Levanter, validated at
LLaMA-Factory parity (MATH-500 48.0 vs 44.2) on the Delphi Levanter-SFT parity experiment. It is a
plain ``SFTSpec`` that passes in the Delphi v0 chat template, so the launcher itself stays
model-agnostic.

  * math-strong source: magpie ``Magpie-Align/Magpie-Llama-3.3-Pro-500K-Filtered`` (ShareGPT
    ``conversations``/``from``/``value``) at weight 0.9, blended with the fixed ``delphi_warmup``
    CoT slice ``laion/llama-nemotron-science-reasoning-on-le3000tok-100k`` (OpenAI ``messages``) at
    0.1.
  * math-weak variant (``SPEC_WC50M``): swaps the first ``DatasetSpec`` for
    ``nyu-dice-lab/wildchat50m-rewild-sft-385700`` at the same weights and LR, so the instruction
    dataset is the only varying factor.

``model_ref``/``tokenizer_path`` point at a *prepared* Delphi checkpoint + tokenizer (reserved
slots renamed so ``<|start_think|>``=128002 … are single ids, embeddings reinitialized), staged per
prefix before launch. :func:`delphi_prepared_checkpoint_step` regenerates that prepared checkpoint
reproducibly from the raw ``laion/delphi-*`` base — run it to (re)stage the artifact this config
points at. A future config can instead depend on such a step directly (``PreparedModel``) to build
its inputs from a clean prefix.

Remaining prerequisites:
  * ``revision`` pins are placeholders (``"main"``); resolve the 7-char commit per dataset so the
    transform fingerprints are content-stable.
  * ``num_train_steps`` is a packed 1-epoch count (magpie 313,007,558 tokens / 4096 / 0.9 / 16 =
    5307), recomputed per dataset because chat packing puts several conversations per sequence.
    Deriving it from ``num_train_epochs`` against a chat ``TokenizedCache`` is tracked in #7244.

Launch on a TPU slice or CoreWeave H100s::

    python -m experiments.sft.configs.delphi_1e22 --accelerator v4-64
    python -m experiments.sft.configs.delphi_1e22 --accelerator 8xH100
"""

from marin.execution.lazy import ArtifactStep

from experiments.sft.delphi_chat_template import DELPHI_RESERVED_TOKEN_RENAMES, DELPHI_V0_CHAT_TEMPLATE
from experiments.sft.launcher import DatasetSpec, HFModel, SFTSpec, run_sft_cli
from experiments.sft.prepare_checkpoint import prepare_checkpoint_step

# Prepared (reserved-slot-renamed + embedding-reinit'd) checkpoint + tokenizer. Stage per prefix
# (gs:// for TPU, s3:// for CoreWeave) before launch; build it with delphi_prepared_checkpoint_step.
DELPHI_1E22_PREPARED_CKPT = "laion/delphi-1e22-p33m67-32p07b-lr0.67-54770ae7"
DELPHI_PREPARED_TOKENIZER = "laion/delphi-1e22-p33m67-32p07b-lr0.67-54770ae7"

# Raw base for the preparation step: reserved slots still named ``<|reserved_special_token_N|>``.
DELPHI_1E22_BASE_MODEL = "laion/delphi-1e22-p33m67-32p07b-lr0.67-54770ae7"
DELPHI_1E22_BASE_REVISION = "56cc5aaf65cd59d6632962cee636a0423abfbf66"


def delphi_prepared_checkpoint_step() -> ArtifactStep:
    """The ``ArtifactStep`` that reproducibly builds the prepared Delphi checkpoint from the raw base.

    Renames the reserved slots and reinitializes their embedding rows. ``SPEC`` points ``model_ref``
    at a staged copy of this step's output; run it to (re)generate that artifact.
    """
    return prepare_checkpoint_step(
        name="checkpoints/delphi-1e22-prepared",
        version="2026.07.16",
        source_model=DELPHI_1E22_BASE_MODEL,
        source_revision=DELPHI_1E22_BASE_REVISION,
        token_renames=DELPHI_RESERVED_TOKEN_RENAMES,
    )


_MAGPIE = DatasetSpec(
    slug="magpie",  # math-strong
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

# math-weak counterpart: same weights and LR, only the instruction dataset changes.
_WILDCHAT_386K = DatasetSpec(
    slug="wc386k",  # math-weak (wildchat50m)
    hf_dataset_id="nyu-dice-lab/wildchat50m-rewild-sft-385700",
    revision="main",  # TODO: pin 7-char commit
    adapter_kwargs=dict(conversation_column="conversation"),  # role/content, user/assistant defaults
    weight=0.9,
)

# The math-strong DoD config (the parity reproduction).
SPEC = SFTSpec(
    name="checkpoints/delphi-1e22-magpie-warmup-levanter-sft",
    version="2026.07.15",
    model=HFModel(DELPHI_1E22_PREPARED_CKPT, tokenizer_path=DELPHI_PREPARED_TOKENIZER),
    chat_template=DELPHI_V0_CHAT_TEMPLATE,  # the Delphi jinja passed in as a parameter
    datasets=[_MAGPIE, _WARMUP],
    seq_len=4096,
    lr=1e-5,
    batch_size=16,
    num_train_steps=5307,  # packed 1-epoch (magpie); recompute if the dataset/seq_len changes
)

# The math-weak sibling (swap magpie -> wildchat50m; recompute num_train_steps for its token count).
SPEC_WC50M = SFTSpec(
    name="checkpoints/delphi-1e22-wc50m-warmup-levanter-sft",
    version="2026.07.15",
    model=HFModel(DELPHI_1E22_PREPARED_CKPT, tokenizer_path=DELPHI_PREPARED_TOKENIZER),
    chat_template=DELPHI_V0_CHAT_TEMPLATE,
    datasets=[_WILDCHAT_386K, _WARMUP],
    seq_len=4096,
    lr=1e-5,
    batch_size=16,
    num_train_steps=5307,  # TODO: recompute from the wildchat_386k packed token count
)


if __name__ == "__main__":
    run_sft_cli(SPEC)

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""General-purpose SFT launcher on marin's lazy-artifact (``ArtifactStep``) flow.

A parameterized ``sft_step(spec, resources)`` that expresses a chat-SFT run as a lazy
``ArtifactStep[LevanterCheckpoint]``: dataset transform (ShareGPT/OpenAI -> canonical) ->
chat tokenize/pack (a *pluggable* chat template + completions-only masking) -> Levanter
SFT (``initialize_from_hf``) -> HF export. The chat template, dataset mix, model, sequence
length, and packing are all fields of :class:`SFTSpec` — nothing is hardcoded to a particular
model family — and the accelerator is chosen at launch time (``--accelerator``), not baked into
the spec.

``configs/delphi_1e22.py`` is the first worked example: one ``SFTSpec`` that supplies the
Delphi v0 chat template (``delphi_chat_template.DELPHI_V0_CHAT_TEMPLATE``) and the
magpie(math-strong) 90 / warmup 10 mixture.
"""

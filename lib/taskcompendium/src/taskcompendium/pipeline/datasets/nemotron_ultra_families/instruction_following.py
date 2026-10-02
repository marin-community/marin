# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""instruction-following contracts and explicit pinned blend selections."""

from taskcompendium.pipeline.datasets.nemotron_ultra_source import quality_source

FAMILY_MODULE = __name__
CRITERIA = (
    (
        "Identify the underlying content request and verify that all supplied formal constraints "
        "and semantic rubric requirements can hold together."
    ),
    (
        "Preserve every conversation turn. Public schemas/examples are legitimate context. "
        "Distinguish factual extraction from authorized arbitrary schema generation, and compare "
        "private constraints with public instructions."
    ),
)

SOURCES = tuple(
    quality_source(blend, selector, upstream, "instruction-following", CRITERIA, FAMILY_MODULE)
    for blend, selector, upstream in (
        (
            "mopd",
            "ultra_sft_step3200_calendar_v2",
            "https://huggingface.co/datasets/nvidia/Nemotron-RL-Instruction-Following-Calendar-v2",
        ),
        (
            "mopd",
            "ultra_sft_step3200_instruction_following",
            "https://huggingface.co/datasets/nvidia/Nemotron-RL-instruction_following",
        ),
        (
            "mopd",
            "ultra_sft_step3200_multichallenge_len40k",
            "https://huggingface.co/datasets/nvidia/Nemotron-RL-Instruction-Following-MultiTurnChat-v1",
        ),
        (
            "mopd",
            "ultra_sft_step3200_structured_outputs_v2",
            "https://huggingface.co/datasets/nvidia/Nemotron-RL-Instruction-Following-Structured-Outputs-v2",
        ),
        (
            "mopd",
            "ultra_v3_agentic_rl_step73_citation_format_v2",
            "https://huggingface.co/datasets/nvidia/Nemotron-RL-Instruction-Following-Citation-Formatting-v1",
        ),
        (
            "mopd",
            "ultra_v3_agentic_rl_step73_freeform_text_v2",
            "https://huggingface.co/datasets/nvidia/Nemotron-RL-Instruction-Following-Free-Form-Formatting-v1",
        ),
        (
            "mopd",
            "ultra_v3_agentic_rl_step73_structured_outputs_v2",
            "https://huggingface.co/datasets/nvidia/Nemotron-RL-Instruction-Following-Structured-Outputs-v2",
        ),
        (
            "rlvr1",
            "ultra_sft_step3200_calendar_v2",
            "https://huggingface.co/datasets/nvidia/Nemotron-RL-Instruction-Following-Calendar-v2",
        ),
        (
            "rlvr1",
            "ultra_sft_step3200_instruction_following",
            "https://huggingface.co/datasets/nvidia/Nemotron-RL-instruction_following",
        ),
        (
            "rlvr1",
            "ultra_sft_step3200_multichallenge_len40k",
            "https://huggingface.co/datasets/nvidia/Nemotron-RL-Instruction-Following-MultiTurnChat-v1",
        ),
        (
            "rlvr1",
            "ultra_sft_step3200_structured_outputs_v2",
            "https://huggingface.co/datasets/nvidia/Nemotron-RL-Instruction-Following-Structured-Outputs-v2",
        ),
        (
            "rlvr2",
            "ultra_sft_step3200_calendar_v2",
            "https://huggingface.co/datasets/nvidia/Nemotron-RL-Instruction-Following-Calendar-v2",
        ),
        (
            "rlvr2",
            "ultra_sft_step3200_ds2_freeform",
            "https://huggingface.co/datasets/nvidia/Nemotron-RL-Instruction-Following-Free-Form-Formatting-v1",
        ),
        (
            "rlvr2",
            "ultra_sft_step3200_ds3_citation",
            "https://huggingface.co/datasets/nvidia/Nemotron-RL-Instruction-Following-Citation-Formatting-v1",
        ),
        (
            "rlvr2",
            "ultra_sft_step3200_instruction_following",
            "https://huggingface.co/datasets/nvidia/Nemotron-RL-instruction_following",
        ),
        (
            "rlvr2",
            "ultra_sft_step3200_multichallenge_len40k",
            "https://huggingface.co/datasets/nvidia/Nemotron-RL-Instruction-Following-MultiTurnChat-v1",
        ),
        (
            "rlvr2",
            "ultra_sft_step3200_structured_outputs_v2",
            "https://huggingface.co/datasets/nvidia/Nemotron-RL-Instruction-Following-Structured-Outputs-v2",
        ),
        (
            "rlvr2",
            "ultra_sft_step3200_structured_outputs_v3",
            "https://huggingface.co/datasets/nvidia/Nemotron-RL-Instruction-Following-Structured-Outputs-v2",
        ),
    )
)

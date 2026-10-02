# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""tool-use contracts and explicit pinned blend selections."""

from taskcompendium.pipeline.datasets.nemotron_ultra.source import ACTION_COMPARISON_CRITERION, quality_source

FAMILY_MODULE = __name__
CRITERIA = (
    (
        "Check the complete role/tool sequence and advertised schemas against expected_action, "
        "scenario, and source environment state."
    ),
    (
        "Historical observations are public context. Expected future tool calls and reward state "
        "are private evidence; multiple valid actions require the original comparison policy "
        "rather than invented exact matching."
    ),
    (ACTION_COMPARISON_CRITERION),
)

SOURCES = tuple(
    quality_source(blend, selector, upstream, "tool-use", CRITERIA, FAMILY_MODULE)
    for blend, selector, upstream in (
        (
            "mopd",
            "ultra_sft_step3200_toolcall_schema",
            "https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1",
        ),
        (
            "rlvr1",
            "ultra_sft_step3200_tau_pivot",
            "https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-v1",
        ),
        (
            "rlvr1",
            "ultra_sft_step3200_toolcall_schema",
            "https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1",
        ),
        (
            "rlvr2",
            "ultra_sft_step3200_tau_pivot",
            "https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-v1",
        ),
        (
            "rlvr2",
            "ultra_sft_step3200_toolcall_schema",
            "https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1",
        ),
    )
)

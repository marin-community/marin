# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""qa-multiple-choice contracts and explicit pinned blend selections."""

from taskcompendium.pipeline.datasets.nemotron_ultra.source import quality_source

FAMILY_MODULE = __name__
CRITERIA = (
    "Check option labels and answer encoding against the complete public choices and expected_answer.",
    (
        "Knowledge questions can use ordinary external knowledge. Missing referenced passages, "
        "images, or material contradictions are defects; source labels must not be exposed as "
        "public hints."
    ),
)

SOURCES = tuple(
    quality_source(blend, selector, upstream, "qa-multiple-choice", CRITERIA, FAMILY_MODULE)
    for blend, selector, upstream in (
        ("mopd", "ultra_sft_step3200_stem_mcqa", "https://huggingface.co/datasets/nvidia/Nemotron-RL-knowledge-mcqa"),
        (
            "mopd",
            "ultra_sft_step3200_stem_mcqa_cot_rima_new",
            "https://huggingface.co/datasets/nvidia/Nemotron-SFT-Science-v2",
        ),
        ("rlvr1", "ultra_sft_step3200_stem_mcqa", "https://huggingface.co/datasets/nvidia/Nemotron-RL-knowledge-mcqa"),
        (
            "rlvr1",
            "ultra_sft_step3200_stem_mcqa_cot_rima_new",
            "https://huggingface.co/datasets/nvidia/Nemotron-SFT-Science-v2",
        ),
        ("rlvr2", "ultra_sft_step3200_stem_mcqa", "https://huggingface.co/datasets/nvidia/Nemotron-RL-knowledge-mcqa"),
        (
            "rlvr2",
            "ultra_sft_step3200_stem_mcqa_cot_rima_new",
            "https://huggingface.co/datasets/nvidia/Nemotron-SFT-Science-v2",
        ),
    )
)

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Family assignments audited against upstream cards/schema and selected SkyRL loaders."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FamilyAudit:
    family: str
    evidence_url: str


# Reviewed 2026-09-28. Eurus and Nemotron assignments describe the selected subset.
FAMILY_AUDITS = {
    "AI-MO/NuminaMath-CoT": FamilyAudit(
        "math-answer",
        "https://huggingface.co/datasets/AI-MO/NuminaMath-CoT/blob/9d8d210c9f6a36c8f3cd84045668c9b7800ef517/README.md",
    ),
    "Anthropic/hh-rlhf": FamilyAudit(
        "preference",
        "https://huggingface.co/datasets/Anthropic/hh-rlhf/blob/09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa/README.md",
    ),
    "BytedTsinghua-SIA/DAPO-Math-17k": FamilyAudit(
        "math-answer",
        "https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k/blob/65877096c24ffa7abc4e4fa5edb95cf3413a5674/README.md",
    ),
    "ChilleD/SVAMP": FamilyAudit(
        "math-answer",
        "https://huggingface.co/datasets/ChilleD/SVAMP/blob/5e0bf1e5e7c0e9c4bc39180d224f41f3f801b7ef/README.md",
    ),
    "EleutherAI/hendrycks_math": FamilyAudit(
        "math-answer",
        "https://huggingface.co/datasets/EleutherAI/hendrycks_math/blob/21a5633873b6a120296cce3e2df9d5550074f4a3/README.md",
    ),
    "HuggingFaceH4/MATH-500": FamilyAudit(
        "math-answer",
        "https://huggingface.co/datasets/HuggingFaceH4/MATH-500/blob/6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be/README.md",
    ),
    "HuggingFaceH4/aime_2024": FamilyAudit(
        "math-answer",
        "https://huggingface.co/datasets/HuggingFaceH4/aime_2024/blob/2fe88a2f1091d5048c0f36abc874fb997b3dd99a/README.md",
    ),
    "Idavidrein/gpqa": FamilyAudit(
        "qa-multiple-choice",
        "https://huggingface.co/datasets/Idavidrein/gpqa/blob/83022cefff930aea54f654c0b282e74b9eeda5c6/README.md",
    ),
    "PRIME-RL/Eurus-2-RL-Data": FamilyAudit(
        "competitive-programming",
        "https://huggingface.co/datasets/PRIME-RL/Eurus-2-RL-Data/blob/9776b13264b5aaa0b16495fcf086a0a8d86fd655/README.md",
    ),
    "agentica-org/DeepScaleR-Preview-Dataset": FamilyAudit(
        "math-answer",
        "https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset/blob/b6ae8c60f5c1f2b594e2140b91c49c9ad0949e29/README.md",
    ),
    "allenai/RLVR-IFeval": FamilyAudit(
        "instruction-following",
        "https://huggingface.co/datasets/allenai/RLVR-IFeval/blob/47c03c73621c4aab2b824b7818681117d662770e/README.md",
    ),
    "allenai/RLVR-MATH": FamilyAudit(
        "math-answer",
        "https://huggingface.co/datasets/allenai/RLVR-MATH/blob/bd2a93551b503a395fadd1a740d957559cfe6f3c/README.md",
    ),
    "chaochun/nlu-asdiv-dataset": FamilyAudit(
        "math-answer",
        "https://github.com/chaochun/nlu-asdiv-dataset/blob/883f90a9a65bf00304ba8f37423910fe743abc47/README.md",
    ),
    "codeparrot/apps": FamilyAudit(
        "competitive-programming",
        "https://huggingface.co/datasets/codeparrot/apps/blob/21e74ddf8de1a21436da12e3e653065c5213e9d1/README.md",
    ),
    "di-zhang-fdu/AIME_1983_2024": FamilyAudit(
        "math-answer",
        "https://huggingface.co/datasets/di-zhang-fdu/AIME_1983_2024/blob/3e2cc86390666c5c756622afc0eeb9e6194496bc/README.md",
    ),
    "gretelai/synthetic_text_to_sql": FamilyAudit(
        "text-to-sql",
        "https://huggingface.co/datasets/gretelai/synthetic_text_to_sql/blob/740ab236e64503fba51be1101df7a1be83bf455d/README.md",
    ),
    "nvidia/Llama-Nemotron-Post-Training-Dataset": FamilyAudit(
        "instruction-following",
        "https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Dataset/blob/ab2a40d258a6a4d9d4c277d702aeea445081766c/README.md",
    ),
    "nvidia/Nemotron-RL-Ultra-Training-Blends": FamilyAudit(
        "mixed",
        "https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends/blob/79f8eda15ea12e1adf7bb14dcb338a29d391b80e/README.md",
    ),
    "nvidia/OpenScience": FamilyAudit(
        "qa-multiple-choice",
        "https://huggingface.co/datasets/nvidia/OpenScience/blob/7bd0437e4756f761768fe7e5cebeaa75480a4fd6/README.md",
    ),
    "open-r1/verifiable-coding-problems-python": FamilyAudit(
        "competitive-programming",
        "https://huggingface.co/datasets/open-r1/verifiable-coding-problems-python/blob/b761a24a95fa03289a231d2d31c183636ffb9833/README.md",
    ),
    "open-thought/reasoning-gym": FamilyAudit(
        "reasoning-gym",
        "https://github.com/open-thought/reasoning-gym/blob/49b07130b3fcd12f2d064bba7c43869543a0e7e7/README.md",
    ),
    "openai/gsm8k": FamilyAudit(
        "math-answer",
        "https://huggingface.co/datasets/openai/gsm8k/blob/740312add88f781978c0658806c59bc2815b9866/README.md",
    ),
    "pafitis/HARDMath_processed_training": FamilyAudit(
        "math-answer",
        "https://huggingface.co/datasets/pafitis/HARDMath_processed_training/blob/937e9f10356e31e854f6efb9a2507f1e200c8b25/README.md",
    ),
    "trl-lib/kto-mix-14k": FamilyAudit(
        "preference",
        "https://huggingface.co/datasets/trl-lib/kto-mix-14k/blob/4470f033f33364e7d064c9f920c3df54d0cce767/README.md",
    ),
}

BENCHMARK_DATASETS = {
    "HuggingFaceH4/MATH-500",
    "HuggingFaceH4/aime_2024",
    "Idavidrein/gpqa",
    "codeparrot/apps",
    "di-zhang-fdu/AIME_1983_2024",
}

GITHUB_DATASETS = {
    "chaochun/nlu-asdiv-dataset": "master",
    "open-thought/reasoning-gym": "main",
}

CARD_COUNT_DATASETS = {
    "codeparrot/apps",
    "PRIME-RL/Eurus-2-RL-Data",
    "nvidia/Llama-Nemotron-Post-Training-Dataset",
    "nvidia/Nemotron-RL-Ultra-Training-Blends",
}

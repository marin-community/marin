# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned direct-source selections for task curation."""

from taskcompendium.pipeline.datasets import (
    code_contracts,
    gpqa,
    gretel_text_to_sql,
    instruction_tasks,
    math_answers,
    nemo_actions,
    numeric_answers,
    openscience,
    preference_tasks,
)
from taskcompendium.pipeline.datasets.reasoning_gym import generated as reasoning_gym_generated
from taskcompendium.pipeline.inputs import RecipeInputs, SourceFiles, SourceFormat, UrlDownload, hub_inputs
from taskcompendium.pipeline.models import DatasetRecipe, HFSource, IntendedUse, TaskPipeline

ASDIV_REVISION = "883f90a9a65bf00304ba8f37423910fe743abc47"
HH_REVISION = "09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa"
REASONING_GYM_REVISION = "49b07130b3fcd12f2d064bba7c43869543a0e7e7"


def _hub_recipe(
    name: str,
    version: str,
    source: HFSource,
    files: SourceFiles,
    pipeline: TaskPipeline,
    intended_use: IntendedUse,
) -> DatasetRecipe:
    return DatasetRecipe(
        name=name,
        version=version,
        source=source,
        inputs=hub_inputs(source.dataset, source.revision, files),
        pipeline=pipeline,
        intended_use=intended_use,
    )


RECIPES: dict[str, DatasetRecipe] = {
    "math500": _hub_recipe(
        name="math500",
        version="math500-v1",
        source=HFSource("HuggingFaceH4/MATH-500", "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be", "default", "test"),
        files=SourceFiles(("test.jsonl",), SourceFormat.JSONL),
        pipeline=math_answers.math_pipeline(
            math_answers.normalize_math500, math_answers.MATH500_RUBRIC, "math500-controls"
        ),
        intended_use=IntendedUse.EVAL,
    ),
    "aime_1983_2024": _hub_recipe(
        name="aime_1983_2024",
        version="aime_1983_2024-v1",
        source=HFSource("di-zhang-fdu/AIME_1983_2024", "3e2cc86390666c5c756622afc0eeb9e6194496bc", "default", "train"),
        files=SourceFiles(("AIME_Dataset_1983_2024.csv",), SourceFormat.CSV),
        pipeline=math_answers.math_pipeline(
            math_answers.normalize_aime_1983_2024, math_answers.AIME_1983_2024_RUBRIC, "aime_1983_2024-controls"
        ),
        intended_use=IntendedUse.EVAL,
    ),
    "gsm8k": _hub_recipe(
        name="gsm8k",
        version="gsm8k-v1",
        source=HFSource("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866", "main", "train"),
        files=SourceFiles(("main/train-00000-of-00001.parquet",), SourceFormat.PARQUET),
        pipeline=math_answers.math_pipeline(math_answers.normalize_gsm8k, math_answers.GSM8K_RUBRIC, "gsm8k-controls"),
        intended_use=IntendedUse.TRAIN,
    ),
    "asdiv": DatasetRecipe(
        name="asdiv",
        version="asdiv-v1",
        source=HFSource("chaochun/nlu-asdiv-dataset", ASDIV_REVISION, "original-xml", "train"),
        inputs=RecipeInputs(
            SourceFiles(("ASDiv.xml",), SourceFormat.XML, reader=math_answers.asdiv_rows),
            (
                UrlDownload(
                    f"https://raw.githubusercontent.com/chaochun/nlu-asdiv-dataset/{ASDIV_REVISION}/dataset/ASDiv.xml",
                    "ASDiv.xml",
                ),
            ),
        ),
        pipeline=math_answers.math_pipeline(math_answers.normalize_asdiv, math_answers.ASDIV_RUBRIC, "asdiv-controls"),
        intended_use=IntendedUse.TRAIN,
    ),
    "dapo_math": _hub_recipe(
        name="dapo_math",
        version="dapo_math-v1",
        source=HFSource(
            "BytedTsinghua-SIA/DAPO-Math-17k", "65877096c24ffa7abc4e4fa5edb95cf3413a5674", "default", "train"
        ),
        files=SourceFiles(("data/dapo-math-17k.parquet",), SourceFormat.PARQUET),
        pipeline=math_answers.math_pipeline(
            math_answers.normalize_dapo_math, math_answers.DAPO_MATH_RUBRIC, "dapo_math-controls"
        ),
        intended_use=IntendedUse.TRAIN,
    ),
    "rlvr_math": _hub_recipe(
        name="rlvr_math",
        version="rlvr_math-v1",
        source=HFSource("allenai/RLVR-MATH", "bd2a93551b503a395fadd1a740d957559cfe6f3c", "default", "train"),
        files=SourceFiles(("data/train-00000-of-00001.parquet",), SourceFormat.PARQUET),
        pipeline=math_answers.math_pipeline(
            math_answers.normalize_rlvr_math, math_answers.RLVR_MATH_RUBRIC, "rlvr_math-controls"
        ),
        intended_use=IntendedUse.TRAIN,
    ),
    "numina_math": _hub_recipe(
        name="numina_math",
        version="numina_math-v1",
        source=HFSource("AI-MO/NuminaMath-CoT", "9d8d210c9f6a36c8f3cd84045668c9b7800ef517", "default", "train"),
        files=SourceFiles(("data/train-*.parquet",), SourceFormat.PARQUET),
        pipeline=math_answers.math_pipeline(
            math_answers.normalize_numina_math, math_answers.NUMINA_MATH_RUBRIC, "numina_math-controls"
        ),
        intended_use=IntendedUse.TRAIN,
    ),
    "hardmath": _hub_recipe(
        name="hardmath",
        version="hardmath-v1",
        source=HFSource(
            "pafitis/HARDMath_processed_training", "937e9f10356e31e854f6efb9a2507f1e200c8b25", "default", "train"
        ),
        files=SourceFiles(("data/train-00000-of-00001.parquet",), SourceFormat.PARQUET),
        pipeline=math_answers.math_pipeline(
            math_answers.normalize_hardmath, math_answers.HARDMATH_RUBRIC, "hardmath-math-controls"
        ),
        intended_use=IntendedUse.TRAIN,
    ),
    "hendrycks_math": _hub_recipe(
        name="hendrycks_math",
        version="hendrycks-math-algebra-train-v1",
        source=HFSource("EleutherAI/hendrycks_math", "21a5633873b6a120296cce3e2df9d5550074f4a3", "algebra", "train"),
        files=SourceFiles(("algebra/train-00000-of-00001.parquet",), SourceFormat.PARQUET),
        pipeline=math_answers.math_pipeline(
            math_answers.normalize_hendrycks_math, math_answers.HENDRYCKS_MATH_RUBRIC, "hendrycks-math-controls"
        ),
        intended_use=IntendedUse.TRAIN,
    ),
    "deepscaler": _hub_recipe(
        name="deepscaler",
        version="deepscaler-v1",
        source=HFSource(
            "agentica-org/DeepScaleR-Preview-Dataset", "b6ae8c60f5c1f2b594e2140b91c49c9ad0949e29", "default", "train"
        ),
        files=SourceFiles(("deepscaler.json",), SourceFormat.JSON),
        pipeline=math_answers.math_pipeline(
            math_answers.normalize_deepscaler, math_answers.DEEPSCALER_RUBRIC, "deepscaler-math-controls"
        ),
        intended_use=IntendedUse.TRAIN,
    ),
    "aime24": _hub_recipe(
        name="aime24",
        version="aime24-v1",
        source=HFSource("HuggingFaceH4/aime_2024", "2fe88a2f1091d5048c0f36abc874fb997b3dd99a", "default", "train"),
        files=SourceFiles(("data/train-*.parquet",), SourceFormat.PARQUET),
        pipeline=numeric_answers.aime24_pipeline(),
        intended_use=IntendedUse.EVAL,
    ),
    "svamp": _hub_recipe(
        name="svamp",
        version="svamp-v1",
        source=HFSource("ChilleD/SVAMP", "5e0bf1e5e7c0e9c4bc39180d224f41f3f801b7ef", "default", "train"),
        files=SourceFiles(("data/train-*.parquet",), SourceFormat.PARQUET),
        pipeline=numeric_answers.svamp_pipeline(),
        intended_use=IntendedUse.TRAIN,
    ),
    "gpqa": _hub_recipe(
        name="gpqa",
        version="gpqa-v1",
        source=HFSource("Idavidrein/gpqa", "83022cefff930aea54f654c0b282e74b9eeda5c6", "gpqa_diamond", "train"),
        files=SourceFiles(("gpqa_diamond.csv",), SourceFormat.CSV),
        pipeline=gpqa.pipeline(),
        intended_use=IntendedUse.EVAL,
    ),
    "nemotron_if": _hub_recipe(
        name="nemotron_if",
        version="nemotron_if-v1",
        source=HFSource(
            "nvidia/Llama-Nemotron-Post-Training-Dataset",
            "ab2a40d258a6a4d9d4c277d702aeea445081766c",
            "default",
            "instruction_following",
        ),
        files=SourceFiles(("RL/instruction_following/instruction_following.jsonl",), SourceFormat.JSONL),
        pipeline=instruction_tasks.nemotron_if_pipeline(),
        intended_use=IntendedUse.TRAIN,
    ),
    "rlvr_ifeval": _hub_recipe(
        name="rlvr_ifeval",
        version="rlvr_ifeval-v1",
        source=HFSource("allenai/RLVR-IFeval", "47c03c73621c4aab2b824b7818681117d662770e", "default", "train"),
        files=SourceFiles(("data/train-00000-of-00001.parquet",), SourceFormat.PARQUET),
        pipeline=instruction_tasks.rlvr_ifeval_pipeline(),
        intended_use=IntendedUse.TRAIN,
    ),
    "apps": _hub_recipe(
        name="apps",
        version="apps-v1",
        source=HFSource("codeparrot/apps", "21e74ddf8de1a21436da12e3e653065c5213e9d1", "default", "train"),
        files=SourceFiles(("train.jsonl",), SourceFormat.JSONL),
        pipeline=code_contracts.apps_pipeline(),
        intended_use=IntendedUse.TRAIN,
    ),
    "eurus2_code": _hub_recipe(
        name="eurus2_code",
        version="eurus2_code-v1",
        source=HFSource("PRIME-RL/Eurus-2-RL-Data", "9776b13264b5aaa0b16495fcf086a0a8d86fd655", "default", "train"),
        files=SourceFiles(("train.parquet",), SourceFormat.PARQUET, selector=code_contracts.select_eurus_code),
        pipeline=code_contracts.eurus2_code_pipeline(),
        intended_use=IntendedUse.TRAIN,
    ),
    "verifiable_code": _hub_recipe(
        name="verifiable_code",
        version="verifiable_code-v1",
        source=HFSource(
            "open-r1/verifiable-coding-problems-python", "b761a24a95fa03289a231d2d31c183636ffb9833", "default", "train"
        ),
        files=SourceFiles(("data/train-*.parquet",), SourceFormat.PARQUET),
        pipeline=code_contracts.verifiable_code_pipeline(),
        intended_use=IntendedUse.TRAIN,
    ),
    "gretel_text_to_sql": _hub_recipe(
        name="gretel_text_to_sql",
        version="gretel_text_to_sql-v1",
        source=HFSource(
            "gretelai/synthetic_text_to_sql", "740ab236e64503fba51be1101df7a1be83bf455d", "default", "train"
        ),
        files=SourceFiles(("synthetic_text_to_sql_train.snappy.parquet",), SourceFormat.PARQUET),
        pipeline=gretel_text_to_sql.pipeline(),
        intended_use=IntendedUse.TRAIN,
    ),
    "openscience": _hub_recipe(
        name="openscience",
        version="openscience-v1",
        source=HFSource("nvidia/OpenScience", "7bd0437e4756f761768fe7e5cebeaa75480a4fd6", "OS-Q2.5-32B-4", "train"),
        files=SourceFiles(("OS-Q2.5-32B-4.jsonl",), SourceFormat.JSONL),
        pipeline=openscience.pipeline(),
        intended_use=IntendedUse.TRAIN,
    ),
    "nemo_actions": _hub_recipe(
        name="nemo-actions",
        version="nemo-actions-v2",
        source=HFSource(
            "nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1",
            "9643c8103d7bfbc2d7fc4d15991d6739c612ff58",
            "default",
            "train",
        ),
        files=SourceFiles(("train.jsonl",), SourceFormat.JSONL),
        pipeline=nemo_actions.pipeline(),
        intended_use=IntendedUse.TRAIN,
    ),
    "hh_harmless_base": _hub_recipe(
        name="hh_harmless_base",
        version="hh_harmless_base-v1",
        source=HFSource("Anthropic/hh-rlhf", HH_REVISION, "harmless-base", "train"),
        files=SourceFiles(("harmless-base/train.jsonl.gz",), SourceFormat.JSONL),
        pipeline=preference_tasks.hh_pipeline(preference_tasks.HH_HARMLESS_BASE_RUBRIC),
        intended_use=IntendedUse.TRAIN,
    ),
    "hh_helpful_base": _hub_recipe(
        name="hh_helpful_base",
        version="hh_helpful_base-v1",
        source=HFSource("Anthropic/hh-rlhf", HH_REVISION, "helpful-base", "train"),
        files=SourceFiles(("helpful-base/train.jsonl.gz",), SourceFormat.JSONL),
        pipeline=preference_tasks.hh_pipeline(preference_tasks.HH_HELPFUL_BASE_RUBRIC),
        intended_use=IntendedUse.TRAIN,
    ),
    "hh_helpful_online": _hub_recipe(
        name="hh_helpful_online",
        version="hh_helpful_online-v1",
        source=HFSource("Anthropic/hh-rlhf", HH_REVISION, "helpful-online", "train"),
        files=SourceFiles(("helpful-online/train.jsonl.gz",), SourceFormat.JSONL),
        pipeline=preference_tasks.hh_pipeline(preference_tasks.HH_HELPFUL_ONLINE_RUBRIC),
        intended_use=IntendedUse.TRAIN,
    ),
    "hh_helpful_rejection_sampled": _hub_recipe(
        name="hh_helpful_rejection_sampled",
        version="hh_helpful_rejection_sampled-v1",
        source=HFSource("Anthropic/hh-rlhf", HH_REVISION, "helpful-rejection-sampled", "train"),
        files=SourceFiles(("helpful-rejection-sampled/train.jsonl.gz",), SourceFormat.JSONL),
        pipeline=preference_tasks.hh_pipeline(preference_tasks.HH_HELPFUL_REJECTION_SAMPLED_RUBRIC),
        intended_use=IntendedUse.TRAIN,
    ),
    "kto_mix": _hub_recipe(
        name="kto_mix",
        version="kto_mix-v1",
        source=HFSource("trl-lib/kto-mix-14k", "4470f033f33364e7d064c9f920c3df54d0cce767", "default", "train"),
        files=SourceFiles(("data/train-00000-of-00001.parquet",), SourceFormat.PARQUET),
        pipeline=preference_tasks.binary_pipeline(preference_tasks.KTO_MIX_RUBRIC),
        intended_use=IntendedUse.TRAIN,
    ),
    "reasoning_gym_generated": DatasetRecipe(
        name="reasoning_gym_generated",
        version="reasoning-gym-direct-v1",
        source=HFSource("open-thought/reasoning-gym", REASONING_GYM_REVISION, "generated", "generated"),
        inputs=RecipeInputs(
            SourceFiles(
                ("generator.tar.gz",),
                SourceFormat.GENERATED,
                reader=reasoning_gym_generated.GeneratedRows(REASONING_GYM_REVISION),
            ),
            (
                UrlDownload(
                    f"https://api.github.com/repos/open-thought/reasoning-gym/tarball/{REASONING_GYM_REVISION}",
                    "generator.tar.gz",
                ),
            ),
        ),
        pipeline=reasoning_gym_generated.pipeline(),
        intended_use=IntendedUse.TRAIN,
    ),
}

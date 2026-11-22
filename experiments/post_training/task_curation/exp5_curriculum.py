# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exp5's intended eleven-source mix, including the blocked research slot.

This declares sampling policy. It does not admit OpenResearcher to the current
runtime or permit its five percent to be redistributed to another source.
"""

import math
from dataclasses import replace
from types import MappingProxyType

from marin.rl.curriculum import CurriculumEntry, CurriculumValidation, RlCurriculum, SamplingMode, SourcePin

from experiments.post_training.task_curation.curriculum import curriculum_entries
from experiments.post_training.task_curation.exp4_curriculum import (
    AGENTIC_TRAIN_SOURCE,
    CARRY_OVER_SHARES,
    EXP4_CATALOG,
    NEMOTRON_IF_LABEL,
    SEED,
)
from experiments.post_training.task_curation.exp4_curriculum import TARGET_SHARES as EXP4_TARGET_SHARES

IFBENCH_LABEL = "AllenAI:IF_multi_constraints_upto5"
OPENRESEARCHER_LABEL = "OpenResearcher:OpenResearcher-Dataset"
HARDMATH_LABEL = "pafitis:hardmath"

IFBENCH_REPO = "open-athena/allenai-if-multi-constraints-upto5-7bcdabaf-retained-20261116-b-v1"
IFBENCH_REVISION = "1b4076f63af29e34aee43e16781784890b1e9835"
IFBENCH_SHA256 = "9628ff2516d7c1a2f5c2cb7c4584f53a36369d0198dd4aab5a195ae3fcc301c6"
IFBENCH_ROWS = 95_357
IFBENCH_SOURCE = "allenai/IF_multi_constraints_upto5"
IFBENCH_SOURCE_REVISION = "2e3a77407b7fce69f95b248d64a884e3ae1c2423"

OPENRESEARCHER_REPO = "open-athena/openresearcher-openresearcher-dataset-8d7d09e1-retained-20261116-b-v1"
OPENRESEARCHER_REVISION = "5bee39eb26e0fae3e797d9523ac1b8012f5b95a3"
OPENRESEARCHER_SHA256 = "1e75e1396eb1066b1806d83268cd8f9ae6bf6c284d7d89fa0ddb6422ec320e15"

HARDMATH_REPO = "pafitis/HARDMath_processed_training"
HARDMATH_REVISION = "937e9f10356e31e854f6efb9a2507f1e200c8b25"
HARDMATH_FILE = "data/train-00000-of-00001.parquet"
HARDMATH_SHA256 = "27f7acf91b38ca413be2cc03506950142420a19c6ec374b23cf6a334033a18af"
HARDMATH_INPUT_ROWS = 702
HARDMATH_ROWS = 684

TARGET_SHARES = MappingProxyType(
    {
        AGENTIC_TRAIN_SOURCE: 0.20,
        NEMOTRON_IF_LABEL: 0.15,
        **CARRY_OVER_SHARES,
        IFBENCH_LABEL: 0.05,
        OPENRESEARCHER_LABEL: 0.05,
        HARDMATH_LABEL: 0.05,
    }
)
if not math.isclose(sum(TARGET_SHARES.values()), 1.0, rel_tol=0, abs_tol=1e-12):
    raise ValueError("The complete Exp5 target shares must sum to one")

PENDING_SOURCES = MappingProxyType(
    {OPENRESEARCHER_LABEL: "TaskSpec/JudgeSpec schema alignment and openresearcher-bm25:v1 provider admission"}
)


def exp5_rl_curriculum(validation: CurriculumValidation) -> RlCurriculum:
    """Record the complete policy with its recorded holdout.

    ``validation`` is the split the pool step drew for the eleven labels once
    their pools exist — the three added sources are still pending admission, so
    no pool builds it today; the split is drawn when they do."""
    inherited = tuple(
        replace(entry, weight=TARGET_SHARES[entry.label])
        for entry in curriculum_entries(
            "long-loop/exp4-rl", "2026.11.18", weights=dict(EXP4_TARGET_SHARES), catalog=EXP4_CATALOG
        )
    )
    additions = (
        (
            IFBENCH_LABEL,
            IFBENCH_REPO,
            IFBENCH_REVISION,
            "experiments.post_training.long_loop.exp5_data.ifbench_row",
        ),
        (
            OPENRESEARCHER_LABEL,
            OPENRESEARCHER_REPO,
            OPENRESEARCHER_REVISION,
            "experiments.post_training.task_curation.datasets.openresearcher.tasks.convert_question",
        ),
        (
            HARDMATH_LABEL,
            HARDMATH_REPO,
            HARDMATH_REVISION,
            "experiments.post_training.task_curation.datasets.skyrl.math.convert_hardmath",
        ),
    )
    return RlCurriculum(
        name="long-loop/exp5-rl",
        version="2026.11.20",
        mode=SamplingMode.WEIGHTED,
        validation=validation,
        entries=inherited
        + tuple(
            CurriculumEntry(
                label=label,
                source=label,
                version="2026.11.20",
                pin=SourcePin(
                    name=repo, revision=revision, url=f"https://huggingface.co/datasets/{repo}/tree/{revision}"
                ),
                recipe=recipe,
                weight=TARGET_SHARES[label],
            )
            for label, repo, revision, recipe in additions
        ),
        stages=(),
        seed=SEED,
    )

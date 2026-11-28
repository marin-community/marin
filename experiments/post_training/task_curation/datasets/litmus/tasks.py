# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned Litmus-Bench v0.2 training and held-out chemistry questions."""

from finestore.rl.status_taxonomy import DataRestriction
from taskcompendium.pipeline.inputs import SourceFormat
from taskcompendium.pipeline.models import Controls, IntendedUse

from experiments.post_training.task_curation.datasets.environments import GRADER_PACKAGES
from experiments.post_training.task_curation.datasets.litmus.convert import convert
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource, process_rows
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, SourceOrigin, TaskFormat

DATASET = "Nemotron-RL-litmus-bench-v0.2"
REPO = f"nvidia/{DATASET}"
REVISION = "c180425444ebcc6db8dafdb897e51a94927df62a"
RUBRIC = """
Does the prompt supply the molecular structure or measurements needed to answer it?
Check units, requested rounding, comparison direction and answer formatting against
the private row's expected answer, output regex and matching rule.
"""


def sources() -> list[RlDataSource[CurationRecipe]]:
    return [
        RlDataSource(
            info=SourceInfo(
                origin=SourceOrigin.NVIDIA,
                upstream_dataset=DATASET,
                upstream_name="NVIDIA Litmus-Bench v0.2",
                upstream_url=f"https://huggingface.co/datasets/{REPO}/tree/{REVISION}",
                subset=split,
                build_path="NeMo Gym Litmus",
                data_restriction=DataRestriction.OPEN,
                family="chemistry",
                count=count,
                tags=("single-turn", "rlvr", "license:cc-by-sa-4.0", "license:openmdw-1.1", *tags),
                task_format=TaskFormat.MARINSKYRL,
                notes="Each question retains its upstream license (CC BY-SA 4.0 or OpenMDW 1.1).",
            ),
            config=CurationRecipe(
                name=f"litmus-bench-v02-{split}",
                source=HfSource(REPO, REVISION, (f"{split}.jsonl",), SourceFormat.JSONL),
                convert=convert,
                version="2026.11.25.1",
                intended_use=use,
                rubric=RUBRIC,
                controls=Controls(),
                grader=GRADER_PACKAGES,
            ),
            pipeline=process_rows,
        )
        for split, use, count, tags in (
            ("train", IntendedUse.TRAIN, 47_118, ()),
            ("validation", IntendedUse.EVAL, 2_056, ("eval", "benchmark")),
        )
    ]

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SkyRL science multiple-choice sources, graded in process by the verifyit option-letter comparator."""

import hashlib
import re
from dataclasses import replace

from taskcompendium.convert.answers import mcq_task, source_defect, unsupported
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.inputs import ConversionContext, SourceFormat
from taskcompendium.pipeline.models import ImportRejection, IntendedUse, RawRow
from verifyit.modes.extract import extract_boxed

from experiments.post_training.task_curation.pipeline import HfSource, RlDataPipeline, ShellSim
from experiments.post_training.task_curation.source import DataSourceMetadata, RlDataSource

SKYRL_METADATA = DataSourceMetadata(
    id="",
    name="",
    origin="MarinSkyRL",
    revision="e44c4bfcb62c489286a1264094e6d9c883aaf0d2",
    revised_at="2026-10-08T02:13:45Z",
    verifier_revision="0bc12f4e510cd99f34e12ac26599960d96d1ae0a24840eb3fdb9a4e80388e7fb",
    family="qa-multiple-choice",
    environment="mcq",
    type="RLVR",
    turns="Single-turn",
    family_basis="Upstream card/schema and selected SkyRL loader audited 2026-09-28",
    classification_basis="Inferred from SkyRL environment contract; blended sources may contain multiple task types",
    provenance_url=(
        "https://github.com/marin-community/MarinSkyRL/blob/e44c4bfcb62c489286a1264094e6d9c8"
        "83aaf0d2/infra/rl_data/sources.py"
    ),
    license=("cc-by-4.0",),
    verification="two_sided",
    snapshot_safe=True,
    gym_alias="gym/mcq",
    gym_url=(
        "https://github.com/marin-community/MarinSkyRL/blob/e44c4bfcb62c489286a1264094e6d9c883a"
        "af0d2/skyrl-gym/skyrl_gym/envs/__init__.py"
    ),
    gym_entrypoint="skyrl_gym.envs.mcq.env:MCQEnv",
    registry_revised_at="2026-10-01T14:18:17Z",
    verifier_url=(
        "https://github.com/marin-community/MarinSkyRL/tree/e44c4bfcb62c489286a1264094e6d9c88"
        "3aaf0d2/skyrl-gym/skyrl_gym/envs/mcq"
    ),
    verifier_revised_at="2026-10-08T02:13:45Z",
    revision_basis="Latest upstream dataset repository or MarinSkyRL verifier change",
    grading_revision="6a380c062d918ab7f12781b83fb3b6d639084f3223dbb4287f65641116a17c69",
    recorded_at="2026-10-08",
)

GPQA_CHOICES = ("Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3")
OPTION_LABEL = re.compile(r"(?m)^([A-Z]):")

GPQA_RUBRIC = """
Assess every displayed option, units, assumptions, and scientific directionality before judging the key.

Specialist background knowledge is allowed; an omitted experiment, figure, or passage is missing context.

Flag ties, approximate synonyms, or an indefensible key. Technical difficulty alone is not a defect.
"""

OPENSCIENCE_RUBRIC = """
The boxed option is a generated response conclusion, not an independent gold label. Assess each public option and the
reference derivation before accepting it.

Check missing scientific context, overlapping options and incorrect causal claims. Specialist difficulty alone is not
a defect.

The grader compares one option letter.
"""


def _letter(index: int) -> str:
    return chr(ord("A") + index)


def convert_gpqa(row: RawRow, _context: ConversionContext) -> TaskSpec | ImportRejection:
    question = row.data.get("Question")
    choices = [row.data.get(key) for key in GPQA_CHOICES]
    if (
        not isinstance(question, str)
        or not question.strip()
        or not all(isinstance(choice, str) and choice.strip() for choice in choices)
    ):
        return source_defect("missing_prompt_or_choices", "Question and all four choices are required")
    texts = [str(choice).strip() for choice in choices]
    if len(set(texts)) != len(texts):
        return source_defect("duplicate_options", "Choice text repeats after boundary trimming")
    # A per-row hash order avoids always showing the source's correct-first arrangement.
    ordered = sorted(enumerate(texts), key=lambda item: hashlib.sha256(f"{row.id}:{item[0]}".encode()).digest())
    expected = next(_letter(index) for index, (original, _) in enumerate(ordered) if original == 0)
    options = "\n".join(f"{_letter(index)}. {text}" for index, (_, text) in enumerate(ordered))
    prompt = f"{question.strip()}\n\n{options}\n\nChoose one option letter from A through D."
    return mcq_task(row, prompt=prompt, answer=expected, options=len(texts))


def convert_openscience(row: RawRow, _context: ConversionContext) -> TaskSpec | ImportRejection:
    prompt, response = row.data.get("input"), row.data.get("output")
    if not isinstance(prompt, str) or not isinstance(response, str):
        return source_defect("missing_prompt_or_reference", "input and generated output strings are required")
    labels = OPTION_LABEL.findall(prompt)
    expected = extract_boxed(response)
    if not labels or labels != [_letter(index) for index in range(len(labels))] or expected not in labels:
        return unsupported(
            "unsupported_choice_contract", "Contiguous labeled choices and a boxed option conclusion are required"
        )
    return mcq_task(
        row,
        prompt=prompt + "\n\nReturn one option letter.",
        answer=expected,
        options=len(labels),
        evidence={"output": response},
    )


def sources() -> list[RlDataSource]:
    return [
        RlDataSource(
            metadata=replace(
                SKYRL_METADATA,
                id="MarinSkyRL:gpqa",
                name="gpqa",
                display_name="Idavidrein/gpqa · gpqa_diamond",
                url="https://huggingface.co/datasets/Idavidrein/gpqa",
                dataset_id="Idavidrein/gpqa",
                dataset_revision="83022cefff930aea54f654c0b282e74b9eeda5c6",
                task_count=198,
                count_basis="HF card / viewer: gpqa_diamond/train; registry split names the config",
                count_precision="exact",
                count_url="https://datasets-server.huggingface.co/size?dataset=Idavidrein/gpqa",
                split="gpqa_diamond",
                is_benchmark=True,
                benchmark_basis="Upstream dataset card explicitly describes a benchmark",
                family_url=(
                    "https://huggingface.co/datasets/Idavidrein/gpqa/blob/83022cefff930aea54f654c0b2"
                    "82e74b9eeda5c6/README.md"
                ),
                canonical_source="Idavidrein/gpqa · gpqa_diamond",
                canonical_url="https://huggingface.co/datasets/Idavidrein/gpqa",
                dataset_revised_at="2026-09-21T22:29:06.000Z",
            ),
            pipeline=RlDataPipeline(
                name="gpqa",
                source=HfSource(
                    "Idavidrein/gpqa",
                    "83022cefff930aea54f654c0b282e74b9eeda5c6",
                    ("gpqa_diamond.csv",),
                    SourceFormat.CSV,
                ),
                convert=convert_gpqa,
                version="1",
                environment=ShellSim(),
                intended_use=IntendedUse.EVAL,
                rubric=GPQA_RUBRIC,
            ),
        ),
        RlDataSource(
            metadata=replace(
                SKYRL_METADATA,
                id="MarinSkyRL:openscience",
                name="openscience",
                display_name="nvidia/OpenScience",
                url="https://huggingface.co/datasets/nvidia/OpenScience",
                dataset_id="nvidia/OpenScience",
                dataset_revision="7bd0437e4756f761768fe7e5cebeaa75480a4fd6",
                task_count=4476918,
                count_basis=(
                    "HF viewer: all train configurations; partial files use estimated_num_rows. "
                    "Repository-wide count; each SkyRL run selects a configuration explicitly. "
                    "Card audit on 2026-09-28 reports ~6M."
                ),
                count_precision="estimated",
                count_url="https://datasets-server.huggingface.co/size?dataset=nvidia/OpenScience",
                benchmark_basis=(
                    "SkyRL test-only designation or HF benchmark:official tag; false means no " "designation found"
                ),
                family_url=(
                    "https://huggingface.co/datasets/nvidia/OpenScience/blob/7bd0437e4756f761768fe7e"
                    "5cebeaa75480a4fd6/README.md"
                ),
                canonical_source="nvidia/OpenScience",
                canonical_url="https://huggingface.co/datasets/nvidia/OpenScience",
                dataset_revised_at="2025-06-18T19:21:21.000Z",
            ),
            pipeline=RlDataPipeline(
                name="openscience",
                source=HfSource(
                    "nvidia/OpenScience",
                    "7bd0437e4756f761768fe7e5cebeaa75480a4fd6",
                    ("OS-Q2.5-32B-4.jsonl",),
                    SourceFormat.JSONL,
                ),
                convert=convert_openscience,
                version="1",
                environment=ShellSim(),
                intended_use=IntendedUse.TRAIN,
                rubric=OPENSCIENCE_RUBRIC,
            ),
        ),
    ]

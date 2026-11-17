# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grader packages for the prepared ScienceAgentBench Verified environments."""

from pathlib import Path, PurePosixPath

from taskcompendium.grader import GraderPackage
from taskcompendium.importers.scienceagentbench import SUBMISSION_PATH, WORKING_DIRECTORY, VerifiedAnnotation
from taskcompendium.models import (
    ArtifactKind,
    EnvironmentRequirements,
    FileReward,
    ReplayStage,
    RewardFile,
    RewardFileFormat,
    ScriptGrader,
    VerifierArtifact,
)
from taskcompendium.runtime.resources import inline_resource

ASSETS = Path(__file__).parent / "scienceagentbench_assets"
SOURCE_LICENSE = (ASSETS / "ScienceAgentBench.LICENSE").read_bytes()
OUTPUT_DIRECTORY = f"{WORKING_DIRECTORY}/pred_results"
PAPYRUS_ROWS = f"{WORKING_DIRECTORY}/papyrus_rows.jsonl"
VERDICT_PATH = "/logs/verifier/verdict.json"
BINARY_VERDICT = FileReward(files=(RewardFile(path=VERDICT_PATH, format=RewardFileFormat.VERDICT_JSON),), pass_above=0.5)


def deterministic_grader(
    annotation: VerifiedAnnotation,
    *,
    environment: EnvironmentRequirements,
    replay_environment: EnvironmentRequirements,
    evaluator_program: bytes,
    replay_timeout: float,
    verifier_timeout: float,
) -> GraderPackage:
    """Package a deterministic source evaluator; gold assets belong in its private image."""
    if annotation.instance_id == 22:
        raise ValueError("Task 22 requires the pickle-isolating Papyrus grader")
    if b"gpt4_visual_judge" in evaluator_program:
        raise ValueError("Visual tasks require an explicitly configured visual judge")
    output = PurePosixPath(annotation.output_fname)
    if ".." in output.parts or not output.is_relative_to("pred_results"):
        raise ValueError("ScienceAgentBench outputs must be under pred_results")
    grader = ScriptGrader(
        environment=environment,
        argv=(
            "python3",
            "-I",
            "/tests/evaluate.py",
            "/tests/source_evaluator.py",
            f"{WORKING_DIRECTORY}/{output}",
            OUTPUT_DIRECTORY,
            VERDICT_PATH,
        ),
        cwd=WORKING_DIRECTORY,
        answer_path=None,
        timeout=verifier_timeout,
        reward=BINARY_VERDICT,
        replay=ReplayStage(
            environment=replay_environment,
            argv=("python3", SUBMISSION_PATH),
            cwd=WORKING_DIRECTORY,
            timeout=replay_timeout,
            artifacts=(VerifierArtifact(source=OUTPUT_DIRECTORY, target=OUTPUT_DIRECTORY, kind=ArtifactKind.DIRECTORY),),
        ),
    )
    return GraderPackage(
        grader,
        (
            inline_resource("evaluate.py", (ASSETS / "evaluate.py").read_bytes()),
            inline_resource("source_evaluator.py", evaluator_program),
            inline_resource("ScienceAgentBench.LICENSE", SOURCE_LICENSE),
        ),
    )


def papyrus_grader(
    *,
    environment: EnvironmentRequirements,
    replay_environment: EnvironmentRequirements,
    gold_rows: bytes,
    replay_timeout: float,
    verifier_timeout: float,
) -> GraderPackage:
    """Package task 22 with gold rows normalized during trusted image preparation.

    Candidate pickle decoding occurs only in replay. The grading machine gets
    a JSON-lines file, the trusted normalized gold rows, and a JSON-only scorer.
    """
    grader = ScriptGrader(
        environment=environment,
        argv=("python3", "-I", "/tests/papyrus_rows.py", PAPYRUS_ROWS, "/tests/gold_rows.jsonl", VERDICT_PATH),
        cwd=WORKING_DIRECTORY,
        answer_path=None,
        timeout=verifier_timeout,
        reward=BINARY_VERDICT,
        replay=ReplayStage(
            environment=replay_environment,
            argv=(
                "python3",
                "-I",
                "/replay/decode_papyrus.py",
                SUBMISSION_PATH,
                f"{OUTPUT_DIRECTORY}/papyrus_filtered.pkl",
                PAPYRUS_ROWS,
            ),
            cwd=WORKING_DIRECTORY,
            timeout=replay_timeout,
            artifacts=(VerifierArtifact(source=PAPYRUS_ROWS, target=PAPYRUS_ROWS, kind=ArtifactKind.FILE),),
            resources=(
                inline_resource("replay/decode_papyrus.py", (ASSETS / "decode_papyrus.py").read_bytes()),
                inline_resource("replay/ScienceAgentBench.LICENSE", SOURCE_LICENSE),
            ),
        ),
    )
    return GraderPackage(
        grader,
        (
            inline_resource("papyrus_rows.py", (ASSETS / "papyrus_rows.py").read_bytes()),
            inline_resource("gold_rows.jsonl", gold_rows),
            inline_resource("ScienceAgentBench.LICENSE", SOURCE_LICENSE),
        ),
    )

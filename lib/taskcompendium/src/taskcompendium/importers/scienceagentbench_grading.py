# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grader packages for the prepared ScienceAgentBench Verified environments."""

from pathlib import Path, PurePosixPath

from verifyit.spec import JudgeSpec

from taskcompendium.grader import GraderPackage
from taskcompendium.importers.scienceagentbench import SUBMISSION_PATH, WORKING_DIRECTORY, VerifiedAnnotation
from taskcompendium.importers.scienceagentbench_assets.visual_judge import MAX_OUTPUT_TOKENS, MODEL, SOURCE_JUDGE_MODULE
from taskcompendium.models import (
    ArtifactKind,
    EnvironmentRequirements,
    FileReward,
    GraderMachine,
    PrivateVerifierResource,
    ReplayStage,
    RewardFile,
    RewardFileFormat,
    ScriptGrader,
    VerifierArtifact,
)
from taskcompendium.runtime.grading import JUDGE_SPEC_PATH
from taskcompendium.runtime.resources import inline_resource

ASSETS = Path(__file__).parent / "scienceagentbench_assets"
LICENSE_FILENAME = "ScienceAgentBench.LICENSE"
SOURCE_LICENSE = (ASSETS / LICENSE_FILENAME).read_bytes()
OUTPUT_DIRECTORY = f"{WORKING_DIRECTORY}/pred_results"
PAPYRUS_ROWS = f"{WORKING_DIRECTORY}/papyrus_rows.jsonl"
VERDICT_PATH = "/logs/verifier/verdict.json"
BINARY_VERDICT = FileReward(files=(RewardFile(path=VERDICT_PATH, format=RewardFileFormat.VERDICT_JSON),), pass_above=0.5)
EVALUATOR_WRAPPER = "evaluate.py"
SOURCE_EVALUATOR = "source_evaluator.py"
PAPYRUS_SCORER = "papyrus_rows.py"
PAPYRUS_GOLD = "gold_rows.jsonl"
PAPYRUS_DECODER = "decode_papyrus.py"
VISUAL_JUDGE = "visual_judge.py"
VISUAL_PROMPT = "visual_prompt.txt"


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
    if SOURCE_JUDGE_MODULE.encode() in evaluator_program:
        raise ValueError("Visual tasks require an explicitly configured visual judge")
    return _source_grader(
        annotation,
        environment=environment,
        replay_environment=replay_environment,
        evaluator_program=evaluator_program,
        replay_timeout=replay_timeout,
        verifier_timeout=verifier_timeout,
    )


def visual_grader(
    annotation: VerifiedAnnotation,
    *,
    environment: EnvironmentRequirements,
    replay_environment: EnvironmentRequirements,
    evaluator_program: bytes,
    judge_prompt: str,
    replay_timeout: float,
    verifier_timeout: float,
    judge_model: str = MODEL,
) -> GraderPackage:
    """Package a pinned figure evaluator with its three-completion judge."""
    if SOURCE_JUDGE_MODULE.encode() not in evaluator_program:
        raise ValueError("A visual grader requires an upstream figure evaluator")
    package = _source_grader(
        annotation,
        environment=environment,
        replay_environment=replay_environment,
        evaluator_program=evaluator_program,
        replay_timeout=replay_timeout,
        verifier_timeout=verifier_timeout,
    )
    assert isinstance(package.grader, ScriptGrader)
    grader = package.grader.model_copy(
        update={
            "argv": (
                "python3",
                "-I",
                f"/tests/{VISUAL_JUDGE}",
                f"/tests/{VISUAL_PROMPT}",
                JUDGE_SPEC_PATH,
                *package.grader.argv[2:],
            ),
            "judge": JudgeSpec(
                model=judge_model,
                rubric="checklist",
                criteria=(judge_prompt,),
                exact_gate=False,
                max_completion_tokens=MAX_OUTPUT_TOKENS,
            ),
        }
    )
    return GraderPackage(
        grader,
        (
            *package.resources,
            inline_resource(VISUAL_JUDGE, (ASSETS / VISUAL_JUDGE).read_bytes()),
            inline_resource(VISUAL_PROMPT, judge_prompt.encode("utf-8")),
        ),
    )


def _source_grader(
    annotation: VerifiedAnnotation,
    *,
    environment: EnvironmentRequirements,
    replay_environment: EnvironmentRequirements,
    evaluator_program: bytes,
    replay_timeout: float,
    verifier_timeout: float,
) -> GraderPackage:
    output = PurePosixPath(annotation.output_fname)
    if ".." in output.parts or not output.is_relative_to(PurePosixPath(OUTPUT_DIRECTORY).name):
        raise ValueError("ScienceAgentBench outputs must be under pred_results")
    grader = ScriptGrader(
        placement=GraderMachine(environment=environment),
        argv=(
            "python3",
            "-I",
            f"/tests/{EVALUATOR_WRAPPER}",
            f"/tests/{SOURCE_EVALUATOR}",
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
            inline_resource(EVALUATOR_WRAPPER, (ASSETS / EVALUATOR_WRAPPER).read_bytes()),
            inline_resource(SOURCE_EVALUATOR, evaluator_program),
            inline_resource(LICENSE_FILENAME, SOURCE_LICENSE),
        ),
    )


def papyrus_grader(
    *,
    environment: EnvironmentRequirements,
    replay_environment: EnvironmentRequirements,
    gold_rows: bytes | PrivateVerifierResource,
    replay_timeout: float,
    verifier_timeout: float,
) -> GraderPackage:
    """Package task 22 with gold rows normalized during trusted image preparation.

    Candidate pickle decoding occurs only in replay. The grading machine gets
    a JSON-lines file, the trusted normalized gold rows, and a JSON-only scorer.
    """
    if isinstance(gold_rows, PrivateVerifierResource):
        if gold_rows.path != "gold_results/papyrus_filtered_gold.pkl":
            raise ValueError("Expected the published task 22 gold pickle")
        decoder = inline_resource(PAPYRUS_DECODER, (ASSETS / PAPYRUS_DECODER).read_bytes())
        normalize = (
            'python3 -I -c "from pathlib import Path; import runpy; '
            f"runpy.run_path('/tests/{PAPYRUS_DECODER}')['decode_pickle']("
            f"Path('/tests/{gold_rows.path}'), Path('/tests/{PAPYRUS_GOLD}'))\""
        )
        environment = environment.model_copy(update={"setup_commands": (*environment.setup_commands, normalize)})
        private_gold = (gold_rows, decoder)
    else:
        private_gold = (inline_resource(PAPYRUS_GOLD, gold_rows),)
    grader = ScriptGrader(
        placement=GraderMachine(environment=environment),
        argv=("python3", "-I", f"/tests/{PAPYRUS_SCORER}", PAPYRUS_ROWS, f"/tests/{PAPYRUS_GOLD}", VERDICT_PATH),
        cwd=WORKING_DIRECTORY,
        answer_path=None,
        timeout=verifier_timeout,
        reward=BINARY_VERDICT,
        replay=ReplayStage(
            environment=replay_environment,
            argv=(
                "python3",
                "-I",
                f"/replay/{PAPYRUS_DECODER}",
                SUBMISSION_PATH,
                f"{OUTPUT_DIRECTORY}/papyrus_filtered.pkl",
                PAPYRUS_ROWS,
            ),
            cwd=WORKING_DIRECTORY,
            timeout=replay_timeout,
            artifacts=(VerifierArtifact(source=PAPYRUS_ROWS, target=PAPYRUS_ROWS, kind=ArtifactKind.FILE),),
            resources=(
                inline_resource(f"replay/{PAPYRUS_DECODER}", (ASSETS / PAPYRUS_DECODER).read_bytes()),
                inline_resource(f"replay/{LICENSE_FILENAME}", SOURCE_LICENSE),
            ),
        ),
    )
    return GraderPackage(
        grader,
        (
            inline_resource(PAPYRUS_SCORER, (ASSETS / PAPYRUS_SCORER).read_bytes()),
            *private_gold,
            inline_resource(LICENSE_FILENAME, SOURCE_LICENSE),
        ),
    )

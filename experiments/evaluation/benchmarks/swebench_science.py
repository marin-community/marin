# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SWE-bench Science with native isolated grading.

Release tooling and metadata are MIT licensed. Task sources retain their upstream licenses;
the default selection follows the release's unrestricted manifest.
"""

import base64
import re
import shlex
import tomllib
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

from shellbox.machine import NetworkPolicy
from taskcompendium.evaluation.spec import AgentEpisode, EpisodeInputs, EpisodeLimits, PiAgent
from taskcompendium.importers.harbor.package import (
    AGENT_CAPABILITIES,
    INSTRUCTION,
    LOG_DIRECTORIES,
    TEST_SCRIPT,
    harbor_shell,
)
from taskcompendium.importers.harbor.sources import LoweredSource, extend_source
from taskcompendium.models import (
    AnswerType,
    ArtifactKind,
    CommandSemantics,
    ConversationInput,
    EnvironmentRequirements,
    FileReward,
    GraderMachine,
    InlineFile,
    PlainText,
    ResourceGroups,
    RewardFile,
    RewardFileFormat,
    ScriptGrader,
    TaskResource,
    TaskSpec,
    TextMessage,
    VerifierArtifact,
    VerifierCommand,
)
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import IntendedUse, RawRow

from experiments.evaluation.benchmark import EvaluationBenchmark, fixed_benchmarks
from experiments.evaluation.benchmarks.harbor import (
    EPISODE_LABEL,
    MODEL_TURN_TIMEOUT,
    PI_MAX_OUTPUT_TOKENS,
    harbor_item,
    harbor_protocol,
    pi_install,
)
from experiments.evaluation.preparation import EvaluationCorpusRecipe, prepare_evaluation
from experiments.post_training.task_curation.harbor_packages import extract_package, package_rows
from experiments.post_training.task_curation.pipeline import CurationRecipe, HarborRepoSource
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, TaskFormat

DATASET = "OpenMOSS-Team/SWE-bench-Science"
REVISION = "997408298e8e4d09978b1120b95c4540d1a71bcb"
VERSION = "2026.11.24.4"
BENCHMARK = "swebench-science"
# Exact selections/default-96.json in the pinned release.
TASK_IDS = tuple(
    """
    001 002 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 022 024 025
    027 028 029 030 031 033 034 036 037 038 039 040 041 042 043 044 045 046 047 048
    049 050 051 052 053 054 055 056 058 059 060 061 062 063 064 065 067 068 069 070
    071 072 073 076 077 078 079 080 081 086 087 088 089 090 091 092 093 094 095 099
    103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 119
    """.split()
)

# The hosted selection adds these reviewed packages to the upstream default.
HOSTED_TASK_IDS = (*TASK_IDS, *"003 020 021 023 032 057 066 074 075 082 083 084 085 096 097 098 100 118".split())
PI = PiAgent(version="0.87.0", thinking="xhigh", setup_timeout=360)
PATCH = "/logs/artifacts/model.patch"


def _offline_image(env: dict) -> str:
    if (
        env["allow_internet"]
        or env["os"] != "linux"
        or not re.fullmatch(r"[^@]+@sha256:[0-9a-f]{64}", env["docker_image"])
    ):
        raise ValueError("SWE-bench Science requires offline digest-pinned Linux images")
    # Skopeo accepts repository@digest, including registries with ports.
    return re.sub(r":[^/:@]+(?=@)", "", env["docker_image"])


def lower_science_package(directory: Path, row: RawRow) -> TaskSpec:
    """Build a TaskSpec directly from a pinned SWE-bench Science release package."""
    config = tomllib.loads((directory / "task.toml").read_text())
    agent, verifier = config["agent"], config["verifier"]
    solver_env, verifier_env = config["environment"], verifier["environment"]
    if config["schema_version"] != "1.3" or verifier["environment_mode"] != "separate":
        raise ValueError("Expected SWE-bench Science's schema 1.3 separate-verifier release")
    if config.get("artifacts") != [PATCH]:
        raise ValueError(f"Expected the sole Science artifact to be {PATCH}")
    if any(role["network_mode"] != "no-network" for role in (agent, verifier)):
        raise ValueError("SWE-bench Science requires the native no-network protocol")

    solver = LoweredSource(_offline_image(solver_env), None, solver_env["workdir"])
    solver = extend_source(solver, pi_install(PI))
    hook = (directory / "pre_artifacts.sh").read_text()
    command = f"cd {shlex.quote(solver_env['workdir'])} && sh -c {shlex.quote(hook)}"
    episode = EpisodeInputs(
        agent_timeout=agent["timeout_sec"],
        verifier_timeout=verifier["timeout_sec"],
        cpus=solver_env["cpus"],
        memory_mb=solver_env["memory_mb"],
        storage_mb=10240,
        network=NetworkPolicy.DENY,
    )
    return TaskSpec(
        id=row.id,
        source=row.source,
        context=ConversationInput(events=(TextMessage(role="user", content=(directory / INSTRUCTION).read_text()),)),
        environment_requirements=EnvironmentRequirements(
            capabilities=AGENT_CAPABILITIES,
            command_semantics=CommandSemantics.LINUX_PROCESS,
            docker_image=solver.docker_image,
            docker_build=solver.docker_build,
            working_directory=solver.workdir,
            setup_commands=(LOG_DIRECTORIES,),
            environment_variables={**solver_env.get("env", {}), **agent.get("env", {})},
        ),
        answer_type=AnswerType.WORKSPACE_STATE,
        answer_format=PlainText(),
        grader=ScriptGrader(
            argv=("bash", TEST_SCRIPT),
            cwd=verifier_env["workdir"],
            answer_path=None,
            timeout=verifier["timeout_sec"],
            reward=FileReward(files=(RewardFile(path="/logs/verifier/reward.json", format=RewardFileFormat.JSON),)),
            placement=GraderMachine(
                environment=EnvironmentRequirements(
                    command_semantics=CommandSemantics.LINUX_PROCESS,
                    docker_image=_offline_image(verifier_env),
                    working_directory=verifier_env["workdir"],
                    environment_variables={**verifier_env.get("env", {}), **verifier.get("env", {})},
                ),
                collect=(VerifierCommand(argv=harbor_shell(command), timeout=300),),
                artifacts=(VerifierArtifact(source=PATCH, target=PATCH, kind=ArtifactKind.FILE),),
            ),
        ),
        resources=ResourceGroups(
            verifier=(
                TaskResource(
                    path="grader.py",
                    source=InlineFile(
                        content_base64=base64.b64encode((directory / "tests/grader.py").read_bytes()).decode("ascii")
                    ),
                    mode="0644",
                ),
            )
        ),
        tags=("eval",),
        labels={"harbor_path": directory.name, EPISODE_LABEL: episode.model_dump_json()},
    )


def convert(row: RawRow, _context: ConversionContext) -> TaskSpec:
    with TemporaryDirectory(prefix="swe-science-") as root:
        package = Path(root) / row.data["name"]
        extract_package(row.data["archive"], package)
        return lower_science_package(package, row)


def science_source(benchmark: str, task_ids: tuple[str, ...]) -> RlDataSource[EvaluationCorpusRecipe]:
    protocol_name = "pi-xhigh"
    name = f"{benchmark}-{protocol_name}"
    episode = AgentEpisode(
        harness=PI,
        max_input_tokens=65536,
        limits=EpisodeLimits(
            max_turns=None,
            max_total_tokens=None,
            model_turn_timeout=MODEL_TURN_TIMEOUT,
            total_turn_timeout=None,
            verifier_timeout=None,
            attempt_timeout=10800,
        ),
    )
    recipe = CurationRecipe(
        name=f"harbor-{name}",
        version=VERSION,
        intended_use=IntendedUse.EVAL,
        source=HarborRepoSource(
            DATASET, REVISION, read=package_rows, root="tasks", packages=frozenset(f"task_{key}" for key in task_ids)
        ),
        convert=convert,
    )
    return RlDataSource(
        info=SourceInfo(
            id=f"Harbor:{name}",
            title=f"SWE-bench Science ({len(task_ids)} tasks)",
            origin="Harbor",
            family="agentic",
            tags=("eval", "source:harbor", "agentic", "multi-turn", "benchmark"),
            count=len(task_ids),
            task_format=TaskFormat.HARBOR,
        ),
        config=EvaluationCorpusRecipe(
            recipe,
            harbor_protocol(
                benchmark,
                VERSION,
                len(task_ids),
                max_output_tokens=PI_MAX_OUTPUT_TOKENS,
                requires_judge=False,
                trials=1,
            ),
            partial(harbor_item, agent=episode),
            protocol_name=protocol_name,
            agent=episode,
        ),
        pipeline=prepare_evaluation,
    )


def sources() -> list[RlDataSource[EvaluationCorpusRecipe]]:
    return [
        science_source(BENCHMARK, TASK_IDS),
        science_source(f"{BENCHMARK}-hosted114", HOSTED_TASK_IDS),
    ]


def benchmarks() -> list[EvaluationBenchmark]:
    return fixed_benchmarks(sources())

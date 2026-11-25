# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SWE-bench Science with native isolated grading.

Task materials retain their upstream licenses.
"""

import base64
import re
import shlex
import tomllib
from collections.abc import Iterator
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

from finestore.rl.status_taxonomy import DataRestriction
from rigging.filesystem.storage_path import StoragePath
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
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, SourceOrigin, TaskFormat

DATASET = "OpenMOSS-Team/SWE-bench-Science"
REVISION = "997408298e8e4d09978b1120b95c4540d1a71bcb"
VERSION = "2026.11.25.1"
BENCHMARK = "swebench-science"
TASK_COUNT = 118
# The validation workspace accepts 10 GiB per solver. https://github.com/marin-community/marin/issues/12108
SOLVER_DISK_MB = 10 * 1024
PI = PiAgent(version="0.87.0", thinking="xhigh", setup_timeout=360)
PATCH = "/logs/artifacts/model.patch"


def benchmark_rows(path: StoragePath, context: ConversionContext) -> Iterator[dict]:
    # Task 019 requires permission to modify pycisTopic (clause 10(i)).
    # https://github.com/aertslab/pycisTopic/blob/cb5f54111d2affd89ad732c20c14dab6138026f0/LICENSE.txt#L19-L26
    if path.name != "task_019.tar.gz":
        yield from package_rows(path, context)


def _offline_image(image: str, *, os: str, allow_internet: bool) -> str:
    if allow_internet or os != "linux" or not re.fullmatch(r"[^@]+@sha256:[0-9a-f]{64}", image):
        raise ValueError("SWE-bench Science requires offline digest-pinned Linux images")
    # Skopeo accepts repository@digest, including registries with ports.
    return re.sub(r":[^/:@]+(?=@)", "", image)


def lower_package(directory: Path, row: RawRow) -> TaskSpec:
    """Build a TaskSpec directly from a pinned SWE-bench Science release package."""
    config = tomllib.loads((directory / "task.toml").read_text())
    agent, verifier = config["agent"], config["verifier"]
    solver_env, verifier_env = config["environment"], verifier["environment"]
    if config["schema_version"] != "1.3" or verifier["environment_mode"] != "separate":
        raise ValueError("Expected SWE-bench Science's schema 1.3 separate-verifier release")
    if config.get("artifacts") != [PATCH]:
        raise ValueError(f"Expected the sole SWE-bench Science artifact to be {PATCH}")
    if any(role["network_mode"] != "no-network" for role in (agent, verifier)):
        raise ValueError("SWE-bench Science requires the native no-network protocol")

    solver = LoweredSource(
        _offline_image(solver_env["docker_image"], os=solver_env["os"], allow_internet=solver_env["allow_internet"]),
        None,
        solver_env["workdir"],
    )
    solver = extend_source(solver, pi_install(PI))
    hook = (directory / "pre_artifacts.sh").read_text()
    command = f"cd {shlex.quote(solver_env['workdir'])} && sh -c {shlex.quote(hook)}"
    episode = EpisodeInputs(
        agent_timeout=agent["timeout_sec"],
        verifier_timeout=verifier["timeout_sec"],
        cpus=solver_env["cpus"],
        memory_mb=solver_env["memory_mb"],
        storage_mb=SOLVER_DISK_MB,
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
                    docker_image=_offline_image(
                        verifier_env["docker_image"],
                        os=verifier_env["os"],
                        allow_internet=verifier_env["allow_internet"],
                    ),
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
    with TemporaryDirectory(prefix=f"{BENCHMARK}-") as root:
        package = Path(root) / row.data["name"]
        extract_package(row.data["archive"], package)
        return lower_package(package, row)


def benchmarks() -> list[EvaluationBenchmark]:
    protocol_name = "pi-xhigh"
    name = f"{BENCHMARK}-{protocol_name}"
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
        source=HarborRepoSource(DATASET, REVISION, read=benchmark_rows, root="tasks"),
        convert=convert,
    )
    source = RlDataSource(
        info=SourceInfo(
            upstream_dataset=name,
            upstream_name="SWE-bench Science",
            origin=SourceOrigin.HARBOR,
            build_path="Harbor",
            data_restriction=DataRestriction.OPEN,
            family="agentic",
            tags=("eval", "source:harbor", "agentic", "multi-turn", "benchmark"),
            count=TASK_COUNT,
            task_format=TaskFormat.HARBOR,
        ),
        config=EvaluationCorpusRecipe(
            recipe,
            harbor_protocol(
                BENCHMARK,
                VERSION,
                TASK_COUNT,
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

    return fixed_benchmarks([source])

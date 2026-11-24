# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SWE-bench Science with native isolated grading.

Release tooling and metadata are MIT licensed. Task sources retain their upstream licenses;
the default selection follows the release's unrestricted manifest.
"""

import re
import shlex
import shutil
import tomllib
from pathlib import Path
from tempfile import TemporaryDirectory

import tomlkit
from taskcompendium.evaluation.spec import MiniSweAgent, PiAgent
from taskcompendium.importers.harbor.config import TASK_CONFIG
from taskcompendium.importers.harbor.package import (
    REWARD_JSON_PATH,
    HarborTask,
    JobSettings,
    PackageTests,
    lower_package,
)
from taskcompendium.models import (
    ArtifactKind,
    FileReward,
    GraderMachine,
    RewardFile,
    RewardFileFormat,
    ScriptGrader,
    VerifierArtifact,
)
from taskcompendium.pipeline.models import RawRow

from experiments.evaluation.benchmark import EvaluationBenchmark, fixed_benchmarks
from experiments.evaluation.benchmarks.harbor import (
    MAX_OUTPUT_TOKENS,
    PI_MAX_OUTPUT_TOKENS,
    HarborDataset,
    harbor_source,
    pi_install,
)
from experiments.evaluation.preparation import EvaluationCorpusRecipe
from experiments.post_training.task_curation.harbor_packages import package_rows
from experiments.post_training.task_curation.pipeline import HarborRepoSource
from experiments.post_training.task_curation.source import RlDataSource

DATASET = "OpenMOSS-Team/SWE-bench-Science"
REVISION = "997408298e8e4d09978b1120b95c4540d1a71bcb"
VERSION = "2026.11.24.3"
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


def _harbor_config(directory: Path) -> dict:
    """Translate the supported Pier release fields into their Harbor 1.2 equivalents."""
    config = tomllib.loads((directory / TASK_CONFIG).read_text())
    verifier = config["verifier"]
    if config["schema_version"] != "1.3" or verifier["environment_mode"] != "separate":
        raise ValueError("Expected SWE-bench Science's schema 1.3 separate-verifier release")
    if config.get("artifacts") != [PATCH]:
        raise ValueError(f"Expected the sole Science artifact to be {PATCH}")
    for environment in (config["environment"], verifier["environment"]):
        if environment["allow_internet"] or "@sha256:" not in environment["docker_image"]:
            raise ValueError("SWE-bench Science requires offline digest-pinned images")
        # Skopeo's docker transport accepts repository@digest, including registries with ports.
        environment["docker_image"] = re.sub(r":[^/:@]+(?=@)", "", environment["docker_image"])
    if config["agent"]["network_mode"] != "no-network" or verifier["network_mode"] != "no-network":
        raise ValueError("SWE-bench Science requires the native no-network protocol")
    hook = (directory / "pre_artifacts.sh").read_text()
    command = f"cd {shlex.quote(config['environment']['workdir'])} && sh -c {shlex.quote(hook)}"
    # This release's 1.3 fields have 1.2 equivalents: offline environment
    # declarations plus an explicit collection command replace Pier's hook.
    config["schema_version"] = "1.2"
    verifier["collect"] = [{"command": command, "timeout_sec": 300.0}]
    return config


def lower_science_package(directory: Path, row: RawRow, job: JobSettings) -> HarborTask:
    """Convert a Pier 1.3 science package to a Harbor task with isolated patch grading."""
    config = _harbor_config(directory)
    with TemporaryDirectory(prefix="swe-science-package-") as temporary:
        prepared = Path(temporary) / directory.name
        (prepared / "environment").mkdir(parents=True)
        (prepared / "tests").mkdir()
        shutil.copyfile(directory / "instruction.md", prepared / "instruction.md")
        shutil.copyfile(directory / "tests/grader.py", prepared / "tests/grader.py")
        (prepared / TASK_CONFIG).write_text(tomlkit.dumps(config))
        lowered = lower_package(
            prepared, task_id=row.id, source=row.source, rule=PackageTests(), image_digests={}, job=job
        )
    grader = lowered.task.grader
    assert isinstance(grader, ScriptGrader)
    assert isinstance(grader.placement, GraderMachine)
    placement = grader.placement.model_copy(
        update={"artifacts": (VerifierArtifact(source=PATCH, target=PATCH, kind=ArtifactKind.FILE),)}
    )
    # JSON mode reads the native reward field without treating nested diagnostic
    # reports as numeric metrics. The original report is not rewritten.
    grader = grader.model_copy(
        update={
            "placement": placement,
            "reward": FileReward(files=(RewardFile(path=REWARD_JSON_PATH, format=RewardFileFormat.JSON),)),
        }
    )
    return HarborTask(lowered.task.model_copy(update={"grader": grader}), lowered.episode)


def science_source(
    benchmark: str, task_ids: tuple[str, ...], harness: PiAgent | MiniSweAgent
) -> RlDataSource[EvaluationCorpusRecipe]:
    pi = isinstance(harness, PiAgent)
    protocol_name = "pi-xhigh" if pi else "mini-swe-agent"
    job = JobSettings(storage_mb=10240, agent_install=pi_install(harness) if isinstance(harness, PiAgent) else None)
    dataset = HarborDataset(
        recipe=f"{benchmark}-{protocol_name}",
        benchmark=benchmark,
        title=f"SWE-bench Science ({len(task_ids)} tasks)",
        source=HarborRepoSource(
            DATASET, REVISION, read=package_rows, root="tasks", packages=frozenset(f"task_{key}" for key in task_ids)
        ),
        version=VERSION,
        count=len(task_ids),
        attempt_timeout=10800,
    )
    return harbor_source(
        dataset,
        harness,
        lower=lower_science_package,
        job=job,
        max_input_tokens=65536 if pi else None,
        max_output_tokens=PI_MAX_OUTPUT_TOKENS if pi else MAX_OUTPUT_TOKENS,
        protocol=protocol_name,
    )


def sources() -> list[RlDataSource[EvaluationCorpusRecipe]]:
    return [
        science_source(BENCHMARK, TASK_IDS, PI),
        science_source(BENCHMARK, TASK_IDS, MiniSweAgent()),
        science_source(f"{BENCHMARK}-hosted114", HOSTED_TASK_IDS, PI),
    ]


def benchmarks() -> list[EvaluationBenchmark]:
    return fixed_benchmarks(sources())

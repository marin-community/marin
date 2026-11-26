# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Harbor task packages as agent-episode evaluation corpora.

A declaration pins a package source, a grading rule, the agent and its token limits. Each package
lowers to a TaskSpec through ``taskcompendium.importers.harbor``; the package's own timeouts,
resources and network policy become the item's ``EpisodeInputs``. A source's ``JobSettings``
replace a package's timeouts or storage, as a Harbor job's overrides do, and can install an agent,
such as Pi, that the package's image lacks. Container images are pinned by a
checked-in lock, ``harbor_images/<lock>.json``, so a rebuild lowers the same images. Datasets whose
packages name the same images share a lock. Refresh a lock after changing a source revision::

    uv run python -m experiments.evaluation.benchmarks.agentic lock <lock>
"""

import json
import logging
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

from finestore.rl.status_taxonomy import DataRestriction
from taskcompendium.evaluation.corpus import chat_item
from taskcompendium.evaluation.spec import (
    AgentEpisode,
    EpisodeAgent,
    EpisodeInputs,
    EpisodeLimits,
    EvaluationItem,
    EvaluationProtocol,
    Metric,
    MetricKind,
    OperationKind,
    PiAgent,
    Sampling,
    SummaryKind,
    TrialPolicy,
)
from taskcompendium.importers.harbor.config import UnsupportedHarborFeature
from taskcompendium.importers.harbor.images import resolve_image_digests
from taskcompendium.importers.harbor.package import (
    GradingRule,
    JobSettings,
    PackageTests,
    image_references,
    lower_package,
)
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import ImportFailureKind, ImportRejection, IntendedUse, RawRow

from experiments.evaluation.preparation import EvaluationCorpusRecipe, prepare_evaluation
from experiments.post_training.task_curation.campaign import CampaignRuntime
from experiments.post_training.task_curation.harbor_packages import PACKAGE_ARCHIVES, extract_package, package_rows
from experiments.post_training.task_curation.pipeline import (
    CurationRecipe,
    DownloadRequest,
    HarborArchiveSource,
    HarborHubSource,
    HarborRepoSource,
    download_source,
)
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, SourceOrigin, TaskFormat

logger = logging.getLogger(__name__)

EPISODE_LABEL = "harbor_episode"
IMAGE_LOCKS = Path(__file__).with_name("harbor_images")
REWARD = Metric(name="reward", kind=MetricKind.CONTINUOUS, higher_is_better=True)
MODEL_TURN_TIMEOUT = 600
ATTEMPT_TIMEOUT = 4 * 60 * 60
"""The infrastructure backstop of a dataset whose packages' agent and verifier timeouts sum below it."""
MAX_OUTPUT_TOKENS = 8192
"""The reply limit Harbor gives an agent whose policy and model state none."""
PI_MAX_OUTPUT_TOKENS = 16384
"""The reply limit Harbor gives Pi when its policy and model state none."""
PI_PACKAGE = "@earendil-works/pi-coding-agent"
NODE_VERSION = "22.23.5"
"""The Node.js release Pi runs on; Pi requires 22.19 or later."""
NODE_ARCHIVE_SHA256 = "c54118104acc524633b5abe106bbf44a53126ec8b4e5fa105cb92b5d7de8a1a1"
"""The digest of the release's ``linux-x64`` archive, from its ``SHASUMS256.txt``."""
PI_PREFIX = "/opt/pi"

type RuleFor = Callable[[Path], GradingRule]
type PackageSource = HarborRepoSource | HarborArchiveSource | HarborHubSource


@dataclass(frozen=True)
class PackageEdit:
    """Replace the one occurrence of ``old`` with ``new`` in the package file at relative ``path``.

    A dataset edits its packages to remove software that Marin runs must not install, or to change how they grade.
    """

    path: str
    old: str
    new: str

    def apply(self, directory: Path) -> None:
        file = directory / self.path
        text = file.read_text()
        if text.count(self.old) != 1:
            raise ValueError(f"{self.path} must contain the edited text exactly once")
        file.write_text(text.replace(self.old, self.new))


@dataclass(frozen=True)
class PackageCopy:
    """Copy the package file at relative ``source`` to relative ``target``, which must not exist."""

    source: str
    target: str

    def apply(self, directory: Path) -> None:
        target = directory / self.target
        if target.exists():
            raise ValueError(f"{self.target} already exists")
        shutil.copyfile(directory / self.source, target)


type PackageChange = PackageEdit | PackageCopy


def edit_package(directory: Path, edits: tuple[PackageChange, ...]) -> None:
    """Apply ``edits`` to the package at ``directory`` in order."""
    for edit in edits:
        edit.apply(directory)


def package_tests(_directory: Path) -> GradingRule:
    """Grade every package with its own ``tests/test.sh``."""
    return PackageTests()


@dataclass(frozen=True)
class HarborConversion:
    """Lower one package row, after ``edits`` and as ``job`` runs it, with its grading rule and locked image digests."""

    rule: RuleFor
    image_digests: Mapping[str, str]

    def __call__(
        self,
        row: RawRow,
        _context: ConversionContext,
        job: JobSettings = JobSettings(),
        edits: tuple[PackageChange, ...] = (),
    ) -> TaskSpec | ImportRejection:
        with TemporaryDirectory(prefix="harbor-package-") as root:
            directory = Path(root) / row.data["name"]
            extract_package(row.data["archive"], directory)
            edit_package(directory, edits)
            try:
                lowered = lower_package(
                    directory,
                    task_id=row.id,
                    source=row.source,
                    rule=self.rule(directory),
                    image_digests=self.image_digests,
                    job=job,
                )
            except UnsupportedHarborFeature as error:
                return ImportRejection(kind=ImportFailureKind.UNSUPPORTED, reason="harbor_feature", detail=error.reason)
        task = lowered.task
        labels = {**task.labels, EPISODE_LABEL: lowered.episode.model_dump_json()}
        return task.model_copy(update={"labels": labels, "tags": ("eval",)})


def harbor_item(task: TaskSpec, *, agent: AgentEpisode) -> EvaluationItem:
    """Run ``agent`` with the episode inputs the package declared."""
    episode = EpisodeInputs.model_validate_json(task.labels[EPISODE_LABEL])
    return chat_item(task).model_copy(update={"operation": agent, "episode": episode})


def image_lock(recipe: str) -> dict[str, str]:
    return json.loads((IMAGE_LOCKS / f"{recipe}.json").read_text())


def harbor_protocol(
    benchmark: str,
    version: str,
    expected_population: int,
    *,
    max_output_tokens: int,
    requires_judge: bool,
    trials: int,
) -> EvaluationProtocol:
    """Harbor's scoring: the mean of each trial's package reward over ``trials`` trials of every package.

    Agents choose their own sampling, so the protocol sets only the per-reply token limit, which a
    run's smaller budget lowers.
    """
    return EvaluationProtocol(
        benchmark=benchmark,
        requires_judge=requires_judge,
        version=version,
        operation=OperationKind.AGENT_EPISODE,
        sampling=Sampling(
            temperature=None, top_p=None, top_k=None, max_tokens=max_output_tokens, stop=(), chat_template_kwargs={}
        ),
        trials=TrialPolicy(count=trials, seed=None),
        metrics=(REWARD,),
        primary_metric=REWARD.name,
        summary=SummaryKind.MEAN,
        expected_population=expected_population,
    )


@dataclass(frozen=True)
class HarborDataset:
    """One pinned Harbor dataset and how its packages are graded.

    ``benchmark`` is the curated benchmark name, which is also the Evaldash family of the dataset's
    historical Harbor runs. ``requires_judge`` says the rule grades with a host-side judge, so runs
    must supply one. ``attempt_timeout`` is the infrastructure backstop, which must exceed every
    package's agent timeout plus its verifier timeout. ``images`` names the image lock the dataset
    shares with others whose packages name the same images; by default the lock is the recipe's own.
    ``edits`` change each package before it is lowered.
    """

    recipe: str
    benchmark: str
    title: str
    source: PackageSource
    version: str
    count: int
    rule: RuleFor = package_tests
    requires_judge: bool = False
    attempt_timeout: float = ATTEMPT_TIMEOUT
    images: str | None = None
    edits: tuple[PackageChange, ...] = ()

    @property
    def image_lock_name(self) -> str:
        return self.images or self.recipe


def huggingface_packages(repo: str, revision: str) -> HarborRepoSource:
    """The packages of a Hugging Face dataset repository at a full commit, one top-level directory each."""
    return HarborRepoSource(repo, revision, read=package_rows)


def github_packages(
    repository: str, commit: str, sha256: str, root: str, packages: frozenset[str] | None = None
) -> HarborArchiveSource:
    """The packages under ``root`` of a GitHub repository at ``commit``, whose archive has digest ``sha256``.

    This is how Harbor's registry addresses a package: a Git repository, a commit and a path.
    ``packages`` selects the directories a registry dataset lists, when it lists a subset.
    """
    return HarborArchiveSource(
        url=f"https://codeload.github.com/{repository}/tar.gz/{commit}",
        sha256=sha256,
        root=root,
        read=package_rows,
        packages=packages,
    )


def harbor_source(
    dataset: HarborDataset,
    agent: EpisodeAgent,
    *,
    max_input_tokens: int | None = None,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
    trials: int = 1,
    job: JobSettings | None = None,
    protocol: str | None = None,
) -> RlDataSource[EvaluationCorpusRecipe]:
    """An evaluation source that runs ``agent`` ``trials`` times on every package of ``dataset``.

    ``max_input_tokens`` caps the agent's context below the served window; ``max_output_tokens``
    caps each reply. ``job`` holds the Harbor job settings that replace or extend each package's own.
    ``protocol`` names the protocol when the agent's kind does not tell it apart from the benchmark's others.
    """
    convert = HarborConversion(dataset.rule, image_lock(dataset.image_lock_name))
    keywords = {
        **({} if job is None else {"job": job}),
        **({"edits": dataset.edits} if dataset.edits else {}),
    }
    # Harbor's limits: each package sets its agent and verifier timeouts, and Harbor caps no turns or tokens.
    limits = EpisodeLimits(
        max_turns=None,
        max_total_tokens=None,
        model_turn_timeout=MODEL_TURN_TIMEOUT,
        total_turn_timeout=None,
        verifier_timeout=None,
        attempt_timeout=dataset.attempt_timeout,
    )
    episode = AgentEpisode(harness=agent, limits=limits, max_input_tokens=max_input_tokens)
    return RlDataSource(
        info=SourceInfo(
            data_restriction=DataRestriction.OPEN,
            upstream_dataset=dataset.recipe,
            upstream_name=dataset.title,
            subset=None,
            build_path="Harbor",
            origin=SourceOrigin.HARBOR,
            family="agentic",
            tags=("eval", "source:harbor", "agentic", "multi-turn", "benchmark"),
            count=dataset.count,
            task_format=TaskFormat.HARBOR,
        ),
        config=EvaluationCorpusRecipe(
            CurationRecipe(
                name=f"harbor-{dataset.recipe}",
                source=dataset.source,
                # A recipe without job settings or edits keeps the conversion identity of its published corpora.
                convert=partial(convert, **keywords) if keywords else convert,
                version=dataset.version,
                intended_use=IntendedUse.EVAL,
            ),
            harbor_protocol(
                dataset.benchmark,
                dataset.version,
                dataset.count,
                max_output_tokens=max_output_tokens,
                requires_judge=dataset.requires_judge,
                trials=trials,
            ),
            partial(harbor_item, agent=episode),
            protocol_name=protocol or agent.kind,
            agent=episode,
        ),
        pipeline=prepare_evaluation,
    )


def pi_install(agent: PiAgent) -> str:
    """Dockerfile instructions that install ``agent``'s Pi version as ``pi``, with its own Node.js.

    ``pi`` runs with that Node.js first on its ``PATH``, as Harbor ran Pi after ``nvm use``. The image's
    ``PATH`` ends with it, so the Pi agent's ``node`` helper scripts find a Node.js in any image while a
    task's own Node.js stays first. An image without curl gets it from apt, as Harbor's Pi setup did.
    The build fails unless ``pi --version`` reports the pinned version.
    """
    if agent.version is None:
        raise ValueError("Installing Pi requires the agent's pinned Pi version")
    archive = f"https://nodejs.org/dist/v{NODE_VERSION}/node-v{NODE_VERSION}-linux-x64.tar.gz"
    commands = (
        "set -eu",
        "if ! command -v curl >/dev/null; then apt-get update"
        " && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ca-certificates curl; fi",
        f"curl -fsSL -o /tmp/node.tar.gz {archive}",
        f'echo "{NODE_ARCHIVE_SHA256}  /tmp/node.tar.gz" | sha256sum -c -',
        f"mkdir -p {PI_PREFIX}",
        f"tar -xzf /tmp/node.tar.gz -C {PI_PREFIX} --strip-components=1 --no-same-owner",
        "rm /tmp/node.tar.gz",
        f"PATH={PI_PREFIX}/bin:$PATH npm install -g --ignore-scripts {PI_PACKAGE}@{agent.version}",
        f"printf '#!/bin/sh\\nPATH={PI_PREFIX}/bin:$PATH exec {PI_PREFIX}/bin/pi \"$@\"\\n' > /usr/local/bin/pi",
        "chmod 755 /usr/local/bin/pi",
        f'test "$(pi --version | tail -n 1)" = "{agent.version}"',
    )
    return "RUN " + "; \\\n    ".join(commands) + f"\nENV PATH=$PATH:{PI_PREFIX}/bin\n"


def lock_images(dataset: HarborDataset) -> dict[str, str]:
    """Resolve every image the dataset's packages name to the digest its registry serves now."""
    references: set[str] = set()
    with TemporaryDirectory(prefix="harbor-lock-") as directory:
        root = Path(directory)
        download_source(DownloadRequest(dataset.source, str(root / "staged")), campaign=CampaignRuntime())
        for archive in sorted(root.glob(f"staged/{PACKAGE_ARCHIVES}")):
            package = root / "package"
            extract_package(archive.read_bytes(), package)
            references.update(image_references(package))
            shutil.rmtree(package)
    logger.info("Resolving %d images for %s", len(references), dataset.recipe)
    return dict(sorted(resolve_image_digests(references).items()))

#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Advance external projects and package their immutable runtime inputs."""

import argparse
import json
import re
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, quote, urlsplit, urlunsplit

ROOT = Path(__file__).parents[1]
EXTERNAL_ROOT = Path(__file__).with_name("external")
GENERATED_PINS = ROOT / "lib" / "marin" / "src" / "marin" / "external_dependencies.py"
GIT_SUFFIX = ".git"
GIT_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
GENERATED_STRING_CHUNK_WIDTH = 88
VLLM_CONFIG_NAME = "vllm"
VLLM_GPU_RELEASE_CONFIG = EXTERNAL_ROOT / VLLM_CONFIG_NAME / "gpu-release.toml"
TPU_FORKS_CONFIG = EXTERNAL_ROOT / VLLM_CONFIG_NAME / "tpu-forks.toml"


@dataclass(frozen=True)
class ExternalProject:
    config_name: str
    distribution: str
    constant_name: str
    runtime_distributions: tuple[str, ...] = ()

    @property
    def directory(self) -> Path:
        return EXTERNAL_ROOT / self.config_name


@dataclass(frozen=True)
class LockedDependency:
    project: ExternalProject
    repository: str
    version: str
    commit: str
    runtime_requirements: tuple[str, ...]


@dataclass(frozen=True)
class LockedGitSource:
    repository: str
    commit: str


@dataclass(frozen=True)
class UpstreamCommit:
    commit: str
    subject: str


@dataclass(frozen=True)
class DependencyUpdate:
    previous: LockedDependency
    current: LockedDependency
    commits: tuple[UpstreamCommit, ...]


@dataclass(frozen=True)
class VllmGpuWheel:
    architecture: str
    sm_targets: tuple[str, ...]
    url: str
    sha256: str


@dataclass(frozen=True)
class VllmGpuRelease:
    release_tag: str
    source_commit: str
    version: str
    torch_backend: str
    wheels: tuple[VllmGpuWheel, ...]


EXTERNAL_PROJECTS = (
    ExternalProject("evalchemy", "evalchemy", "EVALCHEMY"),
    ExternalProject(
        "harbor",
        "harbor",
        "HARBOR",
        runtime_distributions=("daytona", "pydantic-settings"),
    ),
    ExternalProject("MarinSkyRL", "marinskyrl", "MARIN_SKYRL"),
)


def locked_package(lock_path: Path, distribution: str) -> dict:
    """Read one distribution record from a uv lockfile."""
    lock = tomllib.loads(lock_path.read_text())
    packages = [package for package in lock["package"] if package["name"] == distribution]
    if len(packages) != 1:
        raise ValueError(f"{lock_path}: expected one {distribution!r} package, found {len(packages)}")
    return packages[0]


def locked_git_source(lock_path: Path, distribution: str) -> LockedGitSource:
    """Read and validate one distribution's immutable Git source."""
    package = locked_package(lock_path, distribution)
    source = package["source"]["git"]
    parsed = urlsplit(source)
    commit = parsed.fragment
    if GIT_COMMIT_PATTERN.fullmatch(commit) is None:
        raise ValueError(f"{lock_path}: expected a Git source ending in a full commit, found {source!r}")
    if "subdirectory" in parse_qs(parsed.query):
        raise ValueError(f"{lock_path}: expected a root Git source, found {source!r}")
    repository = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
    return LockedGitSource(repository=repository, commit=commit)


def tpu_fork_source(path: Path, name: str) -> LockedGitSource:
    """Read one forked TPU vLLM pin (repository + commit) from the fork descriptor.

    The TPU vLLM stack (``vllm`` + ``tpu-inference``) runs from an isolated uvx env rather than
    the workspace lock, so its SHAs live in ``config/external/vllm/tpu-forks.toml`` instead of
    ``uv.lock``. See ``marin.inference.vllm_server.IsolatedTpuVllm``.
    """
    config = tomllib.loads(path.read_text())
    if name not in config:
        raise ValueError(f"{path}: missing [{name}] section")
    entry = config[name]
    repository = entry["repository"]
    commit = entry["commit"]
    if not repository.endswith(GIT_SUFFIX):
        raise ValueError(f"{path}: [{name}] repository must be a .git URL, found {repository!r}")
    if GIT_COMMIT_PATTERN.fullmatch(commit) is None:
        raise ValueError(f"{path}: [{name}] commit is not a full Git SHA, found {commit!r}")
    return LockedGitSource(repository=repository, commit=commit)


def locked_dependency(project: ExternalProject) -> LockedDependency:
    """Read an external package and its runtime requirements from its uv lockfile."""
    lock_path = project.directory / "uv.lock"
    package = locked_package(lock_path, project.distribution)
    source = locked_git_source(lock_path, project.distribution)
    requirements = tuple(
        f"{distribution}=={locked_package(lock_path, distribution)['version']}"
        for distribution in project.runtime_distributions
    )
    return LockedDependency(
        project=project,
        repository=source.repository,
        version=package["version"],
        commit=source.commit,
        runtime_requirements=requirements,
    )


def load_vllm_gpu_release(path: Path) -> VllmGpuRelease:
    """Load and validate the promoted vLLM GPU release."""
    config = tomllib.loads(path.read_text())
    release = VllmGpuRelease(
        release_tag=config["release_tag"],
        source_commit=config["source_commit"],
        version=config["version"],
        torch_backend=config["torch_backend"],
        wheels=tuple(
            VllmGpuWheel(
                architecture=wheel["architecture"],
                sm_targets=tuple(wheel["sm_targets"]),
                url=wheel["url"],
                sha256=wheel["sha256"],
            )
            for wheel in config["wheels"]
        ),
    )
    if GIT_COMMIT_PATTERN.fullmatch(release.source_commit) is None:
        raise ValueError(f"{path}: expected a full vLLM source commit, found {release.source_commit!r}")
    if not re.fullmatch(r"cu[0-9]+", release.torch_backend):
        raise ValueError(f"{path}: expected a CUDA torch backend, found {release.torch_backend!r}")
    architectures = tuple(wheel.architecture for wheel in release.wheels)
    if not architectures or len(set(architectures)) != len(architectures):
        raise ValueError(f"{path}: expected one or more unique wheel architectures, found {architectures!r}")
    release_url_prefix = f"https://github.com/marin-community/vllm/releases/download/{release.release_tag}/"
    wheel_filename_prefix = f"vllm-{quote(release.version, safe='')}-"
    wheel_urls = tuple(wheel.url for wheel in release.wheels)
    if len(set(wheel_urls)) != len(wheel_urls):
        raise ValueError(f"{path}: expected a unique asset URL for each wheel")
    for wheel in release.wheels:
        if not wheel.sm_targets:
            raise ValueError(f"{path}: {wheel.architecture} must declare at least one SM target")
        if not wheel.url.startswith(release_url_prefix) or "#" in wheel.url:
            raise ValueError(f"{path}: {wheel.architecture} wheel URL is not an asset from {release.release_tag}")
        if f"/{wheel_filename_prefix}" not in wheel.url or not wheel.url.endswith(f"_{wheel.architecture}.whl"):
            raise ValueError(f"{path}: {wheel.architecture} wheel URL does not match vLLM version {release.version}")
        if SHA256_PATTERN.fullmatch(wheel.sha256) is None:
            raise ValueError(f"{path}: {wheel.architecture} wheel has invalid SHA-256 {wheel.sha256!r}")
    return release


def render_pins(
    dependencies: tuple[LockedDependency, ...],
    *,
    vllm_gpu_release: VllmGpuRelease,
    vllm_fork_requirement: str,
    tpu_inference_fork_requirement: str,
) -> str:
    """Render the packaged pin table consumed by isolated Marin runtimes."""

    def tuple_literal(values: tuple[str, ...]) -> str:
        quoted = ", ".join(f'"{value}"' for value in values)
        # A single-element tuple needs its trailing comma; adding one for 2+ elements is a magic
        # trailing comma that Black would explode across lines, so the generated file would never
        # match a Black pass. Emit the flat form Black leaves alone.
        if not values:
            return "()"
        return f'("{values[0]}",)' if len(values) == 1 else f"({quoted})"

    def wrapped_string_literal(value: str, *, indent: str) -> str:
        chunks = []
        remaining = value
        while len(remaining) > GENERATED_STRING_CHUNK_WIDTH:
            search_end = GENERATED_STRING_CHUNK_WIDTH + 1
            slash = remaining.rfind("/", 0, search_end) + 1
            split_at = slash if slash >= GENERATED_STRING_CHUNK_WIDTH // 2 else remaining.rfind("-", 0, search_end) + 1
            if split_at <= 0:
                split_at = GENERATED_STRING_CHUNK_WIDTH
            chunks.append(remaining[:split_at])
            remaining = remaining[split_at:]
        chunks.append(remaining)
        if len(chunks) == 1:
            return json.dumps(value)
        lines = "\n".join(f"{indent}{json.dumps(chunk)}" for chunk in chunks)
        return f"(\n{lines}\n{indent[:-4]})"

    entries = "\n\n".join(
        f"{dependency.project.constant_name} = ExternalDependency(\n"
        f'    config_name="{dependency.project.config_name}",\n'
        f'    distribution="{dependency.project.distribution}",\n'
        f'    repository="{dependency.repository}",\n'
        f'    version="{dependency.version}",\n'
        f'    commit="{dependency.commit}",\n'
        f"    runtime_requirements={tuple_literal(dependency.runtime_requirements)},\n"
        f")"
        for dependency in dependencies
    )
    constants = "\n".join(f"    {dependency.project.constant_name}," for dependency in dependencies)
    wheel_entries = "\n".join(
        "        VllmGpuWheel(\n"
        f'            architecture="{wheel.architecture}",\n'
        f"            sm_targets={tuple_literal(wheel.sm_targets)},\n"
        f"            url={wrapped_string_literal(wheel.url, indent='                ')},\n"
        f'            sha256="{wheel.sha256}",\n'
        "        ),"
        for wheel in vllm_gpu_release.wheels
    )
    return f'''# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""External requirements generated by ``config/update-external.py``.

Do not edit this file directly. External project revisions come from their
``config/external/<name>/uv.lock`` files, the promoted GPU vLLM release comes
from ``config/external/vllm/gpu-release.toml``, and TPU serving requirements
come from the root ``uv.lock``.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ExternalDependency:
    """An external distribution installed from an immutable Git revision."""

    config_name: str
    distribution: str
    repository: str
    version: str
    commit: str
    runtime_requirements: tuple[str, ...]

    def requirement(self, extras: tuple[str, ...] = ()) -> str:
        """Return a PEP 508 requirement for this revision."""
        distribution = self.distribution
        if extras:
            distribution = f"{{distribution}}[{{','.join(extras)}}]"
        return f"{{distribution}} @ git+{{self.repository}}@{{self.commit}}"


@dataclass(frozen=True)
class VllmGpuWheel:
    """One architecture-specific wheel from a promoted vLLM GPU release."""

    architecture: str
    sm_targets: tuple[str, ...]
    url: str
    sha256: str


@dataclass(frozen=True)
class VllmGpuRelease:
    """Immutable artifacts and ABI for a promoted vLLM GPU release."""

    release_tag: str
    source_commit: str
    version: str
    torch_backend: str
    wheels: tuple[VllmGpuWheel, ...]


{entries}

VLLM_GPU_RELEASE = VllmGpuRelease(
    release_tag="{vllm_gpu_release.release_tag}",
    source_commit="{vllm_gpu_release.source_commit}",
    version="{vllm_gpu_release.version}",
    torch_backend="{vllm_gpu_release.torch_backend}",
    wheels=(
{wheel_entries}
    ),
)

VLLM_FORK_REQUIREMENT = "{vllm_fork_requirement}"
TPU_INFERENCE_FORK_REQUIREMENT = (
    "{tpu_inference_fork_requirement}"
)

EXTERNAL_DEPENDENCIES = (
{constants}
)
'''


def github_repository_path(repository: str) -> str:
    """Return an owner/repository path from a GitHub HTTPS URL."""
    prefix = "https://github.com/"
    repository_path = repository.removesuffix(GIT_SUFFIX).removeprefix(prefix)
    if not repository.startswith(prefix) or repository_path.count("/") != 1:
        raise ValueError(f"expected a GitHub HTTPS repository, found {repository!r}")
    return repository_path


def github_commits(repository: str, base: str, head: str) -> tuple[UpstreamCommit, ...]:
    """Return every commit in a GitHub comparison, oldest first."""
    repository_path = github_repository_path(repository)
    result = subprocess.run(
        [
            "gh",
            "api",
            f"repos/{repository_path}/compare/{base}...{head}",
            "--method",
            "GET",
            "-f",
            "per_page=100",
            "--paginate",
            "--slurp",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    pages = json.loads(result.stdout)
    if not pages or any(page["status"] != "ahead" for page in pages):
        statuses = sorted({page["status"] for page in pages})
        raise ValueError(f"{repository}: expected {base}..{head} to be ahead, found {statuses}")

    commits = tuple(
        UpstreamCommit(
            commit=commit["sha"],
            subject=commit["commit"]["message"].partition("\n")[0],
        )
        for page in pages
        for commit in page["commits"]
    )
    expected_commits = pages[0]["total_commits"]
    if len(commits) != expected_commits:
        raise ValueError(f"{repository}: expected {expected_commits} commits in {base}..{head}, found {len(commits)}")
    return commits


def dependency_updates(
    previous: tuple[LockedDependency, ...],
    current: tuple[LockedDependency, ...],
) -> tuple[DependencyUpdate, ...]:
    """Return changed dependency ranges with their upstream commits."""
    previous_by_project = {dependency.project.config_name: dependency for dependency in previous}
    updates = []
    for current_dependency in current:
        previous_dependency = previous_by_project[current_dependency.project.config_name]
        if previous_dependency.commit == current_dependency.commit:
            continue
        if previous_dependency.repository != current_dependency.repository:
            raise ValueError(
                f"{current_dependency.project.config_name}: cannot compare different repositories "
                f"{previous_dependency.repository!r} and {current_dependency.repository!r}"
            )
        updates.append(
            DependencyUpdate(
                previous=previous_dependency,
                current=current_dependency,
                commits=github_commits(
                    current_dependency.repository,
                    previous_dependency.commit,
                    current_dependency.commit,
                ),
            )
        )
    return tuple(updates)


def render_summary(
    dependencies: tuple[LockedDependency, ...],
    updates: tuple[DependencyUpdate, ...],
) -> str:
    """Render the resolved external revisions for an automated pull request."""
    rows = "\n".join(
        f"| `{dependency.project.config_name}` | `{dependency.project.distribution}` | "
        f"`{dependency.version}` | "
        f"[`{dependency.commit[:12]}`]"
        f"({dependency.repository.removesuffix(GIT_SUFFIX)}/commit/{dependency.commit}) |"
        for dependency in dependencies
    )
    update_sections = "\n\n".join(
        f"### {update.current.project.config_name} "
        f"(`{update.previous.commit[:12]}` → `{update.current.commit[:12]}`)\n\n"
        + "\n".join(
            f"- [`{commit.commit[:12]}`]"
            f"({update.current.repository.removesuffix(GIT_SUFFIX)}/commit/{commit.commit}): "
            f"{commit.subject}"
            for commit in update.commits
        )
        for update in updates
    )
    upstream_commits = f"\n\n## Upstream commits\n\n{update_sections}" if update_sections else ""
    return f"""Advance Marin's isolated external runtimes to the current commits on their configured branches.

| Project | Distribution | Version | Commit |
| --- | --- | --- | --- |
{rows}{upstream_commits}

Generated by `config/update-external.py`. Review the linked upstream changes before merging.
"""


def synchronize_file(path: Path, expected: str, *, check: bool) -> bool:
    """Write expected content unless checking; return whether it already matched."""
    matches = path.exists() and path.read_text() == expected
    if not matches and not check:
        path.write_text(expected)
    return matches


def project_by_name(name: str) -> ExternalProject:
    """Return the configured external project named by the CLI."""
    return next(project for project in EXTERNAL_PROJECTS if project.config_name == name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upgrade external uv locks and package immutable external runtime inputs."
    )
    parser.add_argument(
        "projects",
        nargs="*",
        choices=(*(project.config_name for project in EXTERNAL_PROJECTS), VLLM_CONFIG_NAME),
        metavar="PROJECT",
        help="Git projects to advance, or vllm to regenerate its promoted GPU release (default: all Git projects)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify generated pins without updating locks or files",
    )
    parser.add_argument(
        "--summary-file",
        type=Path,
        help="write a Markdown summary of the resolved versions and revisions",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = tuple(project_by_name(name) for name in args.projects if name != VLLM_CONFIG_NAME)
    if not args.projects:
        selected = EXTERNAL_PROJECTS
    previous_dependencies = tuple(locked_dependency(project) for project in EXTERNAL_PROJECTS)
    if not args.check:
        for project in selected:
            subprocess.run(
                [
                    "uv",
                    "lock",
                    "--project",
                    str(project.directory),
                    "--upgrade-package",
                    project.distribution,
                ],
                check=True,
            )

    dependencies = tuple(locked_dependency(project) for project in EXTERNAL_PROJECTS)
    vllm_gpu_release = load_vllm_gpu_release(VLLM_GPU_RELEASE_CONFIG)
    for dependency in dependencies:
        print(
            f"resolved {dependency.project.config_name}: "
            f"{dependency.project.distribution}=={dependency.version} commit={dependency.commit}"
        )
    print(
        f"resolved vllm GPU release: {vllm_gpu_release.version} "
        f"source={vllm_gpu_release.source_commit} architectures="
        f"{','.join(wheel.architecture for wheel in vllm_gpu_release.wheels)}"
    )
    if args.summary_file is not None:
        updates = dependency_updates(previous_dependencies, dependencies)
        args.summary_file.write_text(render_summary(dependencies, updates))
    vllm_source = tpu_fork_source(TPU_FORKS_CONFIG, "vllm")
    tpu_inference_source = tpu_fork_source(TPU_FORKS_CONFIG, "tpu-inference")
    pins_match = synchronize_file(
        GENERATED_PINS,
        render_pins(
            dependencies,
            vllm_gpu_release=vllm_gpu_release,
            vllm_fork_requirement=f"vllm @ git+{vllm_source.repository}@{vllm_source.commit}",
            tpu_inference_fork_requirement=(
                f"tpu-inference @ git+{tpu_inference_source.repository}@{tpu_inference_source.commit}"
            ),
        ),
        check=args.check,
    )
    if args.check and not pins_match:
        raise SystemExit("external dependency pins are stale; run `uv run config/update-external.py`")


if __name__ == "__main__":
    main()

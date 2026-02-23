# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Image build commands."""

import subprocess
from pathlib import Path

import click

from iris.cluster.platform.bootstrap import collect_all_regions, parse_artifact_registry_tag

GHCR_DEFAULT_ORG = "marin-community"


def _is_verbose(ctx: click.Context) -> bool:
    """Walk up the Click context chain to find the top-level --verbose flag."""
    while ctx:
        if "verbose" in (ctx.params or {}):
            return ctx.params["verbose"]
        ctx = ctx.parent  # type: ignore[assignment]
    return False


def get_git_sha() -> str:
    """Get a short hash representing the current working tree state.

    Uses ``git stash create`` to produce a commit object that captures both
    staged and unstaged changes without side effects. If the tree is clean,
    stash create returns empty and we fall back to HEAD.
    """
    # Try to capture dirty state as a temporary commit hash
    stash = subprocess.run(
        ["git", "stash", "create"],
        capture_output=True,
        text=True,
    )
    stash_ref = stash.stdout.strip()
    if stash_ref:
        # Dirty tree — use the stash commit hash
        result = subprocess.run(
            ["git", "rev-parse", "--short", stash_ref],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return result.stdout.strip()

    # Clean tree — use HEAD
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise click.ClickException("Failed to get git SHA. Are you in a git repository?")
    return result.stdout.strip()


def _default_versioned_tag(image_base: str) -> str:
    """Default image tag: latest + git short hash."""
    return f"{image_base}:latest-{get_git_sha()}"


def find_marin_root() -> Path:
    """Find the marin monorepo root (contains pyproject.toml + lib/iris)."""
    iris_root = find_iris_root()
    # iris root is lib/iris, marin root is two levels up
    marin_root = iris_root.parent.parent
    if (marin_root / "pyproject.toml").exists() and (marin_root / "lib" / "iris").is_dir():
        return marin_root
    raise click.ClickException("Cannot find marin repo root. Expected lib/iris to be inside a marin workspace.")


def find_iris_root() -> Path:
    """Find the iris package root directory containing Dockerfiles.

    Searches in order:
    1. Relative to this file (cli/build.py -> iris root is 4 levels up from src/iris/cli/build.py)
    2. Current working directory
    3. Walking up from cwd until Dockerfile.worker is found
    """
    build_path = Path(__file__).resolve()
    # build.py is at src/iris/cli/build.py, so iris root is 4 levels up
    iris_root = build_path.parent.parent.parent.parent
    if (iris_root / "Dockerfile.worker").exists() and (iris_root / "Dockerfile.controller").exists():
        return iris_root

    cwd = Path.cwd()
    if (cwd / "Dockerfile.worker").exists():
        return cwd

    for parent in cwd.parents:
        if (parent / "Dockerfile.worker").exists():
            return parent

    raise click.ClickException(
        "Cannot find Dockerfile.worker. Run from the iris directory or specify --dockerfile and --context."
    )


def _resolve_image_name_and_version(
    source_tag: str,
    image_name: str | None = None,
    version: str | None = None,
) -> tuple[str, str]:
    """Extract image name and version from a source tag, using overrides if provided."""
    parts = source_tag.split(":")
    if not image_name:
        image_name = parts[0].split("/")[-1]
    if not version:
        version = parts[1] if len(parts) > 1 else "latest"
    return image_name, version


def push_to_ghcr(
    source_tag: str,
    ghcr_org: str = GHCR_DEFAULT_ORG,
    image_name: str | None = None,
    version: str | None = None,
    verbose: bool = False,
) -> None:
    """Push a local Docker image to GitHub Container Registry (ghcr.io)."""
    image_name, version = _resolve_image_name_and_version(source_tag, image_name, version)
    dest_tag = f"ghcr.io/{ghcr_org}/{image_name}:{version}"

    click.echo(f"Pushing {source_tag} to ghcr.io/{ghcr_org}...")

    result = subprocess.run(["docker", "tag", source_tag, dest_tag], check=False)
    if result.returncode != 0:
        click.echo(f"Failed to tag image as {dest_tag}", err=True)
        raise SystemExit(1)

    click.echo(f"Pushing to {dest_tag}...")
    push_cmd = ["docker", "push", dest_tag]
    if not verbose:
        push_cmd.insert(2, "--quiet")
    if verbose:
        result = subprocess.run(push_cmd)
    else:
        result = subprocess.run(push_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        click.echo(f"Failed to push to {dest_tag}", err=True)
        if not verbose:
            if result.stdout:
                click.echo(result.stdout, err=True)
            if result.stderr:
                click.echo(result.stderr, err=True)
        raise SystemExit(1)

    click.echo(f"Successfully pushed to {dest_tag}")
    click.echo("\nDone!")


def push_to_gcp_registries(
    source_tag: str,
    regions: tuple[str, ...],
    project: str,
    image_name: str | None = None,
    version: str | None = None,
    verbose: bool = False,
) -> None:
    """Push a local Docker image to multiple GCP Artifact Registry regions."""
    image_name, version = _resolve_image_name_and_version(source_tag, image_name, version)

    click.echo(f"Pushing {source_tag} to {len(regions)} GCP region(s)...")

    for r in regions:
        dest_tag = f"{r}-docker.pkg.dev/{project}/marin/{image_name}:{version}"

        click.echo(f"\nConfiguring {r}-docker.pkg.dev...")
        result = subprocess.run(
            ["gcloud", "auth", "configure-docker", f"{r}-docker.pkg.dev", "-q"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            click.echo(
                f"Warning: Failed to configure docker auth for {r}-docker.pkg.dev: {result.stderr.strip()}", err=True
            )

        click.echo(f"Tagging as {dest_tag}")
        result = subprocess.run(["docker", "tag", source_tag, dest_tag], check=False)
        if result.returncode != 0:
            click.echo(f"Failed to tag image for {r}", err=True)
            continue

        click.echo(f"Pushing to {r}...")
        push_cmd = ["docker", "push", dest_tag]
        if not verbose:
            push_cmd.insert(2, "--quiet")
        if verbose:
            result = subprocess.run(push_cmd)
        else:
            result = subprocess.run(push_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            click.echo(f"Failed to push to {r}", err=True)
            if not verbose:
                if result.stdout:
                    click.echo(result.stdout, err=True)
                if result.stderr:
                    click.echo(result.stderr, err=True)
            continue

        click.echo(f"Successfully pushed to {dest_tag}")

    click.echo("\nDone!")


def _push_image(
    source_tag: str,
    registry: str,
    image_name: str | None = None,
    version: str | None = None,
    ghcr_org: str = GHCR_DEFAULT_ORG,
    gcp_regions: tuple[str, ...] = (),
    gcp_project: str = "hai-gcp-models",
    verbose: bool = False,
) -> None:
    """Push a local Docker image to the specified registry."""
    if registry == "ghcr":
        push_to_ghcr(source_tag, ghcr_org=ghcr_org, image_name=image_name, version=version, verbose=verbose)
    elif registry == "gcp":
        if not gcp_regions:
            raise click.ClickException("--region is required when pushing to GCP Artifact Registry")
        push_to_gcp_registries(
            source_tag, gcp_regions, gcp_project, image_name=image_name, version=version, verbose=verbose
        )
    else:
        raise click.ClickException(f"Unknown registry: {registry}. Use 'ghcr' or 'gcp'.")


def build_image(
    image_type: str,
    tag: str,
    push: bool,
    dockerfile: str | None,
    context: str | None,
    platform: str,
    registry: str,
    ghcr_org: str,
    gcp_regions: tuple[str, ...],
    gcp_project: str,
    verbose: bool = False,
) -> None:
    """Build a Docker image for Iris (worker or controller).

    Always tags the image with both the git SHA and "latest" so that
    deployments can pin to a specific version while local workflows
    continue to use "latest".
    """
    dockerfile_name = f"Dockerfile.{image_type}"

    iris_root = find_iris_root()
    dockerfile_path = Path(dockerfile) if dockerfile else iris_root / dockerfile_name
    context_path = Path(context) if context else iris_root

    if not dockerfile_path.exists():
        raise click.ClickException(f"Dockerfile not found: {dockerfile_path}")

    # Derive image base name from tag (e.g. "iris-worker:latest" -> "iris-worker")
    image_base = tag.split(":")[0]
    git_sha = get_git_sha()
    sha_tag = f"{image_base}:{git_sha}"
    latest_tag = f"{image_base}:latest"

    click.echo(f"Using Dockerfile: {dockerfile_path}")

    all_tags = dict.fromkeys([tag, sha_tag, latest_tag])
    cmd = ["docker", "buildx", "build", "--platform", platform]
    cmd.extend(["--build-arg", f"IRIS_GIT_HASH={git_sha}"])
    for t in all_tags:
        cmd.extend(["-t", t])
    cmd.extend(["-f", str(dockerfile_path)])
    cmd.extend(["--output", f"type=docker,compression=zstd,compression-level=1,name={tag}"])
    cmd.append(str(context_path))

    extra = [t for t in all_tags if t != tag]
    extra_msg = f" (also tagged as {', '.join(extra)})" if extra else ""
    click.echo(f"Building image: {tag}{extra_msg}")
    click.echo(f"Platform: {platform}")
    click.echo(f"Context: {context_path}")
    click.echo()

    if verbose:
        result = subprocess.run(cmd)
    else:
        result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        click.echo("Build failed", err=True)
        if not verbose:
            if result.stdout:
                click.echo(result.stdout, err=True)
            if result.stderr:
                click.echo(result.stderr, err=True)
        raise SystemExit(1)

    # buildx --output=docker only loads one name; tag the rest manually
    for t in extra:
        subprocess.run(["docker", "tag", tag, t], check=True)

    click.echo("Build successful!")
    click.echo(f"Image available locally as: {', '.join(all_tags)}")

    if push:
        # Push both the SHA-tagged and latest-tagged versions
        _push_image(
            sha_tag,
            registry=registry,
            ghcr_org=ghcr_org,
            gcp_regions=gcp_regions,
            gcp_project=gcp_project,
            verbose=verbose,
        )
        _push_image(
            latest_tag,
            registry=registry,
            ghcr_org=ghcr_org,
            gcp_regions=gcp_regions,
            gcp_project=gcp_project,
            verbose=verbose,
        )


_registry_options = [
    click.option("--registry", type=click.Choice(["ghcr", "gcp"]), default="ghcr", help="Registry to push to"),
    click.option("--ghcr-org", default=GHCR_DEFAULT_ORG, help="GHCR organization"),
    click.option("--region", multiple=True, help="GCP Artifact Registry regions (required for --registry gcp)"),
    click.option("--project", default="hai-gcp-models", help="GCP project ID for registry"),
]


def _add_registry_options(fn):
    for opt in reversed(_registry_options):
        fn = opt(fn)
    return fn


def _build_all(
    push: bool,
    platform: str,
    registry: str,
    ghcr_org: str,
    gcp_regions: tuple[str, ...],
    gcp_project: str,
    verbose: bool = False,
) -> None:
    """Build all Iris images (worker, controller, task).

    Tags are derived automatically: git SHA + latest.
    """
    marin_root = find_marin_root()
    iris_root = find_iris_root()

    for image_type in ("worker", "controller"):
        tag = _default_versioned_tag(f"iris-{image_type}")
        build_image(
            image_type, tag, push, None, None, platform, registry, ghcr_org, gcp_regions, gcp_project, verbose=verbose
        )
        click.echo()

    task_dockerfile = str(iris_root / "Dockerfile.task")
    build_image(
        "task",
        _default_versioned_tag("iris-task"),
        push,
        task_dockerfile,
        str(marin_root),
        platform,
        registry,
        ghcr_org,
        gcp_regions,
        gcp_project,
        verbose=verbose,
    )


@click.group(invoke_without_command=True)
@click.option("--push", is_flag=True, help="Push images to registry after building")
@click.option("--platform", default="linux/amd64", help="Target platform")
@_add_registry_options
@click.pass_context
def build(ctx, push: bool, platform: str, registry: str, ghcr_org: str, region: tuple[str, ...], project: str):
    """Image build commands.

    When invoked without a subcommand, builds all images (worker, controller, task).
    """
    if ctx.invoked_subcommand is None:
        _build_all(push, platform, registry, ghcr_org, gcp_regions=region, gcp_project=project, verbose=_is_verbose(ctx))


@build.command("all")
@click.option("--push", is_flag=True, help="Push images to registry after building")
@click.option("--platform", default="linux/amd64", help="Target platform")
@_add_registry_options
@click.pass_context
def build_all(ctx, push: bool, platform: str, registry: str, ghcr_org: str, region: tuple[str, ...], project: str):
    """Build all Iris images (worker, controller, task)."""
    verbose = _is_verbose(ctx)
    _build_all(push, platform, registry, ghcr_org, gcp_regions=region, gcp_project=project, verbose=verbose)


@build.command("worker-image")
@click.option("--tag", "-t", default=None, help="Image tag (default: latest-<git-short-sha>)")
@click.option("--push", is_flag=True, help="Push image to registry after building")
@click.option("--dockerfile", type=click.Path(exists=True), help="Custom Dockerfile path")
@click.option("--context", type=click.Path(exists=True), help="Build context directory")
@click.option("--platform", default="linux/amd64", help="Target platform")
@_add_registry_options
@click.pass_context
def build_worker_image(
    ctx,
    tag: str,
    push: bool,
    dockerfile: str | None,
    context: str | None,
    platform: str,
    registry: str,
    ghcr_org: str,
    region: tuple[str, ...],
    project: str,
):
    """Build Docker image for Iris worker."""
    verbose = _is_verbose(ctx)
    tag = tag or _default_versioned_tag("iris-worker")
    build_image("worker", tag, push, dockerfile, context, platform, registry, ghcr_org, region, project, verbose=verbose)


@build.command("controller-image")
@click.option("--tag", "-t", default=None, help="Image tag (default: latest-<git-short-sha>)")
@click.option("--push", is_flag=True, help="Push image to registry after building")
@click.option("--dockerfile", type=click.Path(exists=True), help="Custom Dockerfile path")
@click.option("--context", type=click.Path(exists=True), help="Build context directory")
@click.option("--platform", default="linux/amd64", help="Target platform")
@_add_registry_options
@click.pass_context
def build_controller_image(
    ctx,
    tag: str,
    push: bool,
    dockerfile: str | None,
    context: str | None,
    platform: str,
    registry: str,
    ghcr_org: str,
    region: tuple[str, ...],
    project: str,
):
    """Build Docker image for Iris controller."""
    verbose = _is_verbose(ctx)
    tag = tag or _default_versioned_tag("iris-controller")
    build_image(
        "controller", tag, push, dockerfile, context, platform, registry, ghcr_org, region, project, verbose=verbose
    )


@build.command("task-image")
@click.option("--tag", "-t", default=None, help="Image tag (default: latest-<git-short-sha>)")
@click.option("--push", is_flag=True, help="Push image to registry after building")
@click.option("--dockerfile", type=click.Path(exists=True), help="Custom Dockerfile path")
@click.option("--platform", default="linux/amd64", help="Target platform")
@_add_registry_options
@click.pass_context
def build_task_image(
    ctx,
    tag: str,
    push: bool,
    dockerfile: str | None,
    platform: str,
    registry: str,
    ghcr_org: str,
    region: tuple[str, ...],
    project: str,
):
    """Build base task image with system deps and pre-synced marin core deps.

    The build context is the marin repo root so that pyproject.toml and uv.lock
    are available for COPY. The Dockerfile lives at lib/iris/Dockerfile.task.
    """
    marin_root = find_marin_root()
    iris_root = find_iris_root()
    dockerfile_path = Path(dockerfile) if dockerfile else iris_root / "Dockerfile.task"

    if not dockerfile_path.exists():
        raise click.ClickException(f"Dockerfile not found: {dockerfile_path}")

    verbose = _is_verbose(ctx)
    resolved_tag = tag or _default_versioned_tag("iris-task")

    build_image(
        "task",
        resolved_tag,
        push,
        str(dockerfile_path),
        str(marin_root),
        platform,
        registry,
        ghcr_org,
        region,
        project,
        verbose=verbose,
    )


@build.command("push")
@click.argument("source_tag")
@click.option("--registry", type=click.Choice(["ghcr", "gcp"]), default="ghcr", help="Registry to push to")
@click.option("--ghcr-org", default=GHCR_DEFAULT_ORG, help="GHCR organization")
@click.option("--region", "-r", multiple=True, help="GCP Artifact Registry region (required for --registry gcp)")
@click.option("--project", default="hai-gcp-models", help="GCP project ID")
@click.option("--image-name", help="Image name in registry (default: derived from source tag)")
@click.option("--version", help="Version tag (default: derived from source tag)")
@click.pass_context
def build_push(
    ctx: click.Context,
    source_tag: str,
    registry: str,
    ghcr_org: str,
    region: tuple[str, ...],
    project: str,
    image_name: str | None,
    version: str | None,
):
    """Push a local Docker image to a container registry.

    By default pushes to ghcr.io. Use --registry gcp with --region for GCP Artifact Registry.

    Examples:

        iris build push iris-worker:latest --image-name iris-worker

        iris build push iris-task:v1.0 --registry gcp --region us-central1
    """
    verbose = _is_verbose(ctx)
    config = ctx.obj.get("config") if ctx.obj else None
    if registry == "gcp" and config and not region:
        regions = collect_all_regions(config)
        if not regions:
            raise click.ClickException("No GCP regions found in config")

        parsed = parse_artifact_registry_tag(source_tag)
        resolved_project = project
        if parsed:
            _, parsed_project, _, _ = parsed
            resolved_project = parsed_project

        ordered_regions = tuple(sorted(regions))
        click.echo(f"Pushing to {len(ordered_regions)} region(s) from config: {', '.join(ordered_regions)}")
        push_to_gcp_registries(
            source_tag,
            ordered_regions,
            resolved_project,
            image_name=image_name,
            version=version,
        )
        return

    _push_image(
        source_tag,
        registry=registry,
        image_name=image_name,
        version=version,
        ghcr_org=ghcr_org,
        gcp_regions=region,
        gcp_project=project,
        verbose=verbose,
    )

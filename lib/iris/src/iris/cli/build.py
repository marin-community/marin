# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Image build commands."""

import subprocess
from pathlib import Path

import click

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
    """Find the iris package root directory containing the unified Dockerfile.

    Searches in order:
    1. Relative to this file (cli/build.py -> iris root is 4 levels up from src/iris/cli/build.py)
    2. Current working directory
    3. Walking up from cwd until Dockerfile is found
    """
    build_path = Path(__file__).resolve()
    # build.py is at src/iris/cli/build.py, so iris root is 4 levels up
    iris_root = build_path.parent.parent.parent.parent
    if (iris_root / "Dockerfile").exists():
        return iris_root

    cwd = Path.cwd()
    if (cwd / "Dockerfile").exists():
        return cwd

    for parent in cwd.parents:
        if (parent / "Dockerfile").exists():
            return parent

    raise click.ClickException("Cannot find Dockerfile. Run from the iris directory.")


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


def _ensure_protos() -> None:
    """Regenerate protobuf Python bindings from .proto sources.

    Called before ``docker build`` so that COPY always picks up fresh bindings.
    The hatch build hook handles staleness for normal ``uv sync`` / ``uv run``,
    but Docker has no ``npx`` so it cannot regenerate inside the image.
    """
    iris_root = find_iris_root()
    generate_script = iris_root / "scripts" / "generate_protos.py"
    if not generate_script.exists():
        raise click.ClickException(f"Proto generation script not found: {generate_script}")
    click.echo("Regenerating protobuf bindings...")
    result = subprocess.run(
        ["python", str(generate_script)],
        cwd=iris_root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        click.echo(result.stderr, err=True)
        raise click.ClickException("Protobuf generation failed")
    click.echo("Protobuf bindings regenerated.")


def _ensure_dashboard_dist() -> None:
    """Build Vue dashboard assets.

    Called automatically before building controller/worker images so that
    ``COPY dashboard/dist`` in the Dockerfile always has fresh assets.
    """
    iris_root = find_iris_root()
    dashboard_dir = iris_root / "dashboard"
    if not (dashboard_dir / "package.json").exists():
        raise click.ClickException(
            f"Dashboard source not found at {dashboard_dir}. " "Cannot build dashboard assets for Docker image."
        )
    click.echo("Building frontend assets...")
    subprocess.run(["npm", "ci"], cwd=dashboard_dir, check=True)
    result = subprocess.run(["npm", "run", "build"], cwd=dashboard_dir, capture_output=True, text=True)
    if result.returncode != 0:
        click.echo(result.stderr, err=True)
        raise click.ClickException("Dashboard build failed")
    click.echo("Dashboard built successfully.")


def build_image(
    image_type: str,
    tag: str,
    push: bool,
    context: str | None,
    platform: str,
    ghcr_org: str = GHCR_DEFAULT_ORG,
    verbose: bool = False,
) -> None:
    """Build a Docker image for Iris using the unified multi-stage Dockerfile.

    Always tags the image with both the git SHA and "latest" so that
    deployments can pin to a specific version while local workflows
    continue to use "latest".

    When ``push=True``, images are pushed directly via ``docker buildx build --push``
    and the registry cache is updated in the same operation. The images are NOT
    loaded into the local Docker daemon (buildx cannot do both simultaneously).
    """
    # Controller and worker images COPY dashboard/dist — ensure it exists.
    if image_type in ("controller", "worker"):
        _ensure_dashboard_dist()

    iris_root = find_iris_root()
    dockerfile_path = iris_root / "Dockerfile"
    # Controller/worker Dockerfiles expect the marin repo root as build context
    # so that lib/rigging (a workspace-local dep) can be COPY'd in.
    if context:
        context_path = Path(context)
    elif image_type in ("controller", "worker"):
        context_path = find_marin_root()
    else:
        context_path = iris_root

    if not dockerfile_path.exists():
        raise click.ClickException(f"Dockerfile not found: {dockerfile_path}")

    # Derive image base name from tag (e.g. "iris-worker:latest" -> "iris-worker")
    image_base = tag.split(":")[0]
    git_sha = get_git_sha()
    sha_tag = f"{image_base}:{git_sha}"
    latest_tag = f"{image_base}:latest"

    click.echo(f"Using Dockerfile: {dockerfile_path}")

    if push:
        # Fully-qualified GHCR tags for the registry push
        all_tags = dict.fromkeys(
            [
                f"ghcr.io/{ghcr_org}/{tag}",
                f"ghcr.io/{ghcr_org}/{sha_tag}",
                f"ghcr.io/{ghcr_org}/{latest_tag}",
            ]
        )
    else:
        all_tags = dict.fromkeys([tag, sha_tag, latest_tag])

    cmd = ["docker", "buildx", "build", "--platform", platform]
    cmd.extend(["--target", image_type])
    cmd.extend(["--build-arg", f"IRIS_GIT_HASH={git_sha}"])
    for t in all_tags:
        cmd.extend(["-t", t])
    cmd.extend(["-f", str(dockerfile_path)])

    cache_ref = f"ghcr.io/{ghcr_org}/iris-cache:{image_type}"
    cmd.extend(["--cache-from", f"type=registry,ref={cache_ref}"])

    if push:
        cmd.extend(["--cache-to", f"type=registry,ref={cache_ref},mode=max"])
        cmd.extend(["--output", "type=image,compression=zstd,compression-level=3,push=true"])
        cmd.append("--provenance=false")
    else:
        cmd.extend(["--output", f"type=docker,compression=zstd,compression-level=1,name={tag}"])

    cmd.append(str(context_path))

    extra = [t for t in all_tags if t != tag]
    extra_msg = f" (also tagged as {', '.join(extra)})" if extra else ""
    click.echo(f"Building image: {tag}{extra_msg}")
    click.echo(f"Platform: {platform}")
    click.echo(f"Context: {context_path}")
    if push:
        click.echo("Push: enabled (images will be pushed to registry)")
    click.echo()

    if verbose:
        cmd.extend(["--progress", "plain"])
        result = subprocess.run(cmd)
    elif push:
        cmd.extend(["--progress", "plain"])
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

    if not push:
        # buildx --output=docker only loads one name; tag the rest manually
        for t in extra:
            subprocess.run(["docker", "tag", tag, t], check=True)

    click.echo("Build successful!")
    if push:
        click.echo(f"Images pushed to: {', '.join(all_tags)}")
    else:
        click.echo(f"Image available locally as: {', '.join(all_tags)}")


def _build_all(
    push: bool,
    platform: str,
    ghcr_org: str,
) -> None:
    """Build all Iris images (worker, controller, task).

    Tags are derived automatically: git SHA + latest.
    """
    marin_root = find_marin_root()

    _ensure_protos()

    for image_type in ("worker", "controller"):
        tag = _default_versioned_tag(f"iris-{image_type}")
        build_image(image_type, tag, push, None, platform, ghcr_org)
        click.echo()

    # Task target uses the same Dockerfile but needs marin root as context
    build_image(
        "task",
        _default_versioned_tag("iris-task"),
        push,
        str(marin_root),
        platform,
        ghcr_org,
    )


@click.group(invoke_without_command=True)
@click.option("--push", is_flag=True, help="Push images to registry after building")
@click.option("--platform", default="linux/amd64", help="Target platform")
@click.option("--ghcr-org", default=GHCR_DEFAULT_ORG, help="GHCR organization")
@click.pass_context
def build(ctx, push: bool, platform: str, ghcr_org: str):
    """Image build commands.

    When invoked without a subcommand, builds all images (worker, controller, task).
    """
    if ctx.invoked_subcommand is None:
        _build_all(push, platform, ghcr_org)


@build.command("all")
@click.option("--push", is_flag=True, help="Push images to registry after building")
@click.option("--platform", default="linux/amd64", help="Target platform")
@click.option("--ghcr-org", default=GHCR_DEFAULT_ORG, help="GHCR organization")
@click.pass_context
def build_all(
    ctx: click.Context,
    push: bool,
    platform: str,
    ghcr_org: str,
):
    """Build all Iris images (worker, controller, task)."""
    _build_all(push, platform, ghcr_org)


@build.command("worker-image")
@click.option("--tag", "-t", default=None, help="Image tag (default: latest-<git-short-sha>)")
@click.option("--push", is_flag=True, help="Push image to registry after building")
@click.option("--context", type=click.Path(exists=True), help="Build context directory")
@click.option("--platform", default="linux/amd64", help="Target platform")
@click.option("--ghcr-org", default=GHCR_DEFAULT_ORG, help="GHCR organization")
@click.pass_context
def build_worker_image(
    ctx,
    tag: str,
    push: bool,
    context: str | None,
    platform: str,
    ghcr_org: str,
):
    """Build Docker image for Iris worker."""
    verbose = _is_verbose(ctx)
    _ensure_protos()
    tag = tag or _default_versioned_tag("iris-worker")
    build_image("worker", tag, push, context, platform, ghcr_org, verbose=verbose)


@build.command("controller-image")
@click.option("--tag", "-t", default=None, help="Image tag (default: latest-<git-short-sha>)")
@click.option("--push", is_flag=True, help="Push image to registry after building")
@click.option("--context", type=click.Path(exists=True), help="Build context directory")
@click.option("--platform", default="linux/amd64", help="Target platform")
@click.option("--ghcr-org", default=GHCR_DEFAULT_ORG, help="GHCR organization")
@click.pass_context
def build_controller_image(
    ctx,
    tag: str,
    push: bool,
    context: str | None,
    platform: str,
    ghcr_org: str,
):
    """Build Docker image for Iris controller."""
    verbose = _is_verbose(ctx)
    _ensure_protos()
    tag = tag or _default_versioned_tag("iris-controller")
    build_image("controller", tag, push, context, platform, ghcr_org, verbose=verbose)


@build.command("task-image")
@click.option("--tag", "-t", default=None, help="Image tag (default: latest-<git-short-sha>)")
@click.option("--push", is_flag=True, help="Push image to registry after building")
@click.option("--platform", default="linux/amd64", help="Target platform")
@click.option("--ghcr-org", default=GHCR_DEFAULT_ORG, help="GHCR organization")
@click.pass_context
def build_task_image(
    ctx,
    tag: str,
    push: bool,
    platform: str,
    ghcr_org: str,
):
    """Build base task image with system deps and pre-synced marin core deps.

    The build context is the marin repo root so that pyproject.toml and uv.lock
    are available for COPY. Uses the ``task`` target in ``lib/iris/Dockerfile``.
    """
    marin_root = find_marin_root()

    verbose = _is_verbose(ctx)
    _ensure_protos()
    resolved_tag = tag or _default_versioned_tag("iris-task")

    build_image(
        "task",
        resolved_tag,
        push,
        str(marin_root),
        platform,
        ghcr_org,
        verbose=verbose,
    )


@build.command("dashboard")
def build_dashboard():
    """Build Vue dashboard assets via Rsbuild."""
    dashboard_dir = find_iris_root() / "dashboard"
    if not (dashboard_dir / "package.json").exists():
        raise click.ClickException(f"Dashboard source not found at {dashboard_dir}")
    if not (dashboard_dir / "node_modules").exists():
        click.echo("Installing dashboard dependencies...")
        subprocess.run(["npm", "ci"], cwd=dashboard_dir, check=True)
    click.echo("Building dashboard...")
    result = subprocess.run(["npm", "run", "build"], cwd=dashboard_dir, capture_output=True, text=True)
    if result.returncode != 0:
        click.echo(result.stderr, err=True)
        raise click.ClickException("Dashboard build failed")
    click.echo("Dashboard built successfully.")


@build.command("push")
@click.argument("source_tag")
@click.option("--ghcr-org", default=GHCR_DEFAULT_ORG, help="GHCR organization")
@click.option("--image-name", help="Image name in registry (default: derived from source tag)")
@click.option("--version", help="Version tag (default: derived from source tag)")
@click.pass_context
def build_push(
    ctx: click.Context,
    source_tag: str,
    ghcr_org: str,
    image_name: str | None,
    version: str | None,
):
    """Push a local Docker image to GHCR.

    Examples:

        iris build push iris-worker:latest --image-name iris-worker

        iris build push iris-task:v1.0 --ghcr-org my-org
    """
    verbose = _is_verbose(ctx)
    push_to_ghcr(source_tag, ghcr_org=ghcr_org, image_name=image_name, version=version, verbose=verbose)

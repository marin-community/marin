import json
import os
import pty
import shlex
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path


GCP_CLEANUP_POLICY = [
    {
        "name": "delete-stale",
        "action": {"type": "Delete"},
        "condition": {
            "olderThan": "86400s",
            "tagState": "ANY",
        },
    },
    {
        "name": "keep-latest",
        "action": {"type": "Keep"},
        "mostRecentVersions": {
            "keepCount": 5,
        },
    },
]


def _rm(path):
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.is_file():
        os.remove(path)
    elif path.exists():
        raise RuntimeError(f"Remove failed. Path ({path}) is neither a directory nor a file.")


def _cp(src, dst):
    # delete dst if exists
    _rm(dst)

    if src.is_dir():
        shutil.copytree(src, dst)
    elif src.is_file():
        shutil.copy(src, dst)
    else:
        raise RuntimeError(f"Copy failed. Source path ({src}) is neither a directory nor a file. Check if it exists.")


def _run(argv):
    if sys.stdout.isatty():
        output = []

        def read(fd):
            data = os.read(fd, 1024)
            output.append(data)
            return data

        exit_code = pty.spawn(argv, master_read=read)
        if exit_code != 0:
            e = subprocess.CalledProcessError(exit_code, argv)
            e.output = b"".join(output)
            raise e

        return b"".join(output)
    else:
        try:
            return subprocess.check_output(argv, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            # print the output if the command failed, reraising the exception
            print(e.output.decode())
            raise e


def configure_gcp_docker(project_id, region, repository):
    """Setup Artifact registry repository and configure permissions to enable TPU access."""
    # check if the repository already exists
    try:
        _run(
            ["gcloud", "artifacts", "repositories", "describe", f"--location={region}", repository],
        )
        print(f"Found existing artifact registry repository `{repository}`, skipping setup.")
        return
    except subprocess.CalledProcessError as e:
        if b"NOT_FOUND" not in e.output:
            raise

    # Activate artifact registry and setup the repository.
    _run(["gcloud", "services", "enable", "artifactregistry.googleapis.com"])

    try:
        _run(
            [
                "gcloud",
                "artifacts",
                "repositories",
                "create",
                repository,
                f"--location={region}",
                "--repository-format=docker",
            ],
        )
    except subprocess.CalledProcessError as e:
        # Ignore error if repository already exists.
        if b"ALREADY_EXISTS" not in e.output:
            print("Error creating repository: ", e.output)
            raise

    with open("/tmp/cleanup-policy.json", "w") as f:
        json.dump(GCP_CLEANUP_POLICY, f, indent=2)

    _run(
        [
            "gcloud",
            "artifacts",
            "repositories",
            "set-cleanup-policies",
            f"--location={region}",
            "--policy=/tmp/cleanup-policy.json",
            repository,
        ]
    )

    # Grant public read access ('allUsers') for TPU VMs
    _run(
        [
            "gcloud",
            "artifacts",
            "repositories",
            "add-iam-policy-binding",
            "--member=allUsers",
            "--role=roles/artifactregistry.reader",
            f"--location={region}",
            repository,
        ]
    )

    _run(
        [
            "gcloud",
            "--project",
            project_id,
            "artifacts",
            "repositories",
            "add-iam-policy-binding",
            repository,
            "--location",
            region,
            "--member",
            "allUsers",
            "--role",
            "roles/artifactregistry.reader",
        ]
    )

    _run(["gcloud", "auth", "configure-docker", "--quiet", f"{region}-docker.pkg.dev"])


@contextmanager
def copy_extra_ctx(extra_ctx):
    """Context manager to handle copying and cleanup of extra context directory."""
    if extra_ctx is not None:
        mount_dst = Path(".mnt")
        _cp(extra_ctx, mount_dst)
        try:
            yield extra_ctx
        finally:
            _rm(mount_dst)
    else:
        yield None


def build_docker(docker_file, image_name, tag, build_args=None) -> str:
    """Builds a Docker image, enables artifact access, and pushes to Artifact Registry."""
    args = [
        "docker",
        "buildx",
        "build",
        "--platform=linux/amd64",
        # "--progress=plain",
        "-t",
        f"{image_name}:{tag}",
    ]

    if build_args:
        for key, value in build_args.items():
            args.extend(["--build-arg", f"{key}={value}"])

    args.extend(
        [
            "-f",
            docker_file,
            ".",
        ]
    )
    _run(args)

    return f"{image_name}:{tag}"


def push_to_github(local_id, github_user, github_token=None):
    """Pushes a local Docker image to Docker Hub."""

    # Authenticate the docker service with Github if a token exists
    if github_token:
        login_process = subprocess.Popen(
            ["docker", "login", "ghcr.io", "-u", github_user, "--password-stdin"], stdin=subprocess.PIPE
        )
        print(login_process.communicate(input=github_token.encode(), timeout=10))

    remote_name = f"ghcr.io/{github_user}/{local_id}"

    _run(["docker", "tag", local_id, remote_name])
    _run(["docker", "push", remote_name])
    return remote_name


def push_to_gcp(local_id, project_id, region, repository) -> str:
    """Pushes a local Docker image to Artifact Registry."""
    configure_gcp_docker(project_id, region, repository)

    artifact_repo = f"{region}-docker.pkg.dev/{project_id}/{repository}"

    full_image_name = f"{artifact_repo}/{local_id}"
    _run(["docker", "tag", local_id, full_image_name])
    _run(["docker", "push", full_image_name])

    return f"{artifact_repo}/{local_id}"


def split_image_and_tag(docker_base_image):
    if ":" in docker_base_image:
        base_image, base_tag = docker_base_image.rsplit(":", 1)
    else:
        base_image = docker_base_image
        base_tag = "latest"
    return base_image, base_tag


def make_docker_run_command(image_id, command, *, foreground, env, name="levanter"):
    docker_command = [
        "docker",
        "run",
        "-t" if foreground else "-d",
        f"--name={shlex.quote(name)}",
        "--privileged",
        "--shm-size=32gb",
        "--net=host",
        "--init",
        "--mount",
        "type=volume,source=levanter,target=/home/levanter",
        "-v",
        "/tmp:/tmp",
    ]

    # optionally add multislice env vars (if set by ray runtime env vars)
    for v in ["MEGASCALE_COORDINATOR_ADDRESS", "MEGASCALE_NUM_SLICES", "MEGASCALE_PORT", "MEGASCALE_SLICE_ID"]:
        v = shlex.quote(str(v))
        docker_command.extend(["-e", v])

    for k, v in env.items():
        v = shlex.quote(str(v))
        k = shlex.quote(str(k))
        docker_command.extend(["-e", f"{k}={v}"])

    docker_command.extend([image_id, *command])
    return docker_command

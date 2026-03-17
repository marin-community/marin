#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CoreWeave multi-host GPU canary runner.

Default command is `run`: auto-detects whether to cold-boot or warm-reboot,
submits the multi-host canary ferry, and waits for completion.

Usage:
    uv run scripts/canary/coreweave_multihost.py                # run (default)
    uv run scripts/canary/coreweave_multihost.py run             # explicit run
    uv run scripts/canary/coreweave_multihost.py boot            # just boot cluster
    uv run scripts/canary/coreweave_multihost.py teardown        # full teardown
    uv run scripts/canary/coreweave_multihost.py validate        # check prereqs
    uv run scripts/canary/coreweave_multihost.py diagnostics     # dump controller/worker state
    uv run scripts/canary/coreweave_multihost.py loop            # run in a loop until success
"""

import datetime
import logging
import os
import subprocess
import time
from pathlib import Path

import click
import fsspec
import s3fs

from iris.cli.cluster import _build_cluster_images, _pin_latest_images
from iris.cli.job import run_iris_job
from iris.cli.main import _configure_client_s3, create_client_token_provider
from iris.cluster.config import IrisConfig

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = "lib/iris/examples/coreweave-canary-multihost.yaml"
DEFAULT_NAMESPACE = "iris-canary-mh"
DEFAULT_LABEL_PREFIX = "iris-canary-mh"

CANARY_ENV_DEFAULTS = {
    "CANARY_ACCELERATOR": "gpu",
    "CANARY_MULTI_HOST": "true",
    "CANARY_BATCH_SIZE": "32",
    "CANARY_TARGET_TOKENS": "6553600",
    "WANDB_ENTITY": "marin-community",
    "WANDB_PROJECT": "marin",
    "MARIN_PREFIX": "s3://marin-na/marin/",
}

REQUIRED_ENV_VARS = ["R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "WANDB_API_KEY"]


def _kubeconfig_path() -> str:
    return os.environ.get("KUBECONFIG", str(Path.home() / ".kube/coreweave-iris"))


def _kubectl(namespace: str, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run kubectl against the CoreWeave cluster."""
    cmd = ["kubectl", "--kubeconfig", _kubeconfig_path(), "-n", namespace, *args]
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def _controller_deployment_exists(namespace: str) -> bool:
    result = _kubectl(
        namespace,
        "get",
        "deployment",
        "-l",
        "app=iris-controller",
        "--no-headers",
        check=False,
    )
    return bool(result.stdout.strip())


def _generate_run_id() -> str:
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"mh-canary-{ts}"


def validate_prereqs(config_path: str) -> None:
    errors: list[str] = []

    if not Path(config_path).exists():
        errors.append(f"Config file not found: {config_path}")

    kubeconfig = os.environ.get("KUBECONFIG", str(Path.home() / ".kube/coreweave-iris"))
    if not Path(kubeconfig).exists():
        errors.append(f"Kubeconfig not found: {kubeconfig}")

    for var in REQUIRED_ENV_VARS:
        if not os.environ.get(var):
            errors.append(f"Environment variable not set: {var}")

    if errors:
        for e in errors:
            click.echo(f"  FAIL: {e}", err=True)
        raise SystemExit(1)

    result = subprocess.run(
        ["kubectl", "--kubeconfig", kubeconfig, "cluster-info"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        click.echo(f"  FAIL: cannot reach CoreWeave cluster: {result.stderr.strip()}", err=True)
        raise SystemExit(1)

    IrisConfig.load(config_path)
    click.echo("All prereqs OK")


def _make_s3fs(endpoint: str) -> s3fs.S3FileSystem:
    """Create an S3FileSystem configured for the given R2/S3 endpoint."""
    return s3fs.S3FileSystem(
        endpoint_url=endpoint,
        key=os.environ.get("R2_ACCESS_KEY_ID"),
        secret=os.environ.get("R2_SECRET_ACCESS_KEY"),
        client_kwargs={"region_name": "auto"},
        skip_instance_cache=True,
    )


def clear_controller_state(config_path: str) -> None:
    """Delete the S3 controller checkpoint so the next boot starts fresh."""
    iris_config = IrisConfig.load(config_path)
    _configure_client_s3(iris_config.proto)
    state_dir = iris_config.proto.storage.remote_state_dir
    if not state_dir:
        click.echo("No remote_state_dir configured, skipping state clear")
        return
    endpoint = iris_config.proto.platform.coreweave.object_storage_endpoint
    checkpoint_dir = f"{state_dir}/controller-state"
    _, path = fsspec.core.url_to_fs(checkpoint_dir)
    fs = _make_s3fs(endpoint)
    if fs.exists(path):
        fs.rm(path, recursive=True)
        click.echo(f"Cleared controller state: {checkpoint_dir}")
    else:
        click.echo(f"No controller state to clear: {checkpoint_dir}")


def cold_boot(config_path: str, verbose: bool = False) -> str:
    click.echo("=== Cold boot: creating all resources ===")
    iris_config = IrisConfig.load(config_path)
    config = iris_config.proto

    _pin_latest_images(config)
    _build_cluster_images(config, verbose=verbose)

    platform = iris_config.platform()
    address = platform.start_controller(config)
    click.echo(f"Controller started at {address}")
    return address


def warm_reboot(config_path: str, namespace: str, label_prefix: str, verbose: bool = False) -> str:
    click.echo("=== Warm reboot: wiping workers, restarting controller ===")

    managed_label = f"iris-{label_prefix}-managed=true"
    _kubectl(namespace, "delete", "pods", "-l", managed_label, "--ignore-not-found", check=False)
    _kubectl(namespace, "delete", "configmaps", "-l", managed_label, "--ignore-not-found", check=False)

    clear_controller_state(config_path)

    iris_config = IrisConfig.load(config_path)
    config = iris_config.proto

    _pin_latest_images(config)
    _build_cluster_images(config, verbose=verbose)

    platform = iris_config.platform()
    address = platform.start_controller(config)
    click.echo(f"Controller restarted at {address}")
    return address


def boot_cluster(config_path: str, namespace: str, label_prefix: str, verbose: bool = False) -> str:
    if _controller_deployment_exists(namespace):
        return warm_reboot(config_path, namespace, label_prefix, verbose=verbose)
    return cold_boot(config_path, verbose=verbose)


def teardown_cluster(config_path: str, namespace: str, label_prefix: str) -> None:
    click.echo("=== Teardown ===")
    iris_config = IrisConfig.load(config_path)
    platform = iris_config.platform()
    try:
        names = platform.stop_all(iris_config.proto)
        click.echo(f"Stopped {len(names)} resource(s)")
    except Exception as e:
        click.echo(f"cluster stop failed (continuing): {e}", err=True)
    finally:
        platform.shutdown()

    _kubectl(namespace, "delete", "nodepool", "-l", f"{label_prefix}-managed=true", "--ignore-not-found", check=False)
    click.echo("Teardown complete")


def submit_canary(config_path: str, run_id: str) -> int:
    click.echo(f"=== Submitting multi-host canary: {run_id} ===")

    iris_config = IrisConfig.load(config_path)
    _configure_client_s3(iris_config.proto)

    env_vars: dict[str, str] = {}
    for key, default in CANARY_ENV_DEFAULTS.items():
        env_vars[key] = os.environ.get(key, default)
    env_vars["RUN_ID"] = run_id

    for key in ["WANDB_API_KEY", "HF_TOKEN", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"]:
        val = os.environ.get(key, "")
        if val:
            env_vars[key] = val

    platform = iris_config.platform()
    controller_address = iris_config.controller_address()
    if not controller_address:
        controller_address = platform.discover_controller(iris_config.proto.controller)

    with platform.tunnel(address=controller_address) as controller_url:
        token_provider = None
        if iris_config.proto.HasField("auth"):
            token_provider = create_client_token_provider(iris_config.proto.auth)

        return run_iris_job(
            command=["python", "-m", "experiments.ferries.canary_ferry"],
            env_vars=env_vars,
            controller_url=controller_url,
            cpu=1,
            memory="16GB",
            disk="16GB",
            wait=True,
            job_name=f"multihost-canary-{run_id}",
            extras=["cpu"],
            token_provider=token_provider,
        )


def dump_diagnostics(namespace: str, label_prefix: str) -> None:
    click.echo("=== Controller logs ===")
    result = _kubectl(namespace, "logs", "-l", "app=iris-controller", "--tail=200", check=False)
    click.echo(result.stdout or "(no logs)")

    click.echo("\n=== Worker pods ===")
    result = _kubectl(namespace, "get", "pods", "-l", f"{label_prefix}-role=worker", "-o", "wide", check=False)
    click.echo(result.stdout or "(no worker pods)")

    click.echo("\n=== Error events ===")
    result = _kubectl(
        namespace, "get", "events", "--sort-by=.lastTimestamp", "--field-selector=type!=Normal", check=False
    )
    lines = result.stdout.strip().split("\n")
    click.echo("\n".join(lines[-30:]) if lines else "(no error events)")


@click.group(invoke_without_command=True)
@click.option("--config", "config_path", default=DEFAULT_CONFIG, show_default=True, help="Iris cluster config file")
@click.option("--namespace", default=DEFAULT_NAMESPACE, show_default=True)
@click.option("--label-prefix", default=DEFAULT_LABEL_PREFIX, show_default=True)
@click.option("-v", "--verbose", is_flag=True, help="Verbose logging")
@click.pass_context
def cli(ctx, config_path: str, namespace: str, label_prefix: str, verbose: bool):
    """CoreWeave multi-host GPU canary runner.

    Default (no subcommand): boots cluster, submits canary, waits for result.
    """
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config_path
    ctx.obj["namespace"] = namespace
    ctx.obj["label_prefix"] = label_prefix
    ctx.obj["verbose"] = verbose

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if verbose:
        logging.getLogger("iris").setLevel(logging.DEBUG)
        logging.getLogger(__name__).setLevel(logging.DEBUG)

    if ctx.invoked_subcommand is None:
        ctx.invoke(run)


@cli.command()
@click.pass_context
def validate(ctx):
    """Check prerequisites: config, credentials, cluster connectivity."""
    validate_prereqs(ctx.obj["config_path"])


@cli.command()
@click.pass_context
def boot(ctx):
    """Start cluster (cold boot or warm reboot as needed)."""
    validate_prereqs(ctx.obj["config_path"])
    boot_cluster(
        ctx.obj["config_path"],
        ctx.obj["namespace"],
        ctx.obj["label_prefix"],
        verbose=ctx.obj["verbose"],
    )


@cli.command()
@click.pass_context
def teardown(ctx):
    """Full teardown: stop cluster and delete NodePools."""
    teardown_cluster(ctx.obj["config_path"], ctx.obj["namespace"], ctx.obj["label_prefix"])


@cli.command()
@click.pass_context
def diagnostics(ctx):
    """Dump controller logs, worker pods, and error events."""
    dump_diagnostics(ctx.obj["namespace"], ctx.obj["label_prefix"])


@cli.command()
@click.pass_context
def run(ctx):
    """Boot cluster, submit multi-host canary, wait for result (default command)."""
    config_path = ctx.obj["config_path"]
    namespace = ctx.obj["namespace"]
    label_prefix = ctx.obj["label_prefix"]

    validate_prereqs(config_path)
    boot_cluster(config_path, namespace, label_prefix, verbose=ctx.obj["verbose"])

    run_id = _generate_run_id()
    exit_code = submit_canary(config_path, run_id)
    if exit_code == 0:
        click.echo("=== Multi-host canary PASSED ===")
    else:
        click.echo("=== Multi-host canary FAILED ===", err=True)
        dump_diagnostics(namespace, label_prefix)
    raise SystemExit(exit_code)


@cli.command()
@click.option("--max-attempts", default=0, help="Max attempts (0 = unlimited)")
@click.pass_context
def loop(ctx, max_attempts: int):
    """Run canary in a loop until success (warm rebooting between attempts)."""
    config_path = ctx.obj["config_path"]
    namespace = ctx.obj["namespace"]
    label_prefix = ctx.obj["label_prefix"]
    verbose = ctx.obj["verbose"]

    validate_prereqs(config_path)
    attempt = 0
    while max_attempts == 0 or attempt < max_attempts:
        attempt += 1
        click.echo(f"\n{'='*20} Attempt {attempt} {'='*20}")

        boot_cluster(config_path, namespace, label_prefix, verbose=verbose)
        run_id = _generate_run_id()
        exit_code = submit_canary(config_path, run_id)

        if exit_code == 0:
            click.echo(f"Multi-host canary PASSED on attempt {attempt}")
            raise SystemExit(0)

        click.echo(f"Failed on attempt {attempt}, dumping diagnostics...")
        dump_diagnostics(namespace, label_prefix)
        click.echo("Retrying in 10s...")
        time.sleep(10)

    click.echo(f"Exhausted {max_attempts} attempts", err=True)
    raise SystemExit(1)


if __name__ == "__main__":
    cli()

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pulumi-backed Iris controller rollouts and paired checkpoint rollback."""

import hashlib
import json
import subprocess
import tempfile
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import click
from iris.cli.build import CARGO_PROFILES, DEFAULT_CARGO_PROFILE
from iris.cli.cluster import (
    build_and_pin_deploy_images,
    use_prebuilt_kubernetes_images,
)
from iris.cli.connect import IRIS_CLUSTER_CONFIG_DIRS, connect_controller, take_controller_checkpoint
from iris.cluster.composer import provider_bundle
from iris.cluster.config import IrisClusterConfig, load_config
from iris.cluster.controller.rollout import (
    ROLLOUT_RECORD_FILENAME,
    RolloutPhase,
    RolloutRecord,
    read_rollout_record,
    write_rollout_record,
)
from iris.cluster.platforms.factory import ProviderBundle
from iris.cluster.platforms.gcp.handles import GcpStandaloneWorkerHandle
from iris.cluster.platforms.gcp.ssh import ssh_impersonate_service_account
from iris.cluster.platforms.k8s.controller import K8sControllerProvider, configure_client_s3
from iris.cluster.platforms.vm_lifecycle import controller_restart_plan
from rigging.config_discovery import resolve_cluster_config
from rigging.timing import Timestamp

from marin_deploy.gce import GceVmTarget, StartupScriptPersistence, activate_startup_script
from marin_deploy.pulumi import run_pulumi_up

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
IRIS_PULUMI_DIRECTORY = REPOSITORY_ROOT / "infra" / "iris"
GCE_ACTIVATION_TIMEOUT = 900
ACTIVATION_MARKER_PREFIX = "marin-iris-activation-"
MILLISECONDS_PER_SECOND = 1000


class IrisDeployError(RuntimeError):
    """An Iris rollout failed or could not restore its paired rollback state."""


class IrisActivationError(RuntimeError):
    """A Pulumi update failed, with whether controller mutation had started."""

    def __init__(self, message: str, *, started: bool) -> None:
        super().__init__(message)
        self.started = started


@dataclass(frozen=True)
class IrisActivationSpec:
    """Non-secret inputs consumed by one Pulumi controller activation."""

    cluster: str
    controller_image: str
    worker_image: str
    task_image: str
    activation_id: str

    @classmethod
    def from_config(cls, cluster: str, config: IrisClusterConfig) -> "IrisActivationSpec":
        return cls(
            cluster=cluster,
            controller_image=config.controller.image,
            worker_image=config.defaults.worker.docker_image,
            task_image=config.defaults.worker.default_task_image,
            activation_id=uuid.uuid4().hex,
        )

    def apply(self, config: IrisClusterConfig) -> IrisClusterConfig:
        activated = config.model_copy(deep=True)
        activated.controller.image = self.controller_image
        activated.defaults.worker.docker_image = self.worker_image
        activated.defaults.worker.default_task_image = self.task_image
        return activated

    def with_controller_image(self, image: str) -> "IrisActivationSpec":
        return replace(self, controller_image=image, activation_id=uuid.uuid4().hex)

    def to_json(self) -> str:
        """Serialize the activation for Pulumi resource state."""
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, value: str) -> "IrisActivationSpec":
        """Load an activation from Pulumi resource state."""
        return cls(**json.loads(value))

    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()


def activation_marker_path(spec: IrisActivationSpec) -> Path:
    """Return the marker written immediately before controller mutation."""
    activation_hash = hashlib.sha256(spec.activation_id.encode()).hexdigest()
    return Path(tempfile.gettempdir()) / f"{ACTIVATION_MARKER_PREFIX}{activation_hash}.started"


RolloutWriter = Callable[[str, RolloutRecord], None]
ActivationRunner = Callable[[IrisActivationSpec], None]


def _record_rollout(
    remote_state_dir: str,
    record: RolloutRecord,
    *,
    required: bool,
    writer: RolloutWriter,
) -> None:
    if not remote_state_dir:
        if required:
            raise IrisDeployError(f"Recording {record.phase} requires config.storage.remote_state_dir")
        return
    try:
        writer(remote_state_dir, record)
    except OSError as error:
        raise IrisDeployError(f"Could not record {record.phase} rollout state: {error}") from error
    click.echo(f"Rollout record: {record.phase} -> {remote_state_dir}/{ROLLOUT_RECORD_FILENAME}")


def _updated_at_ms() -> int:
    return Timestamp.now().epoch_ms()


def _request_and_apply_rollback(
    *,
    remote_state_dir: str,
    activation: IrisActivationSpec,
    image: str,
    checkpoint: str | None,
    apply: ActivationRunner,
    writer: RolloutWriter,
) -> None:
    rollback = activation.with_controller_image(image)
    _record_rollout(
        remote_state_dir,
        RolloutRecord(
            phase=RolloutPhase.ROLLBACK_REQUESTED,
            image=image,
            previous_image=None,
            rollback_checkpoint=checkpoint,
            updated_at_ms=_updated_at_ms(),
        ),
        required=True,
        writer=writer,
    )
    apply(rollback)


def run_forward_activation(
    *,
    remote_state_dir: str,
    activation: IrisActivationSpec,
    prior_record: RolloutRecord | None,
    checkpoint: str | None,
    apply_candidate: ActivationRunner,
    apply_rollback: ActivationRunner,
    writer: RolloutWriter = write_rollout_record,
) -> None:
    """Apply one candidate and restore its paired previous state on failure."""
    previous_image = prior_record.image if prior_record is not None else None
    if prior_record is not None:
        _record_rollout(
            remote_state_dir,
            RolloutRecord(
                phase=RolloutPhase.PENDING,
                image=activation.controller_image,
                previous_image=previous_image,
                rollback_checkpoint=checkpoint,
                updated_at_ms=_updated_at_ms(),
            ),
            required=False,
            writer=writer,
        )

    try:
        apply_candidate(activation)
    except Exception as failure:
        if isinstance(failure, IrisActivationError) and not failure.started:
            if prior_record is not None:
                _record_rollout(
                    remote_state_dir,
                    prior_record,
                    required=True,
                    writer=writer,
                )
            raise IrisDeployError(f"Controller activation failed before mutation: {failure}") from failure
        if previous_image is None:
            raise IrisDeployError(
                "Controller activation failed and no previous image is recorded; deploy known-good code"
            ) from failure

        try:
            _request_and_apply_rollback(
                remote_state_dir=remote_state_dir,
                activation=activation,
                image=previous_image,
                checkpoint=checkpoint,
                apply=apply_rollback,
                writer=writer,
            )
        except Exception as rollback_failure:
            raise IrisDeployError(
                f"Controller activation failed and recovery to {previous_image} also failed: {rollback_failure}"
            ) from failure
        raise IrisDeployError(
            f"Controller activation failed; restored {previous_image} with checkpoint {checkpoint or '(local DB)'}"
        ) from failure

    _record_rollout(
        remote_state_dir,
        RolloutRecord(
            phase=RolloutPhase.COMMITTED,
            image=activation.controller_image,
            previous_image=previous_image,
            rollback_checkpoint=checkpoint,
            updated_at_ms=_updated_at_ms(),
        ),
        required=False,
        writer=writer,
    )


def run_rollback_activation(
    *,
    remote_state_dir: str,
    activation: IrisActivationSpec,
    prior_record: RolloutRecord,
    apply: ActivationRunner,
    writer: RolloutWriter = write_rollout_record,
) -> None:
    """Activate the previous image and request its recorded checkpoint when present."""
    if not prior_record.previous_image:
        raise IrisDeployError(f"No deploy to roll back to in {remote_state_dir}/{ROLLOUT_RECORD_FILENAME}")
    _request_and_apply_rollback(
        remote_state_dir=remote_state_dir,
        activation=activation,
        image=prior_record.previous_image,
        checkpoint=prior_record.rollback_checkpoint,
        apply=apply,
        writer=writer,
    )


def _config_for_cluster(cluster: str) -> IrisClusterConfig:
    try:
        path = resolve_cluster_config(cluster, dirs=IRIS_CLUSTER_CONFIG_DIRS)
    except FileNotFoundError as error:
        raise click.ClickException(f"Unknown Iris cluster {cluster!r}") from error
    config = load_config(path)
    configure_client_s3(config)
    return config


def _take_checkpoint(cluster: str, timeout: int) -> str:
    click.echo(f"Taking checkpoint (timeout {timeout}s)...")
    with connect_controller(cluster_name=cluster) as endpoint:
        response = take_controller_checkpoint(
            endpoint.url,
            endpoint.credentials,
            timeout_ms=timeout * MILLISECONDS_PER_SECOND,
        )
    checkpoint_summary = (
        f"Checkpoint: {response.checkpoint_path} ({response.job_count} jobs, {response.worker_count} workers)"
    )
    click.echo(checkpoint_summary)
    return response.checkpoint_path


def apply_pulumi_activation(spec: IrisActivationSpec, *, yes: bool) -> None:
    stack_config = IRIS_PULUMI_DIRECTORY / f"Pulumi.{spec.cluster}.yaml"
    if not stack_config.is_file():
        raise IrisActivationError(
            f"No Iris Pulumi stack config for cluster {spec.cluster!r}",
            started=False,
        )

    activation_marker = activation_marker_path(spec)
    activation_marker.unlink(missing_ok=True)
    config = (
        f"iris:controller_image={spec.controller_image}",
        f"iris:worker_image={spec.worker_image}",
        f"iris:task_image={spec.task_image}",
        f"iris:activation_id={spec.activation_id}",
        f"iris:activation_digest={spec.digest()}",
    )
    try:
        run_pulumi_up(
            IRIS_PULUMI_DIRECTORY,
            spec.cluster,
            yes=yes,
            config=config,
            stack_config=stack_config,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise IrisActivationError(str(error), started=activation_marker.is_file()) from error
    finally:
        activation_marker.unlink(missing_ok=True)


def _activate_gce_controller(
    config: IrisClusterConfig,
    bundle: ProviderBundle,
    on_activation_start: Callable[[], None],
) -> str:
    workers = bundle.workers
    if workers is None:
        raise IrisDeployError("GCE controller activation requires a worker infrastructure provider")
    plan = controller_restart_plan(workers, config, bundle.controller.resolve_image)
    vm = plan.vm
    if not isinstance(vm, GcpStandaloneWorkerHandle):
        raise IrisDeployError(f"Expected a GCE controller VM, found {type(vm).__name__}")

    target = GceVmTarget(
        project=vm.project_id,
        zone=vm.zone,
        instance=vm.gce_vm_name,
        impersonate_service_account=ssh_impersonate_service_account(config.defaults.ssh),
    )
    on_activation_start()
    activate_startup_script(
        target,
        plan.bootstrap_script,
        persistence=StartupScriptPersistence.BEFORE_ACTIVATION,
        timeout=GCE_ACTIVATION_TIMEOUT,
        attempts=3,
    )
    return plan.address


def activate_controller(
    spec: IrisActivationSpec,
    *,
    on_activation_start: Callable[[], None],
) -> str:
    """Apply a Pulumi activation specification and return the controller URL."""
    config = spec.apply(_config_for_cluster(spec.cluster))
    bundle = provider_bundle(config)
    bundle.controller.preflight_controller(config)
    # GCE activation stays here so marin-deploy owns startup-script persistence
    # and SSH. Kubernetes has no VM activation layer to replace.
    if config.platform.platform_kind() == "gcp":
        return _activate_gce_controller(config, bundle, on_activation_start)
    if not isinstance(bundle.controller, K8sControllerProvider):
        raise IrisDeployError(f"Unsupported Pulumi controller provider: {type(bundle.controller).__name__}")
    bundle.controller.verify_prerequisites(config)
    on_activation_start()
    return bundle.controller.restart_controller(config)


def _forward_activation(
    cluster: str,
    *,
    skip_checkpoint: bool,
    checkpoint_timeout: int,
    task_image_platforms: str | None,
    cargo_profile: str,
    prebuilt_tag: str | None,
    yes: bool,
) -> None:
    config = _config_for_cluster(cluster)
    if config.controller.controller_kind() == "local":
        raise click.ClickException("Local Iris controllers are not deployed through Pulumi")

    bundle = provider_bundle(config)
    try:
        bundle.controller.preflight_controller(config)
    except Exception as error:
        raise click.ClickException(f"Controller deploy preflight failed: {error}") from error

    checkpoint = None if skip_checkpoint else _take_checkpoint(cluster, checkpoint_timeout)
    if skip_checkpoint:
        click.echo("Skipping pre-deploy checkpoint.")

    if prebuilt_tag is not None:
        if task_image_platforms is not None:
            raise click.ClickException("--prebuilt-tag cannot be combined with --image-platform")
        if cargo_profile != DEFAULT_CARGO_PROFILE:
            raise click.ClickException("--prebuilt-tag cannot be combined with a non-default --cargo-profile")
        use_prebuilt_kubernetes_images(config, prebuilt_tag)
    else:
        build_and_pin_deploy_images(
            config,
            task_platforms=task_image_platforms,
            cargo_profile=cargo_profile,
            verbose=False,
        )

    remote_state_dir = config.storage.remote_state_dir
    prior_record = read_rollout_record(remote_state_dir) if remote_state_dir else None
    activation = IrisActivationSpec.from_config(cluster, config)

    try:
        run_forward_activation(
            remote_state_dir=remote_state_dir,
            activation=activation,
            prior_record=prior_record,
            checkpoint=checkpoint,
            apply_candidate=lambda candidate: apply_pulumi_activation(candidate, yes=yes),
            apply_rollback=lambda rollback: apply_pulumi_activation(rollback, yes=True),
        )
    except IrisDeployError as error:
        raise click.ClickException(str(error)) from error
    click.echo(f"Iris controller {cluster} deployed as {activation.controller_image}")


@click.group()
def iris() -> None:
    """Deploy Iris controllers."""


@iris.command("rollout")
@click.argument("cluster")
@click.option("--skip-checkpoint", is_flag=True, help="Skip the pre-deploy checkpoint.")
@click.option("--checkpoint-timeout", type=int, default=300, show_default=True)
@click.option("--image-platform", "task_image_platforms", default=None)
@click.option(
    "--cargo-profile",
    type=click.Choice(CARGO_PROFILES),
    default=DEFAULT_CARGO_PROFILE,
    show_default=True,
)
@click.option("--prebuilt-tag", default=None, help="Use an existing immutable Kubernetes image tag.")
@click.option("-y", "--yes", is_flag=True, help="Skip Pulumi confirmation.")
def rollout_cmd(
    cluster: str,
    skip_checkpoint: bool,
    checkpoint_timeout: int,
    task_image_platforms: str | None,
    cargo_profile: str,
    prebuilt_tag: str | None,
    yes: bool,
) -> None:
    """Deploy and verify one controller through Pulumi."""
    _forward_activation(
        cluster,
        skip_checkpoint=skip_checkpoint,
        checkpoint_timeout=checkpoint_timeout,
        task_image_platforms=task_image_platforms,
        cargo_profile=cargo_profile,
        prebuilt_tag=prebuilt_tag,
        yes=yes,
    )


@iris.command("rollback")
@click.argument("cluster")
@click.option("-y", "--yes", is_flag=True, help="Skip confirmation.")
def rollback_cmd(cluster: str, yes: bool) -> None:
    """Restore the previous image and its recorded checkpoint when present."""
    config = _config_for_cluster(cluster)
    remote_state_dir = config.storage.remote_state_dir
    if not remote_state_dir:
        raise click.ClickException("Iris rollback requires config.storage.remote_state_dir")
    prior_record = read_rollout_record(remote_state_dir)
    if prior_record is None or not prior_record.previous_image:
        raise click.ClickException(f"No deploy to roll back to in {remote_state_dir}/{ROLLOUT_RECORD_FILENAME}")
    bundle = provider_bundle(config)
    try:
        bundle.controller.preflight_controller(config)
        with connect_controller(cluster_name=cluster):
            pass
    except Exception as error:
        raise click.ClickException(f"Controller rollback preflight failed: {error}") from error
    click.echo(f"Current image: {prior_record.image}")
    click.echo(f"Restore image: {prior_record.previous_image}")
    click.echo(f"Restore checkpoint: {prior_record.rollback_checkpoint or '(local DB)'}")
    if not yes and not click.confirm("Roll back this Iris controller?"):
        click.echo("Aborted.")
        return

    activation = IrisActivationSpec.from_config(cluster, config)
    try:
        run_rollback_activation(
            remote_state_dir=remote_state_dir,
            activation=activation,
            prior_record=prior_record,
            apply=lambda candidate: apply_pulumi_activation(candidate, yes=True),
        )
    except IrisDeployError as error:
        raise click.ClickException(str(error)) from error
    click.echo(f"Iris controller {cluster} rollback activated")

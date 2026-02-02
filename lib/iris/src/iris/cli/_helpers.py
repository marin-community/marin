# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared helper functions for Iris CLI commands."""

import concurrent.futures
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import click

from iris.cluster.vm.config import (
    create_autoscaler_from_config,
    load_config,
)
from iris.cluster.vm.vm_platform import compute_slice_state_counts
from iris.rpc import cluster_connect, cluster_pb2, vm_pb2
from iris.rpc.proto_utils import format_accelerator_display, vm_state_name
from iris.time_utils import Timestamp


def handle_error(ctx: click.Context, message: str, error: Exception) -> None:
    """Handle an error, optionally showing traceback based on --traceback flag."""
    click.echo(f"{message}: {error}", err=True)
    if ctx.obj and ctx.obj.get("traceback"):
        traceback.print_exc()
    raise SystemExit(1)


def format_timestamp(ms: int) -> str:
    """Format millisecond timestamp as human-readable string."""
    if ms == 0:
        return "-"
    return Timestamp.from_ms(ms).as_formatted_date()


def format_status_table(status: vm_pb2.AutoscalerStatus) -> str:
    """Format autoscaler status as a table showing slice states."""
    header = f"{'Scale Group':<18} {'Booting':>8} {'Initializing':>12} {'Ready':>6} {'Failed':>7} {'Demand':>7}"
    lines = [header]
    for group in status.groups:
        counts = compute_slice_state_counts(group.slices)
        line = (
            f"{group.name:<18} "
            f"{counts.get('booting', 0):>8} "
            f"{counts.get('initializing', 0):>12} "
            f"{counts.get('ready', 0):>6} "
            f"{counts.get('failed', 0):>7} "
            f"{group.current_demand:>7}"
        )
        lines.append(line)
    return "\n".join(lines)


def get_autoscaler_status(controller_url: str) -> vm_pb2.AutoscalerStatus:
    """Fetch autoscaler status from the controller via Connect RPC."""
    client = cluster_connect.ControllerServiceClientSync(controller_url)
    request = cluster_pb2.Controller.GetAutoscalerStatusRequest()
    response = client.get_autoscaler_status(request)
    return response.status


def get_vm_logs(controller_url: str, vm_id: str, tail: int) -> tuple[str, str, int]:
    """Fetch VM logs from the controller via Connect RPC.

    Returns:
        Tuple of (logs, vm_id, state)
    """
    client = cluster_connect.ControllerServiceClientSync(controller_url)
    request = cluster_pb2.Controller.GetVmLogsRequest(vm_id=vm_id, tail=tail)
    response = client.get_vm_logs(request)
    return response.logs, response.vm_id, response.state


def load_autoscaler(config_file: str, dry_run: bool = False):
    """Load Autoscaler from config file and discover existing slices.

    Returns tuple of (autoscaler, config) so caller can access config settings.
    """
    config = load_config(Path(config_file))
    autoscaler = create_autoscaler_from_config(config, dry_run=dry_run)
    autoscaler.reconcile()
    return autoscaler, config


def require_config(ctx: click.Context) -> str:
    """Get config file from context, raising error if not provided."""
    config_file = ctx.obj.get("config_file")
    if not config_file:
        click.echo("Error: --config is required for this command", err=True)
        raise SystemExit(1)
    return config_file


def parse_artifact_registry_tag(image_tag: str) -> tuple[str, str, str, str] | None:
    """Parse a GCP Artifact Registry image tag into components.

    Args:
        image_tag: Full image tag like "europe-west4-docker.pkg.dev/project/repo/image:version"

    Returns:
        Tuple of (region, project, image_name, version) or None if not a valid AR tag.
    """
    if "-docker.pkg.dev/" not in image_tag:
        return None

    parts = image_tag.split("/")
    if len(parts) < 4:
        return None

    registry = parts[0]
    if not registry.endswith("-docker.pkg.dev"):
        return None
    region = registry.replace("-docker.pkg.dev", "")

    project = parts[1]
    image_and_version = parts[3]

    if ":" in image_and_version:
        image_name, version = image_and_version.split(":", 1)
    else:
        image_name = image_and_version
        version = "latest"

    return region, project, image_name, version


@dataclass
class ImageBuildParams:
    """Parameters extracted from config for building an image."""

    image_type: Literal["worker", "controller"]
    region: str
    project: str
    image_name: str
    version: str

    @property
    def local_tag(self) -> str:
        return f"{self.image_name}:{self.version}"


def extract_worker_image_params(config) -> ImageBuildParams | None:
    """Extract worker image build params from config.bootstrap.docker_image."""
    if not config.bootstrap.docker_image:
        return None

    parsed = parse_artifact_registry_tag(config.bootstrap.docker_image)
    if not parsed:
        return None

    region, project, image_name, version = parsed
    return ImageBuildParams(
        image_type="worker",
        region=region,
        project=project,
        image_name=image_name,
        version=version,
    )


def extract_controller_image_params(config) -> ImageBuildParams | None:
    """Extract controller image build params from config.controller_vm.image."""
    if not config.controller_vm.image:
        return None

    parsed = parse_artifact_registry_tag(config.controller_vm.image)
    if not parsed:
        return None

    region, project, image_name, version = parsed
    return ImageBuildParams(
        image_type="controller",
        region=region,
        project=project,
        image_name=image_name,
        version=version,
    )


def build_and_push_image(params: ImageBuildParams) -> None:
    """Build and push a single image using params."""
    from iris.build import build_image, push_to_registries

    click.echo(f"Building {params.image_type} image: {params.local_tag}")
    click.echo(f"  Region: {params.region}")
    click.echo(f"  Project: {params.project}")
    click.echo()

    build_image(
        image_type=params.image_type,
        tag=params.local_tag,
        push=False,
        dockerfile=None,
        context=None,
        platform="linux/amd64",
        region=(),
        project=params.project,
    )

    click.echo()
    push_to_registries(
        source_tag=params.local_tag,
        regions=(params.region,),
        project=params.project,
        image_name=params.image_name,
        version=params.version,
    )


def build_cluster_images(config) -> None:
    """Build and push all cluster images (worker and controller)."""
    for extract_fn in [extract_worker_image_params, extract_controller_image_params]:
        params = extract_fn(config)
        if params:
            build_and_push_image(params)
            click.echo()


def terminate_slices_parallel(autoscaler_obj, slice_ids: tuple[str, ...] | list[str]) -> None:
    """Terminate slices in parallel, printing progress."""

    def terminate_one(slice_id: str) -> tuple[str, Exception | None]:
        try:
            for group in autoscaler_obj.groups.values():
                for vm_group in group.vm_groups():
                    if vm_group.slice_id == slice_id:
                        group.scale_down(slice_id)
                        return slice_id, None
            return slice_id, ValueError(f"Slice {slice_id} not found")
        except Exception as e:
            return slice_id, e

    click.echo(f"Terminating {len(slice_ids)} slice(s) in parallel...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(slice_ids)) as executor:
        futures = {executor.submit(terminate_one, sid): sid for sid in slice_ids}
        for future in concurrent.futures.as_completed(futures):
            slice_id, error = future.result()
            if error:
                click.echo(f"Failed to terminate {slice_id}: {error}", err=True)
            else:
                click.echo(f"Terminated: {slice_id}")


def wait_for_slice_obj(slice_obj, poll_interval: float = 5.0) -> bool:
    """Poll until slice reaches terminal state (all READY or any FAILED).

    Returns True if all VMs are ready, False if any failed.
    """
    import time

    last_states: dict[str, str] = {}
    slice_id = slice_obj.slice_id

    click.echo(f"Waiting for slice {slice_id} to reach READY state...")

    while True:
        status = slice_obj.status()

        for vm in slice_obj.vms():
            state = vm_state_name(vm.info.state)
            if last_states.get(vm.info.vm_id) != state:
                click.echo(f"  {vm.info.vm_id}: {state}")
                last_states[vm.info.vm_id] = state

        if status.all_ready:
            vm_count = len(list(slice_obj.vms()))
            click.echo(f"Slice {slice_id} is READY ({vm_count} VMs)")
            return True

        if status.any_failed:
            click.echo(f"Slice {slice_id} has FAILED VMs:", err=True)
            for vm in slice_obj.vms():
                if vm.info.state == vm_pb2.VM_STATE_FAILED:
                    click.echo(f"  {vm.info.vm_id}: {vm.info.init_error}", err=True)
            return False

        time.sleep(poll_interval)

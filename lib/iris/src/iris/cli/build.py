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

"""Image build commands."""

import click

from iris.build import build_image, push_to_registries


@click.group()
def build():
    """Image build commands."""


@build.command("worker-image")
@click.option("--tag", "-t", default="iris-worker:latest", help="Image tag")
@click.option("--push", is_flag=True, help="Push image to registry after building")
@click.option("--dockerfile", type=click.Path(exists=True), help="Custom Dockerfile path")
@click.option("--context", type=click.Path(exists=True), help="Build context directory")
@click.option("--platform", default="linux/amd64", help="Target platform")
@click.option("--region", multiple=True, help="GCP Artifact Registry regions to push to")
@click.option("--project", default="hai-gcp-models", help="GCP project ID for registry")
def build_worker_image(
    tag: str,
    push: bool,
    dockerfile: str | None,
    context: str | None,
    platform: str,
    region: tuple[str, ...],
    project: str,
):
    """Build Docker image for Iris worker."""
    build_image("worker", tag, push, dockerfile, context, platform, region, project)


@build.command("controller-image")
@click.option("--tag", "-t", default="iris-controller:latest", help="Image tag")
@click.option("--push", is_flag=True, help="Push image to registry after building")
@click.option("--dockerfile", type=click.Path(exists=True), help="Custom Dockerfile path")
@click.option("--context", type=click.Path(exists=True), help="Build context directory")
@click.option("--platform", default="linux/amd64", help="Target platform")
@click.option("--region", multiple=True, help="GCP Artifact Registry regions to push to")
@click.option("--project", default="hai-gcp-models", help="GCP project ID for registry")
def build_controller_image(
    tag: str,
    push: bool,
    dockerfile: str | None,
    context: str | None,
    platform: str,
    region: tuple[str, ...],
    project: str,
):
    """Build Docker image for Iris controller."""
    build_image("controller", tag, push, dockerfile, context, platform, region, project)


@build.command("push")
@click.argument("source_tag")
@click.option("--region", "-r", multiple=True, required=True, help="GCP Artifact Registry region")
@click.option("--project", default="hai-gcp-models", help="GCP project ID")
@click.option("--image-name", default="iris-worker", help="Image name in registry")
@click.option("--version", default="latest", help="Version tag")
def build_push(source_tag: str, region: tuple[str, ...], project: str, image_name: str, version: str):
    """Push a local Docker image to GCP Artifact Registry."""
    push_to_registries(
        source_tag,
        region,
        project,
        image_name=image_name,
        version=version,
    )

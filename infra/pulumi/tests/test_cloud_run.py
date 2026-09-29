# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pulumi
import pulumi_gcp as gcp
from iac.gcp.cloud_run import CloudRunService, CloudRunServiceArgs, SecretEnv, dockerfile_image
from pulumi.runtime import MockCallArgs, MockResourceArgs, Mocks


class RecordingMocks(Mocks):
    def __init__(self) -> None:
        self.resources: list[MockResourceArgs] = []

    def new_resource(self, args: MockResourceArgs):
        self.resources.append(args)
        outputs = dict(args.inputs)
        if args.typ == "docker-build:index:Image":
            outputs["digest"] = "sha256:" + "a" * 64
            outputs["ref"] = f"{outputs['tags'][0]}@sha256:" + "a" * 64
        return f"{args.name}_id", outputs

    def call(self, args: MockCallArgs) -> tuple[dict, list[tuple[str, str]] | None]:
        outputs = dict(args.args)
        if args.token == "gcp:organizations/getProject:getProject":
            outputs["number"] = "123456789"
        return outputs, []


@pulumi.runtime.test
def test_dockerfile_image_uses_dedicated_registry_cache() -> pulumi.Output[None]:
    mocks = RecordingMocks()
    pulumi.runtime.set_mocks(mocks, project="marin-iac", stack="test", preview=False)
    provider = gcp.Provider("gcp", project="example")
    parent = pulumi.ComponentResource("test:index:Parent", "parent")
    image = dockerfile_image(
        image_name="service",
        description="Service images",
        project="example",
        region="us-central1",
        build_context="/tmp/service",
        dockerfile="Dockerfile",
        parent=parent,
        gcp_provider=provider,
    )

    def check(_: str) -> None:
        resource = next(resource for resource in mocks.resources if resource.typ == "docker-build:index:Image")
        cache_ref = "us-central1-docker.pkg.dev/example/service/service:buildcache"
        assert resource.inputs["cacheFrom"] == [{"registry": {"ref": cache_ref}}]
        (cache_to,) = resource.inputs["cacheTo"]
        registry = cache_to["registry"]
        assert registry["ref"] == cache_ref
        assert registry["mode"] == "max"
        assert registry["compression"] == "zstd"
        assert registry["compressionLevel"] == 3
        assert registry["ociMediaTypes"] is True
        assert registry["imageManifest"] is True

    return image.ref.apply(check)


@pulumi.runtime.test
def test_cloud_run_components_leave_iam_to_the_infrastructure_stack() -> pulumi.Output[None]:
    mocks = RecordingMocks()
    pulumi.runtime.set_mocks(mocks, project="marin-iac", stack="test", preview=False)
    provider = gcp.Provider("gcp", project="example")
    service = CloudRunService(
        "service",
        CloudRunServiceArgs(
            project="example",
            region="us-central1",
            service_name="service",
            build_context="/tmp/service",
            secrets=(SecretEnv(name="TOKEN", secret="service-token"),),
            cloudsql_instances=("example:us-central1:database",),
        ),
        gcp_provider=provider,
    )

    def check(_: list[str]) -> None:
        iam_resource_types = {
            "gcp:cloudrunv2/jobIamMember:JobIamMember",
            "gcp:cloudrunv2/serviceIamMember:ServiceIamMember",
            "gcp:iap/webCloudRunServiceIamMember:WebCloudRunServiceIamMember",
            "gcp:projects/iAMMember:IAMMember",
            "gcp:secretmanager/secretIamMember:SecretIamMember",
        }
        assert iam_resource_types.isdisjoint(resource.typ for resource in mocks.resources)

    return service.uri.apply(lambda uri: check([uri]))


@pulumi.runtime.test
def test_cloud_run_service_can_boost_startup_with_request_based_cpu() -> pulumi.Output[None]:
    mocks = RecordingMocks()
    pulumi.runtime.set_mocks(mocks, project="marin-iac", stack="test", preview=False)
    provider = gcp.Provider("gcp", project="example")
    service = CloudRunService(
        "service",
        CloudRunServiceArgs(
            project="example",
            region="us-central1",
            service_name="service",
            build_context="/tmp/service",
            cpu_always_allocated=False,
            startup_cpu_boost=True,
        ),
        gcp_provider=provider,
    )

    def check(_: str) -> None:
        resource = next(resource for resource in mocks.resources if resource.typ == "gcp:cloudrunv2/service:Service")
        resources = resource.inputs["template"]["containers"][0]["resources"]
        assert resources["cpuIdle"] is True
        assert resources["startupCpuBoost"] is True

    return service.uri.apply(check)

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path

import pulumi
from pulumi.runtime import MockCallArgs, MockResourceArgs, Mocks

STACK_PROGRAM = Path(__file__).parents[1] / "__main__.py"
DOMAIN_MAPPING = "gcp:cloudrun/domainMapping:DomainMapping"
CLOUD_RUN_SERVICE = "gcp:cloudrunv2/service:Service"


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


def run_stack(config: dict[str, object]) -> RecordingMocks:
    """Run the Marina stack program against recording mocks and return its registrations."""
    mocks = RecordingMocks()
    pulumi.runtime.set_mocks(mocks, project="marin-marina", stack="test", preview=True)
    pulumi.runtime.set_all_config(
        {f"marin-marina:{key}": value if isinstance(value, str) else json.dumps(value) for key, value in config.items()}
    )

    @pulumi.runtime.test
    def program() -> None:
        spec = importlib.util.spec_from_file_location("marina_stack", STACK_PROGRAM)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(importlib.util.module_from_spec(spec))

    program()
    return mocks


def service_env(mocks: RecordingMocks, service_name: str) -> dict[str, str]:
    service = next(
        resource
        for resource in mocks.resources
        if resource.typ == CLOUD_RUN_SERVICE and resource.inputs["name"] == service_name
    )
    return {env["name"]: env.get("value") for env in service.inputs["template"]["containers"][0]["envs"]}


def test_public_applet_hosts_route_to_the_public_reader_and_bind_through_the_service() -> None:
    mocks = run_stack(
        {
            "dns_zone_id": "zone",
            "applet_domains": ["filer.marina.oa.dev"],
            "public_applet_domains": ["Atlas.oa.dev"],
        }
    )

    routes = {
        resource.inputs["name"]: resource.inputs["spec"]["routeName"]
        for resource in mocks.resources
        if resource.typ == DOMAIN_MAPPING
    }
    assert routes["filer.marina.oa.dev"] == "marina"
    assert routes["atlas.oa.dev"] == "marina-public-applets"
    assert routes["public.applets.marina.oa.dev"] == "marina-public-applets"
    assert routes["marina.oa.dev"] == "marina"
    assert service_env(mocks, "marina-public-applets")["MARINA_APPLET_DOMAINS"] == "atlas.oa.dev"
    assert service_env(mocks, "marina")["MARINA_APPLET_DOMAINS"] == "filer.marina.oa.dev,atlas.oa.dev"

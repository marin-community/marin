# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import runpy
from pathlib import Path

import pulumi
from iac.gcp import grafana, inferencex
from iac.gcp.iam import GcpEncryptedMember
from iac.gcp.iam_config import load_iam_config
from pulumi.runtime import MockCallArgs, MockResourceArgs, Mocks


def test_inferencex_reuses_viewers_and_separates_reader_and_writer_access() -> None:
    project = "hai-gcp-models"
    principals = {p.principal_id: GcpEncryptedMember(p.ciphertext) for p in load_iam_config().principals}
    grants = inferencex.iam_grants(project, principals)
    viewers = grants.cloud_run_iap[0].iap_grants
    original = grafana.iam_grants(project, principals).cloud_run_iap[0].iap_grants
    ingest = f"serviceAccount:github-iris@{project}.iam.gserviceaccount.com"
    runtime = f"serviceAccount:marin-inferencex@{project}.iam.gserviceaccount.com"
    assert [(g.role, g.members[:-1]) for g in viewers] == [(g.role, g.members) for g in original]
    assert all(g.members[-1] == ingest for g in viewers)
    secrets = {s.secret: {m for g in s.grants for m in g.members} for s in grants.secrets}
    assert secrets["cloudsql-inferencex-reader-password"] == {runtime}
    assert secrets["cloudsql-inferencex-writer-password"] == {ingest}
    assert secrets["marin-inferencex-invalidate-secret"] == {runtime, ingest}


class InferencexMocks(Mocks):
    def __init__(self) -> None:
        self.resources: list[MockResourceArgs] = []

    def new_resource(self, args: MockResourceArgs):
        self.resources.append(args)
        outputs = dict(args.inputs)
        if args.typ == "pulumi:pulumi:StackReference":
            outputs["outputs"] = {"connection_name": "example:us-central1:metadata"}
        if args.typ == "docker-build:index:Image":
            outputs["digest"] = "sha256:" + "a" * 64
            outputs["ref"] = f"{outputs['tags'][0]}@sha256:" + "a" * 64
        if args.typ == "gcp:cloudrunv2/service:Service":
            outputs["uri"] = "https://inferencex.example.run.app"
        return f"{args.name}_id", outputs

    def call(self, args: MockCallArgs):
        return {**args.args, "number": "123456789"}, []


@pulumi.runtime.test
def test_dashboard_graph_has_iap_socket_and_only_reader_credentials() -> pulumi.Output[None]:
    mocks = InferencexMocks()
    pulumi.runtime.set_mocks(mocks, project="marin-inferencex", stack="test", preview=False)
    pulumi.runtime.set_all_config(
        {
            "marin-inferencex:custom_domain": "performance.oa.dev",
            "marin-inferencex:dns_zone_id": "test-zone",
        }
    )
    program = runpy.run_path(str(Path(__file__).resolve().parents[2] / "inferencex" / "__main__.py"))
    service = program["main"]()

    def check(_: str) -> None:
        resource = next(r for r in mocks.resources if r.typ == "gcp:cloudrunv2/service:Service")
        assert resource.inputs["iapEnabled"] is True
        template = resource.inputs["template"]
        assert resource.inputs["scaling"]["maxInstanceCount"] == 1
        environment = {e["name"]: e for e in template["containers"][0]["envs"]}
        assert environment["DATABASE_USER"]["value"] == "inferencex_reader"
        assert environment["DATABASE_SOCKET_DIR"]["value"] == "/cloudsql/example:us-central1:metadata"
        assert (
            environment["DATABASE_PASSWORD"]["valueSource"]["secretKeyRef"]["secret"]
            == "cloudsql-inferencex-reader-password"
        )
        assert "DATABASE_WRITE_URL" not in environment
        assert all("Iam" not in r.typ and "iAM" not in r.typ and "secretVersion" not in r.typ for r in mocks.resources)

    return service.uri.apply(check)

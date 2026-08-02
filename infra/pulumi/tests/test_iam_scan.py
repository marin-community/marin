# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from iac.gcp.iam import GcpEncryptedMember, GcpRoleGrant, GcpSecretIam, GcpServiceAccountIam
from iac.gcp.iam_scan import (
    MARKER,
    Binding,
    Container,
    FindingKind,
    classify,
    declared_bindings,
    fingerprint,
    redact,
    render_markdown,
    service_agent_api,
)

PROJECT = "hai-gcp-models"
SA = "serviceAccount:worker@hai-gcp-models.iam.gserviceaccount.com"


def _project(role: str, member: str) -> Binding:
    return Binding(Container.PROJECT, PROJECT, role, member)


def test_classify_flags_undeclared_bindings_and_leaves_declared_alone():
    declared = {_project("roles/viewer", SA)}
    live = [
        _project("roles/viewer", SA),  # declared — no finding
        _project("roles/storage.objectAdmin", SA),  # undeclared narrow — one finding
    ]

    findings = classify(live, declared, project=PROJECT, enabled_services=None)

    assert len(findings) == 1
    assert findings[0].kind is FindingKind.UNDECLARED_BINDING
    assert findings[0].role == "roles/storage.objectAdmin"
    assert findings[0].broad is False


def test_classify_ranks_broad_role_above_narrow_undeclared():
    live = [
        _project("roles/storage.objectViewer", SA),
        _project("roles/owner", SA),
    ]

    findings = classify(live, declared=set(), project=PROJECT, enabled_services=None)

    assert [f.role for f in findings] == ["roles/owner", "roles/storage.objectViewer"]
    assert findings[0].broad is True


def test_classify_does_not_diff_human_members():
    live = [_project("roles/owner", "user:person@stanford.edu")]

    findings = classify(live, declared=set(), project=PROJECT, enabled_services=None)

    assert findings == []


def test_classify_flags_orphaned_service_agent_only_when_api_disabled():
    agent = "serviceAccount:service-1@gcp-sa-pubsub.iam.gserviceaccount.com"
    live = [_project("roles/pubsub.publisher", agent)]

    disabled = classify(live, {live[0]}, project=PROJECT, enabled_services=frozenset({"storage.googleapis.com"}))
    enabled = classify(live, {live[0]}, project=PROJECT, enabled_services=frozenset({"pubsub.googleapis.com"}))

    assert [f.kind for f in disabled] == [FindingKind.DISABLED_API_AGENT]
    assert "pubsub.googleapis.com" in disabled[0].detail
    assert enabled == []


def test_classify_skips_orphan_check_without_serviceusage_access():
    agent = "serviceAccount:service-1@gcp-sa-pubsub.iam.gserviceaccount.com"
    live = [_project("roles/pubsub.publisher", agent)]

    # enabled_services=None models a missing serviceusage.services.list permission.
    findings = classify(live, {live[0]}, project=PROJECT, enabled_services=None)

    assert findings == []


def test_service_agent_api_recognizes_managed_agents_and_ignores_the_rest():
    assert service_agent_api("serviceAccount:service-1@gcp-sa-pubsub.iam.gserviceaccount.com", PROJECT) == (
        "pubsub.googleapis.com"
    )
    assert service_agent_api("serviceAccount:service-1@compute-system.iam.gserviceaccount.com", PROJECT) == (
        "compute.googleapis.com"
    )
    # A user-managed SA in this project is not a Google agent.
    assert service_agent_api(SA, PROJECT) is None
    # Another project's SA is out of our control; do not guess its API state.
    assert service_agent_api("serviceAccount:x@other-proj.iam.gserviceaccount.com", PROJECT) is None
    # The core Google APIs agent is never tied to a single toggleable API.
    assert service_agent_api("serviceAccount:1@cloudservices.gserviceaccount.com", PROJECT) is None


def test_declared_bindings_drops_encrypted_and_plain_human_members():
    grant = GcpRoleGrant(
        role="roles/secretmanager.secretAccessor",
        members=(SA, GcpEncryptedMember(ciphertext="abc"), "user:person@stanford.edu"),
    )
    secret = GcpSecretIam(secret="MY_SECRET", grants=(grant,))
    account = GcpServiceAccountIam(email="worker@hai-gcp-models.iam.gserviceaccount.com", grants=())

    bindings = declared_bindings(
        PROJECT,
        "key",
        project_grants=(),
        kms_grants=(),
        secrets=(secret,),
        buckets=(),
        artifact_repositories=(),
        service_accounts=(account,),
    )

    assert bindings == {Binding(Container.SECRET, "MY_SECRET", "roles/secretmanager.secretAccessor", SA)}


def test_fingerprint_is_order_independent_and_content_sensitive():
    a = _project("roles/owner", SA)
    b = _project("roles/viewer", SA)
    findings_ab = classify([a, b], set(), project=PROJECT, enabled_services=None)
    findings_ba = classify([b, a], set(), project=PROJECT, enabled_services=None)

    assert fingerprint(findings_ab) == fingerprint(findings_ba)
    assert fingerprint(findings_ab) != fingerprint([findings_ab[0]])


def test_redact_masks_human_local_part_but_not_service_accounts():
    assert redact("user:alice@stanford.edu") == "user:a***@stanford.edu"
    assert redact(SA) == SA


def test_render_markdown_embeds_marker_and_fingerprint():
    findings = classify([_project("roles/owner", SA)], set(), project=PROJECT, enabled_services=None)

    body = render_markdown(findings, project=PROJECT, run_url="https://run")

    assert MARKER in body
    assert f"fingerprint:{fingerprint(findings)}" in body
    assert SA in body
    assert "https://run" in body

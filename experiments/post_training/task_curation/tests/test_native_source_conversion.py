# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The native source conversion pins FROM bases and refuses static blockers."""

import json
import urllib.error
import urllib.request

import pytest
from finestore.rl.licenses import (
    LICENSE_RESOLVER_VERSION,
    LicenseDecision,
    LicenseDecisionKind,
    LicenseReceipt,
    LicenseSourcePin,
)
from taskcompendium.identity import canonical_sha256, task_spec_sha256
from taskcompendium.models import (
    AnswerType,
    CommandSemantics,
    ConversationInput,
    DockerBuildContext,
    DockerBuildExecution,
    EnvironmentRequirements,
    NoGrader,
    PlainText,
    ResourceGroups,
    ScriptGrader,
    TaskSpec,
    TextMessage,
)
from taskcompendium.runtime.resources import inline_resource, resource_bytes

from experiments.post_training.task_curation import native_source_conversion
from experiments.post_training.task_curation.calibration_inventory import task_blockers
from experiments.post_training.task_curation.harbor_cohort_conversion import CohortConversionRejected
from experiments.post_training.task_curation.license_publication import repin_license_receipt
from experiments.post_training.task_curation.license_resolution import license_evidence
from experiments.post_training.task_curation.native_source_conversion import (
    BasePin,
    BasePinRejected,
    conversion_source,
    convert_native_source_row,
    pin_package_dockerfile,
    pinned_dockerfile,
)

PINS = {
    "python:3.9-bookworm": "python:3.9-bookworm@sha256:" + "a" * 64,
    "ubuntu:22.04": "ubuntu:22.04@sha256:" + "b" * 64,
    "namanjain12/orange3_final:deadbeef": "namanjain12/orange3_final@sha256:" + "c" * 64,
}


def _resolve(reference: str) -> str:
    if reference not in PINS:
        raise ValueError(f"unresolvable in the test: {reference}")
    return PINS[reference]


DOCKERFILE = """\
FROM --platform=linux/amd64 python:3.9-bookworm AS base
RUN true
FROM base AS final
FROM --platform=linux/amd64 namanjain12/orange3_final:deadbeef
RUN true
"""

EXPECTED = """\
FROM --platform=linux/amd64 python:3.9-bookworm@sha256:{digest_a} AS base
RUN true
FROM base AS final
FROM --platform=linux/amd64 namanjain12/orange3_final@sha256:{digest_c}
RUN true
"""


def test_pinned_dockerfile_rewrites_external_bases_only():
    text, pins = pinned_dockerfile(DOCKERFILE, _resolve)
    assert text == EXPECTED.format(digest_a="a" * 64, digest_c="c" * 64)
    assert pins == (
        BasePin(base="python:3.9-bookworm", pinned=PINS["python:3.9-bookworm"]),
        BasePin(base="namanjain12/orange3_final:deadbeef", pinned=PINS["namanjain12/orange3_final:deadbeef"]),
    )


def test_pinned_dockerfile_passes_pinned_and_scratch_references():
    text = f"FROM scratch\nFROM ubuntu:22.04@sha256:{'b' * 64}\n"
    pinned, pins = pinned_dockerfile(text, _resolve)
    assert pinned == text
    assert pins == ()


def test_pinned_dockerfile_refuses_unresolvable_base():
    with pytest.raises(BasePinRejected):
        pinned_dockerfile("FROM ghcr.io/org/image:latest\n", _resolve)


def test_pinned_dockerfile_leaves_heredoc_bodies_untouched():
    """A FROM inside a RUN heredoc is shell data, never a base instruction.

    Machine-authored recipes embed instruction-looking lines in RUN heredocs;
    the scanner must not resolve them (a crash) or rewrite their bytes.
    """
    digest = "c" * 64
    text = (
        "FROM ubuntu:22.04\n"
        "COPY repo /testbed\n"
        "RUN <<'EOF'\n"
        "cat > /tmp/inner.Dockerfile <<'INNER'\n"
        "FROM instructor:2\n"
        "INNER\n"
        "echo FROM ghcr.io/unresolved/thing:1\n"
        "EOF\n"
        "FROM python:3.12-slim\n"
        "RUN true\n"
    )

    def resolve(reference: str) -> str:
        assert reference in ("ubuntu:22.04", "python:3.12-slim"), reference
        return f"{reference}@sha256:{digest}"

    pinned, pins = pinned_dockerfile(text, resolve)
    assert "FROM instructor:2" in pinned
    assert "echo FROM ghcr.io/unresolved/thing:1" in pinned
    assert f"FROM python:3.12-slim@sha256:{digest}" in pinned
    assert f"FROM ubuntu:22.04@sha256:{digest}" in pinned
    assert pins == (
        BasePin(base="ubuntu:22.04", pinned=f"ubuntu:22.04@sha256:{digest}"),
        BasePin(base="python:3.12-slim", pinned=f"python:3.12-slim@sha256:{digest}"),
    )


def _package(dockerfile: str) -> TaskSpec:
    build = DockerBuildContext(
        files=(inline_resource("Dockerfile", dockerfile.encode()),),
        empty_directories=(),
    )
    return TaskSpec(
        id="TaskTrove:native-fixture/task-1",
        source={"dataset": "fixture", "revision": "a" * 40, "row": "fixture/task-1", "importer_revision": "v1"},
        context=ConversationInput(events=(TextMessage(role="user", content="Fix the repository."),)),
        environment_requirements=EnvironmentRequirements(
            command_semantics=CommandSemantics.LINUX_PROCESS,
            docker_build=build,
        ),
        answer_type=AnswerType.WORKSPACE_STATE,
        answer_format=PlainText(),
        grader=NoGrader(reason="The native Harbor Trial executes this package's declared verifier"),
        resources=ResourceGroups(
            verifier=(
                inline_resource("environment/Dockerfile", dockerfile.encode()),
                inline_resource("instruction.md", b"Fix the repository.\n"),
                inline_resource("task.toml", b"[verifier]\ntimeout_sec = 900.0\n"),
                inline_resource(
                    "tests/test.sh",
                    b"#!/bin/bash\nexit 0\n",
                ),
            ),
            oracle=(inline_resource("solution/solve.sh", b"#!/bin/bash\ntrue\n"),),
        ),
        tags=("harbor",),
    )


def test_pin_package_dockerfile_rewrites_only_the_dockerfile_resource():
    task = _package(DOCKERFILE)
    pinned, pins = pin_package_dockerfile(task, _resolve)
    assert len(pins) == 2
    assert {resource.path for resource in pinned.resources.verifier} == {
        resource.path for resource in task.resources.verifier
    }
    dockerfile = next(r for r in pinned.resources.verifier if r.path == "environment/Dockerfile")
    assert resource_bytes(dockerfile).decode() == EXPECTED.format(digest_a="a" * 64, digest_c="c" * 64)


def test_convert_native_source_row_yields_executable_grader_without_blockers():
    task = _package(DOCKERFILE)
    outcome = convert_native_source_row(task, input_task_sha256=task_spec_sha256(task), resolve=_resolve)
    assert isinstance(outcome.task.grader, ScriptGrader)
    assert outcome.base_pins[0].base == "python:3.9-bookworm"
    assert task_blockers(outcome.task, docker_build=DockerBuildExecution.DECLARATIVE) == set()
    assert outcome.input_task_sha256 == task_spec_sha256(task)
    assert outcome.output_task_sha256 != outcome.input_task_sha256


def test_convert_native_source_row_refuses_when_blockers_remain():
    # An oracle-less package still carries a static blocker after pinning; the
    # conversion must reject it instead of silently shrinking the population.
    task = _package(DOCKERFILE)
    stripped = task.model_copy(update={"resources": task.resources.model_copy(update={"oracle": ()})})
    with pytest.raises(CohortConversionRejected):
        convert_native_source_row(stripped, input_task_sha256=task_spec_sha256(task), resolve=_resolve)


def test_resolve_base_digest_caches_per_process():
    """A cohort's tasks share bases; one registry resolution serves every row."""
    native_source_conversion._DIGEST_CACHE.clear()
    calls = []

    def counting_resolve(reference: str) -> str:
        calls.append(reference)
        return PINS[reference]

    original = native_source_conversion._resolve_base_digest
    native_source_conversion._resolve_base_digest = counting_resolve
    try:
        assert native_source_conversion.resolve_base_digest("python:3.9-bookworm") == PINS["python:3.9-bookworm"]
        assert native_source_conversion.resolve_base_digest("python:3.9-bookworm") == PINS["python:3.9-bookworm"]
        assert native_source_conversion.resolve_base_digest("ubuntu:22.04") == PINS["ubuntu:22.04"]
    finally:
        native_source_conversion._resolve_base_digest = original
        native_source_conversion._DIGEST_CACHE.clear()
    assert calls == ["python:3.9-bookworm", "ubuntu:22.04"]


def test_release_receipt_carries_the_exception_lineage():
    """The published card follows the original upstream, and an approved user exception
    survives the re-pin with its original pin, date, scope and upstream grant
    (repackager rule + coordinator ruling 2026-11-26)."""
    card = b"---\nlicense: mit\nlicense_source:\n  dataset_id: SWE-Gym/SWE-Gym\n"
    upstream_card = license_evidence(
        "https://huggingface.co/datasets/SWE-Gym/SWE-Gym/resolve/" + "e" * 40 + "/README.md",
        b"---\nlicense: mit\n---\n",
        "dataset_card",
    )
    previous = LicenseReceipt(
        resolver_version=LICENSE_RESOLVER_VERSION,
        source_id="TaskTrove:example",
        dataset_id="open-athena/example",
        revision="e" * 40,
        resolved_at="2026-11-26T00:00:00+00:00",
        evidence=[upstream_card],
        bundled_third_party=True,
        model_generated_outputs=True,
        decisions=[
            LicenseDecision(
                approved=True,
                who="user",
                when="2026-11-16T06:37:25-05:00",
                source_id="TaskTrove:example",
                dataset_id="open-athena/example",
                kind=LicenseDecisionKind.EXCEPTION,
                rationale="The user approved lifting the standing unverified-origin ban.",
                revision="e" * 40,
                scope=["train"],
                evidence_urls=["https://huggingface.co/datasets/SWE-Gym/SWE-Gym"],
                issue_url="https://github.com/marin-community/marin/issues/11497",
            )
        ],
        resolved_source=LicenseSourcePin(
            dataset_id="SWE-Gym/SWE-Gym",
            revision="e" * 40,
            url="https://huggingface.co/datasets/SWE-Gym/SWE-Gym/tree/" + "e" * 40,
            declaration=upstream_card,
        ),
    )
    bound = repin_license_receipt(previous, "open-athena/example-taskspec-1", "f" * 40, card)
    assert [decision.kind for decision in bound.decisions] == [LicenseDecisionKind.EXCEPTION]
    assert bound.resolved_source.dataset_id == "SWE-Gym/SWE-Gym"
    assert bound.dataset_id == "open-athena/example-taskspec-1"
    assert any(item.kind == "wrapper_source" for item in bound.evidence)
    lineage = native_source_conversion.exception_lineage(bound)
    assert "2026-11-16" in lineage and "train" in lineage and "mit" in lineage


def test_resolve_base_digest_retries_429_honoring_retry_after(monkeypatch):
    """Docker Hub rate limits are expected at R2E-Gym's scale: the resolver retries
    bounded on 429 and honors the server's Retry-After (ruling 2026-11-26)."""
    sleeps = []
    monkeypatch.setattr(native_source_conversion.time, "sleep", sleeps.append)
    attempts = []

    class FakeResponse:
        def __init__(self, headers):
            self.headers = headers

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"token": "t"}'

    def fake_urlopen(request, timeout=None):
        attempts.append(request)
        if len(attempts) < 3:
            url = getattr(request, "full_url", request)
            raise urllib.error.HTTPError(url, 429, "rate limited", {"Retry-After": "7"}, None)
        return FakeResponse({"Docker-Content-Digest": "sha256:" + "e" * 64})

    monkeypatch.setattr(native_source_conversion.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(native_source_conversion, "_DOCKER_HUB_TOKEN", None)
    monkeypatch.setattr(native_source_conversion, "_DIGEST_CACHE", {})
    pinned = native_source_conversion._resolve_base_digest("namanjain12/orange3_final:deadbeef")
    assert pinned == "namanjain12/orange3_final@sha256:" + "e" * 64
    assert len(attempts) == 4  # 2 rate-limited (token + manifest retries interleave), then success
    assert 7.0 in sleeps


def test_resolve_base_digest_remints_the_token_when_a_throttle_outlives_its_ttl(monkeypatch):
    """The native-conversion-r2e-gym-v1-v7 failure: the bearer is minted once and
    reused across a throttle window that outlives the token's 300-second TTL, so the
    registry answers the next admitted request with 401 and the attempt dies. The
    manifest fetch must re-mint before the TTL passes."""
    clock = [1000.0]
    monkeypatch.setattr(native_source_conversion.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    monkeypatch.setattr(native_source_conversion.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(native_source_conversion, "_DOCKER_HUB_TOKEN", None)
    monkeypatch.setattr(native_source_conversion, "_DIGEST_CACHE", {})
    exchanges = []
    manifest_attempts = []

    class FakeResponse:
        def __init__(self, headers, body=b""):
            self.headers = headers
            self._body = body

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return self._body

    def fake_urlopen(request, timeout=None):
        url = request if isinstance(request, str) else request.full_url
        if "/token?" in url:
            token = f"t{len(exchanges)}"
            exchanges.append(token)
            return FakeResponse({}, json.dumps({"token": token}).encode())
        manifest_attempts.append(request.headers["Authorization"])
        if len(manifest_attempts) == 1:
            raise urllib.error.HTTPError(url, 429, "rate limited", {"Retry-After": "600"}, None)
        return FakeResponse({"Docker-Content-Digest": "sha256:" + "e" * 64})

    monkeypatch.setattr(native_source_conversion.urllib.request, "urlopen", fake_urlopen)
    pinned = native_source_conversion._resolve_base_digest("namanjain12/sympy_final:deadbeef")
    assert pinned == "namanjain12/sympy_final@sha256:" + "e" * 64
    assert manifest_attempts == ["Bearer t0", "Bearer t1"]  # the second request carries a re-minted token
    assert len(exchanges) == 2
    assert clock[0] == 1600.0  # the throttle window was slept out, not died in


def test_a_registry_401_fails_closed_without_a_retry_storm(monkeypatch):
    """A 401 against a token minted within the TTL is an identity refusal, not a
    throttle: it is a typed rejection raised immediately, with no re-mint loop."""
    monkeypatch.setattr(
        native_source_conversion.time,
        "sleep",
        lambda seconds: (_ for _ in ()).throw(AssertionError("a refused identity must not sleep-retry")),
    )
    monkeypatch.setattr(native_source_conversion, "_DOCKER_HUB_TOKEN", None)
    monkeypatch.setattr(native_source_conversion, "_DIGEST_CACHE", {})
    calls = []

    class FakeResponse:
        def __init__(self, body):
            self.headers = {}
            self._body = body

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return self._body

    def fake_urlopen(request, timeout=None):
        calls.append(request if isinstance(request, str) else request.full_url)
        if isinstance(request, str):
            return FakeResponse(json.dumps({"token": "t0"}).encode())
        raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", {}, None)

    monkeypatch.setattr(native_source_conversion.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(BasePinRejected, match="refused to serve"):
        native_source_conversion._resolve_base_digest("namanjain12/sympy_final:deadbeef")
    assert len(calls) == 2  # one token exchange, one manifest request, nothing else


def test_refused_credentials_fail_closed_at_the_token_exchange(monkeypatch):
    """Expired or wrong submit-time credentials are a typed rejection naming the
    exchange — never a retry storm and never an error that reads like a missing tag."""
    monkeypatch.setenv("DOCKERHUB_USERNAME", "builder")
    monkeypatch.setenv("DOCKERHUB_TOKEN", "expired-secret")
    monkeypatch.setattr(native_source_conversion, "_DOCKER_HUB_TOKEN", None)
    monkeypatch.setattr(native_source_conversion, "_DIGEST_CACHE", {})
    calls = []

    def fake_urlopen(request, timeout=None):
        calls.append(request if isinstance(request, str) else request.full_url)
        url = request if isinstance(request, str) else request.full_url
        raise urllib.error.HTTPError(url, 401, "Unauthorized", {}, None)

    monkeypatch.setattr(native_source_conversion.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(BasePinRejected, match="submit-time Docker Hub credentials"):
        native_source_conversion._resolve_base_digest("namanjain12/sympy_final:deadbeef")
    assert len(calls) == 1  # the exchange alone; no manifest attempt follows a refused identity


def test_a_missing_base_image_is_a_typed_per_task_exclusion(monkeypatch):
    """A tag that no longer exists upstream excludes its row with evidence — never a
    substitute image, never a silently shrunken population (ruling 2026-11-26)."""
    source = conversion_source(
        {
            "id": "TaskTrove:example",
            "name": "example",
            "dataset_id": "SWE-Gym/SWE-Gym",
            "dataset_revision": "e" * 40,
            "disposition_release": {
                "repo": "open-athena/example",
                "revision": "e" * 40,
                "file_hashes": {"tasks.parquet": "a" * 64},
            },
        },
        None,
    )
    task = _package(DOCKERFILE)
    rows = [{"task_spec": task.model_dump_json()}]

    def missing(reference: str) -> str:
        raise native_source_conversion.BasePinMissing(f"The base image {reference} no longer exists upstream")

    monkeypatch.setattr(native_source_conversion, "load_release_population", lambda source, limit: rows)
    monkeypatch.setattr(
        native_source_conversion,
        "convert_native_source_row",
        lambda task, input_task_sha256, resolve=None: (_ for _ in ()).throw(
            native_source_conversion.BasePinMissing("The base image python:3.9-bookworm no longer exists upstream")
        ),
    )
    converted, receipts, counts = native_source_conversion.convert_population(source, limit=None)
    assert converted == [] and receipts == []
    assert counts["input_rows"] == 1 and counts["converted_rows"] == 0 and counts["excluded_rows"] == 1
    assert counts["exclusions"][0]["reason"] == "base_image_missing"
    assert counts["exclusions"][0]["input_task_sha256"] == canonical_sha256(task.model_dump(mode="json"))
    assert counts["converted_rows"] + counts["excluded_rows"] == counts["input_rows"]

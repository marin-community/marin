# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import httpx
import pytest
from finestore.rl.licenses import (
    LicenseBan,
    LicenseCheck,
    LicenseCheckRule,
    LicenseCheckVerdict,
    LicenseDecision,
    LicenseReceipt,
    LicenseSourcePin,
)
from huggingface_hub.errors import HfHubHTTPError

from experiments.post_training.task_curation import license_publication
from experiments.post_training.task_curation.license_resolution import license_evidence
from experiments.post_training.task_curation.tests.license_publication_fixtures import (
    DATASET,
    PREFIX,
    REVISION,
    source_receipt,
)
from infra.marina.apps.rl_data_catalog.server import licensing


def _conflict_response() -> httpx.Response:
    """The Hub's already-created-repo refusal, as the API raises it."""
    return httpx.Response(409, request=httpx.Request("POST", "https://huggingface.co/api/repos/create"))


@pytest.fixture(autouse=True)
def _clear_ruling_cache():
    """Ban lookups cache the pinned rulings; a staged registry must not leak past its test."""
    yield
    getattr(licensing.license_bans, "cache_clear", lambda: None)()


def test_new_private_pin_cannot_hide_a_resolved_upstream_ban(monkeypatch):
    card = b"---\nlicense: apache-2.0\n---\n"
    receipt = source_receipt(card).model_copy(
        update={
            "dataset_id": "open-athena/input",
            "revision": "b" * 40,
            "resolved_source": LicenseSourcePin(
                dataset_id=DATASET,
                revision=REVISION,
                url=f"https://huggingface.co/datasets/{DATASET}/tree/{REVISION}",
                declaration=license_evidence("https://example.test/wrapper", b"Source pin", "wrapper_source_pin"),
            ),
        }
    )
    ban = LicenseBan(
        source_id="fixture:banned-upstream",
        upstream_id=DATASET,
        upstream_revision=REVISION,
        classification="unknown",
        license_ids=["unknown"],
        reason="Recorded upstream ban",
        evidence_urls=[PREFIX + "README.md"],
        decided_at="2026-11-15T00:00:00+00:00",
        decision_by="Fixture user ruling",
        issue_url="https://github.com/marin-community/marin/issues/11497",
    )
    monkeypatch.setattr(
        licensing,
        "read_registry_json",
        lambda _reference: {"version": "fixture", "bans": [ban.model_dump(mode="json")]},
    )
    licensing.license_bans.cache_clear()
    monkeypatch.setattr(license_publication, "resolve_license", lambda target: receipt)
    with pytest.raises(ValueError, match="Recorded upstream ban"):
        license_publication.check_publication(receipt.source_id, receipt.dataset_id, receipt.revision, "train")
    with pytest.raises(ValueError, match="Recorded upstream ban"):
        license_publication.require_publication_license(
            receipt, receipt.source_id, receipt.dataset_id, receipt.revision, "train"
        )


def lineage_receipt(revision: str, scope: list[str], *, with_source: bool = True) -> LicenseReceipt:
    """An escalating receipt at one pin, carrying a typed lineage exception decision."""
    receipt = source_receipt(b"---\nlicense: other\n---\nCustom terms document.\n").model_copy(
        update={"revision": revision}
    )
    if with_source:
        receipt = receipt.model_copy(
            update={
                "resolved_source": LicenseSourcePin(
                    dataset_id="fixture/upstream-parent",
                    revision="d" * 40,
                    url=f"https://huggingface.co/datasets/fixture/upstream-parent/tree/{'d' * 40}",
                    declaration=license_evidence("https://example.test/declaration", b"declared", "wrapper_source"),
                )
            }
        )
    decision = LicenseDecision(
        kind="exception",
        approved=True,
        who="Fixture user ruling",
        when="2026-11-15T00:00:00+00:00",
        source_id=receipt.source_id,
        dataset_id=receipt.dataset_id,
        revision=revision,
        scope=scope,
        rationale="Fixture lineage exception for the user-cleared parent",
        evidence_urls=["https://huggingface.co/datasets/fixture/upstream-parent/tree/" + "d" * 40],
        issue_url="https://github.com/marin-community/marin/issues/11497",
    )
    return receipt.model_copy(update={"decisions": [decision]})


def registered_lineage(monkeypatch, receipt: LicenseReceipt) -> None:
    monkeypatch.setattr(license_publication, "resolve_license", lambda target: receipt)
    monkeypatch.setattr(license_publication, "source_license_ban", lambda source, dataset, actual: None)
    monkeypatch.setattr(license_publication, "read_license_receipts", lambda path: {receipt.source_id: receipt})


def test_check_publication_passes_on_the_recorded_lineage_exception(monkeypatch):
    receipt = lineage_receipt(REVISION, ["train"])
    registered_lineage(monkeypatch, receipt)
    evidence = license_publication.check_publication(receipt.source_id, receipt.dataset_id, receipt.revision, "train")
    assert evidence["verdict"] == "PASS"
    assert evidence["rule_id"] == "E1"
    assert evidence["lineage_exception"]["scope"] == ["train"]
    assert evidence["lineage_exception"]["approved"] is True


def test_check_publication_clears_a_repackager_fail_on_the_folder_lineage(monkeypatch):
    """The repackager ruling: the folder lineage carries the pass at a repackager pin."""
    receipt = lineage_receipt(REVISION, ["train"])
    registered_lineage(monkeypatch, receipt)
    failing = LicenseCheck(
        source_id=receipt.source_id,
        dataset_id=receipt.dataset_id,
        revision=receipt.revision,
        verdict=LicenseCheckVerdict.FAIL,
        rule_id=LicenseCheckRule.MISSING,
        quote="L1: ---",
        evidence_url=PREFIX + "README.md",
    )
    monkeypatch.setattr(license_publication, "check_license", lambda actual, ban=None: failing)
    lineage = {
        receipt.source_id: {
            "source_id": receipt.source_id,
            "dataset_id": receipt.dataset_id,
            "revision": receipt.revision,
            "decisions": [
                {
                    "kind": "exception",
                    "approved": True,
                    "rationale": "explicitly accepts the folder's upstream as MIT.",
                    "evidence_urls": [PREFIX + "endless_v2"],
                }
            ],
            "evidence": [],
        }
    }
    evidence = license_publication.check_publication(
        receipt.source_id, receipt.dataset_id, receipt.revision, "train", lineage=lineage
    )
    assert evidence["verdict"] == "PASS"
    assert evidence["folder_lineage"]["identifier"] == "MIT"
    assert evidence["folder_lineage"]["upstream"] == f"{receipt.dataset_id}@{receipt.revision}"


@pytest.mark.parametrize(
    ("use", "stage"),
    [
        pytest.param(
            "train",
            lambda registered, monkeypatch: monkeypatch.setattr(
                license_publication,
                "read_license_receipts",
                lambda path: {registered.source_id: registered},
            ),
            id="another-pin",
        ),
        pytest.param("eval", lambda registered, monkeypatch: None, id="out-of-scope"),
        pytest.param(
            "train",
            lambda registered, monkeypatch: monkeypatch.setattr(
                license_publication, "read_license_receipts", lambda path: {}
            ),
            id="missing-receipt",
        ),
    ],
)
def test_check_publication_ignores_a_lineage_exception(tmp_path, monkeypatch, use, stage):
    """A lineage exception clears a FAIL only for the same pin, in use scope, with a receipt."""
    receipt = lineage_receipt(REVISION, ["train"])
    registered = lineage_receipt("e" * 40, ["train"])
    registered_lineage(monkeypatch, receipt)
    stage(registered, monkeypatch)
    with pytest.raises(ValueError, match="Published pin failed"):
        license_publication.check_publication(receipt.source_id, receipt.dataset_id, receipt.revision, use)


def test_check_publication_never_clears_a_restricted_pin(monkeypatch):
    receipt = source_receipt(b"---\nlicense: cc-by-nc-4.0\n---\n").model_copy(
        update={
            "resolved_source": LicenseSourcePin(
                dataset_id="fixture/upstream-parent",
                revision="d" * 40,
                url=f"https://huggingface.co/datasets/fixture/upstream-parent/tree/{'d' * 40}",
                declaration=license_evidence("https://example.test/declaration", b"declared", "wrapper_source"),
            )
        }
    )
    decision = LicenseDecision(
        kind="exception",
        approved=True,
        who="Fixture user ruling",
        when="2026-11-15T00:00:00+00:00",
        source_id=receipt.source_id,
        dataset_id=receipt.dataset_id,
        revision=receipt.revision,
        scope=["train"],
        rationale="Fixture lineage exception",
        evidence_urls=["https://huggingface.co/datasets/fixture/upstream-parent/tree/" + "d" * 40],
        issue_url="https://github.com/marin-community/marin/issues/11497",
    )
    receipt = receipt.model_copy(update={"decisions": [decision]})
    registered_lineage(monkeypatch, receipt)
    with pytest.raises(ValueError, match="Published pin failed"):
        license_publication.check_publication(receipt.source_id, receipt.dataset_id, receipt.revision, "train")


@pytest.mark.parametrize("revision", [REVISION, "b" * 40])
def test_publication_license_gate_calls_current_checker_with_exact_pin(revision):
    receipt = source_receipt(b"---\nlicense: apache-2.0\n---\n")
    if revision != REVISION:
        with pytest.raises(ValueError, match="another source revision"):
            license_publication.require_publication_license(receipt, receipt.source_id, DATASET, revision, "train")
        return
    license_publication.require_publication_license(receipt, receipt.source_id, DATASET, revision, "train")


def test_private_publication_reuses_its_own_prior_attempt_repo(tmp_path, monkeypatch):
    card = b"---\nlicense: mit\n---\n# Dataset\n"
    receipt = source_receipt(card)
    (tmp_path / "README.md").write_bytes(card)
    (tmp_path / "tasks.parquet").write_bytes(b"frozen source task bytes")
    repo = "open-athena/fixture-publication"
    writes = []

    class Hub:
        def create_repo(self, **kwargs):
            assert kwargs["exist_ok"] is False
            raise HfHubHTTPError("409 Conflict: the repo already exists", response=_conflict_response())

        def list_repo_tree(self, dataset, *, repo_type, recursive):
            assert dataset == repo and recursive is True
            return [SimpleNamespace(path="README.md")]

        def repo_info(self, dataset, **kwargs):
            assert dataset == repo
            return SimpleNamespace(private=True, sha="c" * 40)

        def upload_folder(self, **kwargs):
            writes.append("upload")
            return SimpleNamespace(oid="d" * 40)

    monkeypatch.setattr(license_publication, "HfApi", Hub)
    monkeypatch.setattr(license_publication, "hf_hub_download", lambda repo, name, **kwargs: str(tmp_path / name))
    monkeypatch.setattr(license_publication, "check_publication", lambda *args, **kwargs: {"verdict": "PASS"})
    prepared = license_publication.prepare_private_dataset(
        receipt.source_id, repo, tmp_path, ("README.md", "tasks.parquet"), receipt, use="train"
    )
    result = license_publication.publish_private_dataset(prepared)
    assert writes == ["upload"] and result.revision == "d" * 40


def test_private_publication_rejects_a_repo_with_other_content(tmp_path, monkeypatch):
    card = b"---\nlicense: mit\n---\n# Dataset\n"
    receipt = source_receipt(card)
    (tmp_path / "README.md").write_bytes(card)
    (tmp_path / "tasks.parquet").write_bytes(b"frozen source task bytes")
    repo = "open-athena/fixture-publication"

    class Hub:
        def create_repo(self, **kwargs):
            raise HfHubHTTPError("409 Conflict", response=_conflict_response())

        def list_repo_tree(self, dataset, *, repo_type, recursive):
            return [SimpleNamespace(path="README.md"), SimpleNamespace(path="stale.parquet")]

        def repo_info(self, dataset, **kwargs):
            return SimpleNamespace(private=True, sha="c" * 40)

    monkeypatch.setattr(license_publication, "HfApi", Hub)
    monkeypatch.setattr(license_publication, "check_publication", lambda *args, **kwargs: {"verdict": "PASS"})
    prepared = license_publication.prepare_private_dataset(
        receipt.source_id, repo, tmp_path, ("README.md", "tasks.parquet"), receipt, use="train"
    )
    with pytest.raises(ValueError, match="already exists with other content"):
        license_publication.publish_private_dataset(prepared)

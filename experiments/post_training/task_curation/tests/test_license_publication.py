# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from types import SimpleNamespace

import pytest
from finestore.rl.licenses import (
    LICENSE_RESOLVER_VERSION,
    LicenseBan,
    LicenseCheck,
    LicenseCheckRule,
    LicenseCheckVerdict,
    LicenseDecision,
    LicenseReceipt,
    LicenseSourcePin,
)
from finestore.rl.status_taxonomy import DataRestriction
from huggingface_hub import DatasetCard
from huggingface_hub.hf_api import RepoFile
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation import license_publication, license_resolution, native_quick
from experiments.post_training.task_curation.license_resolution import license_evidence
from infra.marina.apps.rl_data_catalog.server import license_registry, licensing

DATASET = "fixture/upstream"
REVISION = "a" * 40
PREFIX = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/"


def source_receipt(card):
    url = PREFIX + "README.md"
    return LicenseReceipt(
        source_id="fixture:source",
        dataset_id=DATASET,
        revision=REVISION,
        resolved_at="2026-11-15T00:00:00+00:00",
        resolver_version=LICENSE_RESOLVER_VERSION,
        evidence=[
            license_evidence(url, card, "dataset_card"),
            license_evidence(url, card, "readme_license"),
            license_evidence(url, card, "readme_notice"),
        ],
        bundled_third_party=False,
        model_generated_outputs=False,
    )


def snapshot_hub(tmp_path, monkeypatch, files):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    for name, content in files.items():
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

    class SnapshotHub:
        def list_repo_tree(self, repo, *, repo_type, revision, recursive):
            # Folder-lineage upstreams keep license documents in subfolders, so the
            # publication listing is recursive and the fixture names are flat paths.
            assert (repo, repo_type, revision, recursive) == (DATASET, "dataset", REVISION, True)
            return [RepoFile(path=name, size=len(data), oid="b" * 40) for name, data in files.items()]

    def download(repo, name, *, repo_type, revision):
        assert (repo, repo_type, revision) == (DATASET, "dataset", REVISION)
        return str(snapshot / name)

    monkeypatch.setattr(license_publication, "HfApi", SnapshotHub)
    monkeypatch.setattr(license_publication, "hf_hub_download", download)
    return snapshot


@pytest.mark.parametrize("use", ["train", "eval"])
def test_private_publication_prepares_without_writes_then_checks_uploaded_pin(tmp_path, monkeypatch, use):
    card = b"---\nlicense: mit\n---\n# Dataset\n"
    receipt = source_receipt(card)
    (tmp_path / "README.md").write_bytes(card)
    (tmp_path / "tasks.parquet").write_bytes(b"frozen source task bytes")
    repo = "open-athena/fixture-publication"
    revision = "b" * 40
    writes = []

    class Hub:
        def dataset_info(self, dataset, *, revision):
            return SimpleNamespace(sha=revision, siblings=[SimpleNamespace(rfilename="README.md")])

        def create_repo(self, **kwargs):
            assert kwargs == {"repo_id": repo, "repo_type": "dataset", "private": True, "exist_ok": False}
            writes.append("create")

        def list_repo_tree(self, dataset, *, repo_type, recursive):
            assert dataset == repo and repo_type == "dataset" and recursive is True
            return [SimpleNamespace(path=name) for name in ()]

        def repo_info(self, dataset, **kwargs):
            assert dataset == repo and kwargs["repo_type"] == "dataset"
            return SimpleNamespace(private=True, sha=kwargs.get("revision", "c" * 40))

        def upload_folder(self, **kwargs):
            assert kwargs["repo_id"] == repo and kwargs["folder_path"] == tmp_path
            assert set(kwargs["allow_patterns"]) == {"README.md", "tasks.parquet"}
            assert kwargs["parent_commit"] == "c" * 40
            writes.append("upload")
            return SimpleNamespace(oid=revision)

    def download(dataset, name, *, repo_type, revision):
        assert (dataset, repo_type, revision) == (repo, "dataset", "b" * 40)
        return str(tmp_path / name)

    monkeypatch.setattr(license_publication, "HfApi", Hub)
    monkeypatch.setattr(license_resolution, "HfApi", Hub)
    monkeypatch.setattr(license_resolution, "_fetch", lambda url, **kwargs: card)
    monkeypatch.setattr(license_publication, "hf_hub_download", download)
    monkeypatch.setattr(licensing, "license_bans", lambda: {})
    prepared = license_publication.prepare_private_dataset(
        receipt.source_id, repo, tmp_path, ("README.md", "tasks.parquet"), receipt, use=use
    )
    assert writes == []
    result = license_publication.publish_private_dataset(prepared)
    assert writes == ["create", "upload"]
    assert result.revision == revision and result.license_check["verdict"] == "PASS"
    assert result.license_check["receipt"]["dataset_id"] == repo
    assert result.license_check["receipt"]["revision"] == revision


def test_hub_license_card_value_maps_unknown_identifiers_to_other():
    """The Hub's card schema rejects identifiers outside its own taxonomy
    outright (the OpenSWE taskspec publication's bsd-2-clause-patent
    evidence); the card's license summary maps them to "other" — the repo
    ships the upstream LICENSE notice — while the checked receipt keeps the
    exact set."""
    value = license_publication.hub_license_card_value(
        ["apache-2.0", "bsd-2-clause", "bsd-2-clause-patent", "cc0-1.0", "mit"]
    )
    assert value == ["apache-2.0", "bsd-2-clause", "cc0-1.0", "mit", "other"]
    assert license_publication.hub_license_card_value(["mit"]) == "mit"
    with pytest.raises(ValueError):
        license_publication.hub_license_card_value(["made-up-licence-1"])


def test_publication_card_maps_unknown_identifiers_to_other(tmp_path, monkeypatch):
    card = (
        b"---\nlicense:\n  - mit\n  - bsd-2-clause-patent\n---\n# Dataset\n\n## License\n"
        b"Licensed under MIT and the BSD 2-Clause Patent licence.\n"
    )
    snapshot_hub(tmp_path, monkeypatch, {"README.md": card, "LICENSE": b"MIT + BSD-2-Clause-Patent text\n"})
    receipt = source_receipt(card)
    publication = license_publication.prepare_license_publication(receipt, tmp_path / "published", use="train")
    # The card's license summary is Hub-valid: the unknown identifier becomes
    # "other"; the checked receipt keeps the exact set.
    assert publication.metadata["license"] == ["mit", "other"]
    evidence_ids = {identifier for item in receipt.evidence for identifier in item.license_ids}
    assert "bsd-2-clause-patent" in evidence_ids
    files = {path.name: path.read_bytes() for path in publication.notice_files}
    assert files["LICENSE"] == b"MIT + BSD-2-Clause-Patent text\n"


def test_publication_states_folder_lineage_license(tmp_path, monkeypatch):
    card = b"---\nlicense: apache-2.0\n---\n# Dataset\n\n## License\nLicensed under Apache License 2.0.\n"
    """The repackager ruling: the card states each included folder's upstream license."""
    folder_notice = b"Folder-lineage attribution.\n"
    snapshot_hub(
        tmp_path, monkeypatch, {"README.md": card, "endless_v2/README.md": card, "endless_v2/NOTICE": folder_notice}
    )
    lineage = {
        "FWeindel:validated-tasks/endless_v2": {
            "source_id": "FWeindel:validated-tasks/endless_v2",
            "dataset_id": "FWeindel/validated-tasks",
            "revision": REVISION,
            "decisions": [
                {
                    "kind": "exception",
                    "approved": True,
                    "rationale": "accepts Endless v1-v3 from obiwan96/endless-terminals@2" * 40 + " as MIT.",
                    "evidence_urls": [PREFIX + "endless_v2"],
                }
            ],
            "evidence": [
                {
                    "url": PREFIX + "endless_v2/README.md",
                    "kind": "dataset_card",
                    "sha256": hashlib.sha256(card).hexdigest(),
                    "license_ids": ["missing"],
                }
            ],
        },
    }
    publication = license_publication.prepare_license_publication(
        source_receipt(card),
        tmp_path / "published",
        use="train",
        lineage=lineage,
        folders=["FWeindel:validated-tasks/endless_v2"],
    )
    assert publication.metadata["license"] == "mit"
    assert len(publication.license_table) == 1
    assert publication.license_table[0]["identifier"] == "MIT"
    assert publication.license_table[0]["folder"] == "FWeindel:validated-tasks/endless_v2"
    files = {path.name: path.read_bytes() for path in publication.notice_files}
    assert files["UPSTREAM_endless_v2_NOTICE"] == folder_notice


def test_publication_states_other_for_divergent_folder_licenses(tmp_path, monkeypatch):
    card = b"---\nlicense: apache-2.0\n---\n# Dataset\n\n## License\nLicensed under Apache License 2.0.\n"
    """Several folders with different accepted identifiers publish license: other."""
    snapshot_hub(tmp_path, monkeypatch, {"README.md": card, "endless_v2/README.md": card})
    lineage = {}
    for folder, identifier in (("endless_v2", "MIT"), ("endless_v3", "Apache-2.0")):
        lineage[f"FWeindel:validated-tasks/{folder}"] = {
            "source_id": f"FWeindel:validated-tasks/{folder}",
            "dataset_id": "FWeindel/validated-tasks",
            "revision": REVISION,
            "decisions": [
                {
                    "kind": "exception",
                    "approved": True,
                    "rationale": f"explicitly accepts the folder as {identifier}.",
                    "evidence_urls": [PREFIX + folder],
                }
            ],
            "evidence": [],
        }
    publication = license_publication.prepare_license_publication(
        source_receipt(card), tmp_path / "published", use="train", lineage=lineage, folders=sorted(lineage)
    )
    assert publication.metadata["license"] == "other"
    assert {row["identifier"] for row in publication.license_table} == {"MIT", "Apache-2.0"}


def test_lineage_folder_identifier_requires_an_approved_decision():
    with pytest.raises(ValueError, match="lacks an approved decision"):
        license_publication.lineage_folder_identifier(
            {
                "source_id": "fixture:folder",
                "decisions": [{"kind": "exception", "approved": False, "rationale": "accepts it as MIT"}],
            }
        )
    with pytest.raises(ValueError, match="does not name"):
        license_publication.lineage_folder_identifier(
            {
                "source_id": "fixture:folder",
                "decisions": [{"kind": "exception", "approved": True, "rationale": "no identifier here"}],
            }
        )


def test_publication_preserves_folder_lineage_license_documents(tmp_path, monkeypatch):
    """Folder-lineage upstreams keep license documents in per-dataset subfolders."""
    card = b"---\nlicense: apache-2.0\n---\n# Dataset\n\n## License\nLicensed under Apache License 2.0.\n"
    folder_notice = b"Folder-lineage attribution.\n"
    snapshot_hub(
        tmp_path, monkeypatch, {"README.md": card, "endless_v2/README.md": card, "endless_v2/NOTICE": folder_notice}
    )
    url = PREFIX + "endless_v2/README.md"
    receipt = source_receipt(card).model_copy(
        update={
            "evidence": [
                *source_receipt(card).evidence,
                license_evidence(url, card, "dataset_card"),
            ],
        }
    )
    publication = license_publication.prepare_license_publication(receipt, tmp_path / "published", use="train")
    files = {path.name: path.read_bytes() for path in publication.notice_files}
    assert files["UPSTREAM_README.md"] == card
    assert files["UPSTREAM_endless_v2_README.md"] == card
    assert files["UPSTREAM_endless_v2_NOTICE"] == folder_notice
    provenance = json.loads(files["publication-provenance.json"])
    assert {row["filename"]: row["source_url"] for row in provenance["documents"]} == {
        "UPSTREAM_README.md": PREFIX + "README.md",
        "UPSTREAM_endless_v2_README.md": PREFIX + "endless_v2/README.md",
        "UPSTREAM_endless_v2_NOTICE": PREFIX + "endless_v2/NOTICE",
    }


@pytest.mark.parametrize(
    "card,expected",
    [
        (
            b"---\nlicense: other\nlicense_name: cc0-with-bytedance-notice\n"
            b"license_link: https://example.test/notice\n---\n"
            b"CC0 1.0 with attribution notice.\nCopyright Example Authors.\n",
            {
                "license": "cc0-1.0",
                "license_name": "cc0-with-bytedance-notice",
                "license_link": "https://example.test/notice",
            },
        ),
        (
            b"---\nlicense: apache-2.0\n---\n# Dataset\n\n## License\nLicensed under Apache License 2.0.\n",
            {"license": "apache-2.0"},
        ),
    ],
)
def test_publication_preserves_custom_notice_and_grant_when_readme_evidence_repeats(
    tmp_path, monkeypatch, card, expected
):
    notice = b"Preserve this original attribution exactly.\n"
    snapshot_hub(tmp_path, monkeypatch, {"README.md": card, "NOTICE": notice})
    publication = license_publication.prepare_license_publication(
        source_receipt(card), tmp_path / "published", use="train"
    )
    rendered = license_publication.licensed_dataset_card("# Private retained tasks", publication, pretty_name="Fixture")
    metadata = DatasetCard(rendered).data.to_dict()
    assert {key: metadata[key] for key in expected} == expected
    assert metadata["license_evidence_urls"] == [PREFIX + "README.md"]
    assert metadata["license_source"] == {
        "dataset_id": DATASET,
        "revision": REVISION,
        "url": f"https://huggingface.co/datasets/{DATASET}/tree/{REVISION}",
    }
    files = {path.name: path.read_bytes() for path in publication.notice_files}
    assert files["UPSTREAM_README.md"] == card
    assert files["NOTICE"] == notice
    provenance = json.loads(files["publication-provenance.json"])
    assert {row["filename"]: row["sha256"] for row in provenance["documents"]} == {
        "UPSTREAM_README.md": hashlib.sha256(card).hexdigest(),
        "NOTICE": hashlib.sha256(notice).hexdigest(),
    }


def test_pass_receipt_card_states_the_checker_verified_license(tmp_path, monkeypatch):
    card = (
        b"---\nlicense: other\nlicense_name: cc0-with-bytedance-notice\n---\n"
        b"License mirrors upstream: ByteDance licenses the dataset under CC0.\n"
    )
    snapshot_hub(tmp_path, monkeypatch, {"README.md": card})
    publication = license_publication.prepare_license_publication(
        source_receipt(card), tmp_path / "published", use="train"
    )
    assert publication.metadata["license"] == "cc0-1.0"
    assert b"license: other" in (tmp_path / "published" / "UPSTREAM_README.md").read_bytes()


def test_publication_license_ids_use_the_hub_card_spelling(tmp_path, monkeypatch):
    card = b"---\nlicense: Apache-2.0\n---\nBody cites the Apache License 2.0 grant.\n"
    snapshot_hub(tmp_path, monkeypatch, {"README.md": card})
    publication = license_publication.prepare_license_publication(
        source_receipt(card), tmp_path / "published", use="train"
    )
    assert publication.metadata["license"] == "apache-2.0"


def test_changed_license_bytes_fail_before_publication_provenance_is_written(tmp_path, monkeypatch):
    card = b"---\nlicense: apache-2.0\n---\n"
    snapshot_hub(tmp_path, monkeypatch, {"README.md": card + b"changed"})
    with pytest.raises(ValueError, match="differs from its checked receipt"):
        license_publication.prepare_license_publication(source_receipt(card), tmp_path / "published", use="train")
    assert not (tmp_path / "published/publication-provenance.json").exists()


def package_receipt(wheel: str, wheel_sha: str, metadata: dict) -> LicenseReceipt:
    url = "https://pypi.org/pypi/reasoning_gym/0.1.25/json"
    content = json.dumps(metadata).encode()
    return LicenseReceipt(
        source_id="Reasoning Gym:reasoning_gym",
        dataset_id=wheel,
        revision=wheel_sha,
        resolved_at="2026-11-15T00:00:00+00:00",
        resolver_version=LICENSE_RESOLVER_VERSION,
        evidence=[license_evidence(url, content, "package_metadata")],
        bundled_third_party=False,
        model_generated_outputs=False,
    )


def test_package_grant_publication_declares_the_checked_wheel_pin(tmp_path, monkeypatch):
    wheel = "reasoning_gym-0.1.25-py3-none-any.whl"
    wheel_sha = "c" * 64
    wheel_url = f"https://files.pythonhosted.org/packages/42/21/deadbeef/{wheel}"
    metadata = {
        "info": {"license_expression": "Apache-2.0"},
        "urls": [{"filename": wheel, "digests": {"sha256": wheel_sha}, "url": wheel_url}],
    }
    content = json.dumps(metadata).encode()
    receipt = package_receipt(wheel, wheel_sha, metadata)
    monkeypatch.setattr(license_publication, "_fetch", lambda url, **kwargs: content)
    publication = license_publication.prepare_license_publication(receipt, tmp_path / "published", use="train")
    assert publication.metadata["license"] == "apache-2.0"
    assert publication.metadata["license_source"] == {"dataset_id": wheel, "revision": wheel_sha, "url": wheel_url}
    assert publication.metadata["license_evidence_urls"] == [receipt.evidence[0].url]
    assert publication.metadata["license_input"] == {"dataset_id": wheel, "revision": wheel_sha}
    preserved = tmp_path / "published" / "UPSTREAM_PACKAGE_METADATA.json"
    assert preserved.read_bytes() == content
    assert not (tmp_path / "published" / "README.md").exists()


def test_package_grant_publication_rejects_unbound_or_moved_wheels(tmp_path, monkeypatch):
    wheel = "reasoning_gym-0.1.25-py3-none-any.whl"
    wheel_url = f"https://files.pythonhosted.org/packages/42/21/deadbeef/{wheel}"
    metadata = {
        "info": {"license_expression": "Apache-2.0"},
        "urls": [{"filename": wheel, "digests": {"sha256": "d" * 64}, "url": wheel_url}],
    }
    receipt = package_receipt(wheel, "c" * 64, metadata)
    monkeypatch.setattr(license_publication, "_fetch", lambda url, **kwargs: json.dumps(metadata).encode())
    with pytest.raises(ValueError, match="does not bind the checked wheel"):
        license_publication.prepare_license_publication(receipt, tmp_path / "published", use="train")


def test_package_grant_publication_fails_when_the_checked_metadata_changed(tmp_path, monkeypatch):
    wheel = "reasoning_gym-0.1.25-py3-none-any.whl"
    receipt = package_receipt(wheel, "c" * 64, {"info": {"license_expression": "Apache-2.0"}, "urls": []})
    monkeypatch.setattr(license_publication, "_fetch", lambda url, **kwargs: b"changed")
    with pytest.raises(ValueError, match="changed since its license check"):
        license_publication.prepare_license_publication(receipt, tmp_path / "published", use="train")


@pytest.mark.parametrize("use", ["train", "eval"])
def test_publication_preflight_honors_the_requested_use(tmp_path, monkeypatch, use):
    card = b"---\nlicense: apache-2.0\n---\n"
    snapshot_hub(tmp_path, monkeypatch, {"README.md": card})
    decision = LicenseDecision(
        kind="ban",
        approved=False,
        who="Fixture user ruling",
        when="2026-11-15T00:00:00+00:00",
        source_id="fixture:source",
        dataset_id=DATASET,
        revision=REVISION,
        scope=["train"],
        rationale="Fixture source is permitted only for evaluation",
        evidence_urls=[PREFIX + "README.md"],
        issue_url="https://github.com/marin-community/marin/issues/11497",
    )
    receipt = source_receipt(card).model_copy(update={"decisions": [decision]})
    if use == "train":
        with pytest.raises(ValueError, match="bans this source"):
            license_publication.prepare_license_publication(receipt, tmp_path / "published", use=use)
        return
    publication = license_publication.prepare_license_publication(receipt, tmp_path / "published", use=use)
    assert publication.metadata["license"] == "apache-2.0"


def test_wrapper_publication_uses_resolved_grant_and_preserves_physical_input_pin(tmp_path, monkeypatch):
    card = b"---\nlicense: apache-2.0\n---\n"
    snapshot_hub(tmp_path, monkeypatch, {"README.md": card})
    wrapper = license_evidence(
        "https://huggingface.co/datasets/open-athena/input/resolve/" + "b" * 40 + "/README.md",
        b"Source pin",
        "wrapper_source_pin",
    )
    receipt = source_receipt(card).model_copy(
        update={
            "dataset_id": "open-athena/input",
            "revision": "b" * 40,
            "resolved_source": LicenseSourcePin(
                dataset_id=DATASET,
                revision=REVISION,
                url=f"https://huggingface.co/datasets/{DATASET}/tree/{REVISION}",
                declaration=wrapper,
            ),
        }
    )
    publication = license_publication.prepare_license_publication(receipt, tmp_path / "published", use="train")
    assert publication.metadata["license_source"]["dataset_id"] == DATASET
    assert publication.metadata["license_input"] == {"dataset_id": "open-athena/input", "revision": "b" * 40}


def test_new_private_pin_rechecks_restrictive_terms(tmp_path, monkeypatch):
    receipt = source_receipt(b"---\nlicense: cc-by-nc-4.0\n---\n")
    monkeypatch.setattr(license_publication, "resolve_license", lambda target: receipt)
    monkeypatch.setattr(license_publication, "source_license_ban", lambda source, dataset, actual, use=None: None)
    with pytest.raises(ValueError, match="Published pin failed"):
        license_publication.check_publication(receipt.source_id, "open-athena/new-private-pin", "c" * 40, "train")


def test_native_publication_binds_the_input_receipt_before_reading_tasks(tmp_path):
    """The QUICK publication gate fails closed on missing licence evidence before any task read."""
    entry = {"id": "fixture:source", "dataset_id": DATASET, "dataset_revision": "c" * 40}
    with pytest.raises(ValueError, match="License evidence is missing"):
        native_quick.publish_quick(
            entry,
            tmp_path / "no-task-files",
            "open-athena/new-private-pin",
            None,
            intended_use=IntendedUse.TRAIN,
            data_restriction=DataRestriction.OPEN,
        )


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
        licensing, "LICENSE_BANS", json.dumps({"version": "fixture", "bans": [ban.model_dump(mode="json")]})
    )
    licensing.license_bans.cache_clear()
    monkeypatch.setattr(license_publication, "resolve_license", lambda target: receipt)
    with pytest.raises(ValueError, match="Recorded upstream ban"):
        license_publication.check_publication(receipt.source_id, receipt.dataset_id, receipt.revision, "train")
    with pytest.raises(ValueError, match="Recorded upstream ban"):
        license_publication.require_publication_license(receipt, receipt.source_id, receipt.dataset_id, "train")


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
    monkeypatch.setattr(license_publication, "source_license_ban", lambda source, dataset, actual, use=None: None)
    monkeypatch.setattr(license_publication, "read_license_receipts", lambda path: {receipt.source_id: receipt})


def test_check_publication_passes_on_the_ingest_record_exception(tmp_path, monkeypatch):
    """A user-cleared exception recorded at ingest carries to every revision of the source."""
    receipt = lineage_receipt(REVISION, ["train"])
    registered_lineage(monkeypatch, receipt)
    evidence = license_publication.check_publication(receipt.source_id, receipt.dataset_id, receipt.revision, "train")
    assert evidence["verdict"] == "PASS"
    assert evidence["rule_id"] == "F2"
    assert evidence["ingest_exception"]["scope"] == ["train"]
    assert evidence["ingest_exception"]["approved"] is True


def test_check_publication_clears_a_repackager_fail_on_the_folder_lineage_at_ingest(tmp_path, monkeypatch):
    """The repackager ruling at ingest: the folder lineage carries the pass at a repackager pin."""
    receipt = lineage_receipt(REVISION, ["train"])
    monkeypatch.setattr(license_publication, "read_license_receipts", lambda path: {})
    monkeypatch.setattr(license_publication, "resolve_license", lambda target: receipt)
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


def test_check_publication_ignores_an_exception_out_of_scope(tmp_path, monkeypatch):
    receipt = lineage_receipt(REVISION, ["eval"])
    registered_lineage(monkeypatch, receipt)
    with pytest.raises(ValueError, match="blocks ingest"):
        license_publication.check_publication(receipt.source_id, receipt.dataset_id, receipt.revision, "train")


def test_publication_license_gate_evaluates_the_source_record():
    """The gate looks the licence up by source_id: any revision inherits the record."""
    receipt = source_receipt(b"---\nlicense: apache-2.0\n---\n")
    license_publication.require_publication_license(receipt, receipt.source_id, DATASET, "train")
    with pytest.raises(ValueError, match="belongs to another source"):
        license_publication.require_publication_license(receipt, "Other:source", DATASET, "train")


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
            raise RuntimeError("409 Conflict: the repo already exists")

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
            raise RuntimeError("409 Conflict")

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


def test_licensed_population_publishes_on_the_ingest_record(tmp_path, monkeypatch):
    """The chokepoint publishes a revision on the source's ingest record: the card declares
    the record's provenance, the new pin passes the gate by source_id lookup, and the
    registry is never written per revision (user ruling 2026-11-27)."""
    card = b"---\nlicense: mit\n---\n# Dataset\n"
    ingest = source_receipt(card)
    (tmp_path / "tasks.parquet").write_bytes(b"population bytes")
    repo = "open-athena/fixture-population"
    revision = "d" * 40
    writes = []

    class Hub:
        def dataset_info(self, dataset, *, revision):
            return SimpleNamespace(sha=revision, siblings=[SimpleNamespace(rfilename="README.md")])

        def list_repo_tree(self, dataset, *, repo_type, revision, recursive):
            assert (dataset, repo_type, revision, recursive) == (DATASET, "dataset", REVISION, True)
            return [RepoFile(path="README.md", size=len(card), oid="b" * 40)]

        def create_repo(self, **kwargs):
            assert kwargs == {"repo_id": repo, "repo_type": "dataset", "private": True, "exist_ok": False}
            writes.append("create")

        def repo_info(self, dataset, **kwargs):
            assert dataset == repo and kwargs["repo_type"] == "dataset"
            return SimpleNamespace(private=True, sha=kwargs.get("revision", "c" * 40))

        def upload_folder(self, **kwargs):
            assert set(kwargs["allow_patterns"]) == {
                "README.md",
                "tasks.parquet",
                "UPSTREAM_README.md",
                "publication-provenance.json",
            }
            writes.append("upload")
            return SimpleNamespace(oid=revision)

    upstream = tmp_path / "upstream" / "README.md"
    upstream.parent.mkdir(parents=True, exist_ok=True)
    upstream.write_bytes(card)

    def download(dataset, name, *, repo_type, revision):
        if dataset == DATASET:
            return str(upstream)
        return str(tmp_path / name)

    monkeypatch.setattr(license_publication, "HfApi", Hub)
    monkeypatch.setattr(license_publication, "hf_hub_download", download)
    monkeypatch.setattr(license_publication, "read_license_receipts", lambda index_path: {ingest.source_id: ingest})
    monkeypatch.setattr(
        license_publication,
        "resolve_license",
        lambda target: (_ for _ in ()).throw(AssertionError("a revision must not re-resolve its card")),
    )
    monkeypatch.setattr(licensing, "license_bans", lambda: {})
    published = license_publication.publish_licensed_population(
        ingest.source_id,
        repo,
        tmp_path,
        ("tasks.parquet",),
        body="# Population\n",
        pretty_name="Fixture population",
        use="train",
    )
    assert writes == ["create", "upload"]
    assert published.revision == revision and published.license_check["verdict"] == "PASS"
    assert published.ingest_record == (DATASET, REVISION)
    declared = {
        "dataset_id": DATASET,
        "revision": REVISION,
        "url": f"https://huggingface.co/datasets/{DATASET}/tree/{REVISION}",
    }
    assert published.license_source == declared
    assert DatasetCard((tmp_path / "README.md").read_text()).data.to_dict()["license_source"] == declared


def test_licensed_population_refuses_a_source_without_an_ingest_record(tmp_path, monkeypatch):
    """Fail closed: a source is licensed once at ingest; no record, no publication."""
    (tmp_path / "tasks.parquet").write_bytes(b"population bytes")
    writes = []

    class Hub:
        def create_repo(self, **kwargs):
            writes.append("create")

    monkeypatch.setattr(license_publication, "HfApi", Hub)
    monkeypatch.setattr(license_publication, "read_license_receipts", lambda index_path: {})
    with pytest.raises(ValueError, match="no ingest licence record"):
        license_publication.publish_licensed_population(
            "fixture:source",
            "open-athena/fixture-population",
            tmp_path,
            ("tasks.parquet",),
            body="# Population\n",
            pretty_name="Fixture population",
            use="train",
        )
    assert writes == []


def test_licensed_population_refuses_a_failing_ingest_record(tmp_path, monkeypatch):
    """A restricted ingest record never publishes; doubt means ban at every revision."""
    ingest = source_receipt(b"---\nlicense: cc-by-nc-4.0\n---\n")
    (tmp_path / "tasks.parquet").write_bytes(b"population bytes")
    writes = []

    class Hub:
        def create_repo(self, **kwargs):
            writes.append("create")

    monkeypatch.setattr(license_publication, "HfApi", Hub)
    monkeypatch.setattr(license_publication, "read_license_receipts", lambda index_path: {ingest.source_id: ingest})
    monkeypatch.setattr(licensing, "license_bans", lambda: {})
    with pytest.raises(ValueError, match="blocks ingest and admission"):
        license_publication.publish_licensed_population(
            ingest.source_id,
            "open-athena/fixture-population",
            tmp_path,
            ("tasks.parquet",),
            body="# Population\n",
            pretty_name="Fixture population",
            use="train",
        )
    assert writes == []


def test_licensed_population_carries_a_repaired_permissive_set(tmp_path, monkeypatch):
    """A source repaired at ingest to permissive-only items carries the whole licence set
    forward: the record holds the set, the card declares it, and the revision passes
    without any per-pin re-resolution (user ruling 2026-11-27, mixed-licence exception)."""
    card = b"---\nlicense: [mit, apache-2.0]\n---\n# Dataset\n"
    ingest = source_receipt(card)
    (tmp_path / "tasks.parquet").write_bytes(b"population bytes")
    repo = "open-athena/fixture-population"
    revision = "d" * 40
    writes = []

    class Hub:
        def list_repo_tree(self, dataset, *, repo_type, revision, recursive):
            return [RepoFile(path="README.md", size=len(card), oid="b" * 40)]

        def create_repo(self, **kwargs):
            writes.append("create")

        def repo_info(self, dataset, **kwargs):
            return SimpleNamespace(private=True, sha=kwargs.get("revision", "c" * 40))

        def upload_folder(self, **kwargs):
            writes.append("upload")
            return SimpleNamespace(oid=revision)

    upstream = tmp_path / "upstream" / "README.md"
    upstream.parent.mkdir(parents=True, exist_ok=True)
    upstream.write_bytes(card)

    def download(dataset, name, *, repo_type, revision):
        return str(upstream if dataset == DATASET else tmp_path / name)

    monkeypatch.setattr(license_publication, "HfApi", Hub)
    monkeypatch.setattr(license_publication, "hf_hub_download", download)
    monkeypatch.setattr(license_publication, "read_license_receipts", lambda index_path: {ingest.source_id: ingest})
    monkeypatch.setattr(
        license_publication,
        "resolve_license",
        lambda target: (_ for _ in ()).throw(AssertionError("a revision must not re-resolve its card")),
    )
    monkeypatch.setattr(licensing, "license_bans", lambda: {})
    published = license_publication.publish_licensed_population(
        ingest.source_id,
        repo,
        tmp_path,
        ("tasks.parquet",),
        body="# Population\n",
        pretty_name="Fixture population",
        use="train",
    )
    assert writes == ["create", "upload"]
    assert published.license_check["verdict"] == "PASS"
    assert published.license_source["dataset_id"] == DATASET
    card_metadata = DatasetCard((tmp_path / "README.md").read_text()).data.to_dict()
    assert card_metadata["license"] == ["apache-2.0", "mit"]


def test_licensed_population_refuses_a_registry_ban_at_a_revision(tmp_path, monkeypatch):
    """A ban recorded after ingest still bans every revision of the source."""
    ingest = source_receipt(b"---\nlicense: mit\n---\n")
    (tmp_path / "tasks.parquet").write_bytes(b"population bytes")
    ban = LicenseBan(
        source_id=ingest.source_id,
        upstream_id=DATASET,
        upstream_revision=REVISION,
        classification="unknown",
        license_ids=["unknown"],
        reason="Recorded source ban",
        evidence_urls=[PREFIX + "README.md"],
        decided_at="2026-11-27T12:00:00+00:00",
        decision_by="Fixture user ruling",
        issue_url="https://github.com/marin-community/marin/issues/11497",
    )
    monkeypatch.setattr(
        licensing, "LICENSE_BANS", json.dumps({"version": "fixture", "bans": [ban.model_dump(mode="json")]})
    )
    licensing.license_bans.cache_clear()
    monkeypatch.setattr(license_publication, "read_license_receipts", lambda index_path: {ingest.source_id: ingest})
    with pytest.raises(ValueError, match="Recorded source ban"):
        license_publication.check_publication(ingest.source_id, "open-athena/fixture-population", "e" * 40, "train")


KEEP_ONLY_DATASET = "open-athena/openswe-oss-keep-only-20261126a"
KEEP_ONLY_REVISION = "20a87df9d9baddf70572d7f0d236b31e46e1384"
OPENSWE_UPSTREAM_REVISION = "a8db93af5335df2c8baac0cd1ff367e4d475d3d7"


def openswe_keep_only_receipt(source_id="TaskTrove:GAIR__OpenSWE__openswe_oss"):
    """The repaired-publication receipt shape: pinned at the keep-only release with
    upstream GAIR/OpenSWE@a8db93af as the resolved source."""
    return source_receipt(b"---\nlicense: other\n---\n").model_copy(
        update={
            "source_id": source_id,
            "dataset_id": KEEP_ONLY_DATASET,
            "revision": KEEP_ONLY_REVISION,
            "resolved_source": LicenseSourcePin(
                dataset_id="GAIR/OpenSWE",
                revision=OPENSWE_UPSTREAM_REVISION,
                url=f"https://huggingface.co/datasets/GAIR/OpenSWE/tree/{OPENSWE_UPSTREAM_REVISION}",
                declaration=license_evidence(
                    "https://example.test/keep-only", b"Keep-only release pin", "wrapper_source_pin"
                ),
            ),
        }
    )


def record_openswe_ruling(monkeypatch):
    """Carry the ruling-48 OpenSWE ban entries from the pinned registry data into the
    isolated unit-test registry."""
    registry = json.loads(license_registry.LICENSE_BANS)
    entries = [
        item
        for item in registry["bans"]
        if item["source_id"] in {"TaskTrove:GAIR__OpenSWE__openswe_oss", "TaskTrove:GAIR__OpenSWE__openswe_other"}
    ]
    monkeypatch.setattr(licensing, "LICENSE_BANS", json.dumps({"version": registry["version"], "bans": entries}))
    licensing.license_bans.cache_clear()
    ban = licensing.license_bans()["TaskTrove:GAIR__OpenSWE__openswe_oss"]
    assert ban.upstream_id == "GAIR/OpenSWE"
    assert ban.exception is not None and ban.exception.use == "train"
    return ban


def test_openswe_keep_only_release_pin_publishes_for_train_use(monkeypatch):
    """Ruling 48 (2026-11-27): OpenSWE ships only as the keep-only permissive release,
    for training use, with upstream GAIR/OpenSWE@a8db93af as the resolved source."""
    record_openswe_ruling(monkeypatch)
    receipt = openswe_keep_only_receipt()
    ban = licensing.source_license_ban(receipt.source_id, receipt.dataset_id, receipt, "train")
    assert ban is None
    licensing.require_unbanned(receipt.source_id, receipt.dataset_id, receipt, "train")


def test_openswe_ban_stays_without_the_sanctioned_pin_or_use(monkeypatch):
    """Every OpenSWE pin outside ruling 48 stays banned: a bare keep-only wrapper pin,
    a different upstream revision, a non-train use, the sibling openswe_other source,
    and any other source pinned into GAIR/OpenSWE."""
    record_openswe_ruling(monkeypatch)

    def refused(receipt, source_id, use):
        with pytest.raises(ValueError, match="License banned"):
            licensing.require_unbanned(source_id, receipt.dataset_id, receipt, use)

    # A bare keep-only pin without the resolved upstream cannot claim the exception.
    refused(
        openswe_keep_only_receipt().model_copy(update={"resolved_source": None}),
        "TaskTrove:GAIR__OpenSWE__openswe_oss",
        "train",
    )
    # A different upstream revision stays banned.
    stale = LicenseSourcePin(
        dataset_id="GAIR/OpenSWE",
        revision="dc6df1f3f949439ad1fdd27ffa034a913184405c",
        url="https://huggingface.co/datasets/GAIR/OpenSWE/tree/dc6df1f3f949439ad1fdd27ffa034a913184405c",
        declaration=license_evidence("https://example.test/stale", b"Stale upstream pin", "wrapper_source_pin"),
    )
    refused(
        openswe_keep_only_receipt().model_copy(update={"resolved_source": stale}),
        "TaskTrove:GAIR__OpenSWE__openswe_oss",
        "train",
    )
    # A non-train use stays banned.
    refused(openswe_keep_only_receipt(), "TaskTrove:GAIR__OpenSWE__openswe_oss", "publish")
    # The sibling openswe_other source carries no exception.
    refused(
        openswe_keep_only_receipt("TaskTrove:GAIR__OpenSWE__openswe_other"),
        "TaskTrove:GAIR__OpenSWE__openswe_other",
        "train",
    )
    # The exception is source-scoped: another source pinned at GAIR/OpenSWE is
    # banned by the upstream-dataset fallback.
    upstream_receipt = source_receipt(b"---\nlicense: other\n---\n").model_copy(
        update={"dataset_id": "GAIR/OpenSWE", "revision": OPENSWE_UPSTREAM_REVISION}
    )
    refused(upstream_receipt, "fixture:other-source", "train")


def test_private_publication_reuse_refuses_another_declared_pin(tmp_path, monkeypatch):
    """A reused destination may only carry this publication's declared licence pin."""
    card = b"---\nlicense: mit\n---\n# Dataset\n"
    receipt = source_receipt(card)
    (tmp_path / "README.md").write_bytes(card)
    (tmp_path / "tasks.parquet").write_bytes(b"population bytes")
    repo = "open-athena/fixture-publication"
    other = (
        b"---\nlicense: mit\nlicense_source:\n  dataset_id: other/upstream\n  revision: "
        + b"e" * 40
        + b"\n  url: https://huggingface.co/datasets/other/upstream/tree/"
        + b"e" * 40
        + b"\n---\n"
    )

    class Hub:
        def dataset_info(self, dataset, *, revision):
            assert dataset == DATASET and revision == REVISION
            return SimpleNamespace(sha=revision, siblings=[SimpleNamespace(rfilename="README.md")])

        def create_repo(self, **kwargs):
            raise RuntimeError("409 Conflict: the repo already exists")

        def list_repo_tree(self, dataset, *, repo_type, recursive):
            assert dataset == repo and recursive is True
            return [SimpleNamespace(path="README.md"), SimpleNamespace(path="tasks.parquet")]

        def repo_info(self, dataset, **kwargs):
            assert dataset == repo
            return SimpleNamespace(private=True, sha="c" * 40)

        def upload_folder(self, **kwargs):
            raise AssertionError("a foreign pin must refuse before any upload")

    def download(dataset, name, *, repo_type, revision):
        assert name == "README.md"
        return str(tmp_path / "existing-card.md")

    (tmp_path / "existing-card.md").write_bytes(other)
    monkeypatch.setattr(license_publication, "HfApi", Hub)
    monkeypatch.setattr(license_publication, "hf_hub_download", download)
    monkeypatch.setattr(license_resolution, "HfApi", Hub)
    monkeypatch.setattr(license_resolution, "_fetch", lambda url, **kwargs: card)
    monkeypatch.setattr(licensing, "license_bans", lambda: {})
    prepared = license_publication.prepare_private_dataset(
        receipt.source_id,
        repo,
        tmp_path,
        ("README.md", "tasks.parquet"),
        receipt,
        use="train",
        license_source={
            "dataset_id": DATASET,
            "revision": REVISION,
            "url": f"https://huggingface.co/datasets/{DATASET}/tree/{REVISION}",
        },
    )
    with pytest.raises(ValueError, match="declares another licence source"):
        license_publication.publish_private_dataset(prepared)

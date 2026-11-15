# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pytest
from finestore.rl.licenses import LICENSE_RESOLVER_VERSION, LicenseBan, LicenseReceipt, LicenseSourcePin
from huggingface_hub import DatasetCard
from huggingface_hub.hf_api import RepoFile

from experiments.post_training.task_curation import license_publication, native_quick
from experiments.post_training.task_curation.license_resolution import license_evidence
from infra.marina.apps.rl_data_catalog.server import licensing

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
        (snapshot / name).write_bytes(content)

    class SnapshotHub:
        def list_repo_tree(self, repo, *, repo_type, revision, recursive):
            assert (repo, repo_type, revision, recursive) == (DATASET, "dataset", REVISION, False)
            return [RepoFile(path=name, size=len(data), oid="b" * 40) for name, data in files.items()]

    def download(repo, name, *, repo_type, revision):
        assert (repo, repo_type, revision) == (DATASET, "dataset", REVISION)
        return str(snapshot / name)

    monkeypatch.setattr(license_publication, "HfApi", SnapshotHub)
    monkeypatch.setattr(license_publication, "hf_hub_download", download)
    return snapshot


@pytest.mark.parametrize(
    "card,expected",
    [
        (
            b"---\nlicense: other\nlicense_name: cc0-with-bytedance-notice\n"
            b"license_link: https://example.test/notice\n---\n"
            b"CC0 1.0 with attribution notice.\nCopyright Example Authors.\n",
            {
                "license": "other",
                "license_name": "cc0-with-bytedance-notice",
                "license_link": "https://example.test/notice",
            },
        ),
        (b"# Dataset\n\n## License\nLicensed under Apache License 2.0.\n", {"license": "Apache-2.0"}),
    ],
)
def test_publication_preserves_custom_notice_and_grant_when_readme_evidence_repeats(
    tmp_path, monkeypatch, card, expected
):
    notice = b"Preserve this original attribution exactly.\n"
    snapshot_hub(tmp_path, monkeypatch, {"README.md": card, "NOTICE": notice})
    publication = license_publication.prepare_license_publication(source_receipt(card), tmp_path / "published")
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


def test_changed_license_bytes_fail_before_publication_provenance_is_written(tmp_path, monkeypatch):
    card = b"---\nlicense: apache-2.0\n---\n"
    snapshot_hub(tmp_path, monkeypatch, {"README.md": card + b"changed"})
    with pytest.raises(ValueError, match="differs from its checked receipt"):
        license_publication.prepare_license_publication(source_receipt(card), tmp_path / "published")
    assert not (tmp_path / "published/publication-provenance.json").exists()


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
    publication = license_publication.prepare_license_publication(receipt, tmp_path / "published")
    assert publication.metadata["license_source"]["dataset_id"] == DATASET
    assert publication.metadata["license_input"] == {"dataset_id": "open-athena/input", "revision": "b" * 40}


def test_new_private_pin_rechecks_restrictive_terms(tmp_path, monkeypatch):
    receipt = source_receipt(b"---\nlicense: cc-by-nc-4.0\n---\n")
    monkeypatch.setattr(license_publication, "resolve_license", lambda target: receipt)
    monkeypatch.setattr(license_publication, "source_license_ban", lambda source, dataset, actual: None)
    with pytest.raises(ValueError, match="Published pin failed"):
        license_publication.check_publication(receipt.source_id, "open-athena/new-private-pin", "c" * 40)


def test_native_publication_binds_the_input_receipt_before_reading_tasks(tmp_path):
    receipt = source_receipt(b"---\nlicense: apache-2.0\n---\n")
    entry = {"id": receipt.source_id, "dataset_id": DATASET, "dataset_revision": "c" * 40}
    with pytest.raises(ValueError, match="License"):
        native_quick.publish_quick(entry, tmp_path / "no-task-files", "open-athena/new-private-pin", receipt)


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
        license_publication.check_publication(receipt.source_id, receipt.dataset_id, receipt.revision)

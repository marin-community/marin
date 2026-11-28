# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner
from finestore.rl.licenses import LicenseBan, LicenseCheckVerdict, check_license, require_license
from requests import HTTPError

from experiments.post_training.task_curation import license_check, license_resolution
from experiments.post_training.task_curation.license_targets import LicenseTarget, license_evidence
from experiments.post_training.task_curation.tests.license_fixtures import fixture_license, openmdw_license_text
from infra.marina.apps.rl_data_catalog.server import licensing

PINNED_CASES = json.loads(Path(__file__).with_name("fixtures").joinpath("license_checks.json").read_text())


@pytest.fixture(autouse=True)
def _isolated_rulings(isolated_rulings):
    """Autouse: every test in this module rules against staged empty rulings."""
    yield


@pytest.mark.parametrize("case", PINNED_CASES, ids=lambda case: case["source_id"])
def test_pinned_primary_license_excerpts(case):
    receipt = fixture_license(case["source_id"], case["dataset_id"], case["revision"]).model_copy(
        update={
            "evidence": [license_evidence(doc["url"], doc["text"].encode(), doc["kind"]) for doc in case["documents"]],
            "bundled_third_party": True,
            "model_generated_outputs": True,
        }
    )
    result = check_license(receipt)
    assert (result.verdict.value, result.rule_id.value) == (case["verdict"], case["rule"])
    assert result.evidence_url in {doc["url"] for doc in case["documents"]}
    if result.verdict is LicenseCheckVerdict.PASS:
        require_license(receipt, receipt.source_id, receipt.dataset_id, receipt.revision, "train", ban=None)
    else:
        with pytest.raises(ValueError, match="blocks ingest and admission"):
            require_license(receipt, receipt.source_id, receipt.dataset_id, receipt.revision, "train", ban=None)


def test_openmdw_license_text_and_card_id_are_recognized():
    license_text = openmdw_license_text()
    assert license_evidence("https://example.test/LICENSE", license_text, "license_file").license_ids == ["openmdw-1.1"]
    card = license_evidence(
        "https://example.test/README.md", b"---\nlicense: openmdw-1.1\n---\n# Dataset\n", "dataset_card"
    )
    assert card.license_ids == ["openmdw-1.1"]
    receipt = fixture_license("fixture", "fixture/data", "b" * 40)
    evidence = [
        card,
        license_evidence("https://example.test/LICENSE", license_text, "license_file"),
    ]
    assert check_license(receipt.model_copy(update={"evidence": evidence})).verdict is LicenseCheckVerdict.PASS


def test_readme_scope_does_not_make_descriptions_or_yaml_a_conflicting_license():
    url = "https://huggingface.co/datasets/Tevatron/browsecomp-plus/blob/144cff8e35b5eaef7e526346aa60774a9deb941f/README.md"
    text = b"---\nlicense: mit\n---\n# Dataset\nSynthetic research-only examples are described here.\n"
    receipt = fixture_license("fixture", "Tevatron/browsecomp-plus", "144cff8e35b5eaef7e526346aa60774a9deb941f")
    receipt = receipt.model_copy(
        update={
            "evidence": [
                license_evidence(url, text, kind) for kind in ("dataset_card", "readme_license", "readme_notice")
            ]
        }
    )
    assert check_license(receipt).verdict is LicenseCheckVerdict.PASS
    title = license_evidence(url, b"## MIT License\n", "readme_license")
    assert check_license(receipt.model_copy(update={"evidence": [title]})).verdict is LicenseCheckVerdict.PASS
    terms = license_evidence(url, text + b"\n## License\nMay not be used to train models.\n", "readme_license")
    assert check_license(receipt.model_copy(update={"evidence": [*receipt.evidence, terms]})).rule_id.value == "F1"


def test_conflicting_custom_terms_and_unidentified_third_party_terms_escalate():
    receipt = fixture_license("fixture", "fixture/data", "a" * 40)
    custom = license_evidence(
        "https://example.test/LICENSE", b"Contact the publisher for applicable terms.", "license_file"
    )
    assert check_license(receipt.model_copy(update={"evidence": [*receipt.evidence, custom]})).rule_id.value == "E2"
    notice = license_evidence(
        "https://example.test/README.md", b"Outputs are subject to separate third-party terms.", "readme_notice"
    )
    uncertain = receipt.model_copy(update={"evidence": [*receipt.evidence, notice]})
    assert check_license(uncertain).rule_id.value == "E3"
    with pytest.raises(ValueError, match="ESCALATE E3"):
        require_license(uncertain, uncertain.source_id, uncertain.dataset_id, uncertain.revision, "train", ban=None)


def test_ban_wins_and_retains_the_underlying_evidence_verdict(monkeypatch):
    receipt = fixture_license("fixture", "fixture/data", "a" * 40)
    ban = LicenseBan(
        source_id="fixture",
        upstream_id="fixture/data",
        upstream_revision="a" * 40,
        classification="missing",
        license_ids=["missing"],
        reason="Previously recorded user ban",
        evidence_urls=["https://example.test/LICENSE"],
        decided_at="2026-11-15T00:00:00+00:00",
        decision_by="user",
        issue_url="https://github.com/marin-community/marin/issues/11497",
    )
    monkeypatch.setattr(license_check, "resolve_target", lambda *_: receipt)
    monkeypatch.setattr(license_check, "target_ban", lambda *_: ban)
    row = license_check.check_target(LicenseTarget("fixture", "fixture/data", "a" * 40, "https://example.test"), None)
    assert (row["verdict"], row["rule_id"], row["evidence_verdict"]["verdict"]) == ("FAIL", "F3", "PASS")
    assert check_license(receipt).verdict is LicenseCheckVerdict.PASS


def test_unreachable_pin_is_a_normal_escalation_record_not_cli_failure(monkeypatch):
    def unreachable(*_):
        raise HTTPError("Pinned metadata is unreachable")

    monkeypatch.setattr(license_check, "resolve_target", unreachable)
    monkeypatch.setattr(license_check, "target_ban", lambda *_: None)
    before = licensing.license_bans()
    result = CliRunner().invoke(
        license_check.main, ["EleutherAI/lambada_openai@900124bf3b8235c6daf21033af9948b3f07346c4"]
    )
    assert result.exit_code == 0, result.output
    row = json.loads(result.stdout.splitlines()[0])
    assert (row["verdict"], row["rule_id"]) == ("ESCALATE", "E4")
    assert "Pinned metadata is unreachable" in row["quote"]
    assert licensing.license_bans() == before


def test_resolver_checks_task_content_without_following_source_or_teacher_repositories(monkeypatch):
    pin = "ca483c6443f17a20ba7d6bac3d274859a1721e51"
    files = {
        "README.md": (
            b"---\nlicense: apache-2.0\n---\nSee https://example.test/teacher and https://example.test/source.\n"
        ),
        "LICENSE": b"Apache License\nVersion 2.0\n",
        "benchmark/task/task.toml": b"version = '1'\n",
        "benchmark/task/LICENSE.content.md": b"Attribution-NonCommercial-ShareAlike 4.0 International\n",
        "benchmark/task/src/COPYING": b"A source repository license is not a task-level grant.\n",
        "node_modules/dependency/LICENSE": b"A dependency license is not a task-level grant.\n",
    }
    monkeypatch.setattr(
        license_resolution.HfApi,
        "dataset_info",
        lambda *_args, **_kwargs: SimpleNamespace(sha=pin, siblings=[SimpleNamespace(rfilename=name) for name in files]),
    )
    fetched = []

    def fetch(url, **_kwargs):
        name = url.split(f"/{pin}/", 1)[1]
        fetched.append(name)
        return files[name]

    monkeypatch.setattr(license_resolution, "_fetch", fetch)
    receipt = license_resolution.resolve_license(
        LicenseTarget("fixture", "fixture/tasks", pin, f"https://huggingface.co/datasets/fixture/tasks/tree/{pin}")
    )
    assert set(fetched) == {"README.md", "LICENSE", "benchmark/task/LICENSE.content.md"}
    assert check_license(receipt).rule_id.value == "F1"


@pytest.fixture
def pinned_dataset_metadata(monkeypatch):
    repositories = {}
    fetched = []
    license_resolution._primary_license.cache_clear()

    def info(_api, repository, revision):
        documents = repositories[repository, revision]
        return SimpleNamespace(sha=revision, siblings=[SimpleNamespace(rfilename=name) for name in documents])

    def fetch(url, **_kwargs):
        path = url.split("/datasets/", 1)[1]
        repository, location = path.split("/resolve/", 1)
        revision, filename = location.split("/", 1)
        fetched.append((repository, revision, filename))
        return repositories[repository, revision][filename]

    monkeypatch.setattr(license_resolution.HfApi, "dataset_info", info)
    monkeypatch.setattr(license_resolution, "_fetch", fetch)
    yield repositories, fetched
    license_resolution._primary_license.cache_clear()


@pytest.mark.parametrize("structured", [False, True], ids=["legacy-source-pin", "license-source-yaml"])
def test_our_wrapper_checks_its_one_declared_source_and_preserves_both_pins(pinned_dataset_metadata, structured):
    repositories, fetched = pinned_dataset_metadata
    wrapper, wrapper_pin = "open-athena/atlas-scale-swe-retained", "a" * 40
    parent, parent_pin = "PrimeIntellect/Scale-SWE-Verified", "2c29b67e0ea4d7819f60d8a10d7475a7ad89ccd6"
    parent_url = f"https://huggingface.co/datasets/{parent}/tree/{parent_pin}"
    if structured:
        declaration = f"license_source:\n  dataset_id: {parent}\n  revision: {parent_pin}\n  url: {parent_url}\n"
        card = f"---\npretty_name: Atlas TaskCompendium retention release\n{declaration}---\n"
        card += f"Source pin: `open-athena/physical-input@{wrapper_pin}`.\n"
    else:
        card = f"---\npretty_name: Atlas TaskCompendium retention release\n---\nSource pin: `{parent}@{parent_pin}`.\n"
    card += "Upstream licenses and third-party asset terms continue to apply.\n"
    repositories[wrapper, wrapper_pin] = {"README.md": card.encode()}
    repositories[parent, parent_pin] = {"README.md": b"---\nlicense: apache-2.0\n---\n# Scale-SWE\n"}
    target = LicenseTarget("PrimeIntellect:fixture", wrapper, wrapper_pin, f"https://huggingface.co/datasets/{wrapper}")
    row = license_check.check_target(target, None)
    assert (row["verdict"], row["rule_id"]) == ("PASS", "P1")
    receipt = license_check.LicenseReceipt.model_validate(row["receipt"])
    require_license(receipt, target.source_id, wrapper, wrapper_pin, "train", ban=None)
    assert (receipt.dataset_id, receipt.revision) == (wrapper, wrapper_pin)
    assert receipt.resolved_source is not None
    assert (receipt.resolved_source.dataset_id, receipt.resolved_source.revision) == (parent, parent_pin)
    assert parent_pin in " ".join(receipt.resolved_source.declaration.quoted_lines)
    assert receipt.resolved_source.declaration.sha256 == license_check.hashlib.sha256(card.encode()).hexdigest()
    assert {repository for repository, _, _ in fetched} == {wrapper, parent}
    assert row["evidence_url"].startswith(f"https://huggingface.co/datasets/{parent}/resolve/{parent_pin}/")


def test_first_party_openmdw_publication_passes_from_its_own_card(pinned_dataset_metadata):
    repositories, fetched = pinned_dataset_metadata
    wrapper, wrapper_pin = "open-athena/atlas-closure-retained", "a" * 40
    parent, parent_pin = "open-athena/Snowball-Smartie-RL-2026.10.15.32-Data", "2c29b67e0ea4d7819f60d8a10d7475a7ad89ccd6"
    parent_url = f"https://huggingface.co/datasets/{parent}/tree/{parent_pin}"
    card = (
        "---\npretty_name: Atlas TaskCompendium retention release\n"
        "license: openmdw-1.1\n"
        f"license_source:\n  dataset_id: {parent}\n  revision: {parent_pin}\n  url: {parent_url}\n---\n"
    )
    repositories[wrapper, wrapper_pin] = {"README.md": card.encode(), "LICENSE": openmdw_license_text()}
    # The declared parent would escalate on its own card; it must never be consulted.
    repositories[parent, parent_pin] = {"README.md": b"---\nlicense: other\nlicense_name: mixed-upstream-terms\n---\n"}
    row = license_check.check_target(
        LicenseTarget("OpenAthena:fixture", wrapper, wrapper_pin, f"https://huggingface.co/datasets/{wrapper}"), None
    )
    assert (row["verdict"], row["rule_id"]) == ("PASS", "P1")
    receipt = license_check.LicenseReceipt.model_validate(row["receipt"])
    assert (receipt.dataset_id, receipt.revision) == (wrapper, wrapper_pin)
    assert receipt.resolved_source is None
    assert {repository for repository, _, _ in fetched} == {wrapper}
    require_license(receipt, "OpenAthena:fixture", wrapper, wrapper_pin, "train", ban=None)
    require_license(receipt, "OpenAthena:fixture", wrapper, wrapper_pin, "eval", ban=None)


def test_own_wrappers_normalize_to_first_dataset_without_following_its_lineage(pinned_dataset_metadata):
    repositories, fetched = pinned_dataset_metadata
    wrapper, parent, grandparent = "open-athena/outer-retained", "open-athena/inner-retained", "fixture/original"
    pin = "b" * 40
    for repository, upstream in [(wrapper, parent), (parent, grandparent)]:
        repositories[repository, pin] = {"README.md": f"Source pin: `{upstream}@{pin}`.\n".encode()}
    repositories[grandparent, pin] = {"README.md": b"---\nlicense: mit\n---\n"}
    row = license_check.check_target(
        LicenseTarget("fixture", wrapper, pin, f"https://huggingface.co/datasets/{wrapper}"), None
    )
    assert (row["verdict"], row["rule_id"]) == ("PASS", "P1")
    assert row["receipt"]["resolved_source"]["dataset_id"] == grandparent
    assert [source["dataset_id"] for source in row["receipt"]["wrapper_chain"]] == [parent]
    assert {repository for repository, _, _ in fetched} == {wrapper, parent, grandparent}


def test_wrapper_restrictions_and_real_separate_terms_still_block(pinned_dataset_metadata):
    repositories, _ = pinned_dataset_metadata
    pin = "c" * 40
    wrapper = "open-athena/restricted-retained"
    repositories[wrapper, pin] = {"README.md": b"---\nlicense: cc-by-nc-4.0\n---\n"}
    row = license_check.check_target(
        LicenseTarget("fixture", wrapper, pin, f"https://huggingface.co/datasets/{wrapper}"), None
    )
    assert (row["verdict"], row["rule_id"]) == ("FAIL", "F1")
    text = b"Upstream licenses and third-party asset terms continue to apply.\n"
    assert license_evidence(
        "https://huggingface.co/datasets/publisher/data/README.md", text, "readme_notice"
    ).unidentified_third_party_terms
    text += b"Outputs are subject to separate third-party terms.\n"
    evidence = license_evidence(f"https://huggingface.co/datasets/{wrapper}/README.md", text, "readme_notice")
    assert evidence.unidentified_third_party_terms
    assert "Outputs are subject" in " ".join(evidence.quoted_lines)
    assert "Upstream licenses" not in " ".join(evidence.quoted_lines)


def test_wrapper_cannot_hide_a_known_parent_ban(pinned_dataset_metadata, monkeypatch):
    repositories, _ = pinned_dataset_metadata
    ban = LicenseBan(
        source_id="fixture",
        upstream_id="fixture/banned",
        upstream_revision="a" * 40,
        classification="missing",
        license_ids=["missing"],
        reason="Recorded user ban",
        evidence_urls=["https://example.test/LICENSE"],
        decided_at="2026-11-15T00:00:00+00:00",
        decision_by="user",
        issue_url="https://github.com/marin-community/marin/issues/11497",
    )
    document = {"version": "fixture-ban", "bans": [ban.model_dump(mode="json")]}
    monkeypatch.setattr(licensing, "read_registry_json", lambda _reference: document)
    licensing.license_bans.cache_clear()
    parent = ban.upstream_id
    pin, wrapper = "d" * 40, "open-athena/renamed-retained"
    repositories[wrapper, pin] = {"README.md": f"Source pin: `{parent}@{pin}`.\n".encode()}
    repositories[parent, pin] = {"README.md": b"---\nlicense: mit\n---\n"}
    target = LicenseTarget("Marin:renamed", wrapper, pin, f"https://huggingface.co/datasets/{wrapper}")
    row = license_check.check_target(target, None)
    assert (row["verdict"], row["rule_id"], row["evidence_verdict"]["verdict"]) == ("FAIL", "F3", "PASS")
    receipt = license_check.LicenseReceipt.model_validate(row["receipt"])
    with pytest.raises(ValueError, match="License banned"):
        licensing.require_unbanned(target.source_id, wrapper, receipt)


@pytest.mark.parametrize(
    "filename,path",
    [
        ("selection-manifest.json", "original_selection_plan"),
        ("selection-manifest.json", "prior_dedup_and_lexical_selection"),
        ("selection-manifest.json", "original_release"),
        ("manifest.json", "source_plan_before_oracle_filter/plan"),
        ("filtered-source.json", "prior_retention_plan"),
    ],
)
def test_filtered_wrapper_uses_its_recorded_manifest_source(pinned_dataset_metadata, filename, path):
    repositories, fetched = pinned_dataset_metadata
    pin = "e" * 40
    wrapper, parent = "open-athena/filtered-fixture", "fixture/original"
    manifest = {"source_dataset_id": parent, "source_revision": pin}
    if path == "original_release":
        manifest = {"repo": parent, "revision": pin, "url": f"https://huggingface.co/datasets/{parent}/tree/{pin}"}
    for key in reversed(path.split("/")):
        manifest = {key: manifest}
    repositories[wrapper, pin] = {
        "README.md": b"---\npretty_name: Atlas oracle-filtered TaskCompendium source\n---\n",
        filename: json.dumps(manifest).encode(),
    }
    repositories[parent, pin] = {"README.md": b"---\nlicense: mit\n---\n"}
    row = license_check.check_target(
        LicenseTarget("fixture", wrapper, pin, f"https://huggingface.co/datasets/{wrapper}"), None
    )
    assert (row["verdict"], row["rule_id"]) == ("PASS", "P1")
    source = row["receipt"]["resolved_source"]
    assert (source["dataset_id"], source["revision"]) == (parent, pin)
    assert source["declaration"]["url"].endswith(f"{pin}/{filename}")
    assert (
        source["declaration"]["sha256"] == license_check.hashlib.sha256(repositories[wrapper, pin][filename]).hexdigest()
    )
    assert {repository for repository, _, _ in fetched} == {wrapper, parent}


def test_repair_lineage_does_not_choose_between_conflicting_original_pins(pinned_dataset_metadata):
    repositories, _ = pinned_dataset_metadata
    wrapper, pin = "open-athena/repair-oracle-fixture", "e" * 40
    manifest = {
        "original_selection_plan": {"source_dataset_id": "fixture/first", "source_revision": "a" * 40},
        "original_release": {
            "repo": "fixture/second",
            "revision": "b" * 40,
            "url": "https://huggingface.co/datasets/fixture/second/tree/" + "b" * 40,
        },
    }
    repositories[wrapper, pin] = {
        "README.md": b"---\npretty_name: Atlas repaired TaskCompendium cohort\n---\n",
        "selection-manifest.json": json.dumps(manifest).encode(),
    }
    row = license_check.check_target(
        LicenseTarget("fixture", wrapper, pin, f"https://huggingface.co/datasets/{wrapper}"), None
    )
    assert (row["verdict"], row["rule_id"]) == ("ESCALATE", "E4")
    assert row["receipt"] is None
    assert "exactly one unambiguous source pin" in row["quote"]


@pytest.mark.parametrize(
    "invalid",
    [
        {},
        ["fixture/original", "a" * 40],
        {"repo": "fixture/original", "revision": "a" * 40},
        {"repo": "fixture/original", "revision": "a" * 40, "url": "https://huggingface.co/datasets/fixture/other"},
        {"repo": "fixture/original", "revision": "main", "url": "https://huggingface.co/datasets/fixture/original"},
        {
            "repo": "fixture/original",
            "revision": "a" * 40,
            "url": "https://huggingface.co/datasets/fixture/original/tree/" + "b" * 40,
        },
    ],
)
def test_repair_lineage_requires_a_complete_matching_immutable_pin(pinned_dataset_metadata, invalid):
    repositories, _ = pinned_dataset_metadata
    wrapper, pin = "open-athena/repair-oracle-fixture", "e" * 40
    repositories[wrapper, pin] = {
        "README.md": b"---\npretty_name: Atlas repaired TaskCompendium cohort\n---\n",
        "selection-manifest.json": json.dumps({"original_release": invalid}).encode(),
    }
    row = license_check.check_target(
        LicenseTarget("fixture", wrapper, pin, f"https://huggingface.co/datasets/{wrapper}"), None
    )
    assert (row["verdict"], row["rule_id"]) == ("ESCALATE", "E4")
    assert row["receipt"] is None


def test_package_license_is_bound_to_the_exact_wheel_without_downloading_it(monkeypatch):
    pin = "7f17a3eddb13c015d7d4a755ed576a061df889faf9468bcc2cca334ebe9e0435"
    filename = "reasoning_gym-0.1.25-py3-none-any.whl"
    wheel = f"https://files.pythonhosted.org/packages/fixture/{filename}"
    fetched = []
    metadata = {"info": {"license_expression": "MIT"}, "urls": [{"filename": filename, "digests": {"sha256": pin}}]}

    def fetch(url, **_kwargs):
        fetched.append(url)
        return json.dumps(metadata).encode()

    license_resolution._primary_license.cache_clear()
    monkeypatch.setattr(license_resolution, "_fetch", fetch)
    target = LicenseTarget("Reasoning Gym:fixture", filename, pin, wheel)
    receipt = license_resolution.resolve_license(target)
    assert check_license(receipt).verdict is LicenseCheckVerdict.PASS
    assert fetched == ["https://pypi.org/pypi/reasoning_gym/0.1.25/json"]
    metadata["urls"][0]["digests"]["sha256"] = "0" * 64
    license_resolution._primary_license.cache_clear()
    with pytest.raises(ValueError, match="does not bind"):
        license_resolution.resolve_license(target)
    license_resolution._primary_license.cache_clear()

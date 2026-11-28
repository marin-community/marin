# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Landed-source attestation covers worker code and rejects literal source state."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from experiments.post_training.task_curation.ingest_stages.guard import (
    STAGE_DIRECTORY,
    assert_bundle_allowed,
    assert_checkout_allowed,
    assert_stage_sources_clean,
    checkout_witness,
)
from experiments.post_training.task_curation.ingest_stages.ingest_config import load_ingest_config
from experiments.post_training.task_curation.ingest_stages.planning import derive_execution

CONFIGS = Path(__file__).parents[1] / "ingest_stages/configs"


@pytest.fixture
def checkout(tmp_path):
    root = tmp_path / "checkout"
    stage = root / STAGE_DIRECTORY / "stage.py"
    stage.parent.mkdir(parents=True)
    stage.write_text("def run(value):\n    return value\n")
    policy = load_ingest_config(CONFIGS / "litmus-train.json")
    converter = root / Path(*policy.converter_module.split(".")).with_suffix(".py")
    converter.parent.mkdir(parents=True)
    converter.write_text("def convert(row, context):\n    return row\n")
    helper = root / "lib/fixture/src/helper.py"
    helper.parent.mkdir(parents=True)
    helper.write_text("VALUE = 1\n")
    (root / "config.json").write_text(policy.model_dump_json(indent=2) + "\n")
    subprocess.run(["git", "init", "-q", "-b", "atlas-env-audit"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Worker"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "worker@example.com"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "Seed the worker sources"], cwd=root, check=True)
    return root, policy


def test_checkout_and_copied_bundle_share_verified_source_identity(checkout, tmp_path):
    root, policy = checkout
    execution = derive_execution(root / "config.json", root, ("python", "-m", "worker"), {})
    witness = checkout_witness(root, policy)
    landing = tmp_path / "landings.jsonl"
    landing.write_text(json.dumps({"sha": execution.source_revision}) + "\n")
    assert_checkout_allowed(root, policy, witness, landing)
    bundle = tmp_path / "bundle"
    shutil.copytree(root, bundle, ignore=shutil.ignore_patterns(".git"))

    assert_bundle_allowed(bundle, policy, witness, landing)

    assert not (bundle / ".git").exists()
    assert checkout_witness(root, policy) == witness
    assert execution.source_revision == witness.source_revision


@pytest.mark.parametrize("source", ["converter", "helper"])
def test_copied_bundle_refuses_changed_converter_and_shared_helper(checkout, tmp_path, source):
    root, policy = checkout
    witness = checkout_witness(root, policy)
    landing = tmp_path / "landings.jsonl"
    landing.write_text(json.dumps({"sha": witness.source_revision}) + "\n")
    bundle = tmp_path / "bundle"
    shutil.copytree(root, bundle, ignore=shutil.ignore_patterns(".git"))
    relative = (
        Path(*policy.converter_module.split(".")).with_suffix(".py")
        if source == "converter"
        else Path("lib/fixture/src/helper.py")
    )
    (bundle / relative).write_text("CHANGED = True\n")

    with pytest.raises(ValueError, match="worker source bytes differ"):
        assert_bundle_allowed(bundle, policy, witness, landing)


def test_uncommitted_code_cannot_submit_even_with_a_matching_landing_record(checkout, tmp_path):
    root, policy = checkout
    witness = checkout_witness(root, policy)
    landing = tmp_path / "landings.jsonl"
    landing.write_text(json.dumps({"sha": witness.source_revision}) + "\n")
    (root / STAGE_DIRECTORY / "stage.py").write_text("CHANGED = True\n")

    with pytest.raises(ValueError, match="clean committed checkout"):
        assert_checkout_allowed(root, policy, witness, landing)


@pytest.mark.parametrize(
    "literal",
    [
        'INPUT = "s3://fixture/raw-copy/tasks.parquet"',
        'INPUT = "https://example.invalid/datasets/source/train.jsonl"',
        'SOURCE_REVISION = "0123456789abcdef0123456789abcdef01234567"',
        'SNAPSHOT = "r39-source-snapshot"',
        "CENSUS = dict(measured_tasks=47118)",
    ],
)
def test_bundled_stage_literals_refuse_execution_despite_a_landing_record(checkout, tmp_path, literal):
    root, policy = checkout
    (root / STAGE_DIRECTORY / "stage.py").write_text(literal + "\n")
    witness = checkout_witness(root, policy)
    landing = tmp_path / "landings.jsonl"
    landing.write_text(json.dumps({"sha": witness.source_revision}) + "\n")

    with pytest.raises(ValueError, match="ingest stage source guard refused"):
        assert_bundle_allowed(root, policy, witness, landing)


def test_copied_bundle_refuses_new_unattested_stage_code(checkout, tmp_path):
    root, policy = checkout
    witness = checkout_witness(root, policy)
    landing = tmp_path / "landings.jsonl"
    landing.write_text(json.dumps({"sha": witness.source_revision}) + "\n")
    (root / STAGE_DIRECTORY / "new.py").write_text("VALUE = 1\n")

    with pytest.raises(ValueError, match="unattested stage or converter files"):
        assert_bundle_allowed(root, policy, witness, landing)


def test_actual_ingest_stage_sources_pass_the_literal_guard():
    assert_stage_sources_clean(Path(__file__).resolve().parents[4])

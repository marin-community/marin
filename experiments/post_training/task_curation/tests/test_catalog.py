# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from dataclasses import replace

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from taskcompendium.pipeline.inputs import SourceFormat

from experiments.post_training.task_curation.count_inputs import parquet_counts
from experiments.post_training.task_curation.export_catalog import catalog_document, source_row
from experiments.post_training.task_curation.grading_catalog import annotate_catalog_grading
from experiments.post_training.task_curation.pipeline import HfSource
from experiments.post_training.task_curation.source import (
    DataSourceReview,
    GradingSelection,
    RlDataSource,
    SourceInfo,
    SourceReference,
)
from experiments.post_training.task_curation.sources import all_pipelines
from experiments.post_training.task_curation.tests import (
    test_arc,
    test_nemotron_ultra,
    test_reasoning_gym,
    test_skyrl,
    test_tasktrove_code,
    test_tasktrove_text,
)

FAMILY_TESTS = (test_arc, test_nemotron_ultra, test_reasoning_gym, test_skyrl, test_tasktrove_code, test_tasktrove_text)


def test_every_declaration_has_a_fixture_row():
    covered = [name for module in FAMILY_TESTS for name in module.ROWS]
    assert len(covered) == len(set(covered))
    assert set(covered) == set(all_pipelines())


def test_parquet_count_export_identifies_the_counted_input(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    pq.write_table(pa.table({"question": ["one", "two", "three"]}), snapshot / "tasks.parquet")
    metadata = snapshot / ".cache/huggingface/download/tasks.parquet.metadata"
    metadata.parent.mkdir(parents=True)
    metadata.write_text("pinned-input\nfile-etag\n0\n")
    count = parquet_counts(snapshot, "pinned-input", "*.parquet")["tasks.parquet"]
    pipeline = replace(
        all_pipelines()["math500"],
        source=HfSource("local/questions", "pinned-input", ("tasks.parquet",), SourceFormat.PARQUET),
    )
    source = RlDataSource(
        info=SourceInfo(id="local:questions", title="Questions", origin="local", count=count),
        pipeline=pipeline,
    )
    row = source_row(source)
    assert row["task_count"] == 3
    assert row["dataset_revision"] == row["pipeline"]["revision"] == "pinned-input"
    assert row["url"] == "https://huggingface.co/datasets/local/questions/tree/pinned-input"
    assert row["pipeline"]["files"] == ("tasks.parquet",)
    assert source_row(replace(source, info=replace(source.info, count=None)))["task_count"] is None
    with pytest.raises(ValueError, match="expected revision"):
        parquet_counts(snapshot, "another-revision", "*.parquet")


def test_catalog_export_is_deterministic_and_keeps_metadata_only_sources():
    reviewed = RlDataSource(
        info=SourceInfo(id="local:reviewed", title="Reviewed", origin="local", count=7),
        review=DataSourceReview(grade="good", evidence_url="https://example.org/review", reviewed_at="2026-10-08"),
    )
    excluded = RlDataSource(info=SourceInfo(id="local:excluded", title="Excluded", origin="local", tags=("excluded",)))
    document = catalog_document([reviewed, excluded])
    assert document == catalog_document([excluded, reviewed])
    rows = json.loads(json.dumps(document))["sources"]
    assert [(row["id"], row["status"], row["pipeline"]) for row in rows] == [
        ("local:excluded", "Excluded", None),
        ("local:reviewed", "Available", None),
    ]
    assert rows[1]["quality"] == "good"
    assert rows[1]["review_url"] == "https://example.org/review"
    assert (
        document["revision"]
        == hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    )
    changed = replace(reviewed, info=replace(reviewed.info, count=8))
    assert catalog_document([changed, excluded])["revision"] != document["revision"]


def test_generated_grading_tracks_scorer_changes_and_ignores_unrelated_methods():
    selection = GradingSelection("legacy", (), "a" * 40, "b" * 40)
    source = RlDataSource(
        info=SourceInfo(
            id="local:math",
            title="Math",
            origin="local",
            verifier=SourceReference("aime", "raw-revision", "https://example.org/verifier", selection),
        )
    )
    env_source = (
        "class AimeEnv:\n def __init__(self):\n  self.expected=1\n"
        " def step(self,x):\n  return x==self.expected\n def unrelated_metrics(self):\n  return 4\n"
    )
    modules = {
        "skyrl-gym/skyrl_gym/envs/__init__.py": "register(id='aime',entry_point='skyrl_gym.envs.aime.env:AimeEnv')\n",
        "skyrl-gym/skyrl_gym/envs/aime/env.py": env_source,
        "skyrl-gym/skyrl_gym/envs/base_text_env.py": (
            "class BaseTextEnv:\n def init(self,x):\n  return x\n def close(self):\n  pass\n"
            " def set_rollout_evidence(self,x):\n  pass\n"
        ),
        "skyrl-train/skyrl_train/trajectory_runners/skyrl_gym_contracts.py": (
            "def verification_from_env_step(x):\n return x\ndef fold_verification_results(x):\n return x\n"
        ),
        "skyrl-gym/skyrl_gym/__init__.py": "",
        "skyrl-train/skyrl_train/__init__.py": "",
    }

    def transport(request):
        path = request.url.path.split("/", 4)[4]
        if path == "skyrl-gym/pyproject.toml":
            content = (
                '[project]\ndependencies=["verifyit @ git+https://github.com/marin-community/marin.git@'
                + "c" * 40
                + '#subdirectory=lib/verifyit"]\n'
            )
        elif path == "lib/verifyit/pyproject.toml":
            content = (
                '[project]\ndependencies=["harbor-config @ git+https://github.com/marin-community/harbor@'
                + "d" * 40
                + '#subdirectory=packages/harbor-config"]\n'
            )
        elif path.endswith("pyproject.toml"):
            content = "[project]\ndependencies=[]\n"
        elif path.endswith("uv.lock"):
            content = "package=[]\n"
        elif path in modules:
            content = modules[path]
        else:
            return httpx.Response(404)
        return httpx.Response(200, text=content)

    with httpx.Client(transport=httpx.MockTransport(transport)) as client:
        original = catalog_document([source])
        annotate_catalog_grading(original, client)
        modules["skyrl-gym/skyrl_gym/envs/aime/env.py"] = env_source.replace("return 4", "return 5")
        unrelated = catalog_document([source])
        annotate_catalog_grading(unrelated, client)
        assert original["sources"][0]["grading_revision"] == unrelated["sources"][0]["grading_revision"]
        modules["skyrl-gym/skyrl_gym/envs/aime/env.py"] = env_source.replace("self.expected=1", "self.expected=2")
        changed = catalog_document([source])
        annotate_catalog_grading(changed, client)
        assert original["sources"][0]["grading_revision"] != changed["sources"][0]["grading_revision"]
        assert original["sources"][0]["verifier_revision"] == changed["sources"][0]["verifier_revision"]

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from marin.test_time_scaling import (
    PromptManifest,
    PromptManifestRecord,
    PromptMessage,
    ScoringMode,
    load_prompt_manifest,
    write_prompt_manifest,
)


def test_prompt_manifest_round_trip(tmp_path):
    manifest = PromptManifest(
        manifest_id="demo-manifest",
        task_name="demo-math",
        records=(
            PromptManifestRecord(
                prompt_id="p0",
                messages=(PromptMessage(role="user", content="What is 2 + 2? Put the answer in \\boxed{}."),),
                expected_answer="\\boxed{4}",
                scoring_mode=ScoringMode.MATH_BOXED,
                metadata={"split": "test"},
            ),
        ),
        metadata={"suite": "unit"},
    )

    output_dir = tmp_path / "manifest"
    write_prompt_manifest(str(output_dir), manifest)
    loaded_manifest = load_prompt_manifest(str(output_dir))

    assert loaded_manifest.manifest_id == manifest.manifest_id
    assert loaded_manifest.task_name == manifest.task_name
    assert loaded_manifest.metadata == manifest.metadata
    assert len(loaded_manifest.records) == 1
    assert loaded_manifest.records[0].prompt_id == "p0"
    assert loaded_manifest.records[0].messages[0].role == "user"
    assert loaded_manifest.records[0].expected_answer == "\\boxed{4}"
    assert loaded_manifest.records[0].scoring_mode == ScoringMode.MATH_BOXED


def test_load_prompt_manifest_from_relative_manifest_file(tmp_path, monkeypatch):
    manifest = PromptManifest(
        manifest_id="demo-manifest",
        task_name="demo-math",
        records=(
            PromptManifestRecord(
                prompt_id="p0",
                messages=(PromptMessage(role="user", content="What is 2 + 2? Put the answer in \\boxed{}."),),
                expected_answer="\\boxed{4}",
                scoring_mode=ScoringMode.MATH_BOXED,
            ),
        ),
    )

    write_prompt_manifest(str(tmp_path), manifest)
    monkeypatch.chdir(tmp_path)

    loaded_manifest = load_prompt_manifest("manifest.json")

    assert loaded_manifest.manifest_id == manifest.manifest_id
    assert loaded_manifest.records[0].expected_answer == "\\boxed{4}"

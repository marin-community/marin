# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments import exp5095_diff_patch_ppl as diff_patch


def test_build_diff_patch_raw_validation_sets_emits_two_sources_and_split_metrics() -> None:
    datasets = diff_patch.build_diff_patch_raw_validation_sets(raw_root="gs://example-bucket/raw/diff_patch")

    swe_patch = datasets["swe_bench/issue_to_patch_patch_text"]
    swe_context = datasets["swe_bench/issue_to_patch_context_plus_patch"]
    commitpack_patch = datasets["commitpack/commit_message_plus_diff_patch_text"]

    assert swe_patch.input_path == "gs://example-bucket/raw/diff_patch/swe_bench/issue_to_patch_patch_text.jsonl.gz"
    assert (
        swe_context.input_path
        == "gs://example-bucket/raw/diff_patch/swe_bench/issue_to_patch_context_plus_patch.jsonl.gz"
    )
    assert commitpack_patch.input_path == (
        "gs://example-bucket/raw/diff_patch/commitpack/commit_message_plus_diff_patch_text.jsonl.gz"
    )
    assert swe_patch.tags is not None and "metric:patch_text" in swe_patch.tags
    assert swe_context.tags is not None and "metric:context_plus_patch" in swe_context.tags


def test_build_swe_bench_issue_to_patch_eval_text_masks_provenance_fields() -> None:
    row = {
        "instance_id": "django__django-12345",
        "repo": "django/django",
        "base_commit": "abc123",
        "problem_statement": "Fix template regression when None is rendered.",
        "hints_text": "Regression introduced in parser cleanup.",
        "patch": "diff --git a/a.py b/a.py\n+return 'fixed'\n",
    }

    rendered = diff_patch.build_swe_bench_issue_to_patch_eval_text(row)

    patch_only = rendered[diff_patch.DiffPatchMetric.PATCH_TEXT]
    context_plus_patch = rendered[diff_patch.DiffPatchMetric.CONTEXT_PLUS_PATCH]

    assert patch_only.startswith("diff --git a/a.py b/a.py")
    assert "Issue:\nFix template regression when None is rendered." in context_plus_patch
    assert "Hints:\nRegression introduced in parser cleanup." in context_plus_patch
    assert "repo" not in context_plus_patch
    assert "django__django-12345" not in context_plus_patch
    assert "base_commit" not in context_plus_patch


def test_build_commitpack_eval_text_separates_patch_and_commit_message() -> None:
    row = {
        "repo_name": "org/repo",
        "commit_hash": "f00dbabe",
        "url": "https://example.invalid/commit/f00dbabe",
        "commit_message": "Fix null handling in adapter.",
        "diff": "diff --git a/adapter.py b/adapter.py\n+if value is None:\n+    return ''\n",
    }

    rendered = diff_patch.build_commitpack_commit_message_plus_diff_eval_text(row)

    assert rendered[diff_patch.DiffPatchMetric.PATCH_TEXT] == (
        "diff --git a/adapter.py b/adapter.py\n+if value is None:\n+    return ''"
    )
    assert rendered[diff_patch.DiffPatchMetric.CONTEXT_PLUS_PATCH].startswith(
        "Commit Message:\nFix null handling in adapter."
    )
    assert "repo_name" not in rendered[diff_patch.DiffPatchMetric.CONTEXT_PLUS_PATCH]
    assert "f00dbabe" not in rendered[diff_patch.DiffPatchMetric.CONTEXT_PLUS_PATCH]


def test_diff_patch_raw_validation_sets_prefixes_namespace() -> None:
    prefixed = diff_patch.diff_patch_raw_validation_sets()
    assert "diff_patch/swe_bench/issue_to_patch_patch_text" in prefixed
    assert "diff_patch/commitpack/commit_message_plus_diff_context_plus_patch" in prefixed

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import pytest
import xxhash

from experiments.datakit.store.provider_reconstruction import (
    ProviderTransform,
    ReconstructionRecipe,
    reconstruct_selected_document,
    reconstruction_code_digests,
)


def _recipe(transform: ProviderTransform, id_field: str = "id") -> ReconstructionRecipe:
    normalizer, adapter = reconstruction_code_digests(transform)
    return ReconstructionRecipe(transform, "text", id_field, 128, normalizer, adapter)


def test_provider_parallel_text_reconstruction_matches_selected_content_and_rejects_edits():
    # The chosen provider ID makes the pinned fold emit English, then original.
    raw = {"id": "0", "og_full_text": "bonjour", "translated_text": "hello"}
    expected_text = "hello\n\nbonjour"
    expected_id = xxhash.xxh3_128_hexdigest(expected_text.encode())
    recipe = _recipe(ProviderTransform.FINETRANSLATIONS)
    assert reconstruct_selected_document(recipe, [raw], expected_id)["text"] == expected_text
    with pytest.raises(ValueError, match="did not match"):
        reconstruct_selected_document(recipe, [{**raw, "translated_text": "edited"}], expected_id)
    with pytest.raises(ValueError, match="code digests"):
        reconstruct_selected_document(replace(recipe, transform_sha256="outdated"), [raw], expected_id)


def test_direct_provider_reconstruction_retains_source_id_and_normalizes_whitespace():
    raw = {"key": 17, "text": "hello" + " " * 140 + "world"}
    expected_text = "hello" + " " * 128 + "world"
    expected_id = xxhash.xxh3_128_hexdigest(expected_text.encode())
    result = reconstruct_selected_document(_recipe(ProviderTransform.DIRECT, "key"), [raw], expected_id)
    assert (result["id"], result["text"], result["source_id"]) == (expected_id, expected_text, 17)


def test_davinci_original_messages_replay_success_tag_and_tool_calls():
    raw = {
        "sample_name": "original-trajectory",
        "success": True,
        "messages": [
            {"role": "user", "content": "Fix bug"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"function": {"name": "bash", "arguments": '{"command":"pytest -q"}'}}],
            },
            {"role": "tool", "content": "passed"},
        ],
    }
    expected = (
        "This trajectory solved the task successfully.\n\n<user>\nFix bug\n</user>\n\n"
        "<assistant>\n<tool_call:bash>\n  command: pytest -q\n</tool_call:bash>\n</assistant>\n\n"
        "<tool>\npassed\n</tool>"
    )
    expected_id = xxhash.xxh3_128_hexdigest(expected.encode())
    with pytest.raises(ValueError, match="did not match"):
        reconstruct_selected_document(_recipe(ProviderTransform.DIRECT), [raw], expected_id)
    recovered = reconstruct_selected_document(_recipe(ProviderTransform.DAVINCI_ENV), [raw], expected_id)
    assert recovered["text"] == expected
    with pytest.raises(ValueError, match="did not match"):
        reconstruct_selected_document(_recipe(ProviderTransform.DAVINCI_ENV), [{**raw, "success": False}], expected_id)


def test_davinci_context_original_pr_fields_replay_as_the_frozen_markdown():
    raw = {
        "repo_name": "example/repo",
        "repo_description": "tools",
        "title": "Fix",
        "body": "Patch",
        "related_issue": {"title": "Bug", "body": "Details", "comments": [{"author": "dev", "body": "agreed"}]},
        "pr_summary": "Summary",
        "relevant_files": [{"path": "a.py", "content": "old"}],
        "commits": [{"refined_message": "Improve", "diffs": [{"path": "a.py", "patch": "-old\n+new"}]}],
    }
    expected = (
        "# Repository Context\nName: example/repo\nDescription: tools\n\n# Issue\n## Bug\n\nDetails\n\n"
        "### Comment by dev\nagreed\n\n# Pull Request\n## Fix\nPatch\n\n# Summary\nSummary\n\n"
        "# Relevant Files Found\n\n## a.py\n```\nold\n```\n\n# Edits\n\nImprove\n\n"
        "## a.py\n```diff\n-old\n+new\n```"
    )
    expected_id = xxhash.xxh3_128_hexdigest(expected.encode())
    with pytest.raises(ValueError, match="did not match"):
        reconstruct_selected_document(_recipe(ProviderTransform.DIRECT), [raw], expected_id)
    assert reconstruct_selected_document(_recipe(ProviderTransform.DAVINCI_CTX), [raw], expected_id)["text"] == expected
    with pytest.raises(ValueError, match="did not match"):
        reconstruct_selected_document(_recipe(ProviderTransform.DAVINCI_CTX), [{**raw, "commits": []}], expected_id)


def test_gpt_oss_original_prompt_reasoning_and_response_replay_without_staged_text():
    raw = {"user_content": "Calculate", "assistant_thinking": "Check the inputs", "assistant_content": "Done"}
    expected = (
        "<user>\nCalculate\n</user>\n\n<thinking>\nCheck the inputs\n</thinking>\n\n<assistant>\nDone\n</assistant>"
    )
    expected_id = xxhash.xxh3_128_hexdigest(expected.encode())
    with pytest.raises(ValueError, match="did not match"):
        reconstruct_selected_document(_recipe(ProviderTransform.DIRECT), [raw], expected_id)
    assert reconstruct_selected_document(_recipe(ProviderTransform.GPT_OSS), [raw], expected_id)["text"] == expected
    with pytest.raises(ValueError, match="did not match"):
        reconstruct_selected_document(
            _recipe(ProviderTransform.GPT_OSS), [{**raw, "assistant_thinking": "changed"}], expected_id
        )

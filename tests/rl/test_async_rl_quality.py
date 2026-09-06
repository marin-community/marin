# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import collections
import copy
import hashlib
import json
from pathlib import Path

import pytest
from fsspec.implementations.local import LocalFileSystem
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from experiments.post_training import async_rl_quality_audit as capture
from experiments.post_training.async_rl_quality import AnswerStatus, extract_numeric_answer, normalize_numeric_answer


@pytest.mark.parametrize(
    "text,expected",
    [
        ('The final answer is "#### 18".', "18"),
        ("Therefore, #### 1,234.", "1234"),
        (r"The result is \boxed{18}", "18"),
        ("The final answer is #### 18", "18"),
        ("The final answer is 18.", "18"),
        ("The final answer is\n18", "18"),
        ("#### 18\nThe final answer is 18.", "18"),
        ("#### -0.50", "-1/2"),
        (r"\boxed{1/2}", "1/2"),
    ],
)
def test_numeric_answer_accepts_concluding_prose_and_equivalent_numeric_forms(text, expected):
    result = extract_numeric_answer(text)
    assert result.status == AnswerStatus.EXTRACTED
    assert result.value == expected


@pytest.mark.parametrize(
    "text,status",
    [
        ("#### 18\nActually, the final answer is 19.", AnswerStatus.CONFLICTING),
        ("#### 18\nUser: Why?\nAssistant: #### 18", AnswerStatus.ROLE_CONTINUATION),
        ("7 plus 11 gives 18 students.", AnswerStatus.MISSING),
        ("#### 1,23", AnswerStatus.MALFORMED),
        ("#### 18abc", AnswerStatus.MALFORMED),
        (r"\boxed{1/0}", AnswerStatus.MALFORMED),
        ("#### 18 or 19", AnswerStatus.AMBIGUOUS),
        ("#### 18, 19", AnswerStatus.AMBIGUOUS),
        ("#### 18 + 1", AnswerStatus.AMBIGUOUS),
        ("#### 18 or nineteen", AnswerStatus.AMBIGUOUS),
    ],
)
def test_numeric_answer_reports_coverage_failures_without_reference_matching(text, status):
    result = extract_numeric_answer(text)
    assert result.status == status
    if status != AnswerStatus.EXTRACTED:
        assert result.value is None


def test_numeric_answer_reports_tail_without_certifying_semantics():
    text = "#### 18\nMore discussion with 19."
    result = extract_numeric_answer(text)
    assert result.value == "18"
    assert result.characters_after_last_candidate == len("\nMore discussion with 19.")
    # The unmarked 19 is intentionally not treated as a competing final answer.
    # Independent adjudication must measure this extractor's semantic coverage.


@pytest.mark.parametrize(
    "text",
    [
        "Let's compute the sum:\n12. Glass 12: 0.018915648\n13. Glass 13: 0.011349408\n1",
        "We calculated 7 + 11.\n18",
        "The running totals are:\n5\n8\n9.8",
        "There are several quantities still to combine.\n**18**",
    ],
)
def test_trailing_reasoning_number_without_answer_cue_is_missing(text):
    result = extract_numeric_answer(text)
    assert result.status == AnswerStatus.MISSING
    assert result.value is None


@pytest.mark.parametrize(
    "text,expected",
    [("18", "18"), ("\n 18 \n", "18"), ("1,234", "1234"), ("-0.50", "-1/2"), ("1/2", "1/2")],
)
def test_wholly_numeric_answer_retains_bare_value_support(text, expected):
    result = extract_numeric_answer(text)
    assert (result.status, result.value) == (AnswerStatus.EXTRACTED, expected)


@pytest.mark.parametrize("text", ["#### 18\n19", "The final answer is 18.\n1", "\\boxed{18}\n19"])
def test_numeric_tail_conflicting_with_explicit_answer_abstains(text):
    result = extract_numeric_answer(text)
    assert result.status == AnswerStatus.AMBIGUOUS
    assert result.value is None


def test_equal_numeric_tail_does_not_replace_explicit_answer_evidence():
    result = extract_numeric_answer("#### 18\n18")
    assert (result.status, result.value) == (AnswerStatus.EXTRACTED, "18")
    assert result.candidate_count == 1
    assert result.characters_after_last_candidate == len("\n18")


@pytest.mark.parametrize(
    "text", ["1,23", "18abc", "NaN", "Infinity", "1/0", "2 + 3", "1e9999999", ".2.3", "1e" + "9" * 80]
)
def test_numeric_normalization_rejects_partial_or_nonfinite_values(text):
    assert normalize_numeric_answer(text) is None


@pytest.fixture
def decoder():
    vocabulary = [
        "[UNK]",
        "18",
        "####",
        "reasoning",
        "<|start_think|>",
        "<|end_think|>",
        "<|endoftext|>",
        "<|im_end|>",
        "<|im_start|>",
    ]
    return Tokenizer(WordLevel(dict(zip(vocabulary, range(len(vocabulary)), strict=True)), unk_token="[UNK]"))


def test_thinking_boundary_uses_tokens_and_preserves_only_the_completed_final_segment(decoder):
    tokens = [3, 2, 1, 5, 2, 1, 6]
    original = list(tokens)
    segment, boundary = capture.final_assistant_segment(decoder, tokens, thinking=True, prompt_tokens=[4])
    assert tokens == original
    assert (segment, boundary) == ("#### 18", "resolved")
    assert extract_numeric_answer(segment).candidate_count == 1


@pytest.mark.parametrize(
    "prompt,tokens,thinking,expected",
    [
        ([8, 4], [3, 5, 2, 1, 6], True, "<|start_think|> reasoning <|end_think|> #### 18 <|endoftext|>"),
        ([8, 4], [3, 2, 1], True, "<|start_think|> reasoning #### 18"),
        ([8, 4, 3], [2, 1], True, "#### 18"),
        ([8, 4], [2, 1, 6], False, "#### 18 <|endoftext|>"),
    ],
)
def test_blind_text_preserves_full_generated_turn_and_only_a_valid_inherited_prefix(
    decoder, prompt, tokens, thinking, expected
):
    assert capture.assistant_turn_text(decoder, prompt, tokens, thinking=thinking) == expected


@pytest.mark.parametrize(
    "tokens,thinking,prompt,status",
    [
        ([2, 1], True, [4], "missing_thinking_end"),
        ([5, 2, 1, 5], True, [4], "multiple_thinking_ends"),
        ([5, 2, 1, 4], True, [4], "unexpected_thinking_start"),
        ([5, 2, 1], True, [3], "missing_thinking_prompt"),
        ([5, 2, 1], True, [4, 3], "invalid_thinking_prompt"),
        ([4, 2, 1], False, [3], "unexpected_thinking_tokens"),
        ([2, 1, 7, 1], False, [3], "role_or_thinking_continuation"),
        ([8, 2, 1], False, [3], "role_or_thinking_continuation"),
    ],
)
def test_unresolved_token_boundaries_do_not_promote_a_numeric_answer(decoder, tokens, thinking, prompt, status):
    assert capture.final_assistant_segment(decoder, tokens, thinking=thinking, prompt_tokens=prompt) == (None, status)


def test_adjudication_balances_endpoints_and_obeys_a_whole_case_byte_budget():
    candidates = {
        (f"arm-{arm}", step): {
            ("extracted", True): [{"text": f"answer-{i}", "label": f"arm-{arm}", "step": step} for i in range(128)]
        }
        for arm in range(6)
        for step in (0, 100)
    }
    selected, excluded = capture.select_adjudication(copy.deepcopy(candidates), 17)
    assert selected == capture.select_adjudication(copy.deepcopy(candidates), 17)[0]
    assert len(selected) == 48 and excluded == 0
    assert set(collections.Counter((row["label"], row["step"]) for row in selected).values()) == {4}
    oversized = {("arm", 0): {("missing", False): [{"text": "x" * capture.MAX_ADJUDICATION_BYTES}, {"text": "small"}]}}
    selected, excluded = capture.select_adjudication(oversized, 17)
    assert [row["text"] for row in selected] == ["small"]
    assert excluded == 1


@pytest.fixture
def quality_capture(tmp_path, monkeypatch, decoder):
    fs = LocalFileSystem(auto_mkdir=True)

    def fs_path(uri):
        assert uri.startswith(capture.REGIONAL_PREFIX)
        path = tmp_path / uri.removeprefix(capture.REGIONAL_PREFIX)
        path.parent.mkdir(parents=True, exist_ok=True)
        return fs, str(path)

    monkeypatch.setattr(capture.audit, "fs_path", fs_path)
    root, data = capture.REGIONAL_PREFIX + "run", capture.REGIONAL_PREFIX + "data"
    request = {
        "attempt_id": "hidden-attempt",
        "output": {"export_root": root},
        "model": {"tokenizer_uri": "fixture/model", "tokenizer_revision": "a" * 40},
        "validation_data": [{"uri": data, "relative_path": "validation.parquet"}],
        "config_yaml": "generator: {}",
    }
    manifest_uri = data + "/selection.json"
    capture.write_json(
        manifest_uri,
        {"dataset": "openai/gsm8k", "revision": "pinned", "rows": {"test": [f"test/{i}" for i in range(128)]}},
    )
    metrics = {f"eval/{source}/{metric}": 1.0 for source in ("all", "gsm8k") for metric in ("avg_score", "pass_at_1")}
    proofs = {}
    for step in (0, 100):
        uri = root + f"/dumped_evals/global_step_{step}_evals"
        capture.write_json(uri + "/aggregated_results.jsonl", metrics)
        with fs.open(fs_path(uri + "/rows.jsonl")[1], "wb") as output:
            for ordinal in range(128):
                row = {
                    "row_ordinal": ordinal,
                    "uid": f"row-{ordinal}",
                    "token_provenance": "finalized_trajectory",
                    "prompt_token_ids": [3],
                    "response_ids": [2, 1, 6],
                    "response_length": 3,
                    "score": 1.0,
                    "stop_reason": "stop",
                    "data_source": "gsm8k",
                    "output_response": "#### 18 <|endoftext|>",
                    "env_extras": {"reward_spec": {"ground_truth": "18" if ordinal % 2 == 0 else "19"}},
                }
                for key in ("prompt_token_ids", "response_ids"):
                    row[key + "_sha256"] = capture.audit.canonical_sha(row[key])
                output.write((json.dumps(row) + "\n").encode())
        proofs[str(step)] = capture.audit.audit_eval_dump(root, step, 128, 1, metrics, True)[0]
    run = {
        "label": "hidden-arm",
        "expected_eval_rows": 128,
        "expected_steps": 100,
        "quality_steps": [0, 100],
        "expected_eval_steps": [0, 100],
        "thinking": False,
        "envelope": {"request": request},
        "prior_dump_proofs": proofs,
    }
    monkeypatch.setattr(
        capture.audit,
        "audit_storage",
        lambda _run: (
            request,
            {"hydra_args": ["++generator.chat_template_kwargs.enable_thinking=false"]},
            {"clean_end_to_end": True},
        ),
    )
    monkeypatch.setattr(capture, "load_decoder", lambda _model: (decoder, "fixture-tokenizer-sha"))
    return (
        {"runs": [run], "sample_seed": 17, "output_prefix": capture.REGIONAL_PREFIX + "qualification"},
        fs_path,
        manifest_uri,
    )


def test_capture_rechecks_real_dump_proofs_preserves_raw_reward_and_blinds_case_metadata(quality_capture):
    specification, fs_path, _manifest = quality_capture
    result = capture.qualify(specification)
    evaluations = result["runs"]["hidden-arm"]["evaluations"]
    assert [evaluation["counts"]["raw_correct"] for evaluation in evaluations] == [128, 128]
    assert [evaluation["counts"]["extracted_correct"] for evaluation in evaluations] == [64, 64]
    blind = json.loads(Path(fs_path(specification["output_prefix"] + "/adjudication-blind.json")[1]).read_text())
    key = json.loads(Path(fs_path(specification["output_prefix"] + "/adjudication-key.json")[1]).read_text())
    assert len(blind["rows"]) == 48
    assert all(set(row) == {"blind_id", "text"} for row in blind["rows"])
    assert all(row["text"] == "#### 18 <|endoftext|>" for row in blind["rows"])
    assert {row["blind_id"] for row in blind["rows"]} == {row["blind_id"] for row in key["rows"]}
    assert all(row["label"] == "hidden-arm" and row["prediction"]["value"] == "18" for row in key["rows"])
    assert result["qualification_complete"] is False
    with pytest.raises(ValueError, match="output already exists"):
        capture.qualify(specification)


def test_capture_rejects_a_128_row_heldout_slice_before_any_response_read(quality_capture, monkeypatch):
    specification, fs_path, manifest_uri = quality_capture
    path = Path(fs_path(manifest_uri)[1])
    manifest = json.loads(path.read_text())
    manifest["rows"]["test"] = [f"test/{i}" for i in range(128, 256)]
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(
        capture.audit, "audit_storage", lambda _run: pytest.fail("must reject before storage/dump reads")
    )
    with pytest.raises(ValueError, match="exactly test"):
        capture.qualify(specification)


def test_capture_rejects_a_dump_changed_between_proof_and_quality_passes(quality_capture, monkeypatch):
    specification, fs_path, _manifest = quality_capture
    original = capture.audit.audit_eval_dump

    def audited_then_changed(*args, **kwargs):
        result = original(*args, **kwargs)
        path = Path(fs_path(args[0] + f"/dumped_evals/global_step_{args[1]}_evals/rows.jsonl")[1])
        lines = path.read_text().splitlines()
        changed = json.loads(lines[0])
        changed["score"] = 0.0
        lines[0] = json.dumps(changed)
        path.write_text("\n".join(lines) + "\n")
        return result

    monkeypatch.setattr(capture.audit, "audit_eval_dump", audited_then_changed)
    with pytest.raises(ValueError, match="differs from verified dump"):
        capture.qualify(specification)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("#### <22.5>", "45/2"),
        ("#### $ 1,200 $ pets.", "1200"),
        ("The final answer is **18**.", "18"),
        (r"The final answer is \(\boxed{18}\). Do not use \boxed{}.", "18"),
        ("### Final Answer:\nThe count is **18**.\n#### 18", "18"),
        (r'The final answer is "#### 18". Do not use \boxed{} or additional text.', "18"),
        (r"\boxed{18}. The \boxed{} is not used as per the instructions.", "18"),
    ],
)
def test_candidate_wrappers_and_explicit_empty_format_instructions(text, expected):
    result = extract_numeric_answer(text)
    assert (result.status, result.value) == (AnswerStatus.EXTRACTED, expected)


@pytest.mark.parametrize(
    "text",
    [
        "The answer is 1000 feet. Wait, this is confusing. The answer is",
        "#### 18\nActually, the final answer is **",
        "#### 18\nThe answer is 19 or 20.",
        "#### 18\n#### <18abc>",
        "#### 18\n#### $1,23$",
        r"#### 18\n\boxed{not a number}",
        r"#### 18\n\boxed{}",
        "#### 18\n#### <18 or 19>",
    ],
)
def test_new_incomplete_or_substantive_malformed_candidate_cannot_reuse_earlier_answer(text):
    result = extract_numeric_answer(text)
    assert result.status != AnswerStatus.EXTRACTED
    assert result.value is None


@pytest.mark.parametrize(
    "text",
    [
        "#### <18>abc",
        "#### **18**xyz",
        r"\boxed{18}+2",
        "The answer is 18. Wait, this is confusing. Let me think again.",
        r"#### 18. Do not use \boxed{19oops}.",
        r"#### 18. The final answer is \(\boxed{19",
        r"#### 18. \boxed{19",
        r"\boxed{18}abc",
        "Final Answer:\nNaN\n#### 18",
        "#### 18..",
    ],
)
def test_numeric_candidate_does_not_drop_suffixes_or_unresolved_reconsideration(text):
    assert extract_numeric_answer(text).value is None


def test_fresh_capture_excludes_prior_text_in_every_endpoint_without_changing_metrics(quality_capture):
    specification, fs_path, _ = quality_capture
    specification["exclude_response_text_sha256"] = [hashlib.sha256(b"#### 18 <|endoftext|>").hexdigest()]
    result = capture.qualify(specification)
    assert result["sample_rows"] == 0
    assert result["sample_candidates_excluded_previous_by_endpoint"] == {"hidden-arm/0": 128, "hidden-arm/100": 128}
    assert [row["counts"]["raw_correct"] for row in result["runs"]["hidden-arm"]["evaluations"]] == [128, 128]
    key = json.loads(Path(fs_path(specification["output_prefix"] + "/adjudication-key.json")[1]).read_text())
    assert key["rows"] == []

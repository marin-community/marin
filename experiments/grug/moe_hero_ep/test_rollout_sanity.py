# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import numpy as np
import pytest
from rigging.filesystem.conditional_object import ConditionalWriteError

from experiments.grug.moe_hero_ep.format_eval import EOT_ID
from experiments.grug.moe_hero_ep.format_sample import NativePanelRequest
from experiments.grug.moe_hero_ep.ops.vibe_check.completions import SampleResult
from experiments.grug.moe_hero_ep.rollout_sanity import (
    RolloutSanityRequest,
    check_disjoint,
    frozen_panel,
    sample_waves,
)


@pytest.mark.parametrize("cap, expected_stop", [(2, "eos"), (1, "max_new_tokens")])
def test_partial_final_wave_preserves_order_seeds_scores_and_hashes(tmp_path, suite_inputs, cap, expected_stop):
    suite, tokenizer = suite_inputs
    case = suite.panels[0].panel.cases[0]
    panel = suite.panels[0].panel.model_copy(
        update={
            "max_new_tokens": cap,
            "cases": tuple(
                case.model_copy(update={"id": f"response-{index}", "seed": index + 101}) for index in range(5)
            ),
        }
    )
    native = NativePanelRequest(
        panel=panel,
        checkpoint=suite.checkpoint,
        model=suite.model,
        serialization="marin",
        target_cluster="test",
        expert_axis_size=1,
        replica_axis_size=1,
    )
    request = RolloutSanityRequest(
        native_request=native,
        source_model=suite.model,
        destination=str(tmp_path / "result"),
        batch_size=2,
        physical_padding=256,
        overlap_evidence={},
    )
    answer = tokenizer.encode("x")[0]
    observed_shapes = []

    def logits(tokens, positions):
        observed_shapes.append(tokens.shape)
        scores = np.full((len(tokens), len(tokenizer)), -100.0)
        for row in range(len(tokens)):
            scores[row, EOT_ID if tokens[row, positions[row]] == answer else answer] = 0
        return scores

    manifest = sample_waves(request, tokenizer, logits, request.destination)
    root = tmp_path / "result"
    raw = (root / "samples.json").read_bytes()
    result = SampleResult.model_validate_json(raw)
    slim = json.loads((root / "transcripts.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    assert [row.prompt_id for row in result.completions] == [f"response-{index}" for index in range(5)]
    assert [row.samples[0].seed for row in result.completions] == list(range(101, 106))
    assert all(row.samples[0].token_ids == ((answer, EOT_ID) if cap == 2 else (answer,)) for row in result.completions)
    assert [wave["cases"] for wave in manifest["waves"]] == [2, 2, 1]
    assert observed_shapes == [(2, 256)] * (6 if cap == 2 else 3)
    assert summary["stop_counts"][expected_stop] == 5
    assert all(not row["role_leak_flag"] for row in summary["rollouts"])
    if expected_stop == "max_new_tokens":
        assert summary["capped_unfinished_cases"] == 5
        assert all(row["completion_status"] == "unfinished_censored" for row in summary["rollouts"])
    assert slim["source_samples_sha256"] == hashlib.sha256(raw).hexdigest()
    for original, derived in zip(result.completions, slim["samples"]["completions"], strict=True):
        assert original.samples[0].token_scores
        assert derived["samples"][0]["token_scores"] is None
        assert derived["samples"][0]["token_ids"] == list(original.samples[0].token_ids)
        assert derived["samples"][0]["text"] == original.samples[0].text
    for name, expected in manifest["file_sha256"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected
    for wave in manifest["waves"]:
        for name, expected in wave["file_sha256"].items():
            assert hashlib.sha256((root / "waves" / str(wave["wave"]) / name).read_bytes()).hexdigest() == expected
    # Immutable result publication must not overwrite earlier probability evidence.
    with pytest.raises(ConditionalWriteError):
        sample_waves(request, tokenizer, logits, request.destination)
    assert (root / "samples.json").read_bytes() == raw


@pytest.mark.parametrize("overlap", ["raw-math-question", "history-user-turn"])
def test_exact_disjointness_catches_normalized_raw_questions_and_history_turns(tmp_path, overlap):
    panel, references = frozen_panel("a" * 40)
    if overlap == "raw-math-question":
        text = references[0]["raw_question"]
        expected_id = "math-gsm8k-0500"
    else:
        case = next(case for case in panel.cases if case.id == "chat-pronoun")
        text = case.messages[0]["content"]
        expected_id = case.id
    training = tmp_path / "train.jsonl"
    heldout = tmp_path / "heldout.jsonl"
    training.write_text(
        json.dumps({"messages": [{"role": "user", "content": "\n  ".join(text.upper().split())}]}) + "\n"
    )
    heldout.write_text("")
    with pytest.raises(ValueError, match=expected_id):
        check_disjoint(panel, references, {"train": training, "heldout": heldout})
    # A genuine nonmatching split records exact input bytes, without claiming fuzzy disjointness.
    training.write_text(
        json.dumps({"messages": [{"role": "user", "content": "unrelated supplied training turn"}]}) + "\n"
    )
    evidence = check_disjoint(panel, references, {"train": training, "heldout": heldout})
    assert evidence["splits"]["train"]["sha256"] == hashlib.sha256(training.read_bytes()).hexdigest()
    assert evidence["splits"]["train"]["conversations"] == 1
    assert evidence["splits"]["heldout"]["conversations"] == 0

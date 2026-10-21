# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import sys
from dataclasses import asdict

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast
from verifyit.closure_scoring import ClosureFormat, score_closure

from experiments.post_training.structure_closure.benchmark import (
    ClosureCase,
    Family,
    task_cases,
)
from experiments.post_training.structure_closure.token_data import ToolPrompt, prepare_tokens, tokenized_case_record
from experiments.post_training.structure_closure.tool_prompt_probe import main as probe_main


def test_native_tool_prompts_preserve_content_and_continuation_controls(tokenizer):
    cases = task_cases("sample_task", [{"search": {"query": 'quote=" slash=\\'}}], "train", "source")
    for case in cases:
        native = tokenized_case_record(case, tokenizer, tool_prompt=ToolPrompt.NATIVE_DECLARED)
        compact = tokenized_case_record(case, tokenizer, tool_prompt=ToolPrompt.COMPACT)
        if case.family not in (Family.TOOL, Family.REPEATED_OPENER):
            assert native == compact
            continue
        prompt = tokenizer.decode(native["prompt_token_ids"], clean_up_tokenization_spaces=False)
        target = tokenizer.decode(native["target_token_ids"], clean_up_tokenization_spaces=False)
        [advertised] = json.loads(prompt.splitlines()[0].partition("=")[2])
        assert advertised["function"]["name"] == "write_result"
        parameters = advertised["function"]["parameters"]
        assert set(parameters["required"]) == {"task", "result"}
        assert parameters["properties"]["result"]["type"] == "array"
        assert native["prefix"].startswith("<tool_call>\n")
        assert native["prefix"] + target == native["expected"]
        assert json.loads(native["expected"].removeprefix("<tool_call>").removesuffix("</tool_call>")) == json.loads(
            case.expected.removeprefix("<tool_call>").removesuffix("</tool_call>")
        )
        assert score_closure(native["prefix"] + target, native["expected"], ClosureFormat.TOOL).success
        if case.control == "close_required":
            assert target == "</tool_call>"
        else:
            assert not score_closure(native["prefix"] + "</tool_call>", native["expected"], ClosureFormat.TOOL).success
            assert native["instruction"].endswith(native["expected"])


def test_conditional_target_keeps_prompt_tokens_when_joint_bpe_would_merge():
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(
        ['"abcdef" instructions ASSISTANT:'] * 10,
        trainers.BpeTrainer(vocab_size=400, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()),
    )
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend)
    tokenizer.chat_template = "{{ messages[0]['content'] }}\nASSISTANT:"
    case = ClosureCase(
        "case", "task", "train", Family.STRING, "continue_required", "instructions", '"abc', '"abcdef"', "source"
    )
    record = tokenized_case_record(case, tokenizer, tool_prompt=ToolPrompt.COMPACT)
    prompt = "instructions\nASSISTANT:" + case.prefix
    assert tokenizer.decode(record["prompt_token_ids"], clean_up_tokenization_spaces=False) == prompt
    assert (
        tokenizer.decode(record["prompt_token_ids"] + record["target_token_ids"], clean_up_tokenization_spaces=False)
        == "instructions\nASSISTANT:" + case.expected
    )
    assert not record["canonical_joint_encoding_matches"]


def test_prepare_tokens_reports_malformed_rows_and_continues(tmp_path, tokenizer):
    initializer = tmp_path / "initializer"
    tokenizer.save_pretrained(initializer)
    source = tmp_path / "cases"
    source.mkdir()
    valid = asdict(
        ClosureCase(
            "valid", "task", "train", Family.STRING, "continue_required", "instructions", '"abc', '"abcdef"', "source"
        )
    )
    invalid = ["{", json.dumps(valid | {"family": "unknown"}), "{}", "[]", json.dumps(valid | {"unexpected": True})]
    payload = "\n".join([*invalid, json.dumps(valid)]) + "\n"
    hashes = {}
    for split in ("train", "validation"):
        (source / f"{split}.jsonl").write_text(payload)
        hashes[split] = hashlib.sha256(payload.encode()).hexdigest()
    (source / "manifest.json").write_text(json.dumps({"hashes": hashes}))
    output = tmp_path / "tokens"
    report = prepare_tokens(source, initializer, output, 4096, tool_prompt=ToolPrompt.COMPACT)
    assert report["status"] == "failed"
    assert len(report["failures"]) == 10
    assert [failure["line"] for failure in report["failures"]] == [1, 2, 3, 4, 5] * 2
    assert json.loads((output / "manifest.json").read_text()) == report
    for split in ("train", "validation"):
        assert [json.loads(line)["case_id"] for line in (output / f"{split}.jsonl").read_text().splitlines()] == [
            "valid"
        ]


def test_probe_rejects_partial_token_cohort_with_matching_hash(tmp_path, monkeypatch):
    tokens = tmp_path / "tokens"
    tokens.mkdir()
    payload = b'{"case_id":"survivor","family":"tool_call_delimiters","control":"close_required"}\n'
    (tokens / "validation.jsonl").write_bytes(payload)
    (tokens / "manifest.json").write_text(
        json.dumps({"status": "failed", "hashes": {"validation": hashlib.sha256(payload).hexdigest()}})
    )
    output = tmp_path / "probe"
    monkeypatch.setattr(
        sys,
        "argv",
        ["probe", "--tokens", str(tokens), "--initializer", str(tmp_path), "--output", str(output), "--tasks", "1"],
    )
    with pytest.raises(ValueError):
        probe_main()
    assert not output.exists()

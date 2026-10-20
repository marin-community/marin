# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import draccus
import numpy as np
from levanter.data.text.formats import ChatProcessor
from levanter.testing.tokenizer import stage_gpt2_tokenizer
from levanter.tokenizers import load_tokenizer
from marin.datakit.chat_template import MARIN_CHAT_TEMPLATE
from marin.execution.fingerprint import canonical_json
from tokenizers import Tokenizer

from experiments.grug.moe_hero_ep import format_data
from experiments.grug.moe_hero_ep.format_data import (
    CatalogSelection,
    DataRecipe,
    SourceSpec,
    materialize_catalog_sources,
    prepare_dataset,
    shifted_chat_targets,
    validate_chat_mask,
)
from experiments.grug.moe_hero_ep.format_eval import AcquisitionDiagnostic, build_acquisition_panel
from experiments.grug.moe_hero_ep.ops.vibe_check.completions import digest


def test_shared_renderer_masks_tool_reasoning_and_shifted_eot(tmp_path, monkeypatch):
    tokenizer_path = stage_gpt2_tokenizer(Path("lib/levanter/tests"), tmp_path)
    underlying = Tokenizer.from_file(str(tokenizer_path / "tokenizer.json"))
    underlying.add_special_tokens(
        ["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "<|start_think|>", "<|end_think|>"]
    )
    underlying.save(str(tokenizer_path / "tokenizer.json"))
    tokenizer = load_tokenizer(str(tokenizer_path))
    row = {
        "messages": [
            {"role": "system", "content": "Be precise."},
            {"role": "user", "content": "What is the observation?"},
            {
                "role": "assistant",
                "reasoning_content": "Need observation.",
                "content": "",
                "tool_calls": [{"type": "function", "function": {"name": "lookup", "arguments": {"key": "x"}}}],
            },
            {"role": "tool", "name": "lookup", "content": "observation-secret"},
            {"role": "assistant", "content": "The value is seven."},
        ],
        "chat_template_kwargs": {"enable_thinking": True},
    }
    processed = ChatProcessor(tokenizer, chat_template=MARIN_CHAT_TEMPLATE)([row])[0]
    validate_chat_mask(processed, tokenizer)
    supervised = tokenizer.decode(processed["input_ids"][processed["assistant_masks"].astype(bool)].tolist())
    assert "Need observation." in supervised and "<|start_think|>" in supervised
    assert '<tool_call>\n{"name": "lookup", "arguments": {"key": "x"}}\n</tool_call>' in supervised
    assert "The value is seven." in supervised
    assert "observation-secret" not in supervised and "Be precise." not in supervised
    assert "What is the observation?" not in supervised and "<|start_header_id|>" not in supervised
    eot = tokenizer.encode("<|eot_id|>")[0]
    inputs, targets, weights = shifted_chat_targets(processed)
    assert np.array_equal(inputs[1:], targets[:-1])
    assert targets[-1] == eot and weights[-1] == 1
    assert ((targets == eot) & (weights == 1)).sum() == 2
    selected = tuple(tokenizer.encode(token)[0] for token in ("<|eot_id|>", "<|start_think|>", "<|end_think|>"))
    acquisition = build_acquisition_panel(
        [{**row, "id": "training-row"}],
        tokenizer,
        tokenizer_revision="test-revision",
        source_artifact_digest="source-digest",
        selected_token_ids=selected,
        max_examples=1,
    )
    assert len(acquisition.cases) == 4
    # Regression: a Pydantic object inside this callback could not fingerprint or cross launch JSON.
    callback = AcquisitionDiagnostic.from_panel(acquisition)
    monkeypatch.setenv("MARIN_FINGERPRINT_STRICT", "true")
    restored = draccus.decode(AcquisitionDiagnostic, draccus.encode(callback))
    assert canonical_json(restored) == canonical_json(callback)
    altered = AcquisitionDiagnostic.from_panel(
        acquisition.model_copy(update={"source_artifact_digest": "different-source"})
    )
    assert canonical_json(altered) != canonical_json(callback)
    assert {case.target_token_id for case in acquisition.cases} == set(selected)
    assert all(case.source_id == "training-row" for case in acquisition.cases)
    assert all(
        list(case.prefix_token_ids) == processed["input_ids"][: len(case.prefix_token_ids)].tolist()
        for case in acquisition.cases
    )


def test_dataset_deduplicates_across_sources_and_excludes_long_conversations(tmp_path):
    tokenizer_path = tmp_path / "tokenizer"
    tokenizer_path.mkdir()
    stage_gpt2_tokenizer(Path("lib/levanter/tests"), tokenizer_path)
    underlying = Tokenizer.from_file(str(tokenizer_path / "tokenizer.json"))
    underlying.add_special_tokens(["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"])
    underlying.save(str(tokenizer_path / "tokenizer.json"))
    tokenizer = load_tokenizer(str(tokenizer_path))
    conversations = [
        {
            "id": str(i),
            "messages": [
                {"role": "user", "content": f"Question {i}?"},
                {"role": "assistant", "content": f"Answer {i}."},
            ],
        }
        for i in range(100)
    ]
    long = {
        "messages": [
            {"role": "user", "content": "too long " * 1000},
            {"role": "assistant", "content": "Never truncate me."},
        ]
    }
    first, second = tmp_path / "first.jsonl", tmp_path / "second.jsonl"
    malformed = {
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "One"},
            {"role": "assistant", "content": "Two"},
        ]
    }
    first.write_text("\n".join(map(json.dumps, [*conversations, long, malformed])))
    second.write_text(
        json.dumps({**conversations[0], "id": "unrelated-source-id", "chat_template_kwargs": {"enable_thinking": False}})
        + "\n"
    )
    sources = tuple(
        SourceSpec(
            slug=path.stem,
            path=str(path),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            revision="pinned-test-source",
            trust_basis="Locally authored test conversation",
            in_pretraining=False,
            weight=1,
            max_examples=200,
        )
        for path in (first, second)
    )
    recipe = DataRecipe(
        sources=sources,
        tokenizer_path=str(tokenizer_path),
        tokenizer_revision="a" * 40,
        tokenizer_snapshot_digest=digest(
            {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in tokenizer_path.iterdir()
                if path.is_file()
            }
        ),
        heldout_fraction=0.2,
        split_seed=7,
        max_tokens=80,
    )
    destination = tmp_path / "prepared"
    manifest = prepare_dataset(recipe, destination, tokenizer)
    train = [json.loads(line) for line in (destination / "train.jsonl").read_text().splitlines()]
    heldout = [json.loads(line) for line in (destination / "heldout.jsonl").read_text().splitlines()]
    assert len(train) + len(heldout) == 100
    assert {row["id"] for row in train}.isdisjoint(row["id"] for row in heldout)
    assert {json.dumps(row["messages"], sort_keys=True) for row in train + heldout} == {
        json.dumps(row["messages"], sort_keys=True) for row in conversations
    }
    assert manifest["sources"]["first"]["excluded_long"] == 1
    assert manifest["sources"]["first"]["excluded_invalid_turns"] == 1
    assert manifest["sources"]["second"]["duplicates"] == 1
    assert manifest["tokens"]["train"]["non_padding"] > manifest["tokens"]["train"]["assistant_supervised"]
    repeated = prepare_dataset(recipe, tmp_path / "repeated", tokenizer)
    assert repeated["outputs"]["train"]["sha256"] == manifest["outputs"]["train"]["sha256"]
    assert repeated["outputs"]["heldout"]["sha256"] == manifest["outputs"]["heldout"]["sha256"]


def test_catalog_materialization_bounds_remote_rows_and_preserves_content(tmp_path, monkeypatch):
    raw = [
        {
            "messages": [
                {"role": "system", "content": "Keep literal text."},
                {"role": "user", "content": "Explain."},
                {"role": "assistant", "content": "<think>literal</think> answer"},
            ]
        },
        {"messages": []},
        {
            "messages": [
                {"role": "user", "content": "Excluded by scanned-row limit"},
                {"role": "assistant", "content": "Excluded"},
            ]
        },
    ]
    # Both substitutions are external service boundaries; adaptation is the real repository code.
    monkeypatch.setattr(
        format_data, "HfApi", lambda: SimpleNamespace(dataset_info=lambda *args, **kwargs: SimpleNamespace(sha="a" * 40))
    )
    monkeypatch.setattr(format_data.datasets, "load_dataset", lambda *args, **kwargs: iter(raw))
    selection = CatalogSelection(
        name="HuggingFaceH4/no_robots",
        max_scanned_rows=2,
        weight=1,
        trust_basis="Existing reviewed SFT catalog",
        in_pretraining=False,
    )
    sources = materialize_catalog_sources([selection], tmp_path)
    rows = [json.loads(line) for line in Path(sources[0].path).read_text().splitlines()]
    assert len(rows) == sources[0].max_examples == 1
    assert [message["content"] for message in rows[0]["messages"]] == [
        "Keep literal text.",
        "Explain.",
        "<think>literal</think> answer",
    ]
    assert sources[0].revision == "a" * 40
    assert sources[0].sha256 == hashlib.sha256(Path(sources[0].path).read_bytes()).hexdigest()
    assert json.loads((tmp_path / "sources.json").read_text())[0]["path"] == sources[0].path

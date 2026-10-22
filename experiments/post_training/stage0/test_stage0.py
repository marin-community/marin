# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Protocol tests: the shared template, the token maps, and the converter.

The hermetic tests use a stub tokenizer and jinja rendering. The real-tokenizer
tests (skipif offline) validate every token map against its pinned tokenizer and
run the token-level span checks over streamed source rows — the checks the review
requires at token granularity, not text presence.
"""

import gzip
import json
import os

import jinja2
import pytest

from experiments.post_training.chat_template import (
    FAMILY_TOKEN_MAPS,
    GRUG_TOKENS,
    NEMOTRON_TOKENS,
    QWEN_TOKENS,
    ChatTokens,
    shared_template,
    template_sha256,
    validate_token_map,
)
from experiments.post_training.stage0.convert import (
    MAX_RENDERED_TOKENS,
    STAGE_ORDER,
    SourceSpec,
    adapt_nemotron_science,
    adapt_nemotron_terminal,
    adapt_wildchat_rewild,
    build_manifest,
    convert_source,
    row_id,
    strip_think,
)

_HF_REVISIONS = {
    "grug": ("open-athena/snowball-67b-a2b-base-262k-qk175", "11c9430da3075ef4596f92ee0af142d9d08215c1"),
    "qwen3_moe": ("Qwen/Qwen3-30B-A3B-Base", "1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9"),
    "qwen3_5_moe": ("Qwen/Qwen3.5-35B-A3B-Base", "0f0813072d2358973511097385626f21fcb6d422"),
    "nemotron_h": ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16", "e0ac9ee3dfd02be21b5479edfc2f671ed269d0a2"),
}


def _hf_reachable() -> bool:
    return os.environ.get("HF_HUB_OFFLINE") != "1"


# --- the shared template, hermetically ---


class StubTokenizer:
    """Encodes every distinct string to one distinct id — enough for validation."""

    def __init__(self):
        self._ids: dict[str, int] = {}

    def encode(self, text: str, add_special_tokens=False) -> list[int]:
        if text not in self._ids:
            self._ids[text] = 1000 + len(self._ids)
        return [self._ids[text]]


def _render(messages: list[dict], tokens: ChatTokens, **kwargs) -> str:
    template = jinja2.Environment().from_string(shared_template(tokens))
    return template.render(messages=messages, **kwargs)


def test_every_token_map_validates_against_a_single_token_vocabulary():
    for family, tokens in FAMILY_TOKEN_MAPS.items():
        lengths = validate_token_map(tokens, StubTokenizer())
        assert all(length == 1 for length in lengths.values()), family


def test_multi_token_delimiter_is_rejected():
    class AlmostTokenizer(StubTokenizer):
        def encode(self, text: str, add_special_tokens=False) -> list[int]:
            if text == self.turn_close:
                return [1, 2]
            return super().encode(text, add_special_tokens=False)

    tokenizer = AlmostTokenizer()
    tokenizer.turn_close = GRUG_TOKENS.turn_close
    with pytest.raises(ValueError, match="turn_close"):
        validate_token_map(GRUG_TOKENS, tokenizer)


def test_duplicate_delimiters_are_rejected():
    broken = ChatTokens(
        bos="<|begin_of_text|>",
        turn_open_prefix="<|start_header_id|>",
        turn_open_suffix="<|end_header_id|>\n",
        turn_close="<|eot_id|>",
        think_open="<|eot_id|>",
        think_close="<|end_think|>",
    )
    with pytest.raises(ValueError, match="distinct"):
        validate_token_map(broken, StubTokenizer())


CONVERSATION = [
    {"role": "system", "content": "be brief"},
    {"role": "user", "content": "run ls"},
    {
        "role": "assistant",
        "content": "Listing.",
        "reasoning_content": "I should list the files.",
        "tool_calls": [{"function": {"name": "bash", "arguments": {"command": "ls"}}}],
    },
    {"role": "tool", "content": "a.txt"},
    {"role": "assistant", "content": "One file: a.txt."},
]


def test_shared_template_renders_reasoning_tools_and_turns_in_order():
    rendered = _render(CONVERSATION, GRUG_TOKENS)
    assert rendered.startswith("<|begin_of_text|>")
    assert "<|start_header_id|>system<|end_header_id|>\nbe brief<|eot_id|>" in rendered
    assert (
        "<|start_header_id|>assistant<|end_header_id|>\n<|start_think|>I should list the files.<|end_think|>" in rendered
    )
    assert '<tool_call>\n{"name": "bash", "arguments": {"command": "ls"}}\n</tool_call>' in rendered
    assert "<|start_header_id|>tool<|end_header_id|>\n<tool_response>\na.txt\n</tool_response><|eot_id|>" in rendered
    # No name= attribute on tool responses, and exactly one think block per
    # reasoning-bearing turn — never an injected empty one.
    assert 'name="' not in rendered
    assert rendered.count("<|start_think|>") == 1
    assert "<|start_think|><|end_think|>" not in rendered


def test_qwen_and_nemotron_instances_differ_only_in_bos():
    qwen = _render(CONVERSATION, QWEN_TOKENS)
    nemotron = _render(CONVERSATION, NEMOTRON_TOKENS)
    assert "<|im_start|>assistant\n<think>I should list the files.</think>" in qwen
    assert not qwen.startswith("<s>")
    assert nemotron.startswith("<s>")
    assert nemotron.removeprefix("<s>") == qwen
    assert "<think></think>" not in qwen


def test_generation_prompt_and_no_system_turn():
    rendered = _render([{"role": "user", "content": "hi"}], GRUG_TOKENS, add_generation_prompt=True)
    assert rendered.endswith("<|start_header_id|>assistant<|end_header_id|>\n")
    assert "system" not in rendered


def test_template_hash_distinguishes_families():
    assert template_sha256(GRUG_TOKENS) != template_sha256(QWEN_TOKENS)
    assert template_sha256(QWEN_TOKENS) == template_sha256(QWEN_TOKENS)


# --- the converter ---


def test_adapters_strip_both_marker_styles_into_reasoning_content():
    science = adapt_nemotron_science(
        {
            "messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "<|start_think|>\nwork it out\n<|end_think|>\nThe answer is 4."},
            ]
        }
    )
    assert science[1]["reasoning_content"] == "work it out"
    assert science[1]["content"] == "The answer is 4."
    assert "<|start_think|>" not in science[1]["content"]

    terminal = adapt_nemotron_terminal(
        {
            "conversations": [
                {"role": "user", "content": "task"},
                {"role": "assistant", "content": "<think>\nplan\n</think>run ls"},
                {"role": "user", "content": "New Terminal Output: files"},
            ]
        }
    )
    assert terminal[1]["reasoning_content"] == "plan"
    assert terminal[1]["content"] == "run ls"
    assert "<think>" not in terminal[1]["content"]


def test_unclosed_think_block_raises():
    with pytest.raises(ValueError, match="unclosed"):
        strip_think("<|start_think|>\nnever closed", "<|start_think|>", "<|end_think|>")


def test_convert_source_filters_one_row_id_set_over_the_cap(tmp_path):
    rows = iter(
        [
            {"messages": [{"role": "user", "content": "short"}, {"role": "assistant", "content": "ok"}]},
            {"messages": [{"role": "user", "content": "long"}, {"role": "assistant", "content": "too long"}]},
            {"messages": [{"role": "user", "content": "also short"}, {"role": "assistant", "content": "ok"}]},
        ]
    )

    def fake_load_dataset(repo, config, revision, split, streaming):
        return rows

    def fake_render_lengths(messages):
        return {
            family: MAX_RENDERED_TOKENS + 1 if messages[0]["content"] == "long" else 10 for family in FAMILY_TOKEN_MAPS
        }

    spec = SourceSpec(
        name="nemotron-science-reasoning",
        repo="laion/llama-nemotron-science-reasoning-on-canonical-think-full",
        revision="bae881d",
        config=None,
        adapt=adapt_nemotron_science,
    )
    entry = convert_source(spec, str(tmp_path), load_dataset=fake_load_dataset, render_lengths=fake_render_lengths)
    assert entry["rows"] == 2
    assert entry["dropped_over_cap"] == 1
    assert entry["dropped_share"] == pytest.approx(1 / 3)
    assert entry["dropped_row_ids_sha256"]
    with gzip.open(tmp_path / "nemotron-science-reasoning.jsonl.gz", "rt") as handle:
        lines = [json.loads(line) for line in handle]
    assert all(line["messages"][0]["content"] != "long" for line in lines)
    assert [line["row_id"] for line in lines] == [
        row_id("nemotron-science-reasoning", line["messages"]) for line in lines
    ]


def test_convert_source_raises_when_everything_is_dropped(tmp_path):
    def fake_load_dataset(repo, config, revision, split, streaming):
        return iter([{"messages": [{"role": "user", "content": "long"}, {"role": "assistant", "content": "x"}]}])

    spec = SourceSpec(
        name="nemotron-science-reasoning",
        repo="laion/llama-nemotron-science-reasoning-on-canonical-think-full",
        revision="bae881d",
        config=None,
        adapt=adapt_nemotron_science,
    )
    with pytest.raises(ValueError, match="nothing to train on"):
        convert_source(
            spec,
            str(tmp_path),
            load_dataset=fake_load_dataset,
            render_lengths=lambda messages: {family: MAX_RENDERED_TOKENS + 1 for family in FAMILY_TOKEN_MAPS},
        )


def test_manifest_records_hashes_template_hashes_and_stage_order(tmp_path):
    specs = {
        name: SourceSpec(
            name=name, repo=f"example/{name}", revision="feedface", config=None, adapt=adapt_wildchat_rewild
        )
        for name in STAGE_ORDER
    }
    calls: list[str] = []

    def fake_load_dataset(repo, config, revision, split, streaming):
        name = repo.split("/")[-1]
        calls.append(name)

        def iterator():
            for index in range(3):
                yield {"conversation": [{"role": "user", "content": f"q{index}"}]}

        return iterator()

    import experiments.post_training.stage0.convert as convert_module  # noqa: PLC0415 - monkeypatched by this test

    original_sources = convert_module.STAGE0_SOURCES
    convert_module.STAGE0_SOURCES = tuple(specs[name] for name in STAGE_ORDER)
    try:
        manifest = build_manifest(
            str(tmp_path), load_dataset=fake_load_dataset, render_lengths=lambda m: {f: 8 for f in FAMILY_TOKEN_MAPS}
        )
    finally:
        convert_module.STAGE0_SOURCES = original_sources
    assert manifest["stage_order"] == list(STAGE_ORDER)
    assert [entry["name"] for entry in manifest["sources"]] == list(STAGE_ORDER)
    assert manifest["total_rows"] == 9
    assert set(manifest["template_sha256"]) == set(FAMILY_TOKEN_MAPS)
    assert calls == list(STAGE_ORDER), "sources convert in curriculum order"


# --- real tokenizers, real rows (skipif offline) ---


@pytest.mark.parametrize("family", list(FAMILY_TOKEN_MAPS))
def test_token_maps_are_single_token_on_their_pinned_tokenizers(family):
    if not _hf_reachable():
        pytest.skip("HF hub unreachable")
    from transformers import AutoTokenizer  # noqa: PLC0415 - optional dep guard

    repo, revision = _HF_REVISIONS[family]
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
    lengths = validate_token_map(FAMILY_TOKEN_MAPS[family], tokenizer)
    assert all(length == 1 for length in lengths.values())


def _synthetic_rows(source_name: str, count: int = 100) -> list[dict]:
    """Protocol rows with reasoning on every assistant turn, as the converter emits."""
    rows = []
    for index in range(count):
        if source_name == "wildchat-rewild":
            messages = [
                {"role": "user", "content": f"chat question {index}"},
                {"role": "assistant", "content": f"chat answer {index}"},
            ]
        elif source_name == "nemotron-science-reasoning":
            messages = [
                {"role": "user", "content": f"science question {index}"},
                {"role": "assistant", "content": f"answer {index}", "reasoning_content": f"science reasoning {index}"},
            ]
        else:
            messages = [
                {"role": "user", "content": f"terminal task {index}"},
                {
                    "role": "assistant",
                    "content": f"ls output {index}",
                    "reasoning_content": f"terminal reasoning {index}",
                    "tool_calls": [{"function": {"name": "bash", "arguments": {"command": f"ls {index}"}}}],
                },
                {"role": "tool", "content": f"out{index}"},
                {"role": "assistant", "content": f"done {index}"},
            ]
        rows.append({"source": source_name, "messages": messages})
    return rows


@pytest.mark.parametrize("family", list(FAMILY_TOKEN_MAPS))
@pytest.mark.parametrize("source_name", list(STAGE_ORDER))
def test_render_is_token_exact_on_real_tokenizers(family, source_name):
    """100 rows per source per family: reasoning spans sit between that family's
    single think tokens; no turn carries a spurious empty think block."""
    if not _hf_reachable():
        pytest.skip("HF hub unreachable")
    from transformers import AutoTokenizer  # noqa: PLC0415 - optional dep guard

    tokens = FAMILY_TOKEN_MAPS[family]
    repo, revision = _HF_REVISIONS[family]
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
    think_open = tokenizer.encode(tokens.think_open, add_special_tokens=False)
    think_close = tokenizer.encode(tokens.think_close, add_special_tokens=False)
    assert len(think_open) == 1 and len(think_close) == 1
    think_open, think_close = think_open[0], think_close[0]

    for row in _synthetic_rows(source_name):
        rendered = tokenizer.apply_chat_template(
            row["messages"], chat_template=shared_template(tokens), tokenize=True, return_dict=True
        )
        ids = rendered["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        reasoning_turns = sum(1 for m in row["messages"] if m.get("reasoning_content"))
        assert ids.count(think_open) == reasoning_turns, (family, source_name, row["messages"])
        assert ids.count(think_close) == reasoning_turns
        for index, token in enumerate(ids[:-1]):
            if token == think_open:
                assert ids[index + 1] != think_close, "empty think block injected"
        # Every turn closes, so the end-of-turn gate has an exact token to count.
        close = tokenizer.encode(tokens.turn_close, add_special_tokens=False)
        assert ids.count(close[0]) >= len(row["messages"])

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Probe-set tests: screening against training data, seeded selection, T budgets, the
P_agent Harbor config, and token-level buckets over rendered conversations."""

import gzip
import json

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from experiments.post_training.chat_template import FAMILY_TOKEN_MAPS, GRUG_TOKENS, QWEN_TOKENS, shared_template
from experiments.post_training.probe_sets.buckets import Bucket, tag_conversation
from experiments.post_training.probe_sets.builder import (
    HARBOR_SWE_BENCH_REF,
    P_AGENT_SIZE,
    P_CHAT_SIZE,
    P_THINK_OLYMPIADBENCH_SIZE,
    T_DOMAINS,
    T_TOKENS_PER_DOMAIN,
    ScreeningInputs,
    build_p_chat,
    build_p_think,
    build_t,
    p_agent_harbor_config,
    p_agent_instance_ids,
    require_pairwise_disjoint,
    screen,
    seeded_selection,
)
from experiments.post_training.probe_sets.sources import ProbeRow, SourceSpec, multi_turn_chats

LONG = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho"


def row(probe_id: str, text: str, **metadata) -> ProbeRow:
    return ProbeRow(probe_id=probe_id, source="test", messages=({"role": "user", "content": text},), metadata=metadata)


def distinct(index: int) -> str:
    return " ".join(f"word{index}x{position}" for position in range(20))


@pytest.fixture
def inputs(tmp_path):
    stage0 = tmp_path / "stage0.jsonl.gz"
    with gzip.open(stage0, "wt") as handle:
        handle.write(json.dumps({"messages": [{"role": "user", "content": f"prefix {LONG} suffix"}]}) + "\n")
    x_train = tmp_path / "x_train.jsonl"
    x_train.write_text(json.dumps({"prompt": distinct(900), "template_name": "mod_loop", "parameter": 5}) + "\n")
    r_train = tmp_path / "r_train.jsonl"
    r_train.write_text(json.dumps({"prompt": distinct(901)}) + "\n")
    return ScreeningInputs(stage0_files=(stage0,), task_train_files=(x_train, r_train), x_train_file=x_train)


def test_screen_drops_stage0_and_task_train_overlap(inputs):
    rows = [row("clean", distinct(1)), row("stage0", f"say {LONG}"), row("train", distinct(901))]
    kept, drops = screen(rows, inputs, x_task_rows=False)
    assert [r.probe_id for r in kept] == ["clean"]
    assert drops == {"dropped_stage0_overlap": 1, "dropped_task_train_overlap": 1}


def test_x_probe_rows_screen_by_function_identity(inputs):
    rows = [
        row("same-function", distinct(2), template_name="mod_loop", parameter=5),
        row("sibling-function", distinct(3), template_name="mod_loop", parameter=6),
    ]
    kept, drops = screen(rows, inputs, x_task_rows=True)
    assert [r.probe_id for r in kept] == ["sibling-function"]
    assert drops["dropped_task_train_overlap"] == 1


def test_seeded_selection_is_reproducible_and_raises_on_shortfall():
    rows = [row(f"r{index}", distinct(index)) for index in range(50)]
    assert seeded_selection(rows, 10, seed=1, name="P") == seeded_selection(rows, 10, seed=1, name="P")
    assert seeded_selection(rows, 10, seed=1, name="P") != seeded_selection(rows, 10, seed=2, name="P")
    with pytest.raises(ValueError, match="only 50 survive"):
        seeded_selection(rows, 51, seed=1, name="P")


def test_p_chat_keeps_every_constraint_prompt_and_fills_with_chats(inputs):
    constraints = [row(f"c{index}", distinct(index)) for index in range(40)] + [row("c-dirty", f"x {LONG}")]
    chats = [row(f"m{index}", distinct(1000 + index)) for index in range(P_CHAT_SIZE)]
    rows, counts = build_p_chat(constraints, chats, inputs, seed=0)
    assert len(rows) == P_CHAT_SIZE
    assert counts["constraint_prompts"] == 40 and counts["multi_turn_chats"] == P_CHAT_SIZE - 40
    assert "c-dirty" not in {r.probe_id for r in rows}


def test_p_think_raises_when_a_fixed_benchmark_is_contaminated(inputs):
    def benchmark(name, count, offset):
        return [row(f"{name}-{index}", distinct(offset + index)) for index in range(count)]

    benchmarks = {
        "aime_2024": benchmark("aime_2024", 30, 0),
        "aime_2025": benchmark("aime_2025", 30, 100),
        "hmmt_feb_2025": benchmark("hmmt_feb_2025", 30, 200),
        "olympiadbench": benchmark("olympiadbench", 300, 300),
    }
    rows, counts = build_p_think(benchmarks, inputs, seed=0)
    assert len(rows) == 90 + P_THINK_OLYMPIADBENCH_SIZE
    assert counts["olympiadbench"] == P_THINK_OLYMPIADBENCH_SIZE
    benchmarks["aime_2025"][3] = row("aime_2025-3", f"find {LONG}")
    with pytest.raises(ValueError, match="aime_2025 overlaps training data"):
        build_p_think(benchmarks, inputs, seed=0)


def test_probe_sets_must_be_pairwise_disjoint():
    require_pairwise_disjoint({"A": [row("a", distinct(1))], "B": [row("b", distinct(2))]})
    with pytest.raises(ValueError, match="A and B overlap"):
        require_pairwise_disjoint({"A": [row("a", f"one {LONG}")], "B": [row("b", f"two {LONG}")]})


def test_t_fills_every_domain_budget_from_whole_documents():
    stream = [(domain, "tok " * 60_000) for _ in range(5) for domain in (*T_DOMAINS, "Enron Emails")]
    rows, tokens = build_t(stream, lambda text: len(text.split()))
    assert set(tokens) == set(T_DOMAINS)
    assert all(T_TOKENS_PER_DOMAIN <= count < T_TOKENS_PER_DOMAIN + 60_000 for count in tokens.values())
    assert {r["domain"] for r in rows} == set(T_DOMAINS)
    with pytest.raises(ValueError, match="ended before every domain"):
        build_t(stream[:11], lambda text: len(text.split()))


def test_p_agent_tasks_are_seeded_and_the_harbor_config_is_narrowed_and_pinned():
    instance_ids = [f"repo__repo-{index}" for index in range(500)]
    chosen = p_agent_instance_ids(instance_ids, seed=4)
    assert len(chosen) == P_AGENT_SIZE and chosen == sorted(chosen)
    assert chosen == p_agent_instance_ids(list(reversed(instance_ids)), seed=4)
    config = p_agent_harbor_config(chosen)
    [dataset] = config["datasets"]
    assert dataset["ref"] == HARBOR_SWE_BENCH_REF and dataset["task_names"] == chosen
    assert config["agents"] == [{"name": "terminus-2"}]


def test_source_specs_require_a_commit():
    with pytest.raises(ValueError, match="40-character commit"):
        SourceSpec("org/data", "main", "train")


def test_multi_turn_chats_cut_before_the_last_reply_and_need_two_user_turns():
    conversations = [
        {
            "prompt_id": "two",
            "messages": [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "q2"},
                {"role": "assistant", "content": "a2"},
            ],
        },
        {"prompt_id": "one", "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]},
    ]
    [chat] = multi_turn_chats(load_dataset=lambda *args, **kwargs: conversations)
    assert chat.probe_id == "ultrachat-two" and chat.messages[-1] == {"role": "user", "content": "q2"}
    assert len(chat.messages) == 3


# --- token-level buckets ---


def tiny_tokenizer(tokens) -> PreTrainedTokenizerFast:
    backend = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation()])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    delimiters = [
        tokens.bos,
        tokens.turn_open_prefix,
        tokens.turn_open_suffix,
        tokens.turn_close,
        tokens.think_open,
        tokens.think_close,
    ]
    tokenizer.add_special_tokens({"additional_special_tokens": [token for token in delimiters if token]})
    return tokenizer


def bucket_words(tokenizer, family, messages, tagged) -> dict[Bucket, list[str]]:
    text = tokenizer.apply_chat_template(
        messages, chat_template=shared_template(FAMILY_TOKEN_MAPS[family]), tokenize=False
    )
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    words: dict[Bucket, list[str]] = {}
    for (start, end), bucket in zip(offsets, tagged.buckets, strict=True):
        words.setdefault(bucket, []).append(text[start:end])
    return words


TOOL_TRAJECTORY = [
    {"role": "user", "content": "fix the failing test"},
    {
        "role": "assistant",
        "reasoning_content": "inspect the repo first",
        "content": "",
        "tool_calls": [{"function": {"name": "bash", "arguments": '{"cmd": "ls"}'}}],
    },
    {"role": "tool", "content": "setup.py tests"},
    {"role": "assistant", "reasoning_content": "done now", "content": "the patch is applied"},
]


@pytest.mark.parametrize("tokens,family", [(GRUG_TOKENS, "grug"), (QWEN_TOKENS, "qwen3_moe")])
def test_structured_tool_trajectory_buckets(tokens, family):
    tokenizer = tiny_tokenizer(tokens)
    tagged = tag_conversation(tokenizer, family, TOOL_TRAJECTORY)
    words = bucket_words(tokenizer, family, TOOL_TRAJECTORY, tagged)
    assert words[Bucket.REASONING] == ["inspect", "the", "repo", "first", "done", "now"]
    assert words[Bucket.TOOL_NAME] == ['"', "bash", '"']
    assert words[Bucket.TOOL_ARGUMENTS] == ["{", '"', "cmd", '"', ":", '"', "ls", '"', "}"]
    assert words[Bucket.FINAL_MESSAGE] == ["the", "patch", "is", "applied"]
    think = [tokens.think_open, tokens.think_close]
    assert words[Bucket.DELIMITER] == [*think, tokens.turn_close, *think, tokens.turn_close]
    assert "setup" in words[Bucket.CONTEXT] and "fix" in words[Bucket.CONTEXT]
    assert tagged.assistant_mask == [int(bucket is not Bucket.CONTEXT) for bucket in tagged.buckets]
    assert {turn for turn, bit in zip(tagged.assistant_turn, tagged.assistant_mask, strict=True) if bit} == {0, 1}
    assert {turn for turn, bit in zip(tagged.assistant_turn, tagged.assistant_mask, strict=True) if not bit} == {-1}


def test_terminus_json_reply_buckets_and_reasoning_deciles():
    reply = (
        '{"analysis": "tests fail on import", "plan": "list files", '
        '"commands": [{"keystrokes": "ls -la", "duration": 0.1}]}'
    )
    messages = [
        {"role": "user", "content": "fix the failing test"},
        {"role": "assistant", "content": reply},
        {"role": "user", "content": "setup.py tests"},
        {
            "role": "assistant",
            "content": '{"analysis": "all good", "plan": "stop", "commands": [], "task_complete": true}',
        },
    ]
    tokenizer = tiny_tokenizer(QWEN_TOKENS)
    tagged = tag_conversation(tokenizer, "qwen3_moe", messages)
    words = bucket_words(tokenizer, "qwen3_moe", messages, tagged)
    assert words[Bucket.REASONING] == ["tests", "fail", "on", "import", "list", "files", "all", "good", "stop"]
    assert words[Bucket.TOOL_ARGUMENTS] == ["ls", "-", "la"]
    assert "analysis" in words[Bucket.ASSISTANT_TEXT] and "complete" in words[Bucket.FINAL_MESSAGE]
    first_turn = [
        d
        for d, b, t in zip(tagged.reasoning_decile, tagged.buckets, tagged.assistant_turn, strict=True)
        if b is Bucket.REASONING and t == 0
    ]
    assert first_turn == sorted(first_turn) and first_turn[0] == 0 and first_turn[-1] == 8

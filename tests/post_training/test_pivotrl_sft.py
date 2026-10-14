# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from marin.rl.nemotron_pivot import VALIDATION_FILENAME
from marin.rl.pass_rates import PASS_RATES_FILENAME

from experiments.post_training.pivotrl.sft import (
    EXAMPLES_FILENAME,
    ROWS_PER_BATCH,
    PassBand,
    band_mask,
    example_tokens_bound,
    grid_bands,
    write_sft_examples,
    write_validation_examples,
)
from experiments.post_training.pivotrl.sft_templates import GRUG_FINAL_TURN_TEMPLATE, QWEN3_FINAL_TURN_TEMPLATE

TOOLS = [{"type": "function", "name": "bash", "parameters": {"type": "object", "properties": {}}}]


def _call(command: str) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": command, "type": "function", "function": {"name": "bash", "arguments": command}}],
    }


def _swe_row(trajectory_id: int, prompt: list[dict], command: str, passed: int = 0, total: int = 8) -> dict:
    record = {"expected_action": {"type": "function_call", "name": "bash", "arguments": {"command": command}}}
    request = {"input": [], "tools": TOOLS, "max_output_tokens": 512}
    return {
        "passed": passed,
        "total": total,
        "prompt": prompt,
        "extra_info": {
            "source_id": f"swe:{trajectory_id}",
            "trajectory_id": trajectory_id,
            "index": len(prompt),
            "nemotron_ultra": {"record_json": json.dumps(record), "request_json": json.dumps(request)},
        },
    }


@pytest.mark.parametrize("template", [GRUG_FINAL_TURN_TEMPLATE, QWEN3_FINAL_TURN_TEMPLATE])
def test_final_turn_template_supervises_only_the_last_turn(gpt2_tokenizer, template):
    messages = [
        {"role": "user", "content": "Fix the bug."},
        _call("EARLIER_ACTION"),
        {"role": "tool", "tool_call_id": "EARLIER_ACTION", "content": "a.py"},
        _call("FINAL_ACTION"),
    ]

    encoded = gpt2_tokenizer.apply_chat_template_with_masks([messages], chat_template=template, tools=None)
    ids, mask = encoded["input_ids"][0], encoded["assistant_masks"][0]
    supervised = gpt2_tokenizer.decode([token for token, kept in zip(ids, mask, strict=True) if kept])

    assert "FINAL_ACTION" in supervised
    assert "EARLIER_ACTION" not in supervised
    assert "EARLIER_ACTION" in gpt2_tokenizer.decode(ids)


def _write_pass_rates(directory, rows: list[dict], prompt_tokens: list[int]) -> None:
    """Pass rates for ``rows``, and samples recording each row's served prompt length."""
    pq.write_table(pa.Table.from_pylist(rows), directory / PASS_RATES_FILENAME)
    (directory / "samples").mkdir()
    samples = [
        {"row_id": row["extra_info"]["source_id"], "prompt_tokens": tokens}
        for row, tokens in zip(rows, prompt_tokens, strict=True)
    ]
    pq.write_table(pa.Table.from_pylist(samples), directory / "samples" / "chunk-000000.parquet")


def _write_examples(directory, band: PassBand, max_tokens: int = 1024) -> list[dict]:
    output = directory / "out"
    write_sft_examples(
        pass_rates_path=str(directory),
        samples_path=str(directory),
        band=band,
        output_path=str(output),
        task="experiments.post_training.pivotrl.task:PIVOT_TOOL_CALL",
        enable_thinking=False,
        max_tokens=max_tokens,
    )
    with gzip.open(output / EXAMPLES_FILENAME, "rt") as stream:
        return [json.loads(line) for line in stream]


def test_write_sft_examples_appends_the_expert_action_and_drops_rows_that_may_not_fit(tmp_path):
    prompt = [{"role": "user", "content": "Fix the bug."}]
    rows = [_swe_row(1, prompt, "ls"), _swe_row(2, prompt, "cat a.py")]
    # The second prompt was served at nearly the whole sequence length, leaving no room for its action.
    _write_pass_rates(tmp_path, rows, prompt_tokens=[100, 1000])

    (example,) = _write_examples(tmp_path, PassBand("all", 0, 8))

    assert example["messages"][:-1] == prompt
    (call,) = example["messages"][-1]["tool_calls"]
    assert call["function"] == {"name": "bash", "arguments": '{"command": "ls"}'}
    # Tools are the chat-completions form the policy is served with.
    assert example["chat_template_kwargs"]["tools"][0]["function"]["name"] == "bash"
    assert example["chat_template_kwargs"]["enable_thinking"] is False
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert (manifest["examples"], manifest["too_long"]) == (1, 1)


def test_example_length_bound_covers_the_rendered_example(gpt2_tokenizer):
    prompt = [{"role": "user", "content": "Fix the bug. " + "naïve context 🙂 " * 200}]
    action = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "r", "type": "function", "function": {"name": "bash", "arguments": '{"command": "ls é"}'}}
        ],
    }
    tokenizer = gpt2_tokenizer.with_chat_template(GRUG_FINAL_TURN_TEMPLATE)
    served = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tools=None)
    example = tokenizer.apply_chat_template([*prompt, action], tools=None)

    assert len(example) <= example_tokens_bound(len(served), action)


def test_grid_bands_select_by_pass_count_and_skip_excluded_rows():
    passed = [0, 1, 3, 4, 7, 8, 15, 16]
    rows = [{"row": p, "passed": p, "total": 16} for p in passed] + [{"row": -1, "passed": 0, "total": 0}]
    table = pa.Table.from_pylist(rows)

    selected = {band.name: table.filter(band_mask(table, band))["row"].to_pylist() for band in grid_bands(16)}

    assert selected == {
        "all": passed,
        "mixed": [1, 3, 4, 7, 8, 15],
        "pivots": [1, 3, 4, 7],
        "hard": [1, 3],
        "unsolved": [0],
    }


def test_band_rejects_rows_sampled_fewer_times_than_it_counts():
    table = pa.Table.from_pylist([{"passed": 3, "total": 8}])

    with pytest.raises(ValueError, match="does not fit"):
        band_mask(table, PassBand("all", 0, 16))


def test_write_sft_examples_selects_the_band_across_read_batches(tmp_path):
    prompt = [{"role": "user", "content": "Fix the bug."}]
    rows = [_swe_row(index, prompt, f"echo {index}", passed=index % 9) for index in range(3 * ROWS_PER_BATCH)]
    _write_pass_rates(tmp_path, rows, prompt_tokens=[100] * len(rows))

    examples = _write_examples(tmp_path, PassBand("pivots", 1, 3))

    commands = [example["messages"][-1]["tool_calls"][0]["function"]["arguments"] for example in examples]
    expected = [index for index in range(3 * ROWS_PER_BATCH) if 1 <= index % 9 <= 3]
    assert commands == [json.dumps({"command": f"echo {index}"}) for index in expected]


def test_write_validation_examples_measures_rows_and_drops_long_ones(tmp_path, gpt2_tokenizer_path):
    short = [{"role": "user", "content": "Fix the bug."}]
    long = [{"role": "user", "content": "Fix the bug. " + "context " * 2000}]
    rows = [_swe_row(1, short, "ls"), _swe_row(2, long, "cat a.py")]
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / VALIDATION_FILENAME)
    output = tmp_path / "out"

    write_validation_examples(
        rows_path=str(tmp_path),
        output_path=str(output),
        task="experiments.post_training.pivotrl.task:PIVOT_TOOL_CALL",
        tokenizer=gpt2_tokenizer_path,
        chat_template=GRUG_FINAL_TURN_TEMPLATE,
        enable_thinking=False,
        max_tokens=1024,
    )

    with gzip.open(output / EXAMPLES_FILENAME, "rt") as stream:
        (example,) = [json.loads(line) for line in stream]
    assert example["messages"][:-1] == short
    assert json.loads((output / "manifest.json").read_text())["too_long"] == 1

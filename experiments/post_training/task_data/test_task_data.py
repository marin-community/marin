# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Task-data tests: X synthesis and direction-dispatched grading, R exclusion with n-gram
overlap, rendered-token trace caps, and tokenized M1 rows with assistant masks."""

import ast
import json
import os

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from experiments.post_training.chat_template import GRUG_TOKENS, QWEN_TOKENS
from experiments.post_training.task_data.code_x import (
    DIRECTION_INPUT,
    DIRECTION_OUTPUT,
    MAX_CALLS_PER_FUNCTION,
    PARAMETER_RANGE,
    PROBE_SPLIT,
    TEMPLATE_FAMILIES,
    TRAIN_SPLIT,
    exclude_public_cruxeval,
    extract_input_call,
    function_split,
    grade_x_task,
    synthesize_x_tasks,
    write_split as write_x_split,
)
from experiments.post_training.task_data.expert_outputs import (
    CompletionRow,
    grade_r_task,
    sample_completions,
    tokenized_conversation,
    write_m1_rows,
)
from experiments.post_training.task_data.expert_traces import (
    ExpertResponse,
    chat_completion,
    collect_traces,
    write_trace_batch,
)
from experiments.post_training.task_data.math_r import (
    R_POOL_SPEC,
    build_r_tasks,
    draw_r_tasks,
    is_reserved_for_probe,
    load_r_pool,
    ngrams,
    normalize_problem,
    r_prompt,
)


def _valid_input(code: str, answer: str) -> int | None:
    """One input that produces the answer without echoing the answer literal."""
    namespace: dict = {}
    exec(code, namespace)  # noqa: S102 - synthesized pool code under test
    target = ast.literal_eval(answer)
    for candidate in list(range(-5000, 5001)) + list(range(-150_000, 150_001, 7)):
        if namespace["f"](candidate) == target and str(candidate) != answer.strip():
            return candidate
    return None


class FakeChatClient:
    """An OpenAI-compatible chat client that replays canned responses and records request bodies."""

    def __init__(self, messages: list[dict], finish_reasons: list[str] | None = None, failures: int = 0):
        self.messages = list(messages)
        self.finish_reasons = list(finish_reasons or ["stop"] * len(messages))
        self.failures = failures
        self.bodies: list[dict] = []

    def complete(self, body):
        self.bodies.append(dict(body))
        if self.failures:
            self.failures -= 1
            raise ConnectionError("transient")
        return {"choices": [{"message": self.messages.pop(0), "finish_reason": self.finish_reasons.pop(0)}]}


def word_count(text: str) -> int:
    return len(text.split())


def tiny_tokenizer(tokens) -> PreTrainedTokenizerFast:
    """A whitespace word-level tokenizer whose vocabulary holds the family's delimiters as single tokens."""
    words = "the answer is boxed 4 compute 2 + add them step one assistant user \\boxed{4} $ . , : ? \n".split(" ")
    vocab = {"[UNK]": 0, **{word: index + 1 for index, word in enumerate(dict.fromkeys(words))}}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    delimiters = [tokens.turn_open_prefix, tokens.turn_open_suffix, tokens.turn_close, tokens.think_open, tokens.think_close]
    tokenizer.add_special_tokens({"additional_special_tokens": [token for token in [tokens.bos, *delimiters] if token]})
    return tokenizer


# --- X synthesis, function-disjoint splits, prompts ---


def test_synth_meets_both_quotas_with_function_disjoint_splits():
    tasks = synthesize_x_tasks(train_count=60, probe_count=20, seed=7)
    assert sum(task.split == TRAIN_SPLIT for task in tasks) == 60
    assert sum(task.split == PROBE_SPLIT for task in tasks) == 20
    for task in tasks:
        assert task.split == function_split(7, task.template_name, task.parameter)
    train_functions = {(t.template_name, t.parameter) for t in tasks if t.split == TRAIN_SPLIT}
    probe_functions = {(t.template_name, t.parameter) for t in tasks if t.split == PROBE_SPLIT}
    assert not train_functions & probe_functions, "P_X and X train can never share a function"
    assert len({task.task_id for task in tasks}) == len(tasks)
    assert {t.direction for t in tasks} == {DIRECTION_OUTPUT, DIRECTION_INPUT}


def test_the_probe_split_holds_enough_functions_for_p_x():
    probe_functions = [
        (name, parameter)
        for name, _ in TEMPLATE_FAMILIES
        for parameter in PARAMETER_RANGE
        if function_split(3, name, parameter) == PROBE_SPLIT
    ]
    assert len(probe_functions) >= 80, "P_X's 1,000 prompts must span many functions, not a handful"
    assert len(probe_functions) * MAX_CALLS_PER_FUNCTION >= 1_000


def test_synth_is_seeded():
    assert synthesize_x_tasks(train_count=8, probe_count=2, seed=7) == synthesize_x_tasks(
        train_count=8, probe_count=2, seed=7
    )
    assert synthesize_x_tasks(train_count=8, probe_count=2, seed=8) != synthesize_x_tasks(
        train_count=8, probe_count=2, seed=7
    )


def test_x_prompts_hide_what_the_model_must_produce():
    for task in synthesize_x_tasks(train_count=20, probe_count=5, seed=7):
        question = task.prompt.rsplit("```", 1)[-1].replace(task.call, "")
        if task.direction == DIRECTION_OUTPUT:
            assert task.call in task.prompt and task.answer not in question
        else:
            assert task.answer in task.prompt and task.call not in task.prompt


def test_cruxeval_exclusion_requires_the_file_and_compares_code_inside_prompts(tmp_path):
    tasks = synthesize_x_tasks(train_count=3, probe_count=1, seed=3)
    with pytest.raises(FileNotFoundError, match="required CruxEval exclusion input"):
        exclude_public_cruxeval(tasks, tmp_path / "missing.jsonl")
    wrapped = "You are given a Python function.\n\n" + tasks[0].code + "\n\nWhat does f(2) return?"
    blocklist = tmp_path / "cruxeval_prompts.jsonl"
    blocklist.write_text(json.dumps({"prompt": wrapped}) + "\n")
    kept, excluded = exclude_public_cruxeval(tasks, blocklist)
    same_function = [task for task in tasks if task.code == tasks[0].code]
    assert excluded == len(same_function)
    assert [task.task_id for task in kept] == [task.task_id for task in tasks if task.code != tasks[0].code]


# --- X grading: direction dispatch, extraction, no echo rewards ---


OUTPUT_TASK = {"direction": DIRECTION_OUTPUT, "code": "def f(x):\n    return x + 1\n", "call": "f(41)", "answer": "42"}
INPUT_TASK = {"direction": DIRECTION_INPUT, "code": "def f(x):\n    return x % 7\n", "call": "f(12)", "answer": "5"}


def test_output_direction_grades_the_boxed_answer_only():
    assert grade_x_task(OUTPUT_TASK, "Tracing the loop: the return value is $\\boxed{42}$.")
    assert not grade_x_task(OUTPUT_TASK, "The return value is $\\boxed{43}$.")
    assert not grade_x_task(OUTPUT_TASK, "42"), "a bare literal is not a stated answer"
    assert not grade_x_task(OUTPUT_TASK, "The function returns 42."), "the prompt asks for a boxed answer"


def test_input_direction_requires_a_real_call_and_rejects_echoes():
    assert grade_x_task(INPUT_TASK, "Tracing: f(12) gives 5.")
    assert not grade_x_task(INPUT_TASK, "5"), "the bare echoed output literal must not pass"
    assert not grade_x_task(INPUT_TASK, "f(5)"), "the call must differ from echoing the output"
    assert not grade_x_task(INPUT_TASK, "f(3)"), "a call that executes to the wrong value fails"
    assert extract_input_call("first f(3) then finally f(12)", answer="5") == "f(12)"
    assert extract_input_call("g(12) is not our function", answer="5") is None


def test_written_x_splits_grade_reference_answers(tmp_path):
    tasks = synthesize_x_tasks(train_count=30, probe_count=10, seed=7)
    counts = write_x_split(tasks, tmp_path)
    assert counts == {TRAIN_SPLIT: 30, PROBE_SPLIT: 10}
    for split in (TRAIN_SPLIT, PROBE_SPLIT):
        for line in (tmp_path / f"x_{split}.jsonl").read_text().splitlines():
            row = json.loads(line)
            if row["direction"] == DIRECTION_OUTPUT:
                assert grade_x_task(row, f"Tracing carefully, the value is $\\boxed{{{row['answer']}}}$.")
                assert not grade_x_task(row, f"The value is \\boxed{{9{row['answer']}}}.")
            else:
                valid = _valid_input(row["code"], row["answer"])
                assert valid is not None, f"no valid input found for {row['task_id']}"
                assert grade_x_task(row, f"The input is f({valid}).")
                assert not grade_x_task(row, f"f({row['answer']})")


# --- R: pool loader, exclusions, seeded splits, verifier-parsable answers ---


def test_reserved_sources_and_ngram_overlap_exclude_reserved_problems(tmp_path):
    assert is_reserved_for_probe("amc_aime")
    assert is_reserved_for_probe("OlympiadBench")
    assert not is_reserved_for_probe("olympiads"), "NuminaMath's olympiad sources are not OlympiadBench"

    reserved = (
        "In triangle ABC the measure of angle A equals thirty degrees while the circumradius equals four so "
        "compute the exact area of the triangle as a single simplified radical expression"
    )
    rephrased = (
        "Suppose triangle ABC has angle A of thirty degrees and a circumradius of four so compute the exact area "
        "of the triangle as a single simplified radical expression"
    )
    disjoint = (
        "Compute the sum of every perfect square below one thousand and then take that remainder modulo seven "
        "and report the resulting integer in your final boxed answer statement"
    )
    reserved_pool = tmp_path / "reserved.jsonl"
    reserved_pool.write_text(json.dumps({"problem": reserved}) + "\n")
    pool = [
        {"problem": reserved, "answer": "4", "source": "olympiads"},
        {"problem": rephrased, "answer": "4", "source": "math_train"},
        {"problem": disjoint, "answer": "6", "source": "olympiads"},
        {"problem": "An AIME problem", "answer": "1", "source": "amc_aime"},
        {"problem": "", "answer": "3", "source": "math_train"},
    ]
    tasks, counts = draw_r_tasks(pool, seed=1, reserved_pool_path=reserved_pool)
    assert [task.problem for task in tasks] == [disjoint]
    assert counts["dropped_p_think_overlap"] == 2
    assert counts["dropped_reserved_source"] == 1
    assert counts["dropped_malformed"] == 1


def test_missing_reserved_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="reserved-benchmark exclusion input"):
        draw_r_tasks([{"problem": "p", "answer": "1", "source": "math"}], seed=1, reserved_pool_path=tmp_path / "gone")


def test_draw_is_seeded_and_reproducible():
    pool = [{"problem": f"problem number {index} plus one", "answer": str(index), "source": "math"} for index in range(60)]
    tasks_a, _ = draw_r_tasks(pool, seed=3, reserved_pool_path=os.devnull)
    tasks_b, _ = draw_r_tasks(pool, seed=3, reserved_pool_path=os.devnull)
    tasks_c, _ = draw_r_tasks(pool, seed=4, reserved_pool_path=os.devnull)
    assert tasks_a == tasks_b
    assert tasks_a != tasks_c
    assert {t.split for t in tasks_a} == {TRAIN_SPLIT, PROBE_SPLIT}


def test_unparsable_answers_are_dropped_and_counted():
    pool = [
        {"problem": "compute two plus two", "answer": "4", "source": "math"},
        {"problem": "the answer is malformed latex", "answer": "\\", "source": "math"},
    ]
    tasks, counts = draw_r_tasks(pool, seed=1, reserved_pool_path=os.devnull)
    assert [t.answer for t in tasks] == ["4"]
    assert counts["dropped_unparsable_answer"] == 1


def test_pool_loader_records_the_pinned_revision_and_drops_unanswered_rows():
    rows = [
        {"problem": "p1", "solution": "work ... \\boxed{7}", "source": "math"},
        {"problem": "p2", "solution": "no boxed answer here", "source": "math"},
    ]

    def fake_load_dataset(repo, config, revision, split, streaming):
        assert (repo, revision, split) == (R_POOL_SPEC.repo, R_POOL_SPEC.revision, R_POOL_SPEC.split)
        return iter(rows)

    pool, entry = load_r_pool(load_dataset=fake_load_dataset)
    assert pool == [{"problem": "p1", "answer": "7", "source": "math"}]
    assert entry["revision"] == R_POOL_SPEC.revision and entry["dropped_without_boxed_answer"] == 1


def test_build_r_tasks_writes_prompted_splits_and_the_manifest(tmp_path):
    def fake_load_dataset(repo, config, revision, split, streaming):
        return iter(
            {"problem": f"Compute {i} times three minus one.", "solution": f"... \\boxed{{{3 * i - 1}}}", "source": "m"}
            for i in range(1, 41)
        )

    manifest = build_r_tasks(tmp_path, seed=9, reserved_pool_path=os.devnull, load_dataset=fake_load_dataset)
    assert json.loads((tmp_path / "manifest.json").read_text()) == manifest
    assert manifest["pool"]["revision"] == R_POOL_SPEC.revision and manifest["splits"]["probe"] > 0
    for split in (TRAIN_SPLIT, PROBE_SPLIT):
        for line in (tmp_path / f"r_{split}.jsonl").read_text().splitlines():
            row = json.loads(line)
            assert row["prompt"] == r_prompt(row["problem"])
            assert grade_r_task(row, f"Working it out, the final answer is $\\boxed{{{row['answer']}}}$.")
            assert not grade_r_task(row, "The final answer is $\\boxed{999999}$.")


def test_normalize_problem_collides_retypeset_duplicates():
    assert normalize_problem("Compute  2 + 2!") == normalize_problem("compute 2 + 2")
    assert ngrams(normalize_problem("\\frac{1}{2}")) == ngrams(normalize_problem("{1}{2}"))


# --- endpoint client: sampling params, reasoning, retries, truncation ---


def test_chat_completion_sends_sampling_params_keeps_reasoning_and_retries():
    client = FakeChatClient([{"content": "the answer", "reasoning_content": "the thinking"}], failures=2)
    response = chat_completion(
        client, model="glm-5.3", messages=[{"role": "user", "content": "p"}], temperature=0.6, max_tokens=20_000
    )
    assert response == ExpertResponse(reasoning="the thinking", content="the answer", truncated=False)
    assert client.bodies[-1] == {
        "model": "glm-5.3",
        "messages": [{"role": "user", "content": "p"}],
        "temperature": 0.6,
        "max_tokens": 20_000,
    }
    with pytest.raises(ConnectionError):
        chat_completion(FakeChatClient([{}], failures=3), model="m", messages=[], temperature=0.0, max_tokens=8)


def test_trace_cap_counts_rendered_thinking_and_drops_truncated_traces():
    client = FakeChatClient(
        [
            {"content": "done quickly", "reasoning_content": "short thought"},
            {"content": "done slowly", "reasoning_content": "an enormous wall of thinking text that overflows"},
            {"content": "cut off", "reasoning_content": "x"},
        ],
        finish_reasons=["stop", "stop", "length"],
    )
    batch = collect_traces(
        [{"task_id": f"t{index}", "prompt": "p"} for index in range(3)],
        client=client,
        model="glm-5.3",
        family="qwen3_moe",
        temperature=0.7,
        max_tokens=32_000,
        token_cap=6,
        token_count=word_count,
        max_concurrent=1,
    )
    assert [trace["task_id"] for trace in batch.traces] == ["t0"]
    assert batch.traces[0]["reasoning"] == "short thought" and batch.traces[0]["content"] == "done quickly"
    report = write_trace_batch(batch, os.devnull, temperature=0.7, max_tokens=32_000)
    assert report["dropped"] == 2 and report["dropped_share"] == pytest.approx(2 / 3)


def test_collect_traces_rejects_max_tokens_at_or_below_the_cap():
    with pytest.raises(ValueError, match="truncate"):
        collect_traces(
            [],
            client=FakeChatClient([]),
            model="m",
            family="grug",
            temperature=0.0,
            max_tokens=16_384,
            token_cap=16_384,
            token_count=word_count,
            max_concurrent=1,
        )


# --- M1 rows: graded, capped, tokenized with assistant masks ---


def test_sample_completions_grades_content_and_drops_over_cap():
    client = FakeChatClient(
        [
            {"content": "$\\boxed{4}$", "reasoning_content": "added them"},
            {"content": "$\\boxed{5}$", "reasoning_content": ""},
            {"content": "$\\boxed{4}$", "reasoning_content": "far too many words of thinking for this cap"},
        ]
    )
    tasks = [{"task_id": f"r-{index}", "prompt": "Compute 2+2.", "answer": "4"} for index in range(3)]
    batch = sample_completions(
        tasks,
        family="grug",
        temperature=0.6,
        max_tokens=100,
        token_cap=8,
        client=client,
        model="expert-r",
        grade=grade_r_task,
        token_count=word_count,
        max_concurrent=1,
    )
    assert client.bodies[0]["temperature"] == 0.6 and client.bodies[0]["max_tokens"] == 100
    assert [(row.task_id, row.graded_correct) for row in batch.rows] == [("r-0", True), ("r-1", False)]
    assert batch.dropped_over_cap == 1


@pytest.mark.parametrize("tokens,family", [(GRUG_TOKENS, "grug"), (QWEN_TOKENS, "qwen3_moe")])
def test_tokenized_conversation_masks_reasoning_answer_and_turn_close(tokens, family):
    tokenizer = tiny_tokenizer(tokens)
    prompt = [{"role": "user", "content": "compute 2 + 2"}]
    response = ExpertResponse(reasoning="add them", content="the answer is \\boxed{4}", truncated=False)
    row = tokenized_conversation(tokenizer, family, prompt, response)
    ids, mask = row["input_ids"], row["assistant_masks"]
    supervised = [token for token, bit in zip(tokenizer.convert_ids_to_tokens(ids), mask, strict=True) if bit]
    assert supervised == [
        tokens.think_open, "add", "them", tokens.think_close, "the", "answer", "is", "\\boxed{4}", tokens.turn_close
    ]
    assert ids[-1] == tokenizer.convert_tokens_to_ids(tokens.turn_close)
    assert mask == sorted(mask), "the supervised span is one suffix"


def test_m1_writer_keeps_graded_rows_with_prompt_ids_and_tokens(tmp_path):
    tokenizer = tiny_tokenizer(GRUG_TOKENS)
    prompt = [{"role": "user", "content": "compute 2 + 2"}]
    rows = [
        CompletionRow("r-1", prompt, ExpertResponse("add them", "the answer is \\boxed{4}", False), True),
        CompletionRow("r-2", prompt, ExpertResponse("", "the answer is 2", False), False),
    ]
    output = tmp_path / "m1.jsonl"
    report = write_m1_rows(
        rows,
        str(output),
        tokenizer=tokenizer,
        family="grug",
        data_source="task_R",
        teacher_route="E_R",
        temperature=0.6,
        max_tokens=20_000,
    )
    assert report["kept"] == 1 and report["dropped_ungraded"] == 1
    [line] = [json.loads(text) for text in output.read_text().splitlines()]
    assert line["task_id"] == "r-1" and line["prompt"] == prompt
    assert line["data_source"] == "task_R" and line["teacher_route"] == "E_R"
    assert len(line["input_ids"]) == len(line["assistant_masks"]) and any(line["assistant_masks"])

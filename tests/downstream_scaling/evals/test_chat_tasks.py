# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import json

import pytest
from zephyr.dataset import ShardInfo

from experiments.downstream_scaling.evals.framework.schema import read_prompt_rows
from experiments.downstream_scaling.evals.framework.xregion import ledger
from experiments.downstream_scaling.evals.tasks import (
    gsm8k_chat,
    gsm8k_chat_reference,
    gsm8k_two_solution,
    humaneval_chat,
)


class FakeChatTokenizer:
    bos_token = "<bos>"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        history = "".join(f"<{message['role']}>{message['content']}</{message['role']}>" for message in messages)
        return f"{self.bos_token}{history}<assistant>"


class FakeTask:
    def __init__(self, docs):
        self.docs = docs

    def test_docs(self):
        return self.docs


def configure_tokenizer(monkeypatch, module):
    monkeypatch.setattr(module, "discover_hf_checkpoints", lambda _: ["/tokenizer/checkpoint"])
    monkeypatch.setattr(module, "load_tokenizer", lambda _: FakeChatTokenizer())


def test_render_two_solution_exemplars_preserves_solutions_and_blank_lines():
    exemplars = [
        gsm8k_two_solution.TwoSolutionExemplar("What is 20 + 22?", "Add them.\n#### 42", "Double 21.\n#### 42"),
        gsm8k_two_solution.TwoSolutionExemplar("What is 3 * 4?", "Multiply.\n#### 12", "Add four three times.\n#### 12"),
    ]

    assert gsm8k_two_solution.render_exemplar_block(exemplars) == (
        "Question: What is 20 + 22?\n"
        "Solution 1: Add them.\n#### 42\n"
        "Solution 2: Double 21.\n#### 42\n\n"
        "Question: What is 3 * 4?\n"
        "Solution 1: Multiply.\n#### 12\n"
        "Solution 2: Add four three times.\n#### 12\n\n"
    )


@pytest.fixture
def two_solution_prompts_config(tmp_path):
    return gsm8k_two_solution.TwoSolutionGSM8KPromptsConfig(
        output_path=str(tmp_path),
        advisor_model_path="/advisor",
        n_problems=1,
        num_exemplars=1,
        num_fewshot=5,
        fewshot_seed=1234,
        n_samples=8,
        temperature=0.4,
        top_k=16,
        max_tokens=512,
        max_model_len=8192,
        seed=42,
        stop=("Question:",),
        worker_pools=(),
        heartbeat_timeout=120.0,
        poll_backoff=0.0,
    )


def test_two_solution_worker_writes_prompts_once_on_resume(tmp_path, monkeypatch, two_solution_prompts_config):
    ledger_path = str(tmp_path / "ledger")
    ledger.ensure_manifest(ledger_path, [{"chunk_id": 0}])
    prompt_path = tmp_path / "prompts.jsonl.gz"
    row = {"id": "gsm8k/test/0", "prompt": "Solution 2:", "ground_truth": "42", "metadata": {}}

    def write_prompts(_config, _llm):
        with gzip.open(prompt_path, "at") as f:
            f.write(json.dumps(row) + "\n")

    monkeypatch.setattr(gsm8k_two_solution, "_load_advisor", lambda _: object())
    monkeypatch.setattr(gsm8k_two_solution, "write_gsm8k_two_solution_prompts", write_prompts)
    for _ in range(2):
        list(
            gsm8k_two_solution.run_two_solution_prompts_worker(
                iter([0]),
                ShardInfo(shard_idx=0, total_shards=1),
                config=two_solution_prompts_config,
                ledger_path=ledger_path,
                pool_id="test",
            )
        )

    assert list(read_prompt_rows(str(prompt_path))) == [row]
    assert ledger.summarize(ledger_path) == ledger.LedgerSummary(total=1, claimed=0, done=1)


@pytest.mark.parametrize("failure_stage", ["load", "write"])
def test_two_solution_worker_failure_leaves_work_claimable(
    tmp_path, monkeypatch, two_solution_prompts_config, failure_stage
):
    ledger_path = str(tmp_path / "ledger")
    ledger.ensure_manifest(ledger_path, [{"chunk_id": 0}])

    def fail(*_args):
        raise RuntimeError("advisor failure")

    monkeypatch.setattr(gsm8k_two_solution, "_load_advisor", fail if failure_stage == "load" else lambda _: object())
    monkeypatch.setattr(gsm8k_two_solution, "write_gsm8k_two_solution_prompts", fail)
    with pytest.raises(RuntimeError, match="advisor failure"):
        list(
            gsm8k_two_solution.run_two_solution_prompts_worker(
                iter([0]),
                ShardInfo(shard_idx=0, total_shards=1),
                config=two_solution_prompts_config,
                ledger_path=ledger_path,
                pool_id="test",
            )
        )

    assert ledger.done_chunk_ids(ledger_path) == []
    if failure_stage == "load":
        assert ledger.read_chunk_state(ledger_path, 0) is None
    with ledger.claim_next_chunk(ledger_path, "replacement") as claim:
        assert claim is not None


def test_write_gsm8k_chat_prompts_writes_chat_context_and_answer_prefill(tmp_path, monkeypatch):
    configure_tokenizer(monkeypatch, gsm8k_chat)
    monkeypatch.setattr(
        gsm8k_chat,
        "_load_gsm8k_task",
        lambda: FakeTask([{"question": "What is 20 + 22?", "answer": "Add them.\n#### 42"}]),
    )

    gsm8k_chat.write_gsm8k_chat_prompts(
        gsm8k_chat.ChatGSM8KPromptsConfig(
            output_path=str(tmp_path),
            tokenizer_path="/tokenizer",
            n_problems=None,
        )
    )

    [row] = read_prompt_rows(str(tmp_path / "prompts.jsonl.gz"))
    assert row["id"] == "gsm8k/test/0"
    assert row["ground_truth"] == "42"
    assert row["prompt"] == ("<user>What is 20 + 22?\n\nEnd your answer with: #### <number></user><assistant>Answer:")


@pytest.mark.parametrize(
    "message_templates,prefill_template,expected_prompt",
    [
        (
            gsm8k_chat_reference.TWO_SOLUTION_MESSAGES,
            gsm8k_chat_reference.TWO_SOLUTION_PREFILL,
            "<user>Generate two independent, correct solutions to this problem.\n\n"
            "What is 20 + 22?\n\nEnd each solution with: #### <number></user>"
            "<assistant>Solution 1: Add them.\n#### 42\nSolution 2:",
        ),
        (
            gsm8k_chat_reference.FOLLOWUP_MESSAGES,
            gsm8k_chat_reference.FOLLOWUP_PREFILL,
            "<user>What is 20 + 22?\n\nEnd your answer with: #### <number></user>"
            "<assistant>Answer: Add them.\n#### 42</assistant>"
            "<user>Great. Can you solve it another way?</user><assistant>Answer:",
        ),
    ],
    ids=["two_solution", "followup"],
)
def test_write_gsm8k_chat_reference_prompts_preserves_reference_and_turns(
    tmp_path, monkeypatch, message_templates, prefill_template, expected_prompt
):
    configure_tokenizer(monkeypatch, gsm8k_chat_reference)
    monkeypatch.setattr(
        gsm8k_chat_reference,
        "_load_gsm8k_task",
        lambda: FakeTask([{"question": "What is 20 + 22?", "answer": "Add them.\n#### 42"}]),
    )

    gsm8k_chat_reference.write_gsm8k_chat_reference_prompts(
        gsm8k_chat_reference.ReferenceChatGSM8KPromptsConfig(
            output_path=str(tmp_path),
            tokenizer_path="/tokenizer",
            n_problems=None,
            message_templates=message_templates,
            prefill_template=prefill_template,
        )
    )

    [row] = read_prompt_rows(str(tmp_path / "prompts.jsonl.gz"))
    assert row["id"] == "gsm8k/test/0"
    assert row["ground_truth"] == "42"
    assert row["prompt"] == expected_prompt


def test_write_humaneval_chat_prompts_preserves_raw_prompt_suffix(tmp_path, monkeypatch):
    configure_tokenizer(monkeypatch, humaneval_chat)
    raw_prompt = 'def answer():\n    """Return 42."""\n    '
    monkeypatch.setattr(
        humaneval_chat,
        "_load_humaneval_task",
        lambda: FakeTask(
            [
                {
                    "task_id": "HumanEval/0",
                    "prompt": raw_prompt,
                    "canonical_solution": "return 42\n",
                    "entry_point": "answer",
                    "test": "assert answer() == 42",
                }
            ]
        ),
    )

    humaneval_chat.write_humaneval_chat_prompts(
        humaneval_chat.ChatHumanEvalPromptsConfig(
            output_path=str(tmp_path),
            tokenizer_path="/tokenizer",
            n_problems=None,
        )
    )

    [row] = read_prompt_rows(str(tmp_path / "prompts.jsonl.gz"))
    assert row["id"] == "humaneval/test/HumanEval/0"
    assert row["prompt"] == (
        "<user>Write a solution to the following problem and make sure that it passes the tests:\n"
        f"```python\n{raw_prompt}\n```\n</user><assistant>```python\n{raw_prompt}"
    )

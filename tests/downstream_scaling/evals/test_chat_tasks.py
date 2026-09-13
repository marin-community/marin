# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.downstream_scaling.evals.framework.schema import read_prompt_rows
from experiments.downstream_scaling.evals.tasks import gsm8k_chat, humaneval_chat


class FakeChatTokenizer:
    bos_token = "<bos>"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        return f"{self.bos_token}<user>{messages[0]['content']}</user><assistant>"


class FakeTask:
    def __init__(self, docs):
        self.docs = docs

    def test_docs(self):
        return self.docs


def configure_tokenizer(monkeypatch, module):
    monkeypatch.setattr(module, "discover_hf_checkpoints", lambda _: ["/tokenizer/checkpoint"])
    monkeypatch.setattr(module, "load_tokenizer", lambda _: FakeChatTokenizer())


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

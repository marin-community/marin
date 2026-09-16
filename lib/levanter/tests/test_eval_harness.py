# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json
import math
import os
import subprocess
import sys
import textwrap

import jax
import jmp
import pytest
from haliax import Axis
from levanter.testing.helpers import skip_if_module_missing
from transformers import AutoTokenizer

from levanter.data.packing import PromptCompletion
from levanter.eval_harness import (
    LmEvalHarnessConfig,
    SampleLoggingConfig,
    _call_with_retry,
    _iterate_tokenized_requests,
    run_lm_eval_harness,
)
from levanter.eval_harness_config import TaskConfig
from levanter.models.gpt2 import Gpt2Config, Gpt2LMHeadModel
from levanter.testing.helpers import use_test_mesh
from levanter.eval_harness import _paged_attention_max_seqs


def test_eval_entry_point_preserves_dependency_import_failure(tmp_path):
    dependency = tmp_path / "lm_eval"
    dependency.mkdir()
    (dependency / "__init__.py").write_text("raise AttributeError('removed_transformers_api')\n")
    probe = textwrap.dedent(
        """\
        from levanter.eval_harness import EvalHarnessMainConfig, LmEvalHarnessConfig, run_eval_harness_main

        config = EvalHarnessMainConfig(
            eval_harness=LmEvalHarnessConfig(task_spec=[]),
            tokenizer="/nonexistent-tokenizer",
            checkpoint_path="/nonexistent-checkpoint",
        )
        try:
            run_eval_harness_main(config)
        except ImportError as exc:
            assert isinstance(exc.__cause__, AttributeError)
            assert str(exc.__cause__) == "removed_transformers_api"
        else:
            raise AssertionError("Evaluation accepted a broken lm-eval dependency")
        """
    )
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(tmp_path), os.environ.get("PYTHONPATH", "")]))
    result = subprocess.run([sys.executable, "-c", probe], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@skip_if_module_missing("lm_eval")
@pytest.mark.parametrize("smooth_metrics", [False, True])
def test_lm_eval_harness_scores_and_retains_local_multiple_choice_samples(
    tmp_path, local_gpt2_tokenizer, smooth_metrics
):
    # Identical inputs with opposite gold labels give accuracy 1/2 for any deterministic model.
    documents = [{"question": "Answer:", "choices": ["yes", "no"], "gold": gold} for gold in (0, 1)]
    data_path = tmp_path / "questions.jsonl"
    data_path.write_text("\n".join(json.dumps(doc) for doc in documents))
    task = TaskConfig(
        task="local_multiple_choice",
        dataset_path="json",
        dataset_kwargs={"data_files": {"test": str(data_path)}},
        test_split="test",
        output_type="multiple_choice",
        doc_to_text="{{question}}",
        doc_to_choice="{{choices}}",
        doc_to_target="{{gold}}",
        num_fewshot=0,
        metric_list=[{"metric": "acc", "aggregation": "mean", "higher_is_better": True}],
    )
    if smooth_metrics:
        task = dataclasses.replace(
            task,
            metric_list=task.metric_list
            + [
                {"metric": name, "aggregation": "mean", "higher_is_better": name != "bpb"}
                for name in ("bpb", "logprob", "choice_logprob", "choice_prob_norm", "choice_logprob_norm")
            ],
        )
    config = LmEvalHarnessConfig(
        task_spec=[task], max_length=16, log_samples=True, sample_logging=SampleLoggingConfig(log_all=True)
    )
    model_config = Gpt2Config(
        max_seq_len=16, hidden_dim=16, num_layers=1, num_heads=2, resid_pdrop=0.0, use_flash_attention=False
    )
    with use_test_mesh():
        model = Gpt2LMHeadModel.init(Axis("vocab", len(local_gpt2_tokenizer)), model_config, key=jax.random.PRNGKey(0))
        result = run_lm_eval_harness(config, model, local_gpt2_tokenizer, 1, {}, jmp.get_policy("f32"))

    assert result is not None
    assert result["results"]["local_multiple_choice"]["acc,none"] == 0.5
    assert result["n-samples"]["local_multiple_choice"]["effective"] == 2
    samples = result["samples"]["local_multiple_choice"]
    assert len(samples) == 2
    assert all(len(sample["resps"]) == 2 for sample in samples)
    assert all(math.isfinite(response[0][0]) for sample in samples for response in sample["resps"])
    outputs = result["results"]["local_multiple_choice"]["outputs"]
    assert len(outputs) == 4
    assert {sample["generation"] for sample in outputs} == {" yes", " no"}
    if smooth_metrics:
        expected_bpb = []
        for sample in samples:
            gold = sample["doc"]["gold"]
            logprob = sample["resps"][gold][0][0]
            bpb = -logprob / (len(sample["doc"]["choices"][gold].encode("utf-8")) * math.log(2))
            assert sample["bpb"] == pytest.approx(bpb)
            assert sample["logprob"] == logprob
            expected_bpb.append(bpb)
        assert result["results"]["local_multiple_choice"]["bpb,none"] == pytest.approx(sum(expected_bpb) / 2)


@skip_if_module_missing("lm_eval")
def test_iterate_tokenized_requests_with_chat_template():
    """Test the chat template functionality in _iterate_tokenized_requests"""
    from lm_eval.api.instance import Instance  # noqa: PLC0415  # optional dep: lm_eval

    # Load a tokenizer with chat template - Llama 3 has one
    hf_tokenizer = AutoTokenizer.from_pretrained("marin-community/marin-tokenizer")
    if hf_tokenizer.pad_token is None:
        hf_tokenizer.pad_token = hf_tokenizer.eos_token

    # Create chat-like requests with contexts formatted via the chat template, matching harness usage
    raw_contexts = [
        "What's the best way to learn Python?",
        "Explain quantum computing.",
    ]
    completions = [
        " Practice by building small projects.",
        " Quantum computing uses quantum bits or qubits.",
    ]

    formatted_contexts = [
        hf_tokenizer.apply_chat_template(
            [{"role": "user", "content": context}], tokenize=False, add_generation_prompt=True
        )
        for context in raw_contexts
    ]

    requests = [
        Instance(
            request_type="loglikelihood",
            doc={},
            arguments=(formatted_contexts[i], completions[i]),
            idx=i,
            metadata=("test_task", i, None),
        )
        for i in range(len(raw_contexts))
    ]

    # Parameters
    max_len = 100  # Larger max_len to accommodate chat template additions
    batch_size = 2

    # Run with chat template
    results = list(_iterate_tokenized_requests(requests, hf_tokenizer, max_len, batch_size))

    # Verify the results
    assert len(results) == len(requests)

    # Check each result
    for i, result in enumerate(results):
        # Should be a PromptCompletion object
        assert isinstance(result, PromptCompletion)
        assert result.segment_id == i

        # The context should have been transformed by the chat template
        context, completion = requests[i].args

        # Check completion is preserved
        completion_text = completion.strip()
        decoded = hf_tokenizer.decode(result.ids)
        assert completion_text in decoded, f"Completion not found for example {i}"

        # Verify prompt_length is correctly set
        assert result.prompt_length < len(
            result.ids
        ), f"Example {i} has invalid prompt_length ({result.prompt_length} >= {len(result.ids)})"

        # Verify that prompt_length approximately matches the expected chat context length
        # It might not match exactly due to truncation or other processing
        prompt_tokens = result.ids[: result.prompt_length]
        prompt_decoded = hf_tokenizer.decode(prompt_tokens)

        # The prompt should contain key parts of the chat context generated by the template
        if "<|start_header_id|>" in context:
            assert (
                "<|start_header_id|>" in prompt_decoded
            ), f"Chat template formatting not found in prompt for example {i}"

        # Alternative check: the prompt should be longer than the original context
        # due to the chat template adding formatting
        original_context_tokens = hf_tokenizer(raw_contexts[i], truncation=False, padding=False)["input_ids"]
        assert result.prompt_length > len(
            original_context_tokens
        ), f"Prompt length not increased by chat template for example {i}"


@skip_if_module_missing("lm_eval")
def test_iterate_tokenized_requests():
    from lm_eval.api.instance import Instance  # noqa: PLC0415  # optional dep: lm_eval

    hf_tokenizer = AutoTokenizer.from_pretrained("marin-community/marin-tokenizer")
    if hf_tokenizer.pad_token is None:
        hf_tokenizer.pad_token = hf_tokenizer.eos_token

    requests = [
        Instance(
            request_type="loglikelihood",
            doc={},  # Empty dict as it's not used in the function
            arguments=("What is the capital of France?", " Paris"),
            idx=0,
            metadata=("test_task", 0, None),
        ),
        Instance(
            request_type="loglikelihood",
            doc={},
            arguments=("The quick brown fox", " jumps over the lazy dog"),
            idx=1,
            metadata=("test_task", 1, None),
        ),
        Instance(
            request_type="loglikelihood",
            doc={},
            arguments=("To be or not to be,", " that is the question"),
            idx=2,
            metadata=("test_task", 2, None),
        ),
    ]

    # Parameters
    max_len = 50
    batch_size = 2

    # Run the function
    results = list(_iterate_tokenized_requests(requests, hf_tokenizer, max_len, batch_size))

    # Verify results
    assert len(results) == 3  # One result per request

    for i, result in enumerate(results):
        # Check basic structure
        assert isinstance(result, PromptCompletion)
        assert result.segment_id == i

        # Check token content
        context, completion = requests[i].args
        decoded = hf_tokenizer.decode(result.ids)

        # The decoded tokens should contain the original text
        assert context in decoded or decoded.startswith(context.strip())
        assert completion in decoded or decoded.endswith(completion.strip())

        # Verify prompt_length
        prompt_tokens = result.ids[: result.prompt_length]
        decoded_prompt = hf_tokenizer.decode(prompt_tokens)

        # The decoded prompt should match or contain the context
        assert context.strip() in decoded_prompt

        # Sequence length should be within max_len
        assert len(result.ids) <= max_len


@skip_if_module_missing("lm_eval")
def test_task_config():
    task_spec = [
        TaskConfig(
            task="hellaswag",
            task_alias="hellaswag_10shot",
            num_fewshot=10,
        ),
        TaskConfig(
            task="hellaswag",
            task_alias="hellaswag_5shot",
            num_fewshot=5,
        ),
        "lambada_openai",
    ]

    config = LmEvalHarnessConfig(
        task_spec=task_spec,
    )

    q = config.to_task_dict()

    assert len(q) == 3


def test_call_with_retry_does_not_sleep_after_the_last_attempt(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr("levanter.eval_harness.time.sleep", sleeps.append)

    attempts = 0

    def always_fails():
        nonlocal attempts
        attempts += 1
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="Failed after 3 attempts"):
        _call_with_retry(always_fails, max_retries=3, base_delay=1.0)

    assert attempts == 3
    assert len(sleeps) == 2


def test_paged_attention_max_seqs_fits_smem_page_table():
    assert _paged_attention_max_seqs(8192, 8) == 224
    assert _paged_attention_max_seqs(4096, 8) == 256
    assert _paged_attention_max_seqs(8192, 128) == 256
    assert _paged_attention_max_seqs(1 << 20, 8) == 1

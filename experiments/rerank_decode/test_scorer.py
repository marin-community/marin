"""Tests that scorer implementations compute correct logprobs.

Compares KVCacheScorer and VLLMLogprobScorer against a ground truth
HuggingFace forward pass.
"""

import pytest
import openai
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.rerank_decode.scorer import KVCacheScorer, LevanterScorer, VLLMLogprobScorer
from experiments.rerank_decode.serve import launch_vllm_server, wait_for_server, shutdown_servers
from levanter.models.qwen import QwenConfig

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
VLLM_PORT = 8192


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL)


@pytest.fixture(scope="module")
def reference_model():
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    model.eval()
    return model


@pytest.fixture(scope="module")
def _kv_scorer():
    scorer = KVCacheScorer(MODEL, device="cuda")
    yield scorer
    scorer.reset()


@pytest.fixture
def kv_scorer(_kv_scorer):
    _kv_scorer.reset()
    yield _kv_scorer
    _kv_scorer.reset()


@pytest.fixture(scope="module")
def vllm_server():
    proc = launch_vllm_server(model=MODEL, port=VLLM_PORT, gpu_ids=[0])
    try:
        wait_for_server(VLLM_PORT, timeout=300)
        yield proc
    finally:
        shutdown_servers(proc)


@pytest.fixture
def vllm_scorer(vllm_server):
    client = openai.Client(base_url=f"http://localhost:{VLLM_PORT}/v1", api_key="none")
    return VLLMLogprobScorer(client=client, model=MODEL)


@pytest.fixture(scope="module")
def _levanter_scorer():
    scorer = LevanterScorer(
        MODEL,
        model_config=QwenConfig(),
        max_batch_size=4,
        max_prompt_len=128,
        max_completion_len=64,
        max_pages=64,
    )
    yield scorer
    scorer.reset()


@pytest.fixture
def levanter_scorer(_levanter_scorer):
    _levanter_scorer.reset()
    yield _levanter_scorer
    _levanter_scorer.reset()


def ground_truth_logprobs(reference_model, tokenizer, prompt: str, completion: str) -> float:
    """Compute logprobs for completion tokens given prompt via a single forward pass."""
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    completion_ids = tokenizer.encode(completion, add_special_tokens=False)
    all_ids = prompt_ids + completion_ids

    input_ids = torch.tensor([all_ids])
    with torch.no_grad():
        logits = reference_model(input_ids).logits[0]  # [seq_len, vocab]

    log_probs = F.log_softmax(logits, dim=-1)

    total = 0.0
    for i, token_id in enumerate(completion_ids):
        pos = len(prompt_ids) + i - 1
        total += log_probs[pos, token_id].item()

    return total


def prompt_logprobs(reference_model, tokenizer, prompt: str) -> float:
    """Compute logprobs for prompt tokens (excluding the first, which has no conditioning)."""
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor([ids])
    with torch.no_grad():
        logits = reference_model(input_ids).logits[0]
    log_probs = F.log_softmax(logits, dim=-1)
    total = 0.0
    for i in range(1, len(ids)):
        total += log_probs[i - 1, ids[i]].item()
    return total


# --- KVCacheScorer tests ---


def test_kvcache_single_completion(reference_model, tokenizer, kv_scorer):
    prompt = "The capital of France is"
    completion = " Paris, a beautiful city."

    expected = ground_truth_logprobs(reference_model, tokenizer, prompt, completion)
    actual = kv_scorer.score(prompt, [completion])[0]

    assert abs(expected - actual) < 0.5, f"expected {expected:.4f}, got {actual:.4f}"


def test_kvcache_multiple_completions(reference_model, tokenizer, kv_scorer):
    prompt = "Hello"
    completions = [", world!", " there!", " everyone, how are you?"]

    expected = [
        ground_truth_logprobs(reference_model, tokenizer, prompt, c)
        for c in completions
    ]
    actual = kv_scorer.score(prompt, completions)

    for i, (e, a) in enumerate(zip(expected, actual)):
        assert abs(e - a) < 0.5, f"completion {i}: expected {e:.4f}, got {a:.4f}"


def test_kvcache_eos_token(reference_model, tokenizer, kv_scorer):
    prompt = "Hello, world!"
    eos = tokenizer.eos_token
    completion = eos

    expected = ground_truth_logprobs(reference_model, tokenizer, prompt, completion)
    actual = kv_scorer.score(prompt, [completion])[0]

    assert abs(expected - actual) < 0.5, f"expected {expected:.4f}, got {actual:.4f}"


def test_kvcache_completion_with_eos(reference_model, tokenizer, kv_scorer):
    prompt = "What is 2+2?"
    eos = tokenizer.eos_token
    completion = " 4" + eos

    expected = ground_truth_logprobs(reference_model, tokenizer, prompt, completion)
    actual = kv_scorer.score(prompt, [completion])[0]

    assert abs(expected - actual) < 0.5, f"expected {expected:.4f}, got {actual:.4f}"


def test_kvcache_accept_then_score(reference_model, tokenizer, kv_scorer):
    """Test that accept() correctly extends the KV cache for subsequent scoring."""
    prompt = "Once upon a"
    chunk1 = " time there"
    chunk2 = " was a dragon"

    score1 = kv_scorer.score(prompt, [chunk1])[0]
    expected1 = ground_truth_logprobs(reference_model, tokenizer, prompt, chunk1)
    assert abs(expected1 - score1) < 0.5, f"chunk1: expected {expected1:.4f}, got {score1:.4f}"

    kv_scorer.accept(prompt, chunk1)
    score2 = kv_scorer.score(prompt + chunk1, [chunk2])[0]
    expected2 = ground_truth_logprobs(reference_model, tokenizer, prompt + chunk1, chunk2)
    assert abs(expected2 - score2) < 0.5, f"chunk2: expected {expected2:.4f}, got {score2:.4f}"


def test_kvcache_accept_multiple_chunks(reference_model, tokenizer, kv_scorer):
    """Test multiple rounds of accept + score."""
    prompt = "The"
    chunks = [" quick", " brown", " fox"]

    current_prompt = prompt
    for chunk in chunks:
        score = kv_scorer.score(current_prompt, [chunk])[0]
        expected = ground_truth_logprobs(reference_model, tokenizer, current_prompt, chunk)
        assert abs(expected - score) < 0.5, f"'{chunk}': expected {expected:.4f}, got {score:.4f}"

        kv_scorer.accept(current_prompt, chunk)
        current_prompt += chunk


def test_kvcache_reuse_across_prompts_asserts(reference_model, tokenizer, kv_scorer):
    """Test that reusing a scorer across prompts without reset raises AssertionError."""
    kv_scorer.score("The capital of France is", [" Paris"])

    with pytest.raises(AssertionError):
        kv_scorer.score("Hello", [", world!"])


def test_kvcache_reuse_across_prompts_with_reset(reference_model, tokenizer, kv_scorer):
    """Test that a scorer produces correct results when reset between prompts."""
    prompts_and_completions = [
        ("The capital of France is", " Paris"),
        ("Hello", ", world!"),
        ("Once upon a", " time there was"),
    ]

    for prompt, completion in prompts_and_completions:
        kv_scorer.reset()
        score = kv_scorer.score(prompt, [completion])[0]
        expected = ground_truth_logprobs(reference_model, tokenizer, prompt, completion)
        assert abs(expected - score) < 0.5, f"prompt={prompt!r}: expected {expected:.4f}, got {score:.4f}"


# --- LevanterScorer tests ---


@pytest.mark.timeout(600)
def test_levanter_single_completion(reference_model, tokenizer, levanter_scorer):
    prompt = "The capital of France is"
    completion = " Paris, a beautiful city."

    expected = ground_truth_logprobs(reference_model, tokenizer, prompt, completion)
    actual = levanter_scorer.score(prompt, [completion])[0]

    assert abs(expected - actual) < 0.5, f"expected {expected:.4f}, got {actual:.4f}"


@pytest.mark.timeout(600)
def test_levanter_multiple_completions(reference_model, tokenizer, levanter_scorer):
    prompt = "Hello"
    completions = [", world!", " there!", " everyone, how are you?"]

    expected = [
        ground_truth_logprobs(reference_model, tokenizer, prompt, c)
        for c in completions
    ]
    actual = levanter_scorer.score(prompt, completions)

    for i, (e, a) in enumerate(zip(expected, actual)):
        assert abs(e - a) < 0.5, f"completion {i}: expected {e:.4f}, got {a:.4f}"


@pytest.mark.timeout(600)
def test_levanter_eos_token(reference_model, tokenizer, levanter_scorer):
    prompt = "Hello, world!"
    eos = tokenizer.eos_token
    completion = eos

    expected = ground_truth_logprobs(reference_model, tokenizer, prompt, completion)
    actual = levanter_scorer.score(prompt, [completion])[0]

    assert abs(expected - actual) < 0.5, f"expected {expected:.4f}, got {actual:.4f}"


@pytest.mark.timeout(600)
def test_levanter_completion_with_eos(reference_model, tokenizer, levanter_scorer):
    prompt = "What is 2+2?"
    eos = tokenizer.eos_token
    completion = " 4" + eos

    expected = ground_truth_logprobs(reference_model, tokenizer, prompt, completion)
    actual = levanter_scorer.score(prompt, [completion])[0]

    assert abs(expected - actual) < 0.5, f"expected {expected:.4f}, got {actual:.4f}"


@pytest.mark.timeout(600)
def test_levanter_accept_then_score(reference_model, tokenizer, levanter_scorer):
    prompt = "Once upon a"
    chunk1 = " time there"
    chunk2 = " was a dragon"

    score1 = levanter_scorer.score(prompt, [chunk1])[0]
    expected1 = ground_truth_logprobs(reference_model, tokenizer, prompt, chunk1)
    assert abs(expected1 - score1) < 0.5, f"chunk1: expected {expected1:.4f}, got {score1:.4f}"

    levanter_scorer.accept(prompt, chunk1)
    prompt2 = prompt + chunk1
    score2 = levanter_scorer.score(prompt2, [chunk2])[0]
    expected2 = ground_truth_logprobs(reference_model, tokenizer, prompt2, chunk2)
    assert abs(expected2 - score2) < 0.5, f"chunk2: expected {expected2:.4f}, got {score2:.4f}"


@pytest.mark.timeout(600)
def test_levanter_accept_multiple_chunks(reference_model, tokenizer, levanter_scorer):
    prompt = "The"
    chunks = [" quick", " brown", " fox"]

    current_prompt = prompt
    for chunk in chunks:
        score = levanter_scorer.score(current_prompt, [chunk])[0]
        expected = ground_truth_logprobs(reference_model, tokenizer, current_prompt, chunk)
        assert abs(expected - score) < 0.5, f"'{chunk}': expected {expected:.4f}, got {score:.4f}"

        levanter_scorer.accept(current_prompt, chunk)
        current_prompt += chunk


@pytest.mark.timeout(600)
def test_levanter_reuse_across_prompts(reference_model, tokenizer, levanter_scorer):
    prompts_and_completions = [
        ("The capital of France is", " Paris"),
        ("Hello", ", world!"),
        ("Once upon a", " time there was"),
    ]

    for prompt, completion in prompts_and_completions:
        score = levanter_scorer.score(prompt, [completion])[0]
        expected = ground_truth_logprobs(reference_model, tokenizer, prompt, completion)
        assert abs(expected - score) < 0.5, f"prompt={prompt!r}: expected {expected:.4f}, got {score:.4f}"


@pytest.mark.timeout(600)
def test_levanter_reuse_across_prompts_with_reset(reference_model, tokenizer, levanter_scorer):
    prompts_and_completions = [
        ("The capital of France is", " Paris"),
        ("Hello", ", world!"),
        ("Once upon a", " time there was"),
    ]

    for prompt, completion in prompts_and_completions:
        levanter_scorer.reset()
        score = levanter_scorer.score(prompt, [completion])[0]
        expected = ground_truth_logprobs(reference_model, tokenizer, prompt, completion)
        assert abs(expected - score) < 0.5, f"prompt={prompt!r}: expected {expected:.4f}, got {score:.4f}"


# --- VLLMLogprobScorer tests ---


@pytest.mark.timeout(600)
def test_vllm_single_completion(reference_model, tokenizer, vllm_scorer):
    prompt = "The capital of France is"
    completion = " Paris, a beautiful city."

    expected = ground_truth_logprobs(reference_model, tokenizer, prompt, completion)
    actual = vllm_scorer.score(prompt, [completion])[0] - prompt_logprobs(reference_model, tokenizer, prompt)

    assert abs(expected - actual) < 1.0, f"expected {expected:.4f}, got {actual:.4f}"


@pytest.mark.timeout(600)
def test_vllm_multiple_completions(reference_model, tokenizer, vllm_scorer):
    prompt = "Hello"
    completions = [", world!", " there!", " everyone, how are you?"]

    expected = [
        ground_truth_logprobs(reference_model, tokenizer, prompt, c)
        for c in completions
    ]
    actual = vllm_scorer.score(prompt, completions)
    offset = prompt_logprobs(reference_model, tokenizer, prompt)

    for i, (e, a) in enumerate(zip(expected, actual)):
        assert abs(e - (a - offset)) < 1.0, f"completion {i}: expected {e:.4f}, got {a - offset:.4f}"


@pytest.mark.timeout(600)
def test_vllm_eos_token(reference_model, tokenizer, vllm_scorer):
    prompt = "Hello, world!"
    eos = tokenizer.eos_token
    completion = eos

    expected = ground_truth_logprobs(reference_model, tokenizer, prompt, completion)
    actual = vllm_scorer.score(prompt, [completion])[0] - prompt_logprobs(reference_model, tokenizer, prompt)

    assert abs(expected - actual) < 1.0, f"expected {expected:.4f}, got {actual:.4f}"


@pytest.mark.timeout(600)
def test_vllm_completion_with_eos(reference_model, tokenizer, vllm_scorer):
    prompt = "What is 2+2?"
    eos = tokenizer.eos_token
    completion = " 4" + eos

    expected = ground_truth_logprobs(reference_model, tokenizer, prompt, completion)
    actual = vllm_scorer.score(prompt, [completion])[0] - prompt_logprobs(reference_model, tokenizer, prompt)

    assert abs(expected - actual) < 1.0, f"expected {expected:.4f}, got {actual:.4f}"


@pytest.mark.timeout(600)
def test_vllm_accept_then_score(reference_model, tokenizer, vllm_scorer):
    prompt = "Once upon a"
    chunk1 = " time there"
    chunk2 = " was a dragon"

    score1 = vllm_scorer.score(prompt, [chunk1])[0] - prompt_logprobs(reference_model, tokenizer, prompt)
    expected1 = ground_truth_logprobs(reference_model, tokenizer, prompt, chunk1)
    assert abs(expected1 - score1) < 1.0, f"chunk1: expected {expected1:.4f}, got {score1:.4f}"

    vllm_scorer.accept(prompt, chunk1)
    prompt2 = prompt + chunk1
    score2 = vllm_scorer.score(prompt2, [chunk2])[0] - prompt_logprobs(reference_model, tokenizer, prompt2)
    expected2 = ground_truth_logprobs(reference_model, tokenizer, prompt2, chunk2)
    assert abs(expected2 - score2) < 1.0, f"chunk2: expected {expected2:.4f}, got {score2:.4f}"


@pytest.mark.timeout(600)
def test_vllm_accept_multiple_chunks(reference_model, tokenizer, vllm_scorer):
    prompt = "The"
    chunks = [" quick", " brown", " fox"]

    current_prompt = prompt
    for chunk in chunks:
        score = vllm_scorer.score(current_prompt, [chunk])[0] - prompt_logprobs(reference_model, tokenizer, current_prompt)
        expected = ground_truth_logprobs(reference_model, tokenizer, current_prompt, chunk)
        assert abs(expected - score) < 1.0, f"'{chunk}': expected {expected:.4f}, got {score:.4f}"

        vllm_scorer.accept(current_prompt, chunk)
        current_prompt += chunk


@pytest.mark.timeout(600)
def test_vllm_reuse_across_prompts(reference_model, tokenizer, vllm_scorer):
    prompts_and_completions = [
        ("The capital of France is", " Paris"),
        ("Hello", ", world!"),
        ("Once upon a", " time there was"),
    ]

    for prompt, completion in prompts_and_completions:
        score = vllm_scorer.score(prompt, [completion])[0] - prompt_logprobs(reference_model, tokenizer, prompt)
        expected = ground_truth_logprobs(reference_model, tokenizer, prompt, completion)
        assert abs(expected - score) < 1.0, f"prompt={prompt!r}: expected {expected:.4f}, got {score:.4f}"


@pytest.mark.timeout(600)
def test_vllm_reuse_across_prompts_with_reset(reference_model, tokenizer, vllm_scorer):
    prompts_and_completions = [
        ("The capital of France is", " Paris"),
        ("Hello", ", world!"),
        ("Once upon a", " time there was"),
    ]

    for prompt, completion in prompts_and_completions:
        vllm_scorer.reset()
        score = vllm_scorer.score(prompt, [completion])[0] - prompt_logprobs(reference_model, tokenizer, prompt)
        expected = ground_truth_logprobs(reference_model, tokenizer, prompt, completion)
        assert abs(expected - score) < 1.0, f"prompt={prompt!r}: expected {expected:.4f}, got {score:.4f}"

"""Fixed-dataset (prompt, completion) trajectory runner: the static SFT data path."""

import asyncio
import json

import pytest
import torch
from omegaconf import OmegaConf

from marinskyrl.runtime_options import FixedCompletionFormat
from skyrl_train.config.objective_spec import validate_objective
from skyrl_train.config.utils import (
    response_token_cap,
    sequence_token_cap,
    sequences_per_step,
)
from skyrl_train.dataset.dataset import prompt_token_ids
from skyrl_train.dataset.fixed_completions import FixedCompletionsDataset
from skyrl_train.dataset.preprocess import convert_prompts_responses_to_batch_tensors
from skyrl_train.objective.objective import build_objective_micro_batch, compute_policy_objective
from skyrl_train.objective.reduction import step_counts
from skyrl_train.trajectory_runners.fixed_completions import FixedCompletionsTrajectoryRunner
from skyrl_train.trajectory_runners.trajectory_processing import prepare_trajectory_request
from skyrl_train.rollouts.loader import SeededPasses
from skyrl_train.utils.algorithm_registry import PolicyLossRegistry

from tests.cpu.util import fixed_completion_test_config
from tests.cpu.tiny_training.tiny_model import build_tiny_policy

MAX_PROMPT_LENGTH = 64
MAX_COMPLETION_LENGTH = 16


@pytest.fixture(scope="module")
def tiny_policy(tmp_path_factory):
    """The repo's tiny Qwen policy: a real tokenizer and a hand-shaped causal LM."""
    directory = build_tiny_policy(tmp_path_factory.mktemp("fixed-completions") / "model")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    return AutoTokenizer.from_pretrained(directory), AutoModelForCausalLM.from_pretrained(directory)


def write_text_rows(path, num_rows: int = 3):
    with path.open("w") as handle:
        for index in range(num_rows):
            handle.write(
                json.dumps(
                    {
                        "prompt": [{"role": "user", "content": f"Question {index}: what is one?"}],
                        "completion": "#### 1<|im_end|>",
                        "data_source": "task_x" if index % 2 == 0 else "task_m",
                        "teacher_route": f"route_{index % 2}",
                    }
                )
                + "\n"
            )
    return path


def build_dataset(
    tokenizer, data_path, data_format=FixedCompletionFormat.TEXT, max_completion_length=MAX_COMPLETION_LENGTH
):
    return FixedCompletionsDataset(
        tokenizer,
        data_format=data_format,
        datasets=[str(data_path)],
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=max_completion_length,
        num_workers=1,
    )


def build_runner(tokenizer, data_format=FixedCompletionFormat.TEXT, max_generate_length=MAX_COMPLETION_LENGTH):
    return FixedCompletionsTrajectoryRunner(
        tokenizer,
        data_format=data_format,
        max_generate_length=max_generate_length,
        max_input_length=MAX_PROMPT_LENGTH,
        generator_config=OmegaConf.create({}),
    )


def runner_batch(dataset, runner):
    rows = [dataset.collate_fn([dataset[index]])[0] for index in range(len(dataset))]
    request, _ = prepare_trajectory_request(rows, 1, None, "train", 0)
    return asyncio.run(runner.run(request))


def test_runner_emits_completion_tokens_with_one_task_tag_and_teacher_route(tmp_path, tiny_policy):
    tokenizer, _model = tiny_policy
    dataset = build_dataset(tokenizer, write_text_rows(tmp_path / "train.jsonl"))
    batch = runner_batch(dataset, build_runner(tokenizer))

    completion_ids = tokenizer("#### 1<|im_end|>", add_special_tokens=False)["input_ids"]
    prompt_ids = prompt_token_ids(tokenizer, [{"role": "user", "content": "Question 0: what is one?"}])
    assert batch["prompt_token_ids"][0] == prompt_ids
    assert batch["response_ids"][0] == completion_ids
    assert batch["loss_masks"][0] == [1] * len(completion_ids)
    assert batch["rewards"] == [1.0] * len(batch["response_ids"])
    assert batch["data_sources"] == ["task_x", "task_m", "task_x"]
    assert batch["teacher_route_keys"] == ["route_0", "route_1", "route_0"]


def test_tokenized_rows_replay_the_same_conversation_as_text_rows(tmp_path, tiny_policy):
    tokenizer, _model = tiny_policy
    prompt = [{"role": "user", "content": "Question 3: what is one?"}]
    completion = "#### 1<|im_end|>"
    prompt_ids = prompt_token_ids(tokenizer, prompt)
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    row = {
        "input_ids": prompt_ids + completion_ids,
        "assistant_masks": [0] * len(prompt_ids) + [1] * len(completion_ids),
        "data_source": "task_x",
        "teacher_route": "route_0",
    }
    tokenized_path = tmp_path / "tokenized.jsonl"
    tokenized_path.write_text(json.dumps(row) + "\n")
    dataset = build_dataset(tokenizer, tokenized_path, data_format=FixedCompletionFormat.TOKENIZED)
    batch = runner_batch(dataset, build_runner(tokenizer, FixedCompletionFormat.TOKENIZED))

    assert batch["prompt_token_ids"][0] == prompt_ids
    assert batch["response_ids"][0] == completion_ids
    assert batch["loss_masks"][0] == [1] * len(completion_ids)
    assert batch["data_sources"] == ["task_x"]
    assert batch["teacher_route_keys"] == ["route_0"]


def test_tokenized_rows_keep_unsupervised_tool_observation_tokens(tmp_path, tiny_policy):
    """Tokens inside the response span that the expert did not supervise stay in the row, unmasked."""
    tokenizer, _model = tiny_policy
    prompt_ids = prompt_token_ids(tokenizer, [{"role": "user", "content": "Question 4"}])
    completion_ids = tokenizer("#### 1<|im_end|>", add_special_tokens=False)["input_ids"]
    tool_ids = tokenizer("<tool_response>ok</tool_response>", add_special_tokens=False)["input_ids"]
    ids = prompt_ids + completion_ids + tool_ids + completion_ids
    mask = [0] * len(prompt_ids) + [1] * len(completion_ids) + [0] * len(tool_ids) + [1] * len(completion_ids)
    path = tmp_path / "tool.jsonl"
    path.write_text(json.dumps({"input_ids": ids, "assistant_masks": mask}) + "\n")
    dataset = build_dataset(tokenizer, path, data_format=FixedCompletionFormat.TOKENIZED, max_completion_length=64)
    batch = runner_batch(dataset, build_runner(tokenizer, FixedCompletionFormat.TOKENIZED, max_generate_length=64))

    assert batch["response_ids"][0] == completion_ids + tool_ids + completion_ids
    assert batch["loss_masks"][0] == [1] * len(completion_ids) + [0] * len(tool_ids) + [1] * len(completion_ids)


def test_one_sft_step_on_the_fixed_batch_matches_a_hand_computed_nll(tmp_path, tiny_policy):
    """The stack's loss and its parameter gradients equal the completion-token NLL's."""
    tokenizer, model = tiny_policy
    dataset = build_dataset(tokenizer, write_text_rows(tmp_path / "train.jsonl"))
    batch = runner_batch(dataset, build_runner(tokenizer))

    sequences, attention_mask, _action_mask, _rewards, loss_masks, *_rest = convert_prompts_responses_to_batch_tensors(
        tokenizer,
        batch["prompt_token_ids"],
        batch["response_ids"],
        [[1.0] * len(response) for response in batch["response_ids"]],
        batch["loss_masks"],
    )
    response_length = max(len(response) for response in batch["response_ids"])
    max_prompt_length = max(len(prompt) for prompt in batch["prompt_token_ids"])
    logits = model(input_ids=sequences, attention_mask=attention_mask).logits[:, :-1]
    target_logprobs = torch.log_softmax(logits.float(), dim=-1).gather(-1, sequences[:, 1:].unsqueeze(-1)).squeeze(-1)
    window = target_logprobs[:, max_prompt_length - 1 : max_prompt_length - 1 + response_length]

    algorithm = fixed_completion_test_config().trainer.algorithm
    advantages = torch.ones_like(loss_masks)
    micro_batch = build_objective_micro_batch(
        action_log_probs=window,
        old_action_log_probs=window.detach(),
        base_action_log_probs=None,
        advantages=advantages,
        loss_mask=loss_masks,
        rollout_logprobs=None,
        response_span_tags=None,
        token_entropy=torch.zeros_like(loss_masks),
        think_token_weight=1.0,
        teacher=None,
    )
    counts = step_counts(
        [loss_masks],
        [loss_masks],
        [loss_masks],
        [advantages],
        max_seq_len=int(sequences.shape[1]),
        all_reduce_sum=lambda values: values,
    )
    objective = compute_policy_objective(
        micro_batch,
        loss=PolicyLossRegistry.get("sft"),
        counts=counts,
        config=algorithm,
        loss_scale=1.0,
        report_scale=1.0,
    )

    supervised = loss_masks > 0
    hand_nll = -window[supervised].sum() / supervised.sum()
    assert torch.allclose(objective.optimization_loss, hand_nll, rtol=1e-6, atol=1e-7)

    stack_grads = torch.autograd.grad(objective.optimization_loss, model.parameters(), retain_graph=True)
    hand_grads = torch.autograd.grad(hand_nll, model.parameters())
    for stack_grad, hand_grad in zip(stack_grads, hand_grads):
        assert torch.allclose(stack_grad, hand_grad, rtol=1e-6, atol=1e-7)


def test_seeded_epoch_order_is_reproducible(tmp_path, tiny_policy):
    tokenizer, _model = tiny_policy
    dataset = build_dataset(tokenizer, write_text_rows(tmp_path / "train.jsonl", num_rows=8))
    first = SeededPasses(len(dataset), seed=42, shuffle=True)
    second = SeededPasses(len(dataset), seed=42, shuffle=True)
    order = [[first.next_index() for _ in range(len(dataset))], [second.next_index() for _ in range(len(dataset))]]
    assert order[0] == order[1]


def test_each_row_is_consumed_once_per_epoch(tmp_path, tiny_policy):
    tokenizer, _model = tiny_policy
    dataset = build_dataset(tokenizer, write_text_rows(tmp_path / "train.jsonl", num_rows=8))
    passes = SeededPasses(len(dataset), seed=7, shuffle=True)
    for _epoch in range(3):
        indices = [passes.next_index() for _ in range(len(dataset))]
        assert sorted(indices) == list(range(len(dataset)))


def test_m1_and_m4_configs_with_the_same_table_values_match_geometry():
    """A 32-sequence step is 32 fixed rows (M1) or 8 prompts x 4 rollouts (M4), with equal caps."""
    table = {"sequences_per_step": 32, "rollouts_per_prompt": 4, "response_cap": 4096, "prompt_cap": 32768}
    m1 = fixed_completion_test_config(
        **{
            "trainer": {
                "train_batch_size": table["sequences_per_step"],
                "max_prompt_length": table["prompt_cap"],
            },
            "generator": {"sampling_params": {"max_generate_length": table["response_cap"]}},
        }
    )
    m4 = fixed_completion_test_config(
        **{
            "environment": {"env_class": "gsm8k"},
            "trainer": {
                "train_batch_size": table["sequences_per_step"] // table["rollouts_per_prompt"],
                "max_prompt_length": table["prompt_cap"],
            },
            "generator": {
                "sampling_params": {"max_generate_length": table["response_cap"]},
                "n_samples_per_prompt": table["rollouts_per_prompt"],
            },
        }
    )
    assert sequences_per_step(m1) == sequences_per_step(m4) == table["sequences_per_step"]
    assert response_token_cap(m1) == response_token_cap(m4) == table["response_cap"]
    assert sequence_token_cap(m1) == sequence_token_cap(m4) == table["prompt_cap"] + table["response_cap"]


def test_validate_objective_rejects_fixed_completion_configs_that_break_the_matched_budget():
    cases = (
        ({"generator": {"n_samples_per_prompt": 4}}, "n_samples_per_prompt must be 1"),
        ({"trainer": {"use_sample_packing": True}}, "use_sample_packing=false"),
        ({"trainer": {"algorithm": {"loss_reduction": "sequence_mean"}}}, "loss_reduction=token_mean"),
        ({"trainer": {"algorithm": {"policy_loss_type": "regular"}}}, "policy_loss_type=sft"),
        ({"trainer": {"algorithm": {"advantage_estimator": "grpo"}}}, "advantage_estimator=uniform"),
    )
    for overrides, message in cases:
        cfg = fixed_completion_test_config(**overrides)
        with pytest.raises(ValueError, match=message):
            validate_objective(cfg)

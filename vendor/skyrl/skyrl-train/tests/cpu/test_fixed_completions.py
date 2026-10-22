"""Fixed-dataset (prompt, completion) trajectory runner: the static SFT data path."""

import asyncio
import collections
import copy
import json

import pytest
import torch
from omegaconf import OmegaConf, open_dict
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from marinskyrl.runtime_options import FixedCompletionFormat
from skyrl_train.config.objective_spec import validate_objective
from skyrl_train.config.training_budget import ResponseCaps
from skyrl_train.config.utils import get_default_config
from skyrl_train.dataset.dataset import prompt_token_ids
from skyrl_train.dataset.fixed_completions import FixedCompletionsDataset
from skyrl_train.dataset.preprocess import convert_prompts_responses_to_batch_tensors
from skyrl_train.objective.objective import build_objective_micro_batch, compute_policy_objective
from skyrl_train.objective.reduction import step_counts
from skyrl_train.rollouts.context import RolloutRequestSpec, prompt_order_from_config
from skyrl_train.rollouts.loader import PromptLoader
from skyrl_train.trajectory_runners.fixed_completions import FixedCompletionsTrajectoryRunner
from skyrl_train.trajectory_runners.trajectory_processing import prepare_trajectory_request
from skyrl_train.utils.algorithm_registry import PolicyLossRegistry

from tests.cpu.task_specs import lowered_task, task_input, text_task
from tests.cpu.tiny_training.cpu_backend import CausalLMPolicy
from tests.cpu.tiny_training.tiny_model import END_OF_TURN, build_tiny_policy
from tests.cpu.util import fixed_completion_test_config

MAX_PROMPT_LENGTH = 64
MAX_COMPLETION_LENGTH = 16
CAPS = ResponseCaps(default=MAX_COMPLETION_LENGTH)


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory):
    """The repo's tiny byte-level ChatML tokenizer; its EOS is the end-of-turn token."""
    directory = build_tiny_policy(tmp_path_factory.mktemp("fixed-completions") / "model")
    return AutoTokenizer.from_pretrained(directory)


def tiny_qwen3(tokenizer) -> Qwen3ForCausalLM:
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=len(tokenizer),
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=256,
        tie_word_embeddings=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        attn_implementation="eager",
    )
    return Qwen3ForCausalLM(config).float()


def text_row(index: int, *, route: str, data_source: str, answer: str = "1") -> dict:
    return {
        "prompt": [{"role": "user", "content": f"Question {index}: what is {'one ' * (1 + index % 3)}?"}],
        "completion": f"#### {answer}{END_OF_TURN}",
        "data_source": data_source,
        "teacher_route": route,
    }


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def write_text_rows(path, num_rows: int = 3, routes: tuple[str, ...] = ("route_0", "route_1")):
    sources = ("task_x", "task_m")
    return write_rows(
        path,
        [
            text_row(index, route=routes[index % len(routes)], data_source=sources[index % len(sources)])
            for index in range(num_rows)
        ],
    )


def build_dataset(tokenizer, data_path, data_format=FixedCompletionFormat.TEXT, response_caps=CAPS):
    return FixedCompletionsDataset(
        tokenizer,
        data_format=data_format,
        datasets=[str(data_path)],
        max_prompt_length=MAX_PROMPT_LENGTH,
        response_caps=response_caps,
        num_workers=1,
    )


def build_runner(tokenizer, data_format=FixedCompletionFormat.TEXT, response_caps=CAPS):
    return FixedCompletionsTrajectoryRunner(
        tokenizer,
        data_format=data_format,
        response_caps=response_caps,
        max_input_length=MAX_PROMPT_LENGTH,
        generator_config=OmegaConf.create({}),
    )


def runner_batch(dataset, runner, indices=None):
    indices = range(len(dataset)) if indices is None else indices
    rows = [dataset.collate_fn([dataset[index]])[0] for index in indices]
    request, _ = prepare_trajectory_request(rows, 1, None, "train", 0)
    return asyncio.run(runner.run(request))


def consumed_batches(cfg, dataset, num_batches: int) -> list[list[dict]]:
    """Draw ``num_batches`` training batches through the trainer's prompt loader and order."""
    batch_size = cfg.trainer.train_batch_size
    loader = PromptLoader(dataset, prompt_order_from_config(cfg, dataset), batch_size=batch_size)
    batches = []
    for _ in range(num_batches):
        batch, live = [], set()
        for _ in range(batch_size):
            prompt = loader.next_prompt(live)
            live.add(prompt["uid"])
            batch.append(prompt)
        batches.append(batch)
    return batches


def with_open_maps(cfg, overrides: dict):
    """Merge overrides that add keys to maps the base config declares empty, as Hydra ``+key=`` does."""
    with open_dict(cfg):
        cfg.merge_with(overrides)
    return cfg


def uids(batches) -> list[str]:
    return [prompt["uid"] for batch in batches for prompt in batch]


def test_runner_emits_completion_tokens_with_one_task_tag_and_teacher_route(tmp_path, tokenizer):
    dataset = build_dataset(tokenizer, write_text_rows(tmp_path / "train.jsonl"))
    batch = runner_batch(dataset, build_runner(tokenizer))

    completion_ids = tokenizer(f"#### 1{END_OF_TURN}", add_special_tokens=False)["input_ids"]
    prompt_ids = prompt_token_ids(tokenizer, text_row(0, route="", data_source="")["prompt"])
    assert batch["prompt_token_ids"][0] == prompt_ids
    assert batch["response_ids"][0] == completion_ids
    assert batch["loss_masks"][0] == [1] * len(completion_ids)
    assert batch["rewards"] == [1.0] * len(batch["response_ids"])
    assert batch["data_sources"] == ["task_x", "task_m", "task_x"]
    assert batch["teacher_route_keys"] == ["route_0", "route_1", "route_0"]


def test_tokenized_rows_replay_the_same_conversation_as_text_rows(tmp_path, tokenizer):
    row = text_row(3, route="route_0", data_source="task_x")
    prompt_ids = prompt_token_ids(tokenizer, row["prompt"])
    completion_ids = tokenizer(row["completion"], add_special_tokens=False)["input_ids"]
    tokenized_path = write_rows(
        tmp_path / "tokenized.jsonl",
        [
            {
                "input_ids": prompt_ids + completion_ids,
                "assistant_masks": [0] * len(prompt_ids) + [1] * len(completion_ids),
                "data_source": "task_x",
                "teacher_route": "route_0",
            }
        ],
    )
    dataset = build_dataset(tokenizer, tokenized_path, data_format=FixedCompletionFormat.TOKENIZED)
    batch = runner_batch(dataset, build_runner(tokenizer, FixedCompletionFormat.TOKENIZED))

    assert batch["prompt_token_ids"][0] == prompt_ids
    assert batch["response_ids"][0] == completion_ids
    assert batch["loss_masks"][0] == [1] * len(completion_ids)
    assert batch["data_sources"] == ["task_x"]
    assert batch["teacher_route_keys"] == ["route_0"]


def test_tokenized_rows_keep_unsupervised_tool_observation_tokens(tmp_path, tokenizer):
    """Tokens inside the response span that the expert did not supervise stay in the row, unmasked."""
    prompt_ids = prompt_token_ids(tokenizer, [{"role": "user", "content": "Question 4"}])
    completion_ids = tokenizer(f"#### 1{END_OF_TURN}", add_special_tokens=False)["input_ids"]
    tool_ids = tokenizer("<tool_response>ok</tool_response>", add_special_tokens=False)["input_ids"]
    ids = prompt_ids + completion_ids + tool_ids + completion_ids
    mask = [0] * len(prompt_ids) + [1] * len(completion_ids) + [0] * len(tool_ids) + [1] * len(completion_ids)
    path = write_rows(tmp_path / "tool.jsonl", [{"input_ids": ids, "assistant_masks": mask, "data_source": "task_t"}])
    caps = ResponseCaps(default=64)
    dataset = build_dataset(tokenizer, path, data_format=FixedCompletionFormat.TOKENIZED, response_caps=caps)
    batch = runner_batch(dataset, build_runner(tokenizer, FixedCompletionFormat.TOKENIZED, response_caps=caps))

    assert batch["response_ids"][0] == completion_ids + tool_ids + completion_ids
    assert batch["loss_masks"][0] == [1] * len(completion_ids) + [0] * len(tool_ids) + [1] * len(completion_ids)


def test_each_task_keeps_its_own_response_cap_in_the_dataset_and_runner(tmp_path, tokenizer):
    """A joint X+R run keeps X's tighter cap; an over-cap row raises instead of silently leaving the dataset."""
    answer = "1" * 8
    rows = [
        text_row(0, route="route_x", data_source="task_x", answer=answer),
        text_row(1, route="route_r", data_source="task_r", answer=answer),
    ]
    path = write_rows(tmp_path / "joint.jsonl", rows)
    completion_length = len(tokenizer(rows[0]["completion"], add_special_tokens=False)["input_ids"])
    joint = ResponseCaps(default=16, by_data_source={"task_x": completion_length - 1, "task_r": 16})

    with pytest.raises(ValueError, match=rf"row 0 \(task_x\).*{completion_length - 1}\]-token response cap"):
        build_dataset(tokenizer, path, response_caps=joint)

    dataset = build_dataset(tokenizer, path, response_caps=ResponseCaps(default=16))
    joint_runner = build_runner(tokenizer, response_caps=joint)
    assert len(runner_batch(dataset, joint_runner, indices=[1])["response_ids"][0]) == completion_length
    with pytest.raises(ValueError, match="task_x"):
        runner_batch(dataset, joint_runner, indices=[0])


def test_one_sft_step_on_a_fixed_batch_matches_independent_unpadded_nll(tmp_path, tokenizer):
    """The padded training path's loss, gradients, and optimizer step equal per-row unpadded NLL's."""
    rows = [text_row(index, route="route_0", data_source="task_x", answer="1" * (1 + index)) for index in range(3)]
    dataset = build_dataset(tokenizer, write_rows(tmp_path / "train.jsonl", rows))
    batch = runner_batch(dataset, build_runner(tokenizer))
    prompts, responses = batch["prompt_token_ids"], batch["response_ids"]
    assert len({len(prompt) for prompt in prompts}) > 1 and len({len(response) for response in responses}) > 1

    model = tiny_qwen3(tokenizer)
    reference = copy.deepcopy(model)

    sequences, attention_mask, _action_mask, _rewards, loss_masks, *_rest = convert_prompts_responses_to_batch_tensors(
        tokenizer, prompts, responses, [[1.0] * len(response) for response in responses], batch["loss_masks"]
    )
    action_log_probs = CausalLMPolicy(model)(sequences, int(loss_masks.shape[1]), attention_mask)
    advantages = torch.ones_like(loss_masks)
    micro_batch = build_objective_micro_batch(
        action_log_probs=action_log_probs,
        old_action_log_probs=action_log_probs.detach(),
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
        config=fixed_completion_test_config().trainer.algorithm,
        loss_scale=1.0,
        report_scale=1.0,
    )

    token_nlls = []
    for prompt, response in zip(prompts, responses):
        ids = torch.tensor([prompt + response])
        logits = reference(input_ids=ids).logits[0, :-1].float()
        target_logprobs = torch.log_softmax(logits, dim=-1).gather(-1, ids[0, 1:, None]).squeeze(-1)
        token_nlls.append(-target_logprobs[len(prompt) - 1 :])
    independent_nll = torch.cat(token_nlls).mean()
    assert torch.allclose(objective.optimization_loss, independent_nll, rtol=1e-5, atol=1e-6)

    for network, loss in ((model, objective.optimization_loss), (reference, independent_nll)):
        optimizer = torch.optim.AdamW(network.parameters(), lr=1e-2, weight_decay=0.0)
        loss.backward()
        optimizer.step()
    for (name, stepped), expected in zip(model.named_parameters(), reference.parameters()):
        assert torch.allclose(stepped, expected, rtol=1e-5, atol=1e-6), name


def test_seeded_passes_consume_each_row_once_per_epoch_in_a_reproducible_order(tmp_path, tokenizer):
    dataset = build_dataset(tokenizer, write_text_rows(tmp_path / "train.jsonl", num_rows=16, routes=("route_0",)))
    cfg = fixed_completion_test_config(trainer={"train_batch_size": 4, "policy_mini_batch_size": 4, "seed": 7})
    validate_objective(cfg)
    batches_per_epoch = len(dataset) // cfg.trainer.train_batch_size
    batches = consumed_batches(cfg, dataset, 3 * batches_per_epoch)
    epochs = [uids(batches[start : start + batches_per_epoch]) for start in range(0, len(batches), batches_per_epoch)]

    every_uid = sorted(dataset.uid(index) for index in range(len(dataset)))
    assert all(sorted(epoch) == every_uid for epoch in epochs)
    assert epochs[0] != epochs[1] != epochs[2]
    assert uids(consumed_batches(cfg, dataset, len(batches))) == uids(batches)
    # Each pass draws its order from (seed, epoch); seed 8's first pass is not seed 7's second.
    next_seed = OmegaConf.merge(cfg, {"trainer": {"seed": 8}})
    assert uids(consumed_batches(next_seed, dataset, batches_per_epoch)) != epochs[1]


def test_domain_weighted_order_fills_every_batch_with_the_configured_rows_per_route(tmp_path, tokenizer):
    dataset = build_dataset(
        tokenizer, write_text_rows(tmp_path / "train.jsonl", num_rows=128, routes=("route_x", "route_r"))
    )
    cfg = with_open_maps(
        fixed_completion_test_config(trainer={"train_batch_size": 64, "policy_mini_batch_size": 64}),
        {"data": {"sampling": {"kind": "domain-weighted", "domain_weights": {"route_x": 1.0, "route_r": 1.0}}}},
    )
    validate_objective(cfg)
    batches = consumed_batches(cfg, dataset, 4)

    for batch in batches:
        routes = collections.Counter(prompt["input"].metadata.teacher_route for prompt in batch)
        assert routes == {"route_x": 32, "route_r": 32}
    every_uid = sorted(dataset.uid(index) for index in range(len(dataset)))
    assert sorted(uids(batches[:2])) == sorted(uids(batches[2:])) == every_uid
    # Seeded passes would leave each batch's route mix to chance, so a multi-route dataset needs this order.
    with pytest.raises(ValueError, match="domain-weighted"):
        prompt_order_from_config(fixed_completion_test_config(trainer={"train_batch_size": 64}), dataset)


def test_rollout_requests_cap_generation_at_their_task_cap():
    """Rollout training in a joint run gives each task's prompts that task's response cap."""
    cfg = with_open_maps(
        get_default_config(),
        {
            "generator": {
                "sampling_params": {"max_generate_length": 16},
                "max_generate_length_by_data_source": {"task_x": 4, "task_r": 16},
            }
        },
    )
    validate_objective(cfg)
    spec = RolloutRequestSpec.from_config(cfg)

    for source, cap in (("task_x", 4), ("task_r", 16)):
        prompt = {"uid": source, "input": task_input(lowered_task(text_task(source, source=source)))}
        assert spec.request(prompt, 0).sampling_params["max_tokens"] == cap


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

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a bounded native-JAX generation evaluation for the augmented d768 model."""

import json
import os
import re
import statistics
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import draccus
import jax
import jax.numpy as jnp
import jmp
import wandb
from datasets import load_dataset
from haliax.partitioning import set_mesh
from jax.sharding import Mesh, reshard
from jax.sharding import PartitionSpec as P
from levanter.checkpoint import latest_checkpoint_path, load_checkpoint
from levanter.data.text.datasets import DatasetComponent, LmDataConfig
from levanter.grug.sharding import compact_grug_mesh
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from experiments.grug.moe.launch_cw_scale import build_scale_model
from experiments.grug.moe.model import Transformer
from experiments.grug.moe.train import GrugEvalConfig, _apply_qb_betas, build_tagged_evaluator
from experiments.marin_tokenizer import marin_tokenizer

_GSM8K_REVISION = "e53f048"
_DEFAULT_LIMIT = 64
_DEFAULT_BATCH_SIZE = 8
_DEFAULT_MAX_NEW_TOKENS = 192
_DEFAULT_MAX_LENGTH = 512
_DEFAULT_EVAL_CONFIG_RUN = "nest-augdk-e256-4b-r2"
_ANSWER_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


@dataclass(frozen=True)
class GenerationResult:
    prompt: str
    response: str
    expected: str | None
    correct: bool | None


@dataclass(frozen=True)
class InstructionCase:
    prompt: str
    grader: Callable[[str], bool]


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None:
        raise ValueError(f"{name} must be set")
    return value


def _last_number(text: str) -> str | None:
    matches = _ANSWER_NUMBER.findall(text)
    if not matches:
        return None
    return matches[-1].replace(",", "")


def _gsm8k_answer(answer: str) -> str:
    marker = answer.rsplit("####", maxsplit=1)
    if len(marker) != 2:
        raise ValueError(f"GSM8K answer has no final marker: {answer!r}")
    value = _last_number(marker[1])
    if value is None:
        raise ValueError(f"GSM8K answer has no final number: {answer!r}")
    return value


def _instruction_cases() -> tuple[InstructionCase, ...]:
    return (
        InstructionCase("Reply with exactly OK and no punctuation.", lambda text: text.strip() == "OK"),
        InstructionCase(
            'Return only a JSON object with string keys "action" and "reason".',
            lambda text: _is_json_with_exact_keys(text, {"action", "reason"}),
        ),
        InstructionCase(
            "Give exactly three non-empty lines numbered 1., 2., and 3. about checking a training job.",
            lambda text: [line[:2] for line in text.strip().splitlines()] == ["1.", "2.", "3."],
        ),
        InstructionCase(
            "Write exactly one sentence that contains the word checkpoint.",
            lambda text: "checkpoint" in text.lower() and _sentence_count(text) == 1,
        ),
        InstructionCase(
            "Reply with the five words alpha beta gamma delta epsilon, in that order and nothing else.",
            lambda text: text.strip().lower() == "alpha beta gamma delta epsilon",
        ),
        InstructionCase(
            "Return only a JSON array containing the integers 2, 4, and 8.",
            lambda text: _json_value(text) == [2, 4, 8],
        ),
        InstructionCase(
            "State whether 17 is prime using exactly two words.",
            lambda text: len(text.strip().split()) == 2 and "prime" in text.lower(),
        ),
        InstructionCase(
            "Write one line beginning ERROR: followed by a six-word description of a stalled collective.",
            lambda text: len(text.strip().split()) == 7 and text.strip().startswith("ERROR:"),
        ),
    )


def _json_value(text: str) -> object | None:
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return None


def _is_json_with_exact_keys(text: str, keys: set[str]) -> bool:
    value = _json_value(text)
    return isinstance(value, dict) and set(value) == keys and all(isinstance(item, str) for item in value.values())


def _sentence_count(text: str) -> int:
    return len([part for part in re.split(r"[.!?]+", text.strip()) if part.strip()])


def _chat_prompt(tokenizer: PreTrainedTokenizerBase, prompt: str) -> list[int]:
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Follow the user's instructions carefully."},
        {"role": "user", "content": prompt},
    ]
    tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if isinstance(tokens, Mapping):
        tokens = tokens["input_ids"]
    if tokens and isinstance(tokens[0], list):
        tokens = tokens[0]
    return [int(token) for token in tokens]


def _generation_batch(
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    *,
    max_length: int,
    max_new_tokens: int,
) -> tuple[jax.Array, jax.Array]:
    tokenized = [_chat_prompt(tokenizer, prompt) for prompt in prompts]
    max_prompt_length = max_length - max_new_tokens
    tokenized = [tokens[-max_prompt_length:] for tokens in tokenized]
    lengths = jnp.asarray([len(tokens) for tokens in tokenized], dtype=jnp.int32)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("tokenizer has neither a pad nor EOS token")
    token_ids = jnp.full((len(prompts), max_length), int(pad_token_id), dtype=jnp.int32)
    for row, tokens in enumerate(tokenized):
        token_ids = token_ids.at[row, : len(tokens)].set(jnp.asarray(tokens, dtype=jnp.int32))
    return token_ids, lengths


def _eligibility(batch_size: int, expert_count: int, total_experts: int) -> jax.Array | None:
    if expert_count == total_experts:
        return None
    if expert_count <= 0 or expert_count > total_experts:
        raise ValueError(f"expert count must be in [1, {total_experts}], got {expert_count}")
    expert_ids = jnp.arange(total_experts)
    return jnp.broadcast_to(expert_ids < expert_count, (batch_size, total_experts))


def _generate(
    model: Transformer,
    token_ids: jax.Array,
    prompt_lengths: jax.Array,
    *,
    expert_count: int,
    eos_token_id: int,
    pad_token_id: int,
    max_new_tokens: int,
) -> jax.Array:
    batch_size = token_ids.shape[0]
    eligibility = _eligibility(batch_size, expert_count, model.config.num_experts)
    row_ids = jnp.arange(batch_size)

    @jax.jit
    def generate(params: Transformer, initial_ids: jax.Array, lengths: jax.Array) -> jax.Array:
        batch_axes = ("replica_dcn", "data", "expert")
        initial_ids = reshard(initial_ids, P(batch_axes, None))
        lengths = reshard(lengths, P(batch_axes))

        def body(index: int, carry: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
            ids, finished = carry
            positions = lengths - 1 + index
            logits = params.logits(ids, expert_eligibility=eligibility)
            next_logits = logits.at[row_ids, positions].get(out_sharding=P(("replica_dcn", "data", "expert"), "model"))
            next_ids = jnp.argmax(next_logits, axis=-1).astype(jnp.int32)
            next_ids = jnp.where(finished, pad_token_id, next_ids)
            ids = jax.vmap(lambda row, position, value: row.at[position].set(value))(
                ids,
                positions + 1,
                next_ids,
            )
            return ids, finished | (next_ids == eos_token_id)

        generated, _ = jax.lax.fori_loop(
            0,
            max_new_tokens,
            body,
            (
                initial_ids,
                reshard(jnp.zeros((batch_size,), dtype=jnp.bool_), P(batch_axes)),
            ),
        )
        return generated

    return generate(model, token_ids, prompt_lengths).block_until_ready()


def _decode_batch(
    tokenizer: PreTrainedTokenizerBase,
    generated: jax.Array,
    prompt_lengths: jax.Array,
    *,
    max_new_tokens: int,
) -> list[str]:
    host_tokens = jax.device_get(generated)
    host_lengths = jax.device_get(prompt_lengths)
    responses = []
    for tokens, prompt_length in zip(host_tokens, host_lengths, strict=True):
        output = tokens[int(prompt_length) : int(prompt_length) + max_new_tokens]
        responses.append(tokenizer.decode(output, skip_special_tokens=True).strip())
    return responses


def _evaluate_gsm8k(
    model: Transformer,
    tokenizer: PreTrainedTokenizerBase,
    *,
    expert_count: int,
    limit: int,
    batch_size: int,
    max_length: int,
    max_new_tokens: int,
) -> list[GenerationResult]:
    dataset = load_dataset("openai/gsm8k", "main", split=f"test[:{limit}]", revision=_GSM8K_REVISION)
    prompts = [
        f"Solve this problem. Show your reasoning, then give the final numerical answer.\n\n{question}"
        for question in dataset["question"]
    ]
    expected = [_gsm8k_answer(answer) for answer in dataset["answer"]]
    results = []
    for start in range(0, len(prompts), batch_size):
        prompt_batch = prompts[start : start + batch_size]
        expected_batch = expected[start : start + batch_size]
        token_ids, lengths = _generation_batch(
            tokenizer,
            prompt_batch,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
        )
        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        eos_token_id = tokenizer.eos_token_id
        if pad_token_id is None or eos_token_id is None:
            raise ValueError("tokenizer must define EOS and padding token ids")
        generated = _generate(
            model,
            token_ids,
            lengths,
            expert_count=expert_count,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            max_new_tokens=max_new_tokens,
        )
        responses = _decode_batch(tokenizer, generated, lengths, max_new_tokens=max_new_tokens)
        for prompt, response, answer in zip(prompt_batch, responses, expected_batch, strict=True):
            results.append(
                GenerationResult(
                    prompt=prompt,
                    response=response,
                    expected=answer,
                    correct=_last_number(response) == answer,
                )
            )
    return results


def _evaluate_instructions(
    model: Transformer,
    tokenizer: PreTrainedTokenizerBase,
    *,
    expert_count: int,
    max_length: int,
    max_new_tokens: int,
) -> list[GenerationResult]:
    cases = _instruction_cases()
    prompts = [case.prompt for case in cases]
    token_ids, lengths = _generation_batch(
        tokenizer,
        prompts,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
    )
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    eos_token_id = tokenizer.eos_token_id
    if pad_token_id is None or eos_token_id is None:
        raise ValueError("tokenizer must define EOS and padding token ids")
    generated = _generate(
        model,
        token_ids,
        lengths,
        expert_count=expert_count,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        max_new_tokens=max_new_tokens,
    )
    responses = _decode_batch(tokenizer, generated, lengths, max_new_tokens=max_new_tokens)
    return [
        GenerationResult(prompt=case.prompt, response=response, expected=None, correct=case.grader(response))
        for case, response in zip(cases, responses, strict=True)
    ]


def _load_model(checkpoint_root: str, mesh: Mesh) -> Transformer:
    model_config = build_scale_model()

    @jax.jit
    def initialize(key: jax.Array) -> dict[str, object]:
        params = Transformer.init(model_config, key=key)
        return {
            "params": params,
            "pending_qb_betas": jnp.zeros_like(params.stacked_blocks.stacked.mlp.router_bias),
        }

    exemplar = initialize(jax.random.PRNGKey(0))
    checkpoint_path = latest_checkpoint_path(checkpoint_root)
    loaded = load_checkpoint(exemplar, checkpoint_path, mesh=mesh, allow_partial=True)
    params = _apply_qb_betas(loaded["params"], loaded["pending_qb_betas"])
    return jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16").cast_to_compute(params)


def _evaluation_data_config(source_run_id: str) -> LmDataConfig:
    source_run = wandb.Api(timeout=60).run(f"marin-community/marin_moe/{source_run_id}")
    raw_data = source_run.config["data"]
    components = {
        name: draccus.decode(DatasetComponent, raw_component)
        for name, raw_component in raw_data["components"].items()
        if name.startswith(("paloma/", "uncheatable_eval/"))
    }
    if len(components) != 23:
        raise ValueError(f"Expected 23 evaluation components from {source_run_id}, found {len(components)}")
    return LmDataConfig(
        components=components,
        train_weights={name: 0.0 for name in components},
        tokenizer=raw_data["tokenizer"],
        cache_dir=None,
    )


def _evaluate_perplexity(
    model: Transformer,
    data_config: LmDataConfig,
    mesh: Mesh,
    *,
    expert_count: int,
) -> dict[str, object]:
    model_config = model.config
    evaluator = build_tagged_evaluator(
        data_config=data_config,
        max_seq_len=model_config.max_seq_len,
        mesh=mesh,
        eval_cfg=GrugEvalConfig(
            eval_batch_size=128,
            max_eval_batches=8,
            eval_current=True,
            eval_ema=False,
            compute_bpb=True,
        ),
        mp=jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16"),
        nested_expert_count=expert_count if expert_count != model_config.num_experts else None,
    )
    if evaluator is None:
        raise ValueError("Evaluation data config produced no tagged evaluator")
    result = evaluator.evaluate(model)
    return {
        "macro_loss": result.macro_avg_loss,
        "micro_loss": result.micro_avg_loss,
        "paloma_macro_loss": result.tag_macro_losses["paloma"],
        "uncheatable_macro_loss": result.tag_macro_losses["uncheatable_eval"],
        "domain_losses": result.tag_macro_losses,
    }


def main() -> None:
    checkpoint_root = _required_env("AUGDK_EVAL_CHECKPOINT")
    run_id = _required_env("AUGDK_EVAL_RUN_ID")
    expert_count = int(os.environ.get("AUGDK_EVAL_EXPERT_COUNT", "256"))
    limit = int(os.environ.get("AUGDK_EVAL_LIMIT", str(_DEFAULT_LIMIT)))
    batch_size = int(os.environ.get("AUGDK_EVAL_BATCH_SIZE", str(_DEFAULT_BATCH_SIZE)))
    max_new_tokens = int(os.environ.get("AUGDK_EVAL_MAX_NEW_TOKENS", str(_DEFAULT_MAX_NEW_TOKENS)))
    max_length = int(os.environ.get("AUGDK_EVAL_MAX_LENGTH", str(_DEFAULT_MAX_LENGTH)))
    eval_config_run = os.environ.get("AUGDK_EVAL_CONFIG_RUN", _DEFAULT_EVAL_CONFIG_RUN)

    tokenizer = AutoTokenizer.from_pretrained(marin_tokenizer)
    data_config = _evaluation_data_config(eval_config_run)
    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)
    with set_mesh(mesh):
        model = _load_model(checkpoint_root, mesh)
        perplexity = _evaluate_perplexity(
            model,
            data_config,
            mesh,
            expert_count=expert_count,
        )
        gsm8k = _evaluate_gsm8k(
            model,
            tokenizer,
            expert_count=expert_count,
            limit=limit,
            batch_size=batch_size,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
        )
        instructions = _evaluate_instructions(
            model,
            tokenizer,
            expert_count=expert_count,
            max_length=max_length,
            max_new_tokens=min(max_new_tokens, 96),
        )
    results = {
        "checkpoint": checkpoint_root,
        "expert_count": expert_count,
        "perplexity": perplexity,
        "gsm8k": [asdict(result) for result in gsm8k],
        "instructions": [asdict(result) for result in instructions],
        "summary": {
            "gsm8k_exact_match": statistics.fmean(result.correct is True for result in gsm8k),
            "instruction_pass_rate": statistics.fmean(result.correct is True for result in instructions),
            "gsm8k_examples": len(gsm8k),
            "instruction_examples": len(instructions),
            "paloma_macro_loss": perplexity["paloma_macro_loss"],
            "uncheatable_macro_loss": perplexity["uncheatable_macro_loss"],
        },
    }
    run = wandb.init(
        entity="marin-community",
        project="marin_moe_sft",
        id=run_id,
        name=run_id,
        group="NEST-AUGDK-SFT-EVAL",
        tags=["moe", "nested-moe", "aug-dk", "generation-eval", f"e{expert_count}"],
        resume="never",
        config={
            "checkpoint": checkpoint_root,
            "expert_count": expert_count,
            "eval_config_run": eval_config_run,
            "gsm8k_revision": _GSM8K_REVISION,
            "gsm8k_limit": limit,
            "max_new_tokens": max_new_tokens,
        },
    )
    run.log(results["summary"])
    with tempfile.TemporaryDirectory() as temp_dir:
        result_path = Path(temp_dir) / "generation_eval.json"
        result_path.write_text(json.dumps(results, indent=2) + "\n")
        artifact = wandb.Artifact(f"{run_id}-results", type="evaluation")
        artifact.add_file(str(result_path))
        run.log_artifact(artifact)
    run.finish()


if __name__ == "__main__":
    main()

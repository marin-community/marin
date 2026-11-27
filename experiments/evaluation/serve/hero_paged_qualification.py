# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Qualify full-weight Hero paged generation on a 16-device GB200 mesh."""

import argparse
import dataclasses
import json
import logging
import os
import time
from pathlib import Path

import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
from haliax import Axis
from iris.jax.init import initialize_jax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.compat.hf_checkpoints import HFCheckpointConverter, load_tokenizer
from levanter.grug.sharding import compact_grug_mesh
from levanter.inference.engine import InferenceEngine, InferenceEngineConfig, Request
from levanter.inference.jit_scheduler import SeqDecodingParams
from levanter.models.hero import HeroConfig
from levanter.models.hero_model import HeroLMHeadModel
from levanter.models.snowball import GrugMoeHfConfig

_PAGE_SIZE = 128
_EXPERT_AXIS_SIZE = 4
_MODEL_AXIS_SIZE = 4
_OUTPUT_TOKENS = 4
_PROMPT_TEXT = (
    "The observatory logged the time, the direction of the telescope, and the temperature "
    "before starting another measurement. Compare the readings from both instruments.\n"
)
logger = logging.getLogger(__name__)


def _request(prompt: list[int], request_id: int, choices: int = 2) -> Request:
    params = dataclasses.replace(
        SeqDecodingParams.default(), max_num_tokens=jnp.asarray(len(prompt) + _OUTPUT_TOKENS, dtype=jnp.int32)
    )
    return Request(prompt, request_id, params, n_generations=choices)


@jax.jit
def _reference_logits(model: HeroLMHeadModel, tokens: hax.NamedArray) -> jax.Array:
    activations = model.activations(tokens).array[:, -_OUTPUT_TOKENS:, :]
    logits = jnp.einsum("bsd,dv->bsv", activations, model.get_lm_head().array, out_sharding=P())
    return logits[0].astype(jnp.float32)


def _score_reference(
    model: HeroLMHeadModel,
    prompt: list[int],
    generation: list[int],
    mesh: jax.sharding.Mesh,
) -> tuple[np.ndarray, np.ndarray]:
    # The hidden state for the last prompt token predicts the first generated token.
    ids = np.asarray(prompt + generation[:-1], dtype=np.int32)
    batch = np.broadcast_to(ids, (_EXPERT_AXIS_SIZE, ids.size))
    sharding = NamedSharding(mesh, P("expert", None))
    tokens = jax.make_array_from_callback(batch.shape, sharding, lambda index: batch[index])
    named = hax.named(tokens, (Axis("batch", batch.shape[0]), Axis("position", ids.size)))
    logits = _reference_logits(model, named)
    token_ids = jax.make_array_from_callback(
        (_OUTPUT_TOKENS,), NamedSharding(mesh, P()), lambda index: np.asarray(generation, dtype=np.int32)[index]
    )
    scores = jax.nn.log_softmax(logits, axis=-1)[jnp.arange(_OUTPUT_TOKENS), token_ids]
    return jax.device_get(jnp.argmax(logits, axis=-1)), jax.device_get(scores)


def _run(engine, model, mesh, requests):
    choice_slots: dict[tuple[int, int], int] = {}

    def record_slots(_iteration: int) -> None:
        for slot, choice in engine.local_map.items():
            assigned = choice_slots.setdefault(choice, slot)
            if assigned != slot:
                raise AssertionError(f"Choice {choice} moved from slot {assigned} to {slot}")

    start = time.monotonic()
    result = engine.generate(requests, step_callback=record_slots)
    latency = time.monotonic() - start
    if result.logprobs is None:
        raise ValueError("The inference engine omitted generated log probabilities")
    if len(result.tokens) != sum(request.n_generations for request in requests):
        raise ValueError("The engine omitted a generated choice")
    if engine.local_map or sorted(engine.free_slots) != list(range(engine.config.max_seqs)):
        raise AssertionError("The engine failed to release all sequence slots")

    offset = 0
    comparisons = []
    for request in requests:
        first_choice = result.tokens[offset]
        first_scores = np.asarray(result.logprobs[offset])
        hypotheses = {}
        for choice_index in range(offset, offset + request.n_generations):
            choice = result.tokens[choice_index]
            if len(choice) != _OUTPUT_TOKENS:
                raise ValueError(f"Expected {_OUTPUT_TOKENS} output tokens, got {len(choice)}")
            key = tuple(choice)
            if key not in hypotheses:
                hypotheses[key] = _score_reference(model, request.prompt_tokens, choice, mesh)
            expected_tokens, expected_scores = hypotheses[key]
            actual_scores = np.asarray(result.logprobs[choice_index])
            comparisons.append(
                {
                    "request_id": request.request_id,
                    "slot": choice_slots[(request.request_id, choice_index - offset)],
                    "prompt_tokens": len(request.prompt_tokens),
                    "generated_tokens": choice,
                    "reference_greedy_tokens": expected_tokens.tolist(),
                    "token_agreement": int(np.count_nonzero(expected_tokens == choice)),
                    "max_abs_logprob_difference": float(np.max(np.abs(expected_scores - actual_scores))),
                    "finite_logprobs": bool(np.all(np.isfinite(actual_scores)) and np.all(np.isfinite(expected_scores))),
                    "clone_matches_tokens": choice == first_choice,
                    "max_clone_logprob_difference": float(np.max(np.abs(first_scores - actual_scores))),
                }
            )
        offset += request.n_generations
    if jax.process_index() == 0:
        logger.info("Hero parity case %s", json.dumps({"seconds": latency, "choices": comparisons}))
    return result.tokens, {"seconds": latency, "choices": comparisons}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--context", type=int, choices=(65536, 262144), required=True)
    parser.add_argument("--max-logprob-delta", type=float, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    initialize_jax()
    devices = jax.devices()
    if len(devices) != _EXPERT_AXIS_SIZE * _MODEL_AXIS_SIZE or any("GB200" not in d.device_kind for d in devices):
        raise ValueError(f"Expected 16 GB200 devices, got {devices}")
    tokenizer = load_tokenizer(args.tokenizer, revision=args.tokenizer_revision, trust_remote_code=False)
    checkpoint = HFCheckpointConverter(
        HeroConfig, reference_checkpoint=args.weights, HfConfigClass=GrugMoeHfConfig, tokenizer=tokenizer
    )
    config = checkpoint.default_config
    if config.max_seq_len != args.context:
        raise ValueError(f"Expected checkpoint context {args.context}, got {config.max_seq_len}")
    if args.context == 262144 and config.global_query_log_scaling is None:
        raise ValueError("262K qualification requires a checkpoint with global query log scaling")
    config = dataclasses.replace(
        config,
        attention_implementation="gpu_fa4_cute",
        inference_attention_implementation="gpu",
        moe_implementation="ring",
        capacity_factor=_EXPERT_AXIS_SIZE,
    )
    mesh = compact_grug_mesh(replica_axis_size=1, expert_axis_size=_EXPERT_AXIS_SIZE, model_axis_size=_MODEL_AXIS_SIZE)
    with jax.set_mesh(mesh):
        started = time.monotonic()
        model = checkpoint.load_pretrained(HeroLMHeadModel, config=config)
        jax.block_until_ready(model)
        load_time = time.monotonic() - started
        logger.info("Loaded Hero checkpoint in %.1f seconds", load_time)
        engine = InferenceEngine.from_model_with_config(
            model,
            checkpoint.tokenizer,
            InferenceEngineConfig(
                page_size=_PAGE_SIZE,
                max_seq_len=args.context,
                max_pages=args.context // _PAGE_SIZE + 2,
                max_seqs=2,
                max_seqs_in_prefill=2,
                max_prefill_size=args.context,
                max_tokens_per_round=_EXPERT_AXIS_SIZE,
                max_queued_tokens=8,
                max_rounds=4,
            ),
        )
        phrase = checkpoint.tokenizer.encode(_PROMPT_TEXT, add_special_tokens=False)
        if not phrase:
            raise ValueError("The checkpoint tokenizer returned an empty phrase")
        long_size = args.context - 2 * _OUTPUT_TOKENS - 1
        long_prompt = (phrase * ((long_size + len(phrase) - 1) // len(phrase)))[:long_size]
        first = _request(phrase, 0)
        queued = _request(phrase + phrase, 1)
        local_device = jax.local_devices()[0]
        kv_cache_bytes = sum(
            next(
                shard.data.nbytes for shard in layer.kv.kv_pages.array.addressable_shards if shard.device == local_device
            )
            for layer in engine.gen_state.cache.caches
        )
        logger.info("Hero BF16 KV allocation per rank: %d bytes", kv_cache_bytes)
        _, short = _run(engine, model, mesh, [first, queued])
        first_slots = {row["slot"] for row in short["choices"] if row["request_id"] == first.request_id}
        second_slots = {row["slot"] for row in short["choices"] if row["request_id"] == queued.request_id}
        if first_slots != second_slots or len(first_slots) != first.n_generations:
            raise AssertionError(f"Queued requests did not recycle both sequence slots: {first_slots}, {second_slots}")
        first_turn, long = _run(engine, model, mesh, [_request(long_prompt, 2)])
        continuation = long_prompt + first_turn[0] + phrase[:1]
        assert len(continuation) + _OUTPUT_TOKENS == args.context
        _, next_turn = _run(engine, model, mesh, [_request(continuation, 3)])
        try:
            engine.generate([_request(continuation + phrase[:_OUTPUT_TOKENS], 4, choices=1)])
        except ValueError:
            pass
        else:
            raise AssertionError("Generation accepted a prompt that left no room for an output")

    report = {
        "context": args.context,
        "mesh": dict(mesh.shape),
        "tokenizer_revision": args.tokenizer_revision,
        "checkpoint_load_seconds": load_time,
        "kv_cache_bytes_per_rank": kv_cache_bytes,
        "max_logprob_delta": args.max_logprob_delta,
        "short_queue": short,
        "first_turn": long,
        "boundary_continuation": next_turn,
    }
    if jax.process_index() == 0:
        output = Path(os.environ["IRIS_OUTPUT_DIR"]) / "hero-parity.json"
        output.write_text(json.dumps(report, indent=2) + "\n")
    for case in (short, long, next_turn):
        for row in case["choices"]:
            if not (
                row["token_agreement"] == _OUTPUT_TOKENS
                and row["finite_logprobs"]
                and row["max_abs_logprob_difference"] <= args.max_logprob_delta
                and row["clone_matches_tokens"]
                and row["max_clone_logprob_difference"] == 0
            ):
                raise AssertionError(f"Hero generation failed the token and score parity gate: {row}")


if __name__ == "__main__":
    main()

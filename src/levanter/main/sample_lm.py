# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
import logging
import time
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Optional, cast

import equinox as eqx
import haliax as hax
import jax.numpy as jnp
import jax.random as jrandom
from haliax import Axis
from haliax.partitioning import round_axis_for_partitioning

import levanter
from levanter.callbacks import profile_ctx
from levanter.checkpoint import load_checkpoint
from levanter.compat.hf_checkpoints import HFCheckpointConverter, RepoRef, load_tokenizer
from levanter.inference.engine import InferenceEngine, InferenceEngineConfig, Request
from levanter.inference.jit_scheduler import SeqDecodingParams
from levanter.inference.utils import INVALID
from levanter.models.llama import LlamaConfig, LlamaLMHeadModel
from levanter.models.lm_model import LmConfig, LmHeadModel
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import use_cpu_device

logger = logging.getLogger(__name__)


@dataclass
class SampleLmConfig:
    """Configuration for simple text sampling."""

    checkpoint_path: Optional[str] = None
    hf_checkpoint: Optional[RepoRef] = None

    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    model: LmConfig = field(default_factory=LlamaConfig)

    tokenizer: str | None = None

    # Inference service/memory layout configuration
    engine: InferenceEngineConfig = field(default_factory=InferenceEngineConfig)

    prompts: list[str] | str | tuple[str, ...] = (
        "Four score and seven years ago, our",
        # "On the first day of Christmas, my true love gave to me",
        "In a hole in the ground there lived a hobbit, not a nasty, dirty, wet hole",
    ) * 5
    stop_sequence: str | None = "."
    "Stop sequences. Currently only does whole token sequences."
    max_new_tokens: int = 192
    temperature: float = 0.7
    seed: int = 2

    n_generations: int = 1
    n_rounds: int = 4

    # Optional JAX profiling
    profile: bool = False


def _load_model(config: SampleLmConfig, Vocab: Axis, *, key) -> LmHeadModel:
    """Load a model either from a checkpoint or HF repo."""

    if config.checkpoint_path is None and config.hf_checkpoint is None:
        raise ValueError("Must specify either checkpoint_path or hf_checkpoint")
    if config.checkpoint_path is not None and config.hf_checkpoint is not None:
        raise ValueError("Specify only one of checkpoint_path or hf_checkpoint")

    mp = config.trainer.mp

    if config.checkpoint_path is not None:
        with use_cpu_device():
            model = eqx.filter_eval_shape(config.model.build, Vocab, key=key)
            model = load_checkpoint(model, config.checkpoint_path, subpath="model")
            model = mp.cast_to_compute(model)
        return model
    else:
        assert hasattr(config.model, "hf_checkpoint_converter"), "model config lacks HF loader"
        converter: HFCheckpointConverter = config.model.hf_checkpoint_converter()
        converter = converter.replaced(
            reference_checkpoint=config.hf_checkpoint, tokenizer=load_tokenizer(config.tokenizer)
        )
        model = converter.load_pretrained(
            config.model.model_type, ref=config.hf_checkpoint, dtype=config.trainer.mp.compute_dtype
        )
        return model


def main(config: SampleLmConfig):
    levanter.initialize(config)
    tok_string: str | None = config.tokenizer
    if config.tokenizer is None:
        if config.hf_checkpoint is not None:
            # If we have an HF checkpoint, we can load the tokenizer from it
            tok_string = config.hf_checkpoint.model_name_or_path

    if tok_string is None:
        raise ValueError("Must specify a tokenizer or an HF checkpoint with a tokenizer")

    tokenizer = load_tokenizer(config.tokenizer)

    key = jrandom.PRNGKey(config.seed)

    # NB: we use the compute_axis_mapping b/c we're doing inference
    with config.trainer.device_mesh, hax.axis_mapping(config.trainer.compute_axis_mapping):
        vocab_size = len(tokenizer)
        Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), config.trainer.compute_axis_mapping)
        model = _load_model(config, Vocab, key=key)
        assert isinstance(model, LlamaLMHeadModel), "Only LlamaLMHeadModel supported"

        prompts = config.prompts

        if isinstance(prompts, str):
            prompts = [prompts]

        prompt_ids = tokenizer(prompts, add_special_tokens=False)["input_ids"]

        # Initialize a reusable generation service with capacity from config
        service = InferenceEngine.from_model_with_config(model=model, tokenizer=tokenizer, config=config.engine)

        # -------------------------------- Scheduler-based generation --------------------------------

        stop_sequence = config.stop_sequence
        if stop_sequence is not None:
            stop_ids_list = tokenizer(stop_sequence, add_special_tokens=False)["input_ids"]
            if len(stop_ids_list) == 0:
                raise ValueError("Stop sequence must be non-empty")
            stop_ids = hax.named(jnp.asarray(stop_ids_list, dtype=jnp.int32), axis="position").broadcast_axis(
                {"stop_seq": 1}
            )
        else:
            stop_ids = None

        # Optionally enable JAX profiler; no Perfetto link
        with ExitStack() as stack:

            for r in range(config.n_rounds):
                if config.profile:
                    # skip round 0 b/c of compilation unless we're only doing one round
                    if config.n_rounds == 1 or r == 1:
                        run_id = cast(str, config.trainer.id)
                        profile_path = config.trainer.log_dir / run_id / "profiler"
                        stack.enter_context(
                            profile_ctx(
                                str(profile_path),
                                create_perfetto_link=False,
                                host_profile=True,
                                host_profile_topn=40,
                            )
                        )

                for i, toks in enumerate(prompt_ids):
                    print(f"Prompt {i}: {toks}")

                time_in = time.time()
                # Build Requests for this batch
                base_key = jrandom.PRNGKey(config.engine.seed)
                reqs: list[Request] = []
                for ridx, toks in enumerate(prompt_ids):
                    seq_params = SeqDecodingParams(
                        max_num_tokens=jnp.array(len(toks) + config.max_new_tokens, dtype=jnp.int32),
                        stop_tokens=stop_ids,
                        temperature=jnp.array(config.temperature, dtype=jnp.float32),
                        key=jrandom.fold_in(base_key, ridx),
                    )
                    reqs.append(
                        Request(
                            prompt_tokens=list(map(int, toks)),
                            request_id=ridx,  # not used for indexing in this API
                            decode_params=seq_params,
                            n_generations=config.n_generations,
                        )
                    )

                result = service.generate(reqs)
                print(
                    f"Round {r} took {time.time() - time_in:.2f} seconds, "
                    f"generated {result.total_generated} tokens in {len(result.tokens)} sequences."
                )

                # Decode and print outputs
                for seq_id, seq_outputs in enumerate(result.tokens):
                    seq_outputs = [tok for tok in seq_outputs if tok != tokenizer.pad_token_id and tok != INVALID]
                    text = tokenizer.decode(seq_outputs, skip_special_tokens=True)
                    print(f"Tokens for sequence {seq_id} (len: {len(seq_outputs)}: {seq_outputs}")
                    print(f"Generated text for {seq_id}: {text}")
                    tokens_with_text = [
                        f"{tok} ({tokenizer.decode([tok], skip_special_tokens=True)})" for tok in seq_outputs
                    ]
                    print(f"Tokens with text for sequence {seq_id}: {tokens_with_text}")

    levanter.current_tracker().finish()


if __name__ == "__main__":
    levanter.config.main(main)()

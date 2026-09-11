# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Restore a hero checkpoint on one rack and autoregressively sample a completion
for a prompt, one token at a time (no KV cache -- each step re-runs the full
[batch, SEQ_LEN] forward and reads the logits at the current last position). The
forward runs at the EP eval shape so the MoE is happy; every batch row holds the
same growing sequence, and only row 0's logits are gathered to the host, softmaxed
at the requested temperature, and sampled with numpy. `pos` is a traced argument,
so the forward compiles once and every step reuses it.

    uv run iris --config lib/iris/config/marin.yaml job run --no-wait --enable-extra-resources \\
        --target-cluster cw-us-east-08a --priority production \\
        -- python -m experiments.grug.moe_hero_ep.sample_gen --version 2026.08.19.2 \\
           --prompt '<text>' --max-new-tokens 300 --temperature 1.0 --run
"""

import dataclasses
import json
import logging

import click
import fsspec
import jax
import jax.numpy as jnp
import numpy as np
from fray.cluster import ResourceConfig
from haliax.partitioning import set_mesh
from jax.sharding import NamedSharding, PartitionSpec, reshard
from levanter.grug.sharding import compact_grug_mesh
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options

from experiments.grug.checkpointing import restore_grug_state_from_checkpoint
from experiments.grug.dispatch import dispatch_grug_training_run
from experiments.grug.moe_hero_ep.launch_mfu_test import HERO_EP_EXPERT_AXIS_SIZE, HERO_GPUS_PER_NODE
from experiments.grug.moe_hero_ep.launch_scaling_ladder import build_ladder_run
from experiments.grug.moe_hero_ep.small_scale_abl_launch import SEQ_LEN
from experiments.grug.moe_hero_ep.train import (
    GrugRunConfig,
    GrugTrainState,
    MasterParamMode,
    _apply_hero_ep_runtime_defaults,
    initial_state,
)

logger = logging.getLogger(__name__)

HERO_RUN_ID = "hero-12d8b6f0-dee637"
HERO_VERSION = "2026.08.19.2"
RUN_ID_BASE = "hero-12d8b6f0-samplegen-42k"  # main() adds a temp/seed suffix for a distinct executor fingerprint
STEP = "step-42000"
CHECKPOINT_BASE = f"s3://marin-us-east-02a/marin/grug/{HERO_RUN_ID}/{HERO_VERSION}/checkpoints/{STEP}"
OUT_DIR = f"s3://marin-us-east-02a/marin/grug/{HERO_RUN_ID}/analysis"

ONE_RACK_NODES = 16
_ONE_RACK_RESOURCES = ResourceConfig.with_gpu(
    "GB200", count=HERO_GPUS_PER_NODE, cpu=120, ram="850g", disk="1t", replicas=ONE_RACK_NODES
)
_EVAL_BATCH = HERO_EP_EXPERT_AXIS_SIZE  # 64 rows, one per expert device; all identical


def _make_local(prompt: str, max_new_tokens: int, temperature: float, seed: int):
    def _sample_local(config: GrugRunConfig) -> None:
        trainer = config.trainer.trainer
        trainer.initialize()
        optimizer = config.optimizer.build(trainer.num_train_steps)
        tokenizer = config.data.the_tokenizer
        mesh = compact_grug_mesh(
            expert_axis_size=config.trainer.expert_axis_size,
            replica_axis_size=config.trainer.replica_axis_size,
        )
        with set_mesh(mesh):

            @jax.jit
            def _init_state(model_rng: jax.Array) -> GrugTrainState:
                return initial_state(
                    config.model,
                    optimizer=optimizer,
                    mp=trainer.mp,
                    key=model_rng,
                    ema_beta=None,
                    offload_opt_state=False,
                    master_param_mode=MasterParamMode.DISABLED,
                )

            state = _init_state(jax.random.PRNGKey(trainer.seed))
            state = restore_grug_state_from_checkpoint(
                state,
                checkpoint_search_paths=[CHECKPOINT_BASE],
                load_checkpoint_setting=True,
                mesh=mesh,
                allow_partial=True,
            )
            model = state.params
            rep = NamedSharding(mesh, PartitionSpec())

            @jax.jit
            def _logits_at(m, ids, pos):
                hidden, _ = m(ids)  # mask=None -> causal; [B, S, D]
                htgt = hidden.at[:, pos].get(out_sharding=PartitionSpec())  # [B, D] final hidden at pos
                logits = jnp.einsum("bh,hv->bv", htgt, m.output_proj)  # [B, V]; predicts token at pos+1
                return reshard(logits, PartitionSpec())

            prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)  # prepends BOS if defined
            if len(prompt_ids) + max_new_tokens > SEQ_LEN:
                raise ValueError(f"prompt {len(prompt_ids)} + {max_new_tokens} exceeds SEQ_LEN={SEQ_LEN}")
            logger.info("prompt (%d tokens): %r", len(prompt_ids), prompt)

            rng = np.random.default_rng(seed)
            tokens = list(prompt_ids)  # grows in place; row 0 of the batch mirrors it each step
            batch = np.zeros((_EVAL_BATCH, SEQ_LEN), dtype=np.int32)
            new_ids: list[int] = []
            for step in range(max_new_tokens):
                cur_len = len(tokens)
                batch[:, :cur_len] = np.asarray(tokens, dtype=np.int32)  # all rows identical
                token_ids = jax.device_put(jnp.asarray(batch), rep)
                logits = np.asarray(
                    jax.device_get(_logits_at(model, token_ids, jnp.asarray(cur_len - 1, dtype=jnp.int32)))
                )[0].astype(
                    np.float64
                )  # row 0 [V]
                if temperature <= 0.0:
                    tok = int(np.argmax(logits))
                else:
                    z = logits / temperature
                    p = np.exp(z - z.max())
                    p /= p.sum()
                    tok = int(rng.choice(len(p), p=p))
                tokens.append(tok)
                new_ids.append(tok)
                if step < 8 or step % 32 == 0:
                    logger.info("step %d tok=%d %r", step, tok, tokenizer.decode(new_ids))
                if tokenizer.eos_token_id is not None and tok == tokenizer.eos_token_id:
                    logger.info("hit EOS at step %d", step)
                    break

        if jax.process_index() == 0:
            completion = tokenizer.decode(new_ids)
            logger.info("=== COMPLETION ===\n%s%s", prompt, completion)
            payload = {
                "prompt": prompt,
                "completion": completion,
                "full_text": prompt + completion,
                "prompt_ids": list(map(int, prompt_ids)),
                "new_ids": list(map(int, new_ids)),
                "temperature": temperature,
                "seed": seed,
                "checkpoint": CHECKPOINT_BASE,
            }
            out_json = f"{OUT_DIR}/sample_gen_{STEP.replace('-', '')}_t{temperature:g}_s{seed}.json"
            with fsspec.open(out_json, "w") as handle:
                handle.write(json.dumps(payload, indent=2))
            logger.info("wrote %s", out_json)

    return _sample_local


def _make_run(local_entrypoint):
    def run_sample(config: GrugRunConfig) -> None:
        trainer = config.trainer.trainer
        if trainer.id is None:
            raise ValueError("trainer.id must be set before dispatching.")
        _apply_hero_ep_runtime_defaults(inline_watch_enabled=False, processes_per_task=config.processes_per_task)
        dispatch_grug_training_run(
            run_id=trainer.id,
            config=config,
            local_entrypoint=local_entrypoint,
            resources=config.resources,
            processes_per_task=config.processes_per_task,
            max_retries_failure=0,
            max_task_failures=1,
        )

    return run_sample


def _to_one_rack(config: GrugRunConfig) -> GrugRunConfig:
    return dataclasses.replace(
        config,
        resources=_ONE_RACK_RESOURCES,
        processes_per_task=HERO_GPUS_PER_NODE,
        trainer=dataclasses.replace(config.trainer, replica_axis_size=1),
    )


@click.command()
@click.option("--prompt", required=True, help="prompt text to complete")
@click.option("--max-new-tokens", default=300, type=int)
@click.option("--temperature", default=1.0, type=float)
@click.option("--seed", default=0, type=int)
@build_options
def main(prompt: str, max_new_tokens: int, temperature: float, seed: int) -> ArtifactStep:
    run_id = f"{RUN_ID_BASE}-t{temperature:g}-s{seed}"  # distinct fingerprint per temperature/seed
    step = build_ladder_run(run_id=run_id, size="d6144", version=HERO_VERSION)
    original_build_config = step.build_config
    return dataclasses.replace(
        step,
        run=_make_run(_make_local(prompt, max_new_tokens, temperature, seed)),
        build_config=lambda ctx: _to_one_rack(original_build_config(ctx)),
        runtime_args={"train_resources": _ONE_RACK_RESOURCES},
    )


if __name__ == "__main__":
    main()

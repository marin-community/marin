# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Restore a hero checkpoint on one rack and autoregressively sample completions for
several prompts at once -- each prompt occupies its own batch row and is sampled
independently at its own current position, so all prompts advance in a single
[batch, SEQ_LEN] forward per step (no KV cache). Per-row positions are traced, so
the forward compiles once. Rows that emit EOS freeze. Writes one JSON with every
(prompt, completion).

    uv run iris --config lib/iris/config/marin.yaml job run --no-wait --enable-extra-resources \\
        --target-cluster cw-us-east-08a --priority production \\
        -- python -m experiments.grug.moe_hero_ep.sample_gen_multi --version 2026.08.19.2 \\
           --prompts-json '["def add_two_numbers(x, y):", ...]' --temperature 0.2 --run
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
RUN_ID_BASE = "hero-12d8b6f0-samplemulti"  # step + temperature/seed suffix added in main()
CKPT_DIR = f"s3://marin-us-east-02a/marin/grug/{HERO_RUN_ID}/{HERO_VERSION}/checkpoints"
OUT_DIR = f"s3://marin-us-east-02a/marin/grug/{HERO_RUN_ID}/analysis"

ONE_RACK_NODES = 16
_ONE_RACK_RESOURCES = ResourceConfig.with_gpu(
    "GB200", count=HERO_GPUS_PER_NODE, cpu=120, ram="850g", disk="1t", replicas=ONE_RACK_NODES
)
_EVAL_BATCH = HERO_EP_EXPERT_AXIS_SIZE  # 64 rows; the first len(prompts) hold real prompts, rest are filler


def _make_local(
    prompts: list[str], step: str, max_new_tokens: int, temperature: float, seed: int, ckpt_dir: str = CKPT_DIR
):
    checkpoint_base = f"{ckpt_dir}/{step}"

    def _sample_local(config: GrugRunConfig) -> None:
        trainer = config.trainer.trainer
        trainer.initialize()
        optimizer = config.optimizer.build(trainer.num_train_steps)
        tokenizer = config.data.the_tokenizer
        eos = tokenizer.eos_token_id
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
                checkpoint_search_paths=[checkpoint_base],
                load_checkpoint_setting=True,
                mesh=mesh,
                allow_partial=True,
            )
            model = state.params
            rep = NamedSharding(mesh, PartitionSpec())

            @jax.jit
            def _logits_at(m, ids, pos):  # pos: [B] per-row position
                hidden, _ = m(ids)  # [B, S, D]; mask=None -> causal
                b = jnp.arange(ids.shape[0])
                htgt = hidden.at[b, pos].get(out_sharding=PartitionSpec())  # [B, D]
                logits = jnp.einsum("bh,hv->bv", htgt, m.output_proj)  # [B, V]; predicts token at pos+1
                return reshard(logits, PartitionSpec())

            n = len(prompts)
            tokens = [tokenizer.encode(p, add_special_tokens=True) for p in prompts]  # BOS-prefixed
            for i, t in enumerate(tokens):
                logger.info("prompt %d (%d tok): %r", i, len(t), prompts[i])
            done = [False] * n
            new_ids: list[list[int]] = [[] for _ in range(n)]

            rng = np.random.default_rng(seed)
            batch = np.zeros((_EVAL_BATCH, SEQ_LEN), dtype=np.int32)
            for gen_step in range(max_new_tokens):
                for b in range(_EVAL_BATCH):
                    src = tokens[b] if b < n else tokens[0]
                    batch[b, : len(src)] = src
                pos = np.array(
                    [len(tokens[b]) - 1 if b < n else len(tokens[0]) - 1 for b in range(_EVAL_BATCH)], dtype=np.int32
                )
                token_ids = jax.device_put(jnp.asarray(batch), rep)
                logits = np.asarray(
                    jax.device_get(_logits_at(model, token_ids, jax.device_put(jnp.asarray(pos), rep)))
                ).astype(
                    np.float64
                )  # [B, V]
                for b in range(n):
                    if done[b]:
                        continue
                    lg = logits[b]
                    if temperature <= 0.0:
                        tok = int(np.argmax(lg))
                    else:
                        z = lg / temperature
                        p = np.exp(z - z.max())
                        p /= p.sum()
                        tok = int(rng.choice(len(p), p=p))
                    tokens[b].append(tok)
                    new_ids[b].append(tok)
                    if (eos is not None and tok == eos) or len(tokens[b]) >= SEQ_LEN:
                        done[b] = True
                if all(done):
                    logger.info("all prompts hit EOS at gen step %d", gen_step)
                    break

        if jax.process_index() == 0:
            results = []
            for i, p in enumerate(prompts):
                completion = tokenizer.decode(new_ids[i])
                results.append(
                    {
                        "prompt": p,
                        "completion": completion,
                        "full_text": p + completion,
                        "new_ids": list(map(int, new_ids[i])),
                        "hit_eos": done[i],
                    }
                )
                logger.info("=== [%d] %r\n%s%s\n", i, p, p, completion)
            payload = {
                "results": results,
                "temperature": temperature,
                "seed": seed,
                "max_new_tokens": max_new_tokens,
                "checkpoint": checkpoint_base,
            }
            out_json = f"{OUT_DIR}/sample_multi_{step.replace('-', '')}_t{temperature:g}_s{seed}.json"
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
@click.option("--prompts-json", default=None, help="JSON list of prompt strings")
@click.option("--prompts-file", default=None, help="local path to a JSON file with a list of prompt strings")
@click.option("--step", default="step-42000", help="checkpoint step to restore, e.g. step-6000")
@click.option("--max-new-tokens", default=200, type=int)
@click.option("--temperature", default=0.2, type=float)
@click.option("--seed", default=0, type=int)
@click.option("--ckpt-dir", default=CKPT_DIR, help="checkpoint tree to restore from (default: original hero)")
@build_options
def main(
    prompts_json: str, prompts_file: str, step: str, max_new_tokens: int, temperature: float, seed: int, ckpt_dir: str
) -> ArtifactStep:
    if prompts_file:
        with open(prompts_file) as fh:
            prompts = json.load(fh)
    elif prompts_json:
        prompts = json.loads(prompts_json)
    else:
        raise ValueError("provide --prompts-json or --prompts-file")
    if not (0 < len(prompts) <= _EVAL_BATCH):
        raise ValueError(f"need 1..{_EVAL_BATCH} prompts, got {len(prompts)}")
    run_id = f"{RUN_ID_BASE}-{step.replace('-', '')}-t{temperature:g}-s{seed}"
    run_step = build_ladder_run(run_id=run_id, size="d6144", version=HERO_VERSION)
    original_build_config = run_step.build_config
    return dataclasses.replace(
        run_step,
        run=_make_run(_make_local(prompts, step, max_new_tokens, temperature, seed, ckpt_dir)),
        build_config=lambda ctx: _to_one_rack(original_build_config(ctx)),
        runtime_args={"train_resources": _ONE_RACK_RESOURCES},
    )


if __name__ == "__main__":
    main()

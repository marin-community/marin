# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Long-context perplexity eval for the d=1280 EP=8 32k-ctx checkpoint.

Path-2 standalone runner for PR #5923's long-context perplexity slices:
import the dataset definitions, load our grug MoE checkpoint at the same
architecture used at training (sliding_window=2048, long-only YaRN with
mscale=0.1, long_qk_mult=1.5703, seq=32768), iterate up to N documents per
HF source, and compute next-token loss directly via
``Transformer.next_token_loss``. Bypasses the perplexity_gap framework
runner because that one expects a Levanter ``LmConfig`` model.

Only the **raw** long-doc slices are evaluated here (PG19, GovReport,
QuALITY contexts) -- they stream directly from HF without an executor
stage. The supervised QASPER / NarrativeQA / BookSum slices need pre-stage
steps to materialize the target-only renders; can be added later.

Submit (us-central2, v4-32, production priority)::

    .venv/bin/iris --cluster=marin job run --no-wait --region us-central2 \\
        --priority production \\
        -e WANDB_API_KEY "$WANDB_API_KEY" \\
        -- python -m experiments.grug.moe.eval_long_context_d1280_32kctx
"""

import dataclasses
import logging
import math
import os
from dataclasses import dataclass

from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep, this_output_path, versioned

from experiments.grug.dispatch import dispatch_grug_training_run
from experiments.grug.moe.heuristic_muonh import MoeMuonHHeuristic
from experiments.grug.moe.model import GrugModelConfig

logger = logging.getLogger(__name__)

_CKPT_PATH: str = (
    "gs://marin-us-central2/grug/moe_may_compute_opt_d1280_ep8_32kctx_long_yarn_mscale01_halfmix_from13k-"
    "e55766/checkpoints/step-14325"
)
_DIM: int = 1280
_SEQ: int = 32_768
_BS: int = 16  # mesh-required minimum at EP=8 on v4-32 (data*expert=16)
_TOKENIZER_ID: str = "marin-community/marin-tokenizer"
_MAX_DOCS_PER_DATASET: int = 32
_EXPERT_PARALLEL: int = 8

_YARN_OLD_SEQ_LEN: int = 4096
_YARN_ALPHA: int = 1
_YARN_BETA: int = 32
_YARN_MSCALE_COEF: float = 0.1


@dataclass(frozen=True)
class LongContextEvalConfig:
    checkpoint_path: str
    run_id: str
    output_path: str
    resources: ResourceConfig
    seq_len: int
    batch_size: int
    expert_parallel: int
    tokenizer_id: str
    max_docs_per_dataset: int
    mp: str
    seed: int = 0


def _build_model_config() -> GrugModelConfig:
    from levanter.grug.attention import RotaryConfig

    h = MoeMuonHHeuristic()
    base = dataclasses.replace(
        h.build_model_config(_DIM, seq_len=_SEQ),
        sliding_window=2048,
    )
    long_mscale = _YARN_MSCALE_COEF * math.log(_SEQ / _YARN_OLD_SEQ_LEN) + 1.0
    return dataclasses.replace(
        base,
        long_yarn_old_seq_len=_YARN_OLD_SEQ_LEN,
        long_qk_mult=base.qk_mult * long_mscale,
        yarn_alpha=_YARN_ALPHA,
        yarn_beta=_YARN_BETA,
        rope=RotaryConfig(theta=10000.0),
    )


def _run_eval_local(config: LongContextEvalConfig) -> None:
    """Run inside the Fray TPU job: build model, load checkpoint, score datasets."""
    import time
    from datetime import timedelta

    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import jmp
    import wandb
    from datasets import load_dataset
    from jax.sharding import NamedSharding, reshard
    from jax.sharding import PartitionSpec as P
    from levanter.callbacks.profiler import ProfilerConfig
    from levanter.checkpoint import CheckpointerConfig
    from levanter.grug.attention import AttentionMask
    from levanter.tracker.wandb import WandbConfig
    from levanter.trainer import TrainerConfig
    from levanter.utils.mesh import MeshConfig
    from marin.training.training import temporary_checkpoint_base_path
    from transformers import AutoTokenizer

    from experiments.evals.long_context_ppl import long_context_raw_validation_sets
    from experiments.grug.checkpointing import restore_grug_state_from_checkpoint
    from experiments.grug.moe.model import Transformer
    from experiments.grug.moe.train import GrugTrainState, initial_state

    def _load_doc_tokens(ds_cfg, tokenizer, max_docs: int, seq_len: int) -> list[list[int]]:
        ds = load_dataset(
            ds_cfg.hf_dataset_id,
            ds_cfg.hf_dataset_name,
            revision=ds_cfg.hf_dataset_revision,
            split=ds_cfg.split,
            streaming=True,
        )
        out: list[list[int]] = []
        for row in ds:
            text = row.get(ds_cfg.text_key)
            if not isinstance(text, str) or not text.strip():
                continue
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) < 2:
                continue
            out.append(ids[:seq_len])
            if len(out) >= max_docs:
                break
        return out

    def _pad_batch(tok_lists, seq_len):
        B = len(tok_lists)
        tokens = jnp.zeros((B, seq_len), dtype=jnp.int32)
        weight = jnp.zeros((B, seq_len), dtype=jnp.float32)
        for i, ids in enumerate(tok_lists):
            L = min(len(ids), seq_len)
            tokens = tokens.at[i, :L].set(jnp.asarray(ids[:L], dtype=jnp.int32))
            weight = weight.at[i, :L].set(1.0)
        return tokens, weight

    trainer = TrainerConfig(
        id=config.run_id,
        seed=config.seed,
        train_batch_size=config.batch_size,
        num_train_steps=1,
        profiler=ProfilerConfig(),
        mp=jmp.get_policy(config.mp),
        tracker=WandbConfig(
            project="marin_moe",
            name=config.run_id,
            tags=["moe", "long_context_eval", "d1280", "32kctx"],
            group="moe-may-compute-opt",
        ),
        use_explicit_mesh_axes=True,
        mesh=MeshConfig(axes={"expert": config.expert_parallel}),
        require_accelerator=True,
        allow_nondivisible_batch_size=False,
        checkpointer=CheckpointerConfig(
            base_path=os.path.join(config.output_path, "checkpoints"),
            temporary_base_path=temporary_checkpoint_base_path(config.output_path),
            append_run_id_to_base_path=False,
            save_interval=timedelta(minutes=10),
            keep=None,
        ),
        load_checkpoint=True,
        load_checkpoint_path=config.checkpoint_path,
    )
    trainer.initialize()

    model_cfg = _build_model_config()
    # Reuse the train-side initial_state so we get a fully-built GrugTrainState
    # whose ``params`` slot matches the trained checkpoint layout.
    h = MoeMuonHHeuristic()
    # Build a no-op optimizer (we don't take any steps; only the params load matters).
    opt = h.build_muonh_config(config.batch_size, 1.0, _DIM, seq_len=config.seq_len).build(1)
    with trainer.use_device_mesh():

        @jax.jit
        def _init(key):
            return initial_state(model_cfg, optimizer=opt, mp=trainer.mp, key=key, ema_beta=None)

        state: GrugTrainState = _init(jax.random.PRNGKey(config.seed))
        state = restore_grug_state_from_checkpoint(
            state,
            checkpoint_search_paths=[config.checkpoint_path],
            load_checkpoint_setting=True,
            mesh=trainer.device_mesh,
            allow_partial=False,
        )
        model: Transformer = state.params  # type: ignore[assignment]

        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_id)

        batch_pspec = P(("data", "expert"), None)
        batch_sharding = NamedSharding(trainer.device_mesh, batch_pspec)

        @eqx.filter_jit
        def _score_batch(model: Transformer, tokens, weight):
            # Inputs are constructed eager in Python and arrive unsharded; reshard them to
            # the batch pspec the model expects. AttentionMask.segment_ids is a tuple of
            # (q_seg, kv_seg) per the splash-attention kernel; single-document-per-row
            # matches our truncated-doc tokenization.
            tokens = reshard(tokens, batch_sharding)
            weight = reshard(weight, batch_sharding)
            seg = reshard(jnp.zeros(tokens.shape, dtype=jnp.int32), batch_sharding)
            mask = AttentionMask(is_causal=True, sliding_window=None, segment_ids=(seg, seg))
            per_pos = model.next_token_loss(tokens, weight, mask=mask, reduction="none")
            return jnp.sum(per_pos * weight), jnp.sum(weight)

        ds_cfgs = long_context_raw_validation_sets()
        for ds_name, ds_cfg in ds_cfgs.items():
            t_start = time.perf_counter()
            tok_lists = _load_doc_tokens(ds_cfg, tokenizer, config.max_docs_per_dataset, config.seq_len)
            total_loss, total_tokens = 0.0, 0
            for i in range(0, len(tok_lists), config.batch_size):
                batch = tok_lists[i : i + config.batch_size]
                if len(batch) < config.batch_size:
                    batch = batch + [[tokenizer.pad_token_id or 0]] * (config.batch_size - len(batch))
                tokens, weight = _pad_batch(batch, config.seq_len)
                loss_sum, tok_count = _score_batch(model, tokens, weight)
                total_loss += float(loss_sum)
                total_tokens += int(tok_count)
            avg_loss = total_loss / max(total_tokens, 1)
            ppl = math.exp(min(avg_loss, 50.0))
            elapsed = time.perf_counter() - t_start
            logger.info(
                "[%s] docs=%d tokens=%d avg_loss=%.5f ppl=%.3f elapsed=%.1fs",
                ds_name,
                len(tok_lists),
                total_tokens,
                avg_loss,
                ppl,
                elapsed,
            )
            wandb.log(
                {
                    f"long_context/{ds_name}/loss": avg_loss,
                    f"long_context/{ds_name}/ppl": ppl,
                    f"long_context/{ds_name}/tokens": total_tokens,
                    f"long_context/{ds_name}/docs": len(tok_lists),
                }
            )

    wandb.finish()


def run_long_context_eval(config: LongContextEvalConfig) -> None:
    dispatch_grug_training_run(
        run_id=config.run_id,
        config=config,
        local_entrypoint=_run_eval_local,
        resources=config.resources,
    )


_run_id = "moe_may_eval_long_context_d1280_ep8_32kctx_long_yarn_halfmix_from13k"
eval_step = ExecutorStep(
    name=f"grug/{_run_id}",
    fn=run_long_context_eval,
    config=LongContextEvalConfig(
        checkpoint_path=versioned(_CKPT_PATH),
        run_id=_run_id,
        output_path=this_output_path(),
        resources=versioned(ResourceConfig.with_tpu("v4-32")),
        seq_len=versioned(_SEQ),
        batch_size=versioned(_BS),
        expert_parallel=versioned(_EXPERT_PARALLEL),
        tokenizer_id=versioned(_TOKENIZER_ID),
        max_docs_per_dataset=versioned(_MAX_DOCS_PER_DATASET),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
    ),
)


if __name__ == "__main__":
    executor_main(
        steps=[eval_step],
        description=(
            f"Long-context PPL eval of the d=1280 EP=8 32kctx checkpoint over "
            f"3 raw slices (PG19 / GovReport / QuALITY) at "
            f"seq={_SEQ}, bs={_BS}, EP={_EXPERT_PARALLEL}, max_docs/dataset={_MAX_DOCS_PER_DATASET}. "
            f"v4-32 us-central2."
        ),
    )

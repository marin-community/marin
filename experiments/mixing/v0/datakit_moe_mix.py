# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grug-MoE on the datakit us-central2 store mixture (store_8ac06c74).

The store at ``gs://marin-us-central2/datakit/store_8ac06c74`` is partitioned
into 40 lexical clusters x 5 quality buckets = 200 tokenized caches. Each
``cluster=C/quality=Q`` directory is a single Levanter ``TreeCache`` (one
consolidated ``shard_ledger.json`` at the root, data left sharded across
``part-*``), so each bucket loads as a plain ``DatasetComponent``.

With token-proportional weights, a bucket is only sampled if
``weight * mixture_block_size >= 1``. At the maximum block size (65535), 167
buckets clear that floor and mix on their own; the remaining 33 do not and are
concatenated into one ``tail`` component via ``ConcatDatasetComponent``. The
combined tail clears the floor, and ``mixture_block_size`` is pinned to 65535.

Paths are region-relative via ``InputName.hardcoded`` and resolve to
``gs://marin-us-central2/...``; the run must launch in us-central2-b (where we
have v4 quota) or it hard-fails on missing-cache rather than triggering a
cross-region copy.
"""

from fray.cluster import ResourceConfig
from levanter.data.text import (
    ConcatDatasetComponent,
    DatasetComponent,
    LmDataConfig,
    TextLmDatasetFormat,
)
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, InputName, executor_main
from marin.execution.types import this_output_path, versioned
from marin.processing.tokenize import add_validation_sets_to_mixture

from experiments.defaults import default_validation_sets
from experiments.marin_models import marin_tokenizer
from experiments.mixing.v0.datakit_buckets import (
    _MIXABLE_BUCKETS,
    _TAIL_BUCKETS,
    PROPORTIONAL_WEIGHTS,
    TARGET_BUDGET,
    _bucket_name,
)
from experiments.mixing.v0.grug_moe_mix import run_grug_moe_mix
from experiments.mixing.v0.heuristic import build_from_heuristic
from experiments.mixing.v0.launch import GrugMoeLaunchConfig
from experiments.mixing.v0.train import GrugEvalConfig, GrugTrainerConfig

_STORE_PREFIX = "datakit/store_8ac06c74"

# Pinned to the maximum legal block size (MixtureDataset caps at 2**16). The
# 167/33 mixable/tail split in datakit_buckets is computed at this value;
# lowering it would push more buckets below the per-block sampling floor.
_MIXTURE_BLOCK_SIZE = 65535


def _bucket_path(cluster: int, quality: int) -> str:
    return f"{_STORE_PREFIX}/cluster={cluster}/quality={quality}"


def _build_component(name: str, cache_path: str) -> DatasetComponent:
    # flat_cache=True: store's TreeCache lives directly at cache_dir (consolidated
    # shard_ledger.json at the root, no /train subdir).
    cache_dir = InputName.hardcoded(cache_path)
    return DatasetComponent(source=None, cache_dir=cache_dir, format=TextLmDatasetFormat(), tags=[name], flat_cache=True)


_CACHE_BACKED_COMPONENTS: dict[str, DatasetComponent] = {
    _bucket_name(c, q): _build_component(_bucket_name(c, q), _bucket_path(c, q)) for c, q, _ in _MIXABLE_BUCKETS
}

_TAIL_COMPONENT = ConcatDatasetComponent(
    children={
        _bucket_name(c, q): _build_component(f"tail/{_bucket_name(c, q)}", _bucket_path(c, q))
        for c, q, _ in _TAIL_BUCKETS
    },
    tags=["tail"],
)

COMPONENTS: dict[str, DatasetComponent | ConcatDatasetComponent] = {
    **_CACHE_BACKED_COMPONENTS,
    "tail": _TAIL_COMPONENT,
}
assert len(COMPONENTS) == 168

# Smallest grug compute-optimal point (budget, hidden_dim) -- one quick v5p-8 run.
_TARGET_STEPS: int = 2**14
_BUDGET, _HIDDEN_DIM = 2.19e17, 512


def _build_step() -> ExecutorStep:
    model, optimizer, batch, steps = build_from_heuristic(
        budget=_BUDGET,
        hidden_dim=_HIDDEN_DIM,
        target_steps=_TARGET_STEPS,
    )
    experiment_budget = batch * steps * model.max_seq_len
    assert experiment_budget <= TARGET_BUDGET, f"experiment_budget {experiment_budget} exceeds {TARGET_BUDGET}"

    base_mixture = LmDataConfig(
        tokenizer=marin_tokenizer,
        cache_dir=None,
        components=COMPONENTS,
        train_weights=PROPORTIONAL_WEIGHTS,
        auto_build_caches=False,
        mixture_block_size=_MIXTURE_BLOCK_SIZE,
        target_budget=TARGET_BUDGET,
        experiment_budget=experiment_budget,
    )
    data = add_validation_sets_to_mixture(base_mixture, default_validation_sets(tokenizer=marin_tokenizer))

    slug = f"d{_HIDDEN_DIM}-{_BUDGET:.2e}"
    return ExecutorStep(
        name=f"grug/datakit_moe_mix_{slug}",
        fn=run_grug_moe_mix,
        config=GrugMoeLaunchConfig(
            model=versioned(model),
            data=data,
            output_path=this_output_path(),
            run_id=f"datakit_moe_mix_{slug}",
            resources=versioned(ResourceConfig.with_tpu("v4-8", zone="us-central2-b", preemptible=False)),
            steps=versioned(steps),
            batch_size=versioned(batch),
            seed=versioned(0),
            mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
            tracker=WandbConfig(
                project="marin_moe",
                tags=["moe", "datakit_store_mix", slug],
                group="datakit-moe-mix",
                name=None,
            ),
            optimizer=versioned(optimizer),
            grug_trainer=versioned(GrugTrainerConfig(z_loss_weight=1e-4, ema_beta=None, log_every=1)),
            eval=versioned(
                GrugEvalConfig(
                    eval_batch_size=512,
                    steps_per_eval=1000,
                    max_eval_batches=8,
                    eval_current=True,
                    eval_ema=False,
                )
            ),
        ),
    )


datakit_moe_mix_steps: list[ExecutorStep] = [_build_step()]


if __name__ == "__main__":
    executor_main(
        steps=datakit_moe_mix_steps,
        description="Grug MoE on the datakit us-central2 store mixture (167 proportional buckets + tail).",
    )

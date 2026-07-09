# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grug-MoE run on the datakit store with the mixture from ``mixture-3.csv``.

The source CSV was provided at ``/Users/held/Downloads/mixture-3.csv`` with
columns ``bucket,phase_0,phase_1,sum``. The two phase columns are independently
normalized below and used as a two-stage schedule.
"""

import math

from fray.cluster import ResourceConfig
from levanter.data.text import (
    ConcatDatasetComponent,
    DatasetComponent,
    LmDataConfig,
    TextLmDatasetFormat,
    UrlDatasetSourceConfig,
)
from levanter.tracker.wandb import WandbConfig
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.step_runner import StepRunner
from marin.experiment.namespacing import user_namespaced_name
from marin.training.training import LevanterCheckpoint

from experiments.datasets.paloma import paloma_datasets
from experiments.datasets.uncheatable import uncheatable_datasets
from experiments.grug.moe.heuristic import build_from_heuristic
from experiments.grug.moe.launch import GrugMoeLaunchConfig, run_grug_moe_trial
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig
from experiments.marin_tokenizer import marin_tokenizer

_STORE_PREFIX = "datakit/store_8ac06c74"
# Large enough that the smallest nonzero CSV weights still receive at least
# one sequence per block, while aligning cleanly with the heuristic batch size.
_MIXTURE_BLOCK_SIZE = 49_152
_PHASE_1_START_FRACTION = 0.8
ENABLE_SIMULATED_EPOCHING = True

# Natural size of ``datakit/store_8ac06c74`` from Will's datakit-moe-mix branch:
# 167 mixable bucket caches plus the 33-cache tail component.
_TARGET_BUDGET_TOKENS = 10_372_343_704_053

# The datakit store lives in us-central2; pin training there to avoid cross-region I/O.
_TRAIN_RESOURCES = ResourceConfig.with_tpu("v4-8", zone="us-central2-b")

_BUCKET_PHASE_WEIGHTS: tuple[tuple[str, float, float], ...] = (
    ("c01q0", 0.022957, 0.031937),
    ("c01q1", 0.018953, 0.041846),
    ("c01q2", 0.069309, 0.053564),
    ("c01q3", 0.106783, 0.093756),
    ("c01q4", 0.124669, 0.103804),
    ("c02q0", 0.005079, 0.013405),
    ("c02q1", 0.003787, 0.00623),
    ("c02q2", 0.005353, 0.003842),
    ("c02q3", 0.015501, 0.013842),
    ("c02q4", 0.021761, 0.008706),
    ("c03q0", 0.001046, 0.008815),
    ("c03q1", 0.003039, 0.001349),
    ("c03q2", 0.003699, 0.003679),
    ("c03q3", 0.001283, 0.006298),
    ("c03q4", 0.001561, 0.002798),
    ("c05q0", 0.011293, 0.012139),
    ("c05q1", 0.021915, 0.023056),
    ("c05q2", 0.035838, 0.017236),
    ("c05q3", 0.009497, 0.010704),
    ("c05q4", 0.004878, 0.009904),
    ("c06q0", 0.008112, 0.010894),
    ("c06q1", 0.009701, 0.017409),
    ("c06q2", 0.017897, 0.005937),
    ("c06q3", 0.001296, 0.001905),
    ("c06q4", 0.001886, 0.000709),
    ("c08q0", 0.004234, 0.003704),
    ("c08q1", 0.000809, 0.000849),
    ("c08q2", 0.001256, 0.00157),
    ("c08q3", 0.000469, 0.000629),
    ("c08q4", 0.000374, 0.000279),
    ("c10q0", 0.000086, 0.0001),
    ("c11q0", 0.005798, 0.002363),
    ("c11q1", 0.003333, 0.0018),
    ("c11q2", 0.002542, 0.002297),
    ("c11q3", 0.000779, 0.001259),
    ("c11q4", 0.000199, 0.000672),
    ("c12q0", 0.000311, 0.000545),
    ("c12q1", 0.000259, 0.000419),
    ("c12q2", 0.000151, 0.000791),
    ("c12q3", 0.000296, 0.000881),
    ("c12q4", 0.000247, 0.000687),
    ("c13q0", 0.000227, 0.000526),
    ("c13q1", 0.000364, 0.000095),
    ("c13q2", 0.000116, 0.0005),
    ("c13q3", 0.000472, 0.00205),
    ("c13q4", 0.000063, 0.000157),
    ("c14q0", 0.000633, 0.000512),
    ("c14q1", 0.000439, 0.000516),
    ("c14q2", 0.000976, 0.000399),
    ("c14q3", 0.000585, 0.000462),
    ("c14q4", 0.000443, 0.000526),
    ("c15q0", 0.002384, 0.006218),
    ("c15q1", 0.003226, 0.003075),
    ("c15q2", 0.002358, 0.002513),
    ("c15q3", 0.002819, 0.003405),
    ("c15q4", 0.000486, 0.000773),
    ("c16q0", 0.004551, 0.00549),
    ("c16q1", 0.006622, 0.007018),
    ("c16q2", 0.004254, 0.004357),
    ("c16q3", 0.005395, 0.002788),
    ("c16q4", 0.001036, 0.001565),
    ("c17q0", 0.001453, 0.001401),
    ("c17q1", 0.003893, 0.001949),
    ("c17q2", 0.003539, 0.006945),
    ("c17q3", 0.002154, 0.002518),
    ("c17q4", 0.00181, 0.000973),
    ("c18q0", 0.002814, 0.003565),
    ("c18q1", 0.004576, 0.003527),
    ("c18q2", 0.003187, 0.004375),
    ("c18q3", 0.005101, 0.003618),
    ("c18q4", 0.001469, 0.002143),
    ("c19q0", 0.004086, 0.002186),
    ("c19q1", 0.003805, 0.00188),
    ("c19q2", 0.002327, 0.002511),
    ("c19q3", 0.002111, 0.001474),
    ("c19q4", 0.000521, 0.000995),
    ("c20q0", 0.000992, 0.002133),
    ("c20q1", 0.001298, 0.001715),
    ("c20q2", 0.001601, 0.001149),
    ("c20q3", 0.001476, 0.00099),
    ("c20q4", 0.000374, 0.00054),
    ("c21q0", 0.002196, 0.000878),
    ("c21q1", 0.002377, 0.0012),
    ("c21q2", 0.001096, 0.000948),
    ("c21q3", 0.000976, 0.000974),
    ("c21q4", 0.001686, 0.001902),
    ("c22q0", 0.002234, 0.003176),
    ("c22q1", 0.003447, 0.003771),
    ("c22q2", 0.003587, 0.002235),
    ("c22q3", 0.002543, 0.003216),
    ("c22q4", 0.001364, 0.00072),
    ("c23q0", 0.007766, 0.002833),
    ("c23q1", 0.003635, 0.002799),
    ("c23q2", 0.002582, 0.004731),
    ("c23q3", 0.004336, 0.002363),
    ("c23q4", 0.001879, 0.002431),
    ("c24q0", 0.004557, 0.004443),
    ("c24q1", 0.004783, 0.006302),
    ("c24q2", 0.001682, 0.002943),
    ("c24q3", 0.002718, 0.00169),
    ("c24q4", 0.000578, 0.000684),
    ("c25q0", 0.005998, 0.005436),
    ("c25q1", 0.005659, 0.003901),
    ("c25q2", 0.003611, 0.006288),
    ("c25q3", 0.003449, 0.002666),
    ("c25q4", 0.000614, 0.000481),
    ("c26q0", 0.004179, 0.00445),
    ("c26q1", 0.005244, 0.004102),
    ("c26q2", 0.004386, 0.004962),
    ("c26q3", 0.002379, 0.002656),
    ("c26q4", 0.00082, 0.001751),
    ("c27q0", 0.005148, 0.001263),
    ("c27q1", 0.001803, 0.001539),
    ("c27q2", 0.003706, 0.002862),
    ("c27q3", 0.000881, 0.001019),
    ("c27q4", 0.000684, 0.003425),
    ("c28q0", 0.006398, 0.015222),
    ("c28q1", 0.020521, 0.015838),
    ("c28q2", 0.005379, 0.010677),
    ("c28q3", 0.007225, 0.002797),
    ("c28q4", 0.001426, 0.001549),
    ("c29q0", 0.004778, 0.003881),
    ("c29q1", 0.003144, 0.003859),
    ("c29q2", 0.002375, 0.002314),
    ("c29q3", 0.001843, 0.001895),
    ("c29q4", 0.000971, 0.002561),
    ("c30q0", 0.007903, 0.035271),
    ("c30q1", 0.008205, 0.007997),
    ("c30q2", 0.013895, 0.009056),
    ("c30q3", 0.004825, 0.005009),
    ("c30q4", 0.001323, 0.001261),
    ("c31q0", 0.007819, 0.003531),
    ("c31q1", 0.002745, 0.00469),
    ("c31q2", 0.001324, 0.002866),
    ("c31q3", 0.002475, 0.001757),
    ("c31q4", 0.000456, 0.001024),
    ("c32q0", 0.00486, 0.002465),
    ("c32q1", 0.001979, 0.008457),
    ("c32q2", 0.001013, 0.00262),
    ("c32q3", 0.001109, 0.001447),
    ("c32q4", 0.000106, 0.000135),
    ("c33q0", 0.002961, 0.003771),
    ("c33q1", 0.004153, 0.005535),
    ("c33q2", 0.006076, 0.002388),
    ("c33q3", 0.004801, 0.001326),
    ("c33q4", 0.000316, 0.000683),
    ("c34q0", 0.000514, 0.000736),
    ("c34q1", 0.000941, 0.000924),
    ("c34q2", 0.00867, 0.007028),
    ("c34q3", 0.003685, 0.010664),
    ("c34q4", 0.000834, 0.00333),
    ("c35q0", 0.004318, 0.003901),
    ("c35q1", 0.007961, 0.002719),
    ("c35q2", 0.004451, 0.002239),
    ("c35q3", 0.002711, 0.001726),
    ("c35q4", 0.003109, 0.000526),
    ("c36q0", 0.002889, 0.002709),
    ("c36q1", 0.002673, 0.002149),
    ("c36q2", 0.002718, 0.002256),
    ("c36q3", 0.00471, 0.003361),
    ("c36q4", 0.009675, 0.002205),
    ("c37q0", 0.002776, 0.003737),
    ("c37q1", 0.003821, 0.004736),
    ("c37q2", 0.012379, 0.020839),
    ("c37q3", 0.036626, 0.031543),
    ("c37q4", 0.013194, 0.021405),
    ("c38q0", 0.000022, 0.000179),
    ("tail", 0.00007, 0.00003),
)

_TAIL_BUCKETS: tuple[str, ...] = (
    "c00q0",
    "c00q1",
    "c00q2",
    "c00q3",
    "c00q4",
    "c04q0",
    "c04q1",
    "c04q2",
    "c04q3",
    "c04q4",
    "c07q0",
    "c07q1",
    "c07q2",
    "c07q3",
    "c07q4",
    "c09q0",
    "c09q1",
    "c09q2",
    "c09q3",
    "c09q4",
    "c10q1",
    "c10q2",
    "c10q3",
    "c10q4",
    "c38q1",
    "c38q2",
    "c38q3",
    "c38q4",
    "c39q0",
    "c39q1",
    "c39q2",
    "c39q3",
    "c39q4",
)


def _bucket_path(bucket: str) -> str:
    cluster = int(bucket[1:3])
    quality = int(bucket[-1])
    return f"{_STORE_PREFIX}/cluster={cluster}/quality={quality}"


def _bucket_component(bucket: str) -> DatasetComponent:
    return DatasetComponent(
        source=None,
        cache_dir=_bucket_path(bucket),
        format=TextLmDatasetFormat(),
        tags=[bucket],
        flat_cache=True,
    )


def _val_component(cache_dir: str) -> DatasetComponent:
    """Placeholder validation component from a cache dir path."""
    source = UrlDatasetSourceConfig(
        tags=[],
        train_urls=[],
        validation_urls=[],
        cache_dir=cache_dir,
        format=TextLmDatasetFormat(),
    )
    return DatasetComponent(source=source, cache_dir=source.cache_dir, format=source.format, tags=source.tags)


def _normalize(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("mixture weights must sum to a positive value")
    return {name: weight / total for name, weight in weights.items() if weight > 0}


def _phase_weights(phase_index: int) -> dict[str, float]:
    if phase_index == 0:
        return _normalize({bucket: phase_0 for bucket, phase_0, _ in _BUCKET_PHASE_WEIGHTS})
    if phase_index == 1:
        return _normalize({bucket: phase_1 for bucket, _, phase_1 in _BUCKET_PHASE_WEIGHTS})
    raise ValueError(f"unknown phase index {phase_index}")


def _phase_1_start_step(total_steps: int, batch_size: int) -> int:
    requested = max(1, int(total_steps * _PHASE_1_START_FRACTION))
    step_multiple = _MIXTURE_BLOCK_SIZE // math.gcd(_MIXTURE_BLOCK_SIZE, batch_size)
    return max(step_multiple, (requested // step_multiple) * step_multiple)


def _datakit_components() -> dict[str, DatasetComponent | ConcatDatasetComponent]:
    direct = {bucket: _bucket_component(bucket) for bucket, _, _ in _BUCKET_PHASE_WEIGHTS if bucket != "tail"}
    return {
        **direct,
        "tail": ConcatDatasetComponent(
            children={bucket: _bucket_component(bucket) for bucket in _TAIL_BUCKETS},
            tags=["tail"],
        ),
    }


def _simulated_experiment_budget(*, total_steps: int, batch_size: int, max_seq_len: int) -> int:
    return total_steps * batch_size * max_seq_len


def _datakit_data_config(
    *,
    total_steps: int,
    batch_size: int,
    max_seq_len: int,
    enable_simulated_epoching: bool,
    val_components: dict[str, DatasetComponent | ConcatDatasetComponent],
) -> LmDataConfig:
    phase_1_start = _phase_1_start_step(total_steps, batch_size)
    budget_kwargs: dict = {}
    if enable_simulated_epoching:
        experiment_budget = _simulated_experiment_budget(
            total_steps=total_steps,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
        )
        if experiment_budget > _TARGET_BUDGET_TOKENS:
            raise ValueError(f"experiment_budget {experiment_budget} exceeds target_budget {_TARGET_BUDGET_TOKENS}")
        budget_kwargs = {
            "target_budget": _TARGET_BUDGET_TOKENS,
            "experiment_budget": experiment_budget,
        }

    all_components = {**_datakit_components(), **val_components}
    val_zero_weights = {name: 0.0 for name in val_components}

    return LmDataConfig(
        tokenizer=marin_tokenizer,
        cache_dir=None,
        components=all_components,
        train_weights=[
            (0, {**_phase_weights(0), **val_zero_weights}),
            (phase_1_start, {**_phase_weights(1), **val_zero_weights}),
        ],
        auto_build_caches=False,
        mixture_block_size=_MIXTURE_BLOCK_SIZE,
        **budget_kwargs,
    )


_BUDGET: float = 2.19e17
_HIDDEN_DIM: int = 512
_TARGET_STEPS: int = 2**14
_model, _optimizer, _batch_size, _steps = build_from_heuristic(
    budget=_BUDGET,
    hidden_dim=_HIDDEN_DIM,
    target_steps=_TARGET_STEPS,
)

_SLUG = f"d{_HIDDEN_DIM}-{_BUDGET:.2e}"

_VALIDATION = [
    *paloma_datasets(tokenizer=marin_tokenizer).values(),
    *uncheatable_datasets(tokenizer=marin_tokenizer).values(),
]


def build(*, version: str = "dev") -> ArtifactStep[LevanterCheckpoint]:
    """Grug MoE on the us-central2 datakit store with the mixture-3 two-phase bucket schedule."""

    def build_config(ctx: StepContext) -> GrugMoeLaunchConfig:
        if ctx.is_fingerprint:
            val_components = {v.name: _val_component(ctx.artifact_path(v)) for v in _VALIDATION}
        else:
            val_components = {v.name: ctx.resolved(v).as_component() for v in _VALIDATION}
        data = _datakit_data_config(
            total_steps=_steps,
            batch_size=_batch_size,
            max_seq_len=_model.max_seq_len,
            enable_simulated_epoching=ENABLE_SIMULATED_EPOCHING,
            val_components=val_components,
        )
        return GrugMoeLaunchConfig(
            model=_model,
            data=data,
            output_path=ctx.output_path,
            run_id="datakit-moe-mix",
            resources=ctx.runtime_arg("train_resources"),
            steps=_steps,
            batch_size=_batch_size,
            seed=0,
            mp="params=float32,compute=bfloat16,output=bfloat16",
            tracker=WandbConfig(
                project="marin_moe",
                tags=["moe", "datakit_store_mix", _SLUG],
                group="datakit-moe-mix",
                name=None,
            ),
            optimizer=_optimizer,
            grug_trainer=GrugTrainerConfig(z_loss_weight=1e-4, ema_beta=None, log_every=1),
            eval=GrugEvalConfig(
                eval_batch_size=512,
                steps_per_eval=1000,
                max_eval_batches=8,
                eval_current=True,
                eval_ema=False,
            ),
        )

    return ArtifactStep(
        name=user_namespaced_name(f"grug/datakit_moe_mix_{_SLUG}", version),
        version=version,
        artifact_type=LevanterCheckpoint,
        run=run_grug_moe_trial,
        build_config=build_config,
        deps=tuple(_VALIDATION),
        runtime_args={"train_resources": _TRAIN_RESOURCES},
    )


if __name__ == "__main__":
    StepRunner().run([build().lower()])

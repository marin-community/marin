# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grug-MoE on the datakit us-central2 store mixture (store_8ac06c74).

The store at ``gs://marin-us-central2/datakit/store_8ac06c74`` is partitioned
into 40 lexical clusters x 5 quality buckets = 200 tokenized caches. Each
``cluster=C/quality=Q`` directory is a single Levanter ``TreeCache`` (one
consolidated ``shard_ledger.json`` at the root, data left sharded across
``part-*``; see marin PR #5430), so each bucket loads as a plain
``DatasetComponent``.

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
from marin.execution.executor import ExecutorStep, InputName, executor_main, this_output_path, versioned
from marin.processing.tokenize import add_validation_sets_to_mixture

from experiments.defaults import default_validation_sets
from experiments.grug.moe.grug_moe_mix import run_grug_moe_mix
from experiments.grug.moe.heuristic import build_from_heuristic
from experiments.grug.moe.launch import GrugMoeLaunchConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig
from experiments.marin_models import marin_tokenizer

_STORE_PREFIX = "datakit/store_8ac06c74"

# Pinned to the maximum legal block size (MixtureDataset caps at 2**16). The
# 167/33 split below is computed at this value; lowering it would push more
# buckets below the per-block sampling floor.
_MIXTURE_BLOCK_SIZE = 65535

# (cluster, quality, total_tokens) for the 167 buckets large enough to mix on
# their own at _MIXTURE_BLOCK_SIZE.
_MIXABLE_BUCKETS: list[tuple[int, int, int]] = [
    (1, 0, 152613773716),
    (1, 1, 137672798475),
    (1, 2, 158419310113),
    (1, 3, 190944457285),
    (1, 4, 516406017210),
    (2, 0, 17678211181),
    (2, 1, 22467740659),
    (2, 2, 54895261784),
    (2, 3, 149886381691),
    (2, 4, 313258182116),
    (3, 0, 14301444447),
    (3, 1, 12910282546),
    (3, 2, 13980049427),
    (3, 3, 9912603339),
    (3, 4, 3887445282),
    (5, 0, 876858503703),
    (5, 1, 760985570941),
    (5, 2, 460778882472),
    (5, 3, 75204494625),
    (5, 4, 19716864101),
    (6, 0, 415439566202),
    (6, 1, 193106474127),
    (6, 2, 177208361304),
    (6, 3, 21022007258),
    (6, 4, 11261629656),
    (8, 0, 107149207061),
    (8, 1, 10246970923),
    (8, 2, 10677695916),
    (8, 3, 5538127334),
    (8, 4, 2564439699),
    (10, 0, 1062803438),
    (11, 0, 27667903065),
    (11, 1, 34636839701),
    (11, 2, 27344419346),
    (11, 3, 17140262673),
    (11, 4, 3070709048),
    (12, 0, 2902297304),
    (12, 1, 3700425524),
    (12, 2, 3508112515),
    (12, 3, 2899106775),
    (12, 4, 1771372280),
    (13, 0, 1518113682),
    (13, 1, 1696797496),
    (13, 2, 1802188785),
    (13, 3, 1800629369),
    (13, 4, 891504858),
    (14, 0, 4996045284),
    (14, 1, 3317071107),
    (14, 2, 4373858943),
    (14, 3, 2399856768),
    (14, 4, 4649859748),
    (15, 0, 44295778783),
    (15, 1, 59225100684),
    (15, 2, 53911942185),
    (15, 3, 40613404839),
    (15, 4, 9041134954),
    (16, 0, 117973521195),
    (16, 1, 93470606214),
    (16, 2, 68812448880),
    (16, 3, 46704797706),
    (16, 4, 11269401673),
    (17, 0, 21466156258),
    (17, 1, 34940094485),
    (17, 2, 40922307117),
    (17, 3, 44085309158),
    (17, 4, 12067145244),
    (18, 0, 63408140031),
    (18, 1, 85431020441),
    (18, 2, 85631261804),
    (18, 3, 84816368502),
    (18, 4, 36647014639),
    (19, 0, 17675497877),
    (19, 1, 23769179741),
    (19, 2, 21295570916),
    (19, 3, 18684738495),
    (19, 4, 6815365363),
    (20, 0, 20942886326),
    (20, 1, 24370746026),
    (20, 2, 18775063604),
    (20, 3, 13101688889),
    (20, 4, 3417857803),
    (21, 0, 8434905648),
    (21, 1, 11500441287),
    (21, 2, 12408127274),
    (21, 3, 10502600563),
    (21, 4, 3601541920),
    (22, 0, 34302966680),
    (22, 1, 44195585914),
    (22, 2, 43366658092),
    (22, 3, 37705375606),
    (22, 4, 10248994519),
    (23, 0, 28457331904),
    (23, 1, 47123193918),
    (23, 2, 55694772845),
    (23, 3, 56572642039),
    (23, 4, 22175233358),
    (24, 0, 83933830660),
    (24, 1, 74319333037),
    (24, 2, 43669241496),
    (24, 3, 19506376440),
    (24, 4, 2543941398),
    (25, 0, 177819222529),
    (25, 1, 141176737619),
    (25, 2, 81093352129),
    (25, 3, 35256003803),
    (25, 4, 4920301858),
    (26, 0, 113098859487),
    (26, 1, 111826796516),
    (26, 2, 80110230016),
    (26, 3, 47015179764),
    (26, 4, 10562443051),
    (27, 0, 19957517942),
    (27, 1, 21036903099),
    (27, 2, 19084020363),
    (27, 3, 12076550117),
    (27, 4, 4438307683),
    (28, 0, 85490739799),
    (28, 1, 103408530783),
    (28, 2, 96672253452),
    (28, 3, 67571235638),
    (28, 4, 10760485407),
    (29, 0, 61982550303),
    (29, 1, 53134635038),
    (29, 2, 36284094763),
    (29, 3, 20856011632),
    (29, 4, 3379189806),
    (30, 0, 255440254710),
    (30, 1, 233501246542),
    (30, 2, 167040875843),
    (30, 3, 99955116312),
    (30, 4, 14011327379),
    (31, 0, 37864457718),
    (31, 1, 36670405950),
    (31, 2, 25275769660),
    (31, 3, 13289237623),
    (31, 4, 3237330949),
    (32, 0, 24854547047),
    (32, 1, 23211052222),
    (32, 2, 13503199844),
    (32, 3, 7095637963),
    (32, 4, 983150668),
    (33, 0, 34251344129),
    (33, 1, 48219482525),
    (33, 2, 33784852446),
    (33, 3, 15127268124),
    (33, 4, 3249082888),
    (34, 0, 7183726650),
    (34, 1, 10097113862),
    (34, 2, 12647753927),
    (34, 3, 11291078299),
    (34, 4, 2215112289),
    (35, 0, 54182856476),
    (35, 1, 47375127118),
    (35, 2, 35931789104),
    (35, 3, 30479210365),
    (35, 4, 7964903844),
    (36, 0, 33508605181),
    (36, 1, 48315446056),
    (36, 2, 52386416368),
    (36, 3, 59655803032),
    (36, 4, 41886322756),
    (37, 0, 26820167665),
    (37, 1, 41581664595),
    (37, 2, 69206324500),
    (37, 3, 141456572511),
    (37, 4, 183959838290),
    (38, 0, 212463266),
]

# (cluster, quality, total_tokens) for the 33 buckets below the sampling floor;
# concatenated into the single `tail` component.
_TAIL_BUCKETS: list[tuple[int, int, int]] = [
    (0, 0, 89156953),
    (0, 1, 21395593),
    (0, 2, 3985439),
    (0, 3, 5275385),
    (0, 4, 4163280),
    (4, 0, 12750601),
    (4, 1, 3187859),
    (4, 2, 21792058),
    (4, 3, 2804361),
    (4, 4, 4529424),
    (7, 0, 5115667),
    (7, 1, 6050635),
    (7, 2, 6534534),
    (7, 3, 421636),
    (7, 4, 549364),
    (9, 0, 30898239),
    (9, 1, 5584088),
    (9, 2, 36865296),
    (9, 3, 3124586),
    (9, 4, 2704571),
    (10, 1, 47752543),
    (10, 2, 55468536),
    (10, 3, 14295562),
    (10, 4, 11250400),
    (38, 1, 629056),
    (38, 2, 311167),
    (38, 3, 69261),
    (38, 4, 137536),
    (39, 0, 6144138),
    (39, 1, 6605454),
    (39, 2, 5469946),
    (39, 3, 4345123),
    (39, 4, 669737),
]

assert len(_MIXABLE_BUCKETS) == 167
assert len(_TAIL_BUCKETS) == 33


def _bucket_name(cluster: int, quality: int) -> str:
    return f"c{cluster:02d}q{quality}"


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

_TOKEN_COUNTS: dict[str, int] = {_bucket_name(c, q): t for c, q, t in _MIXABLE_BUCKETS}
_TOKEN_COUNTS["tail"] = sum(t for _, _, t in _TAIL_BUCKETS)
TOTAL_TOKENS: int = sum(_TOKEN_COUNTS.values())

# Simulated-epoching anchor: the natural size of the full store.
TARGET_BUDGET: int = TOTAL_TOKENS

PROPORTIONAL_WEIGHTS: dict[str, float] = {name: count / TOTAL_TOKENS for name, count in _TOKEN_COUNTS.items()}

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
        description="Grug MoE on the datakit eu-west4 store mixture (166 proportional buckets + tail).",
    )

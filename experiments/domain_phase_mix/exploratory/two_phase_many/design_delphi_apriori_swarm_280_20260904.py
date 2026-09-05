# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Materialize a prospectively frozen 280-row Delphi 3e18 swarm for the 39-bucket single-phase mixing problem.

Every row is fixed before any surrogate is fitted on the new swarm, and no row is a model-proposed or measured
optimum. The placement rules use bucket metadata (unique-token counts, the bucket type read from its name) and prior
knowledge about repetition; the lattice box and the anchors were chosen with this project's earlier results in view,
so the design is adaptive to that history and frozen from here on. Blocks:

- reserved rows: the proportional, UniMax and uniform baselines, three proportional repeats, and the 39
  leave-one-bucket-out deletions from the proportional control (reused from the current panel);
- type-level anchors B, C, D at full support (B repeated once, as the second seed block's full-support control);
- fixed-share subsampled-pool rows: an anchor mixture with one bucket (or the CC-high group) restricted to a half
  or a quarter of its unique tokens, so that bucket's epochs double or quadruple while every share stays fixed. The
  pilot part uses anchors A and B, four targets, and two paired seed blocks (data seed = subset seed within a block);
- a Latin hypercube over per-type relative epoch levels with per-bucket jitter (shares follow from inventories);
- single-bucket share ladders at anchors B and C;
- Common-Crawl-removal corners, distinct from every anchor after normalization;
- reused random panel rows chosen by farthest-point sampling and the highest-epoch conditional dose ladders that
  carry both targets.

Outputs the launcher-format table (``phase_0_<domain>``/``phase_1_<domain>`` columns, equal across phases;
``pool_fraction_<domain>`` and ``materialized_epochs_<domain>`` columns; provenance, seed block and wave columns),
a per-row summary, and a manifest with seeds, input hashes, a support-separation diagnostic on the surrogate's
share and exposure features, and a post-hoc coverage diagnostic. Nothing is launched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.dolma3_dolmino_top_level_domains import (  # noqa: E402
    TARGET_BUDGET_DOLMA3_COMMON_CRAWL,
    TOP_LEVEL_DOMAIN_TOKEN_COUNTS,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_single_phase_observatory_20260902 as harness,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_round3_heldout_selection_20260903 as selection,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_round4_cap_policies_20260903 as policies,
)
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import (  # noqa: E402
    DOMAIN_NAMES,
    PHASE_NAMES,
)

PANEL = "delphi_3e18_39bucket"
BUDGET_ROWS = 280
SEED = 20_260_904
BASE_DATA_SEED = 662_009  # the matched-seed comparison's data seed; seed block k uses BASE_DATA_SEED + k
TRAINER_SEED = 0
HISTORICAL_MANIFEST = (
    SCRIPT_DIR / "reference_outputs" / "delphi_one_phase_augmented_swarm_3e18_20260715" / "training_manifest.csv"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "delphi_apriori_swarm_280_20260904"
TYPES = ("cc_high", "cc_low", "code", "curated", "math", "synthetic")
LHS_ROWS = 76
LHS_LOG_BOX = {
    "cc_high": (np.log(0.25), np.log(2.0)),
    "cc_low": (np.log(0.25), np.log(2.0)),
    "code": (np.log(0.25), np.log(16.0)),
    "curated": (np.log(0.25), np.log(16.0)),
    "math": (np.log(0.25), np.log(16.0)),
    "synthetic": (np.log(0.25), np.log(16.0)),
}
LHS_JITTER = 0.35
LHS_MAX_EPOCHS = 128.0
ANCHOR_LEVELS = {
    "B_small_pools_forward": {"cc_high": 1.0, "cc_low": 0.5, "code": 2.0, "curated": 4.0, "math": 4.0, "synthetic": 4.0},
    "C_code_math_forward": {"cc_high": 0.5, "cc_low": 0.5, "code": 6.0, "curated": 2.0, "math": 6.0, "synthetic": 2.0},
    "D_cc_half_no_low": {"cc_high": 0.5, "cc_low": 0.0, "code": 4.0, "curated": 4.0, "math": 4.0, "synthetic": 4.0},
}
LADDER_BUCKETS = (
    "dolmino_synth_qa",
    "dolmino_olmocr_pdfs_hq",
    "dolma3_stack_edu",
    "dolmino_stack_edu_fim",
    "dolma3_wikipedia",
    "dolmino_stem_heavy_crawl",
    "dolmino_synth_math",
    "dolmino_synth_code",
)
LADDER_MULTIPLIERS = (0.25, 3.0, 8.0)
CC_HIGH_GROUP = "cc_high_group"
PILOT_TARGETS = ("dolmino_synth_qa", "dolmino_olmocr_pdfs_hq", "dolma3_stack_edu", CC_HIGH_GROUP)
PILOT_ANCHORS = ("A_proportional", "B_small_pools_forward")
PILOT_SEED_BLOCKS = (0, 1)
EXTRA_TARGETS = ("dolmino_stack_edu_fim", "dolma3_wikipedia", "dolmino_stem_heavy_crawl", "dolmino_synth_math")
EXTRA_ANCHOR = "D_cc_half_no_low"
POOL_FRACTIONS = (0.5, 0.25)
CORNER_LEVELS = (  # (cc_high, cc_low, other types); ratios distinct from each other and from anchor D (1:0:8)
    (2.0, 0.0, 4.0),
    (1.0, 0.0, 4.0),
    (0.25, 0.0, 4.0),
    (0.125, 0.0, 4.0),
    (1.0, 0.25, 4.0),
    (0.25, 0.25, 4.0),
    (1.0, 0.5, 4.0),
    (0.5, 0.25, 4.0),
    (0.1, 0.1, 4.0),
)
PROPORTIONAL_REPEATS = 3
REUSED_RANDOM_PANEL_ROWS = 18
REUSED_DOSE_ROWS = 40
REUSED_DOSE_COORDINATES = (  # frozen 2026-09-04: the 40 highest-epoch dose ladders that carried both targets
    "delphi_3e18_39bucket:4d2a9f8e18ebd65e0860eaa7996c71b4bc666ece79a59f99599bf658eaaf1ca6",
    "delphi_3e18_39bucket:39df5a66a6bcbdc5f52f9f37dfd15b1fdd08a415743db8fb17cc8b0512ac963f",
    "delphi_3e18_39bucket:b5ee1f40e4fe0358ac680a969cd93dfe1b00c483ee3b2d4ff4bc73bd6424437b",
    "delphi_3e18_39bucket:479b4c0d010ec8af266c712fad85c52a6b25371b206b9d3bea78ede611674b9e",
    "delphi_3e18_39bucket:9cb37ee04e914fba68e7aa6ea382c6a5dcc2b405ec78b251b537abc8f70b15f2",
    "delphi_3e18_39bucket:f9e5fad6cd843eef1ec963c775019a7ed2d37fb9fc6d6458f98c8bb15b000248",
    "delphi_3e18_39bucket:f387ec3241cb7be9b883e93ed53e1940f5fbf725f14ef13b1c76e1634c9e0704",
    "delphi_3e18_39bucket:e65243af3fc92306e4aa4ea80d2c3f7aa6058bba97560ecd10959066d8d64383",
    "delphi_3e18_39bucket:6a6a57cc90489b1821b2d65721b9c2e1d907405a1cb08fb12b0a637f75a245c7",
    "delphi_3e18_39bucket:71eb42d48f6b6a336390c690ce1e47364dfe485e2a4f06e168075c88b0def0a9",
    "delphi_3e18_39bucket:9a343b0b93b0bad4f87152437bb40a403f7be5b9f572cfae5e7703e50bc7093a",
    "delphi_3e18_39bucket:c721e392dcf2947fea2da8fcf1ef91d058c717ffb8e74e8402734e31f22578fc",
    "delphi_3e18_39bucket:46346afc1e8f516e3cf8db030395bc5bdba3094e0d1d6ce3d9212fbb3736b08f",
    "delphi_3e18_39bucket:d6f6db3e0cc1a598f7eb00a7cdd40d78f821a9336eb0eaafc14da93a04575fc5",
    "delphi_3e18_39bucket:e94c9cbd4ee682a9780bbcea3bf6df3497572253ed953eef164fa75ea4f66a2c",
    "delphi_3e18_39bucket:e283e505de0059b81a2b58a66a9af312bc1f3c7a7440e50c3fdcb3ff8213a34f",
    "delphi_3e18_39bucket:4dcd25089f067b1c4aa3d94b5a6046e58c872c7192b21d095899daf4324f9073",
    "delphi_3e18_39bucket:e0a4d646094bff9aad92ca04cd9c5ca30c296001c0b7d65c3c20a1df0bd8ef08",
    "delphi_3e18_39bucket:d160bb3271ba96668590d3b3c7781ad51718216823831c921585a8f73ffc9cf7",
    "delphi_3e18_39bucket:d82c377f5bda11f52b93db01e55fa25c844255cf016ad96b80cba7cc1e618d90",
    "delphi_3e18_39bucket:d941fe99995a0f675ffc09f1d7883ac2d032a87a00fdfc22e934c5e5ca0b3452",
    "delphi_3e18_39bucket:ec06eca900d6ea9ba81679ee6cca83871b018289aec1505cba76db1fc87b6802",
    "delphi_3e18_39bucket:ff65381e6c6fa2fd41b3444cc7dae32b0a07f1eaf50d99b9dc409da261b0a1ac",
    "delphi_3e18_39bucket:09ab1aa79a519d8cbef31696066513a006253831fb687b610438ea9b39faef2c",
    "delphi_3e18_39bucket:742dd349c1fbc18af63f516a37aaff216490071ad5e744e3cf744b4b49367d40",
    "delphi_3e18_39bucket:0066da643d35e694a311d26516399114a1c69adba1f7795b773b704d93c0e945",
    "delphi_3e18_39bucket:02c70e578e5784e551eefe679886439d9c803cedd53b283f3333a0c5fbe3e5cf",
    "delphi_3e18_39bucket:9dd4959b60a619086be7a961944fc3ecdd885a9e689014e9469cefc96b05d519",
    "delphi_3e18_39bucket:5268fc7970fdd1757c78b3d28d3b4a01e39ae7eb48d9eff786688b6aa8419e2c",
    "delphi_3e18_39bucket:1bd2258fc50c2f2300cfee3ccf288fcf65b55a83f6f4e754166ad83968fd2bc5",
    "delphi_3e18_39bucket:9e9c69194eb0212d58b2cd82c37cfb3cd0425cc53145c90e62e8fdaa0146b72c",
    "delphi_3e18_39bucket:100cd66fd89a88c77157afcbddc20547cc115238f2281abf552b8c0d0374cc36",
    "delphi_3e18_39bucket:7132c13eee4b70ca34a33f9a9ae7366f1bfa0fced8aaf9fa0323bc319b97506a",
    "delphi_3e18_39bucket:1186365305df843132a2b49cf1f4f3279bc6792218502b2ddb658ff6a8909811",
    "delphi_3e18_39bucket:6eff5c2c6d0c6033485ec4cb97a6edae6de89063ea34c89f669ff4e4e0629eea",
    "delphi_3e18_39bucket:1f764babd33705edc89b9384b3ade86c50adefb121d72a48eead725f2f99fdbb",
    "delphi_3e18_39bucket:fafbc2ba18f1c7971ddf8e7eff5f92f1b11469cf1d80181209696d8655812553",
    "delphi_3e18_39bucket:e8e86e247324e07114d17c3fe9b93415a22bc9d717bbeb492e8a949f59ffc574",
    "delphi_3e18_39bucket:435fc59b9f74ea68a12a16abcf01b49d1d751709400f056661ece786c4fb97f4",
    "delphi_3e18_39bucket:037a604380dcbb0bd031ab6fe4d27d6f859ec425fb267f14b984c1fb545aefe5",
)
DOSE_SOURCE = "conditional_epoch_dose_response"
CONDITION_DECIMALS = 8


def shares_from_levels(levels: np.ndarray, inventory: np.ndarray) -> np.ndarray:
    """Relative epoch levels per bucket to normalized shares (share = level / inventory, renormalized)."""
    weights = levels / inventory
    return weights / weights.sum()


def level_vector(spec: dict[str, float], types: np.ndarray) -> np.ndarray:
    return np.array([spec[kind] for kind in types])


def farthest_point(points: np.ndarray, count: int, start: np.ndarray) -> np.ndarray:
    chosen: list[int] = []
    reference = [start]
    for _ in range(count):
        distance = 0.5 * np.abs(points[:, None, :] - np.array(reference)[None, :, :]).sum(-1).min(1)
        distance[chosen] = -1.0
        index = int(np.argmax(distance))
        chosen.append(index)
        reference.append(points[index])
    return np.array(chosen)


def condition_key(weights: np.ndarray, fractions: np.ndarray, seed_block: int) -> tuple[float, ...]:
    """A run condition: shares, pool fractions and the seed block (paired seeds make block-1 rows distinct runs)."""
    return (*np.round(np.concatenate([weights, fractions]), CONDITION_DECIMALS).tolist(), float(seed_block))


def load_dose_frame(panel: harness.BenchPanel) -> pd.DataFrame:
    """Dose-ladder coordinates with both targets measured, their weights, and the run names behind them."""
    coordinates, _components, _hashes = harness.heldout_registry()
    runs = pd.read_csv(harness.HELDOUT_DIR / "heldout_runs.csv", low_memory=False)
    dose = coordinates[
        coordinates["panel"].eq(panel.name) & coordinates["sources"].astype(str).str.contains(DOSE_SOURCE)
    ]
    complete = dose["table9_macro_n"].fillna(0).gt(0) & dose["uncheatable_n"].fillna(0).gt(0)
    dose = dose[complete].copy()
    grouped = runs.groupby("coordinate_id")
    dose["source_run_names"] = dose["coordinate_id"].map(
        grouped["training_wandb_run_id"].apply(lambda s: ";".join(sorted(set(s.astype(str)))))
    )
    for column in ("data_seed", "trainer_seed"):
        unique = grouped[column].apply(lambda s: tuple(sorted(set(s.dropna().astype(int)))))
        dose[column] = dose["coordinate_id"].map(unique.map(lambda values: values[0] if len(values) == 1 else np.nan))
    return dose.reset_index(drop=True)


def historical_seeds(manifest_path: Path) -> dict[str, tuple[int, int]]:
    """(data seed, trainer seed) of the current panel's runs, from the augmented-swarm training manifest."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_path} is required for the reused panel rows' seed provenance")
    manifest = pd.read_csv(manifest_path)
    return {str(row.run_name): (int(row.data_seed), int(row.trainer_seed)) for row in manifest.itertuples()}


def build_design(
    panel: harness.BenchPanel, dose: pd.DataFrame, panel_seeds: dict[str, tuple[int, int]] | None = None
) -> pd.DataFrame:
    seeds = panel_seeds or {}

    def historical_seeds_for(source: str, run_names: str, historical: tuple[int, int] | None) -> tuple[int, int]:
        """Seeds a reused row was trained with; reused rows never get invented seeds."""
        if source == "reused_panel":
            if run_names not in seeds:
                raise ValueError(f"no historical seeds for panel run {run_names!r}; the training manifest is required")
            return seeds[run_names]
        if historical is None:
            raise ValueError(f"reused row {run_names!r} has no historical seeds")
        return historical

    buckets = list(panel.buckets)
    inventory = panel.features.inventory
    types = np.array([policies.bucket_type(bucket) for bucket in buckets])
    weights_panel = panel.features.weights
    names = list(panel.runs)
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, object]] = []

    def add(
        kind: str,
        block: str,
        weights: np.ndarray,
        *,
        anchor: str = "",
        target: str = "",
        multiplier: float = 1.0,
        pool_fractions: dict[str, float] | None = None,
        source: str = "new",
        source_coordinate_id: str = "",
        source_run_names: str = "",
        seed_block: int = 0,
        wave: str = "full",
        repeat_of: str = "",
        historical: tuple[int, int] | None = None,
    ) -> None:
        weights = np.asarray(weights, dtype=float)
        if weights.min() < 0 or not np.isclose(weights.sum(), 1.0, atol=1e-9):
            raise ValueError(f"{kind}: weights must be a nonnegative simplex point")
        rows.append(
            {
                "kind": kind,
                "block": block,
                "anchor": anchor,
                "target_bucket": target,
                "multiplier": multiplier,
                "pool_fractions": dict(pool_fractions or {}),
                "source": source,
                "source_coordinate_id": source_coordinate_id,
                "source_run_names": source_run_names,
                "seed_block": seed_block if source == "new" else -1,
                "data_seed": (
                    BASE_DATA_SEED + seed_block
                    if source == "new"
                    else historical_seeds_for(source, source_run_names, historical)[0]
                ),
                "trainer_seed": (
                    TRAINER_SEED if source == "new" else historical_seeds_for(source, source_run_names, historical)[1]
                ),
                "subset_seed": BASE_DATA_SEED + seed_block if source == "new" else None,
                "wave": wave,
                "repeat_of": repeat_of,
                "weights": weights / weights.sum(),
            }
        )

    proportional_index = names.index("singleavg_fit_000_baseline_proportional")
    proportional = weights_panel[proportional_index]
    add(
        "baseline_proportional",
        "reserved",
        proportional,
        anchor="A_proportional",
        source="reused_panel",
        source_run_names=names[proportional_index],
        wave="pilot",
    )
    unimax_index = names.index("singleavg_fit_001_baseline_unimax")
    add(
        "baseline_unimax",
        "reserved",
        weights_panel[unimax_index],
        source="reused_panel",
        source_run_names=names[unimax_index],
    )
    uniform_index = next(i for i, n in enumerate(names) if "stratified" in n)
    add(
        "baseline_uniform",
        "reserved",
        weights_panel[uniform_index],
        source="reused_panel",
        source_run_names=names[uniform_index],
    )
    for repeat in range(PROPORTIONAL_REPEATS):  # fresh full-support runs of A in seed blocks 0, 1, 2
        add(
            f"proportional_repeat_block{repeat}",
            "reserved",
            proportional,
            anchor="A_proportional",
            seed_block=repeat,
            wave="pilot",
            repeat_of="baseline_proportional",
        )
    for index, name in enumerate(names):
        if "_pctrl_del_" in name:
            add(
                "pctrl_del_" + name.split("_pctrl_del_")[1],
                "reserved",
                weights_panel[index],
                source="reused_panel",
                source_run_names=name,
            )

    anchor_weights = {"A_proportional": proportional}
    for name, spec in ANCHOR_LEVELS.items():
        anchor_weights[name] = shares_from_levels(level_vector(spec, types), inventory)
        pilot = name == "B_small_pools_forward"
        add(f"anchor_{name}", "anchor", anchor_weights[name], anchor=name, wave="pilot" if pilot else "full")
    add(
        "anchor_B_small_pools_forward_repeat",
        "anchor",
        anchor_weights["B_small_pools_forward"],
        anchor="B_small_pools_forward",
        seed_block=1,
        wave="pilot",
        repeat_of="anchor_B_small_pools_forward",
    )

    def members(target: str) -> list[str]:
        if target == CC_HIGH_GROUP:
            return [b for b, t in zip(buckets, types, strict=True) if t == "cc_high"]
        return [target]

    for seed_block in PILOT_SEED_BLOCKS:
        for anchor in PILOT_ANCHORS:
            for target in PILOT_TARGETS:
                for fraction in POOL_FRACTIONS:
                    add(
                        f"subsample_{anchor}_{target}_pool{fraction:g}_block{seed_block}",
                        "subsampled_pool",
                        anchor_weights[anchor],
                        anchor=anchor,
                        target=target,
                        pool_fractions={m: fraction for m in members(target)},
                        seed_block=seed_block,
                        wave="pilot",
                        repeat_of=(
                            f"subsample_{anchor}_{target}_pool{fraction:g}_block{PILOT_SEED_BLOCKS[0]}"
                            if seed_block != PILOT_SEED_BLOCKS[0]
                            else ""
                        ),
                    )
    for target in EXTRA_TARGETS:
        for fraction in POOL_FRACTIONS:
            add(
                f"subsample_{EXTRA_ANCHOR}_{target}_pool{fraction:g}_block0",
                "subsampled_pool",
                anchor_weights[EXTRA_ANCHOR],
                anchor=EXTRA_ANCHOR,
                target=target,
                pool_fractions={m: fraction for m in members(target)},
            )

    sampler = qmc.LatinHypercube(d=len(TYPES), seed=SEED)
    produced = 0
    while produced < LHS_ROWS:
        for unit in sampler.random(LHS_ROWS):
            if produced >= LHS_ROWS:
                break
            levels = {
                kind: float(np.exp(LHS_LOG_BOX[kind][0] + unit[j] * (LHS_LOG_BOX[kind][1] - LHS_LOG_BOX[kind][0])))
                for j, kind in enumerate(TYPES)
            }
            raw = level_vector(levels, types) * np.exp(rng.normal(0.0, LHS_JITTER, len(buckets)))
            weights = shares_from_levels(raw, inventory)
            if (weights * inventory).max() > LHS_MAX_EPOCHS:
                continue
            add(f"lhs_type_epoch_ratios_{produced:03d}", "lhs", weights)
            produced += 1
    for anchor in ("B_small_pools_forward", "C_code_math_forward"):
        base = level_vector(ANCHOR_LEVELS[anchor], types)
        for target in LADDER_BUCKETS:
            for multiplier in LADDER_MULTIPLIERS:
                levels = base.copy()
                levels[buckets.index(target)] *= multiplier
                add(
                    f"ladder_{anchor}_{target}_x{multiplier:g}",
                    "ladder",
                    shares_from_levels(levels, inventory),
                    anchor=anchor,
                    target=target,
                    multiplier=multiplier,
                )
    for cc_high, cc_low, other in CORNER_LEVELS:
        spec = {"cc_high": cc_high, "cc_low": cc_low, "code": other, "curated": other, "math": other, "synthetic": other}
        add(
            f"corner_cchigh{cc_high:g}_cclow{cc_low:g}_others{other:g}",
            "corner",
            shares_from_levels(level_vector(spec, types), inventory),
        )

    random_rows = np.array([i for i, n in enumerate(names) if "_run_" in n])
    for index in random_rows[
        farthest_point(weights_panel[random_rows], REUSED_RANDOM_PANEL_ROWS, weights_panel.mean(0))
    ]:
        add(
            f"random_panel_{names[index].split('_run_')[1]}",
            "reused_random_panel",
            weights_panel[index],
            source="reused_panel",
            source_run_names=names[index],
        )
    by_coordinate = dose.set_index("coordinate_id")
    missing = [cid for cid in REUSED_DOSE_COORDINATES if cid not in by_coordinate.index]
    if missing:
        raise ValueError(f"{len(missing)} frozen dose coordinates are absent or lack a target: {missing[:3]}")
    for rank, cid in enumerate(REUSED_DOSE_COORDINATES):
        row = by_coordinate.loc[cid]
        if not np.isfinite(float(row["data_seed"])) or not np.isfinite(float(row["trainer_seed"])):
            raise ValueError(f"frozen dose coordinate {cid} has missing or conflicting seeds in the registry")
        add(
            f"dose_ladder_reused_{rank:02d}",
            "reused_dose_ladder",
            row[[f"weight::{bucket}" for bucket in buckets]].to_numpy(float),
            source="reused_dose",
            source_coordinate_id=cid,
            source_run_names=str(row["source_run_names"]),
            historical=(int(row["data_seed"]), int(row["trainer_seed"])),
        )

    frame = pd.DataFrame(rows)
    if len(frame) != BUDGET_ROWS:
        raise ValueError(f"design has {len(frame)} rows, budget {BUDGET_ROWS}")
    if frame["kind"].duplicated().any():
        raise ValueError("duplicate row kinds")
    keys = [
        condition_key(w, np.array([f.get(b, 1.0) for b in buckets]), int(block))
        for w, f, block in zip(frame["weights"], frame["pool_fractions"], frame["seed_block"], strict=True)
    ]
    seen: dict[tuple[float, ...], str] = {}
    for key, kind in zip(keys, frame["kind"], strict=True):
        if key in seen:
            raise ValueError(f"{kind} duplicates the run condition of {seen[key]}")
        seen[key] = kind
    frame.insert(0, "run_name", [f"apriori_swarm_{i:03d}_{k}"[:96] for i, k in enumerate(frame["kind"])])
    return frame


def launcher_table(design: pd.DataFrame, buckets: list[str], inventory: np.ndarray) -> pd.DataFrame:
    """Launcher-format columns: equal weights in both phases, pool fractions and materialized epochs per domain."""
    records = []
    for _, row in design.iterrows():
        weights = dict(zip(buckets, row["weights"], strict=True))
        epochs_at_full = dict(zip(buckets, row["weights"] * inventory, strict=True))
        record: dict[str, object] = {
            "run_name": row["run_name"],
            "kind": row["kind"],
            "block": row["block"],
            "wave": row["wave"],
            "anchor": row["anchor"],
            "target_bucket": row["target_bucket"],
            "multiplier": row["multiplier"],
            "source": row["source"],
            "source_coordinate_id": row["source_coordinate_id"],
            "source_run_names": row["source_run_names"],
            "repeat_of": row["repeat_of"],
            "seed_block": row["seed_block"],
            "data_seed": row["data_seed"],
            "trainer_seed": row["trainer_seed"],
            "subset_seed": "" if row["subset_seed"] is None else row["subset_seed"],  # blank for reused rows
        }
        for phase in PHASE_NAMES:
            for domain in DOMAIN_NAMES:
                record[f"{phase}_{domain}"] = weights[domain]
        for domain in DOMAIN_NAMES:
            fraction = float(row["pool_fractions"].get(domain, 1.0))
            record[f"pool_fraction_{domain}"] = fraction
            record[f"materialized_epochs_{domain}"] = epochs_at_full[domain] / fraction
        records.append(record)
    return pd.DataFrame(records)


def row_summary(design: pd.DataFrame, panel: harness.BenchPanel) -> pd.DataFrame:
    buckets = list(panel.buckets)
    inventory = panel.features.inventory
    types = np.array([policies.bucket_type(bucket) for bucket in buckets])
    proportional = panel.features.weights[list(panel.runs).index("singleavg_fit_000_baseline_proportional")]
    records = []
    for _, row in design.iterrows():
        weights = np.asarray(row["weights"])
        fractions = np.array([row["pool_fractions"].get(bucket, 1.0) for bucket in buckets])
        epochs = weights * inventory / fractions
        positive = weights[weights > 0]
        records.append(
            {
                "run_name": row["run_name"],
                "kind": row["kind"],
                "block": row["block"],
                "wave": row["wave"],
                "source": row["source"],
                "seed_block": row["seed_block"],
                "effective_buckets": float(np.exp(-(positive * np.log(positive)).sum())),
                "nonzero_buckets": int((weights > 0).sum()),
                "max_epochs": float(epochs.max()),
                "max_epochs_bucket": buckets[int(np.argmax(epochs))],
                "cc_share": float(weights[np.char.startswith(types.astype(str), "cc")].sum()),
                "synth_qa_share": float(weights[buckets.index("dolmino_synth_qa")]),
                "stack_share": float(
                    weights[[buckets.index("dolma3_stack_edu"), buckets.index("dolmino_stack_edu_fim")]].sum()
                ),
                "tv_to_proportional": float(0.5 * np.abs(weights - proportional).sum()),
            }
        )
    return pd.DataFrame(records)


def support_separation_report(design: pd.DataFrame, panel: harness.BenchPanel) -> dict[str, object]:
    """Can exposure effects be told apart from share effects? Rank of [shares | exposures] and per-bucket conditioning.

    In the current panel exposures equal shares times a constant, so the joint matrix has rank 39 and every
    per-bucket (share, exposure) pair is collinear. The subsampled rows add independent exposure directions.
    """
    buckets = list(panel.buckets)
    inventory = panel.features.inventory
    weights = np.vstack(design["weights"].to_numpy())
    fractions = np.array([[row.get(bucket, 1.0) for bucket in buckets] for row in design["pool_fractions"]])
    exposures = weights * inventory[None, :] / fractions

    def joint_rank(shares: np.ndarray, expo: np.ndarray) -> int:
        matrix = np.hstack([shares, expo / expo.max(axis=0, keepdims=True).clip(1e-12)])
        return int(np.linalg.matrix_rank(matrix, tol=1e-8 * np.linalg.norm(matrix, 2)))

    per_bucket: dict[str, object] = {}
    for column, bucket in enumerate(buckets):
        pair = np.column_stack([weights[:, column], exposures[:, column] / inventory[column]])
        pair = pair - pair.mean(axis=0)
        singular = np.linalg.svd(pair, compute_uv=False)
        per_bucket[bucket] = {
            "rows_with_reduced_pool": int((fractions[:, column] < 1.0).sum()),
            "share_exposure_condition_ratio": float(singular[1] / singular[0]) if singular[0] > 0 else 0.0,
        }
    panel_shares = panel.features.weights
    return {
        "panel_rank_shares_and_exposures": joint_rank(panel_shares, panel.features.exposures),
        "design_rank_shares_and_exposures": joint_rank(weights, exposures),
        "buckets": len(buckets),
        "note": (
            "a rank above 39 counts exposure directions not implied by shares; the per-bucket ratio is the "
            "smaller over the larger singular value of the centred (share, exposure) pair, 0 when collinear"
        ),
        "per_bucket": per_bucket,
    }


def coverage_report(design: pd.DataFrame, panel: harness.BenchPanel, registry_dir: Path) -> dict[str, object]:
    """Post-hoc diagnostic only: distances from measured coordinates to the design; never used to place rows."""
    harness.HELDOUT_DIR = registry_dir.resolve()
    bank = selection.load_bank(panel, "table9")
    _frame, features = harness.heldout_features(panel, "table9")
    top = np.argsort(bank.measured)[:10]
    points = np.vstack(design["weights"].to_numpy())
    new_points = np.vstack(design.loc[design["source"].eq("new"), "weights"].to_numpy())

    def nearest(pts: np.ndarray) -> dict[str, float]:
        distance = (0.5 * np.abs(features.weights[top][:, None, :] - pts[None, :, :]).sum(-1)).min(1)
        return {
            "median_tv_to_top10": float(np.median(distance)),
            "min_tv_to_top10": float(distance.min()),
            "max_tv_to_top10": float(distance.max()),
        }

    return {
        "note": "diagnostic against measured Table-9 coordinates; the design was placed without them",
        "current_panel": nearest(panel.features.weights),
        "design_all_rows": nearest(points),
        "design_new_rows": nearest(new_points),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--registry-dir",
        type=Path,
        default=harness.HELDOUT_DIR,
        help="heldout registry (dose rows to reuse; coverage diagnostic)",
    )
    parser.add_argument("--skip-coverage", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    harness.HELDOUT_DIR = args.registry_dir.resolve()
    panel = harness.load_panel(PANEL)
    _coordinates, _components, registry_hashes = harness.heldout_registry()
    design = build_design(panel, load_dose_frame(panel), historical_seeds(HISTORICAL_MANIFEST))
    buckets = list(panel.buckets)
    table = launcher_table(design, buckets, panel.features.inventory)
    table.to_csv(args.output_dir / "swarm_mixtures.csv", index=False)
    summary = row_summary(design, panel)
    summary.to_csv(args.output_dir / "design_rows.csv", index=False)
    manifest = {
        "design_seed": SEED,
        "base_data_seed": BASE_DATA_SEED,
        "rows": len(design),
        "new_runs": int(design["source"].eq("new").sum()),
        "pilot_runs": int((design["wave"].eq("pilot") & design["source"].eq("new")).sum()),
        "blocks": design.groupby("block")["source"].value_counts().unstack(fill_value=0).to_dict(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "panel_inputs": panel.input_hashes,
        "registry_hashes": registry_hashes,
        "registry_dir": str(harness.HELDOUT_DIR),
        "target_budget_tokens": float(TARGET_BUDGET_DOLMA3_COMMON_CRAWL),
        "unique_tokens": {bucket: int(TOP_LEVEL_DOMAIN_TOKEN_COUNTS[bucket]) for bucket in buckets},
        "reused_dose_rows_require": "table9_macro_n > 0 and uncheatable_n > 0 in the registry",
        "mixtures_sha256": hashlib.sha256((args.output_dir / "swarm_mixtures.csv").read_bytes()).hexdigest(),
        "support_separation": support_separation_report(design, panel),
    }
    if not args.skip_coverage:
        manifest["coverage_diagnostic"] = coverage_report(design, panel, args.registry_dir)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    pd.set_option("display.width", 250)
    print(
        summary.groupby(["block", "source"])
        .agg(
            rows=("run_name", "size"),
            pilot=("wave", lambda s: int((s == "pilot").sum())),
            eff_min=("effective_buckets", "min"),
            eff_median=("effective_buckets", "median"),
            max_epochs_median=("max_epochs", "median"),
            max_epochs_max=("max_epochs", "max"),
            cc_share_min=("cc_share", "min"),
        )
        .round(2)
        .to_string()
    )
    print(
        json.dumps(
            {k: manifest[k] for k in ("rows", "new_runs", "pilot_runs", "coverage_diagnostic") if k in manifest},
            indent=2,
        )
    )
    print(
        "support separation: panel rank",
        manifest["support_separation"]["panel_rank_shares_and_exposures"],
        "design rank",
        manifest["support_separation"]["design_rank_shares_and_exposures"],
    )


if __name__ == "__main__":
    main()

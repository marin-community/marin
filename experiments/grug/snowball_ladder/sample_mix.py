# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build fixed-size, pre-mixed token caches for the Snowball scaling ladder.

The ladder trains on a GPU cluster that cannot stream the full datakit stores from CoreWeave, so
each data phase becomes one flat Levanter cache, built once in ``marin-us-east-02a`` and copied to
the cluster. Training reads it as a single ``DatasetComponent(flat_cache=True)`` at weight 1.

Each recipe samples whole documents from a 200-cell ``cluster=N/quality=Q`` store. Tokens are
copied as stored, so there is no decode or retokenization and every document keeps its EOS (the
loader derives document-masking segment ids from it). Documents are not length-capped.

For cell ``c`` with ``A_c`` stored tokens, output cache ``k`` wants ``p_k = target_k * w_k[c] / A_c``
copies of each document. The caches take consecutive bands ``[a_k, a_k + p_k)`` on one per-document
uniform draw ``u``, and a document lands in cache ``k`` ``ceil(b_k - u) - ceil(a_k - u)`` times. So
the expected tokens per cell hit the mixture exactly, the caches share no documents while the
bands fit in ``[0, 1)``, and a cell that is too small repeats documents instead of falling short.
The draw is seeded by (seed, cell, shard), so results do not depend on how work is split.

Recipes (targets carry a 1% margin so a run never wraps):

* ``harrier``: the hero store ``store_4d2e363d``. ``harrier_main`` uses the September mixture
  (phase ``main``) for Snowball's 2.114T pretraining tokens; ``harrier_cooldown`` uses the hero
  cooldown weights for Snowball's 211B cooldown tokens.
* ``snowball``: Snowball's own store ``store_8ac06c74`` and mixture-3 weights, sized for the
  June-prep d1024 reference run (678B tokens, phase 1 from 80%). Smaller reference runs read a
  subset.

The output is in document order by cell. Train with a full permutation shuffle
(``LmDataConfig(shuffle=True)``), which works at any sequence length.

Phases, run in order (``sample`` as independent Iris tasks, the rest on one machine)::

    python -m experiments.grug.snowball_ladder.sample_mix --phase plan --recipe harrier --out $OUT
    python -m experiments.grug.snowball_ladder.sample_mix --phase sample --out $OUT --num-tasks 64 --task-index $k
    python -m experiments.grug.snowball_ladder.sample_mix --phase merge --out $OUT
    python -m experiments.grug.snowball_ladder.sample_mix --phase validate --out $OUT --du

See ``README.md`` next to this file for launch commands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from levanter.store.cache import CACHE_LAYOUT_SHARDED, CacheLedger, CacheMetadata, TreeCache, _merge_sharded_ledgers
from levanter.store.jagged_array import JaggedArrayStore, set_jagged_array_read_cache_bytes
from rigging.filesystem.factory import url_to_fs
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.datakit.store.bucket_writer import write_bucket_cache
from experiments.grug.moe.launch_datakit_moe_mix import _TAIL_BUCKETS, _phase_weights

logger = logging.getLogger(__name__)

HARRIER_STORE = "s3://marin-us-east-02a/marin/datakit/store_4d2e363d"
SNOWBALL_STORE = "s3://marin-us-east-02a/marin/datakit/store_8ac06c74"

# Snowball (the 2.33T checkpoint, #6811): steps 0-15k at 4096 x 8192, steps 15k-39k at 8192 x 8192,
# then a 3150-step cooldown at 1024 x 65536.
SNOWBALL_PRETRAIN_TOKENS = 15_000 * 4096 * 8192 + 24_000 * 8192 * 8192
SNOWBALL_COOLDOWN_TOKENS = 3_150 * 1024 * 65_536
# June-prep d1024 (#6044): 678B tokens, phase 1 from 80%. The largest reference run on Snowball data.
JUNE_PREP_D1024_TOKENS = 678_000_000_000
TARGET_MARGIN = 1.01

PLAN_FILE = "plan.json"
REPORT_FILE = "report.json"
DONE_DIR = "_done"
STATS_FILE = ".stats.json"
_EXEMPLAR = {"input_ids": np.zeros(0, dtype=np.int32)}
READ_CACHE_BYTES = 256_000_000
READ_BATCH_TOKENS = 64_000_000
READ_BATCH_DOCS = 2_000


@dataclass(frozen=True)
class OutputCache:
    name: str
    target_tokens: int
    # Cell (or group) name -> weight; normalized against the cells that exist.
    weights: dict[str, float]


@dataclass(frozen=True)
class Recipe:
    store: str
    caches: tuple[OutputCache, ...]
    # Group name -> member cells. A group's weight is split across members in proportion to their tokens.
    groups: dict[str, tuple[str, ...]]


def cell_dir(store: str, cell: str) -> str:
    return prefix_join(store, f"cluster={int(cell[1:3])}/quality={int(cell[4])}")


def harrier_recipe() -> Recipe:
    spec = json.loads((Path(__file__).parents[1] / "moe_hero_ep" / "harrier_mix_2026_08_18.json").read_text())
    phases = {phase["name"]: phase["weights"] for phase in spec["phases"]}
    return Recipe(
        store=HARRIER_STORE,
        caches=(
            OutputCache("harrier_main", round(SNOWBALL_PRETRAIN_TOKENS * TARGET_MARGIN), phases["main"]),
            OutputCache("harrier_cooldown", round(SNOWBALL_COOLDOWN_TOKENS * TARGET_MARGIN), phases["cooldown"]),
        ),
        groups={},
    )


def snowball_recipe() -> Recipe:
    return Recipe(
        store=SNOWBALL_STORE,
        caches=(
            OutputCache("snowball_phase0", round(JUNE_PREP_D1024_TOKENS * 0.8 * TARGET_MARGIN), _phase_weights(0)),
            OutputCache("snowball_phase1", round(JUNE_PREP_D1024_TOKENS * 0.2 * TARGET_MARGIN), _phase_weights(1)),
        ),
        groups={"tail": tuple(_TAIL_BUCKETS)},
    )


RECIPES = {"harrier": harrier_recipe, "snowball": snowball_recipe}


def resolve_cell_weights(
    weights: dict[str, float], groups: dict[str, tuple[str, ...]], cell_tokens: dict[str, int]
) -> dict[str, float]:
    """Expand groups to member cells and normalize to a simplex over cells."""
    resolved: dict[str, float] = {}
    for name, weight in weights.items():
        if weight <= 0:
            continue
        if name in groups:
            members = groups[name]
            total = sum(cell_tokens[m] for m in members)
            for member in members:
                resolved[member] = resolved.get(member, 0.0) + weight * cell_tokens[member] / total
        else:
            if name not in cell_tokens:
                raise ValueError(f"weight for {name} but the store has no such cell")
            resolved[name] = resolved.get(name, 0.0) + weight
    total = sum(resolved.values())
    return {cell: w / total for cell, w in resolved.items()}


def cell_bands(
    caches: Sequence[OutputCache], cell_weights: dict[str, dict[str, float]], cell_tokens: dict[str, int]
) -> dict[str, dict[str, tuple[float, float]]]:
    """Per cell, the ``[a, b)`` band on the document draw each cache owns."""
    bands: dict[str, dict[str, tuple[float, float]]] = {}
    for cell, tokens in cell_tokens.items():
        start = 0.0
        cell_band = {}
        for cache in caches:
            weight = cell_weights[cache.name].get(cell, 0.0)
            if weight <= 0 or tokens <= 0:
                continue
            width = cache.target_tokens * weight / tokens
            cell_band[cache.name] = (start, start + width)
            start += width
        if cell_band:
            bands[cell] = cell_band
    return bands


def document_copies(u: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    """How many times each document with draw ``u`` in ``[0, 1)`` falls in ``band``, counted modulo 1."""
    a, b = band
    return (np.ceil(b - u) - np.ceil(a - u)).astype(np.int64)


def document_draws(seed: int, cell: str, shard_key: str, n: int) -> np.ndarray:
    digest = hashlib.blake2b(f"{cell}/{shard_key}".encode(), digest_size=8, key=str(seed).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest, "big")).random(n)


def _cell_items(store: str, cell: str) -> tuple[int, list[dict]]:
    """Total tokens and readable shards of one cell cache."""
    root = cell_dir(store, cell)
    ledger = CacheLedger.load(root)
    tokens = int(ledger.field_counts["input_ids"])
    if ledger.layout == CACHE_LAYOUT_SHARDED:
        shards = [
            {
                "path": prefix_join(root, shard),
                "key": shard,
                "rows": int(ledger.shard_rows[shard]),
                "tokens": int(ledger.field_counts_by_shard.get(shard, {}).get("input_ids", 0)),
            }
            for shard in ledger.finished_shards
        ]
    else:
        shards = [{"path": root, "key": ".", "rows": int(ledger.total_num_rows), "tokens": tokens}]
    return tokens, shards


def recipe_cells(recipe: Recipe) -> list[str]:
    """Cells that any cache gives positive weight, directly or through a group."""
    cells = set()
    for cache in recipe.caches:
        for name, weight in cache.weights.items():
            if weight > 0:
                cells.update(recipe.groups.get(name, (name,)))
    return sorted(cells)


def plan(recipe_name: str, out: str, seed: int, store: str | None, target_overrides: dict[str, int]) -> dict:
    recipe = RECIPES[recipe_name]()
    return plan_recipe(recipe, recipe_name, out, seed, store, target_overrides)


def plan_recipe(
    recipe: Recipe, recipe_name: str, out: str, seed: int, store: str | None, target_overrides: dict[str, int]
) -> dict:
    store = store or recipe.store
    caches = tuple(OutputCache(c.name, target_overrides.get(c.name, c.target_tokens), c.weights) for c in recipe.caches)
    unknown = set(target_overrides) - {c.name for c in caches}
    if unknown:
        raise ValueError(f"--target names unknown caches {sorted(unknown)}")

    cells = recipe_cells(recipe)
    with ThreadPoolExecutor(32) as pool:
        listed = dict(zip(cells, pool.map(lambda c: _cell_items(store, c), cells), strict=True))
    cell_tokens = {cell: tokens for cell, (tokens, _) in listed.items()}
    cell_weights = {c.name: resolve_cell_weights(c.weights, recipe.groups, cell_tokens) for c in caches}
    bands = cell_bands(caches, cell_weights, cell_tokens)

    items = [
        {"cell": cell, **shard} for cell in cells if cell in bands for shard in listed[cell][1] if shard["rows"] > 0
    ]
    result = {
        "recipe": recipe_name,
        "store": store,
        "seed": seed,
        "caches": {c.name: c.target_tokens for c in caches},
        "cells": {
            cell: {
                "tokens": cell_tokens[cell],
                "bands": {name: list(band) for name, band in bands.get(cell, {}).items()},
                "expected_tokens": {name: (b - a) * cell_tokens[cell] for name, (a, b) in bands.get(cell, {}).items()},
            }
            for cell in cells
        },
        "items": items,
    }
    StoragePath(prefix_join(out, PLAN_FILE)).write_text(json.dumps(result, indent=1))
    _log_plan_summary(result)
    return result


def _log_plan_summary(result: dict) -> None:
    store_tokens = sum(cell["tokens"] for cell in result["cells"].values())
    logger.info("store %s: %.3fT tokens, %d shards to read", result["store"], store_tokens / 1e12, len(result["items"]))
    for name, target in result["caches"].items():
        widths = {
            cell: info["bands"][name][1] - info["bands"][name][0]
            for cell, info in result["cells"].items()
            if name in info["bands"]
        }
        repeated = {cell: w for cell, w in widths.items() if w > 1}
        logger.info(
            "%s: target %.1fB tokens, %d cells, max %.3f copies per document (%s), %d cells repeat documents",
            name,
            target / 1e9,
            len(widths),
            max(widths.values()),
            max(widths, key=widths.get),
            len(repeated),
        )
    shared = [
        cell
        for cell, info in result["cells"].items()
        if len(info["bands"]) > 1 and max(b for _, b in info["bands"].values()) > 1
    ]
    if shared:
        logger.info("%d cells share documents across caches: %s", len(shared), shared)


def load_plan(out: str) -> dict:
    return json.loads(StoragePath(prefix_join(out, PLAN_FILE)).read_text())


def _doc_lengths(store: JaggedArrayStore) -> np.ndarray:
    n = len(store)
    offsets = np.asarray(store.offsets[1 : n + 1].read().result(), dtype=np.int64)
    return np.diff(offsets, prepend=0)


def _read_docs(store: JaggedArrayStore, indices: np.ndarray, lengths: np.ndarray) -> Iterator[np.ndarray]:
    """Yield documents in ``indices`` order, batching reads by document count and tokens."""
    start = 0
    while start < len(indices):
        stop = start
        batch_tokens = 0
        while (
            stop < len(indices)
            and stop - start < READ_BATCH_DOCS
            and (stop == start or batch_tokens < READ_BATCH_TOKENS)
        ):
            batch_tokens += int(lengths[indices[stop]])
            stop += 1
        for doc in store.get_batch_sync(indices[start:stop].tolist()):
            yield np.asarray(doc, dtype=np.int32)
        start = stop


def sample_item(the_plan: dict, index: int, out: str) -> dict:
    item = the_plan["items"][index]
    cell = item["cell"]
    bands = the_plan["cells"][cell]["bands"]
    store = JaggedArrayStore.open(prefix_join(item["path"], "input_ids"), mode="r", dtype=np.int32)
    lengths = _doc_lengths(store)
    if len(lengths) != item["rows"]:
        raise ValueError(f"{item['path']}: plan has {item['rows']} rows, store has {len(lengths)}")
    draws = document_draws(the_plan["seed"], cell, item["key"], len(lengths))

    written = {}
    for cache, band in bands.items():
        copies = document_copies(draws, tuple(band))
        indices = np.repeat(np.arange(len(lengths)), copies)
        if not len(indices):
            continue
        path = prefix_join(out, f"{cache}/part-{index:06d}")
        ledger = write_bucket_cache(path, _read_docs(store, indices, lengths), lengths[indices])
        written[cache] = {
            "path": path,
            "rows": int(ledger.total_num_rows),
            "tokens": int(ledger.field_counts["input_ids"]),
        }
        logger.info("item %d (%s): %s %d docs, %d tokens", index, cell, cache, len(indices), written[cache]["tokens"])
    return {"cell": cell, "source": item["path"], "written": written}


def _sidecar(out: str, index: int) -> str:
    return prefix_join(out, f"{DONE_DIR}/item-{index:06d}.json")


def sample(out: str, num_tasks: int, task_index: int) -> None:
    the_plan = load_plan(out)
    indices = range(task_index, len(the_plan["items"]), num_tasks)
    logger.info("sample task %d/%d: %d items", task_index, num_tasks, len(indices))
    for index in indices:
        sidecar = _sidecar(out, index)
        if StoragePath(sidecar).exists():
            logger.info("item %d already done", index)
            continue
        result = sample_item(the_plan, index, out)
        StoragePath(sidecar).write_text(json.dumps(result))


def _load_sidecars(out: str, count: int) -> list[dict]:
    paths = [_sidecar(out, i) for i in range(count)]
    missing = [p for p in paths if not StoragePath(p).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} of {count} items not done, e.g. {missing[0]}")
    with ThreadPoolExecutor(32) as pool:
        return list(pool.map(lambda p: json.loads(StoragePath(p).read_text()), paths))


def merge(out: str) -> dict:
    the_plan = load_plan(out)
    sidecars = _load_sidecars(out, len(the_plan["items"]))
    report: dict = {"caches": {}, "cells": {}}
    for cache, target in the_plan["caches"].items():
        parts = [s["written"][cache] for s in sidecars if cache in s["written"]]
        root = prefix_join(out, cache)
        stubs = [
            CacheLedger(
                total_num_rows=p["rows"],
                shard_rows={},
                finished_shards=[],
                field_counts={},
                metadata=CacheMetadata.empty(),
            )
            for p in parts
        ]
        ledger = _merge_sharded_ledgers(
            root, [p["path"] for p in parts], stubs, [{"input_ids": p["tokens"]} for p in parts], CacheMetadata.empty()
        )
        tokens = int(ledger.field_counts["input_ids"])
        stats = {"total_tokens": tokens, "total_elements": int(ledger.total_num_rows)}
        StoragePath(prefix_join(root, STATS_FILE)).write_text(json.dumps(stats))
        report["caches"][cache] = {**stats, "target_tokens": target, "realized_over_target": tokens / target}
        logger.info(
            "%s: %d docs, %.2fB tokens (%.4f of target)", cache, stats["total_elements"], tokens / 1e9, tokens / target
        )

    realized: dict[str, dict[str, int]] = {}
    for s in sidecars:
        for cache, part in s["written"].items():
            realized.setdefault(s["cell"], {}).setdefault(cache, 0)
            realized[s["cell"]][cache] += part["tokens"]
    for cell, info in the_plan["cells"].items():
        if info["expected_tokens"]:
            report["cells"][cell] = {
                cache: {"expected": expected, "realized": realized.get(cell, {}).get(cache, 0)}
                for cache, expected in info["expected_tokens"].items()
            }
    StoragePath(prefix_join(out, REPORT_FILE)).write_text(json.dumps(report, indent=1))
    return report


def _cache_bytes(path: str) -> int:
    fs, fs_path = url_to_fs(path)
    return int(fs.du(fs_path))


def validate(out: str, measure_bytes: bool, sample_docs: int = 2_000) -> None:
    the_plan = load_plan(out)
    report = json.loads(StoragePath(prefix_join(out, REPORT_FILE)).read_text())
    for cache in the_plan["caches"]:
        root = prefix_join(out, cache)
        stats = report["caches"][cache]
        loaded = TreeCache.load(root, _EXEMPLAR)
        if len(loaded) != stats["total_elements"]:
            raise ValueError(f"{cache}: ledger has {len(loaded)} docs, report has {stats['total_elements']}")
        if stats["realized_over_target"] < 1.0:
            logger.warning("%s: realized tokens are %.4f of target", cache, stats["realized_over_target"])
        rng = np.random.default_rng(0)
        picks = np.sort(rng.choice(len(loaded), size=min(sample_docs, len(loaded)), replace=False))
        docs = [np.asarray(d["input_ids"]) for d in loaded.get_batch_sync(picks.tolist())]
        last_tokens, counts = np.unique([int(d[-1]) for d in docs], return_counts=True)
        eos_share = counts.max() / len(docs)
        logger.info(
            "%s: %d sampled docs, max token id %d, most common final token %d on %.1f%% of docs",
            cache,
            len(docs),
            max(int(d.max()) for d in docs),
            int(last_tokens[counts.argmax()]),
            100 * eos_share,
        )
        if min(len(d) for d in docs) == 0 or eos_share < 0.99:
            raise ValueError(f"{cache}: empty documents or documents without a shared EOS")
        if measure_bytes:
            size = _cache_bytes(root)
            stats["bytes"] = size
            logger.info("%s: %.2f TB on disk, %.2f bytes per token", cache, size / 1e12, size / stats["total_tokens"])
    deviations = [
        (abs(v["realized"] / v["expected"] - 1), cell, cache)
        for cell, caches in report["cells"].items()
        for cache, v in caches.items()
        if v["expected"] >= 1e9
    ]
    if deviations:
        worst = max(deviations)
        logger.info(
            "largest per-cell deviation among cells with >=1B expected tokens: %.2f%% (%s, %s)",
            100 * worst[0],
            worst[1],
            worst[2],
        )
    if measure_bytes:
        StoragePath(prefix_join(out, REPORT_FILE)).write_text(json.dumps(report, indent=1))
    logger.info("VALIDATE OK")


def _parse_targets(values: Sequence[str]) -> dict[str, int]:
    targets = {}
    for value in values:
        name, _, tokens = value.partition("=")
        targets[name] = int(float(tokens))
    return targets


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", required=True, choices=["plan", "sample", "merge", "validate"])
    parser.add_argument("--out", required=True, help="output root; one cache directory per output cache")
    parser.add_argument("--recipe", choices=sorted(RECIPES), help="plan only")
    parser.add_argument("--store", help="plan only: override the recipe's source store")
    parser.add_argument("--target", action="append", default=[], help="plan only: CACHE=TOKENS, e.g. harrier_main=1e12")
    parser.add_argument("--seed", type=int, default=0, help="plan only")
    parser.add_argument("--num-tasks", type=int, default=1)
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--du", action="store_true", help="validate only: measure bytes on disk")
    args = parser.parse_args()

    if args.phase == "plan":
        if not args.recipe:
            parser.error("--phase plan needs --recipe")
        plan(args.recipe, args.out, args.seed, args.store, _parse_targets(args.target))
    elif args.phase == "sample":
        # The shared read cache can be sized only before the first store opens.
        set_jagged_array_read_cache_bytes(READ_CACHE_BYTES)
        sample(args.out, args.num_tasks, args.task_index)
    elif args.phase == "merge":
        merge(args.out)
    else:
        validate(args.out, args.du)


if __name__ == "__main__":
    main()

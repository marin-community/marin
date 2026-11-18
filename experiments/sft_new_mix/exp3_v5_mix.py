# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The Exp3 SFT training mix — data release V5 — as a Marin artifact graph.

Exp3's production SFT stage (job ``/benfeuer/long-loop-exp3-2026.11.16.51``, coordinator
sign-off ``450bc8a4b24613c6b6f00e07da9b3953``) trained on data release V5
(``release-complete-v5.json``, SHA-256 ``3e21619…``) under source revision
``0f5721bbb4953e98898c847896110dee32a5b293``. This module expresses that exact mix as
versioned handles so any run's mix is recoverable from its records:

- **Release seal.** The release document and its cache catalog are adopted at their exact
  URIs, with the pinned SHA-256 digests recorded in the handles' configs.
- **One sealed-cache handle per source.** Each of the 434 mixture sources is a
  ``TokenizedCache`` step whose fingerprinted config pins the source's exact
  content-addressed shard paths, its row and token counts, and its loss mode. The step's
  function verifies every shard still exists; no data is read or recomputed.
- **The mixture.** :func:`exp3_v5_data_config` assembles the ``LmDataConfig`` through
  :func:`marin.experiment.data.mixture`: the 434 sources at their natural weights, the
  three-phase weight schedule (boundaries 0/250/500), the full-permutation feistel shuffle,
  the release's cache catalog, and the per-source IID validation reservation.

The training run's consumed data config is recorded in
``fixtures/exp3_v5_data.json`` (byte-identical to the release member the run consumed);
the per-source seal manifest in ``fixtures/exp3_v5_source_seals.json`` is derived from the
release's ``cache_catalog.json`` and ``source_loss.json``. Both fixtures carry the SHA-256
digests they were verified against, checked at import.

The Exp3 runtime added one packing control this mix records but current ``main`` does not
implement: bounded-lookahead packing (cap 256, lookahead 128) lived on the long-loop fork
only. The geometry is pinned in every seal record; reproducing it in training needs that
packing change, not a different mix.
"""

import hashlib
import json
from dataclasses import replace
from functools import cache
from pathlib import Path

import click
from levanter.data.text.datasets import LmDataConfig
from levanter.store.cache import CacheOptions
from rigging.filesystem.storage_path import StoragePath, prefix_join

from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext, lower, run
from marin.experiment.data import mixture
from marin.processing.tokenize.tokenize import TokenizedCache

EXP3_V5_VERSION = "2026.11.16"
"""The release's creation date; the identity version of every handle in this mix."""

_FIXTURES = Path(__file__).parent / "fixtures"

_RELEASE_URI = (
    "s3://marin-us-east-08a/marin/users/benfeuer/exp3-data-mix/2026.11.15-candidate-v1"
    "/release-v5/release-complete-v5.json"
)
_RELEASE_SHA256 = "3e2161945e1adc60c9edcdb6b0d7395946966dc4e0d16b88660fbc38989603f5"
"""The release document the coordinator signed off and the launcher verified."""


def _load_fixture(filename: str, sha256: str) -> dict:
    raw = (_FIXTURES / filename).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError(f"{filename}: SHA-256 does not match the pinned digest {sha256}")
    return json.loads(raw)


_RELEASE = _load_fixture("exp3_v5_release.json", _RELEASE_SHA256)
"""The byte-identical release document. Its ``artifacts`` pins every member's digest."""

_DATA = _load_fixture(
    "exp3_v5_data.json",
    _RELEASE["artifacts"]["data.json"]["sha256"],
)
"""The release's ``data.json`` — the ``LmDataConfig`` the Exp3 SFT run consumed."""

_MANIFEST = _load_fixture(
    "exp3_v5_source_seals.json",
    "d90720614efd0d66daeff0ba2549dd44db4e8592aa8f7dbff97eb606fc0d4c9b",
)
"""Per-source seal manifest derived from the release's cache catalog and loss spec.

The manifest is a pure function of the two SHA-pinned release members named in its
``derived_from`` block, so regenerating it from an unchanged release is byte-stable."""

_SEALS = _MANIFEST["sources"]

_CACHE_ROOT = _DATA["components"][next(iter(_DATA["components"]))]["cache_dir"]
_CATALOG_URI = _DATA["cache_catalog"]
_CATALOG_SHA256 = _RELEASE["artifacts"]["cache_catalog.json"]["sha256"]
_TOKENIZER = _DATA["tokenizer"]
_SCHEDULE = [[phase, dict(weights)] for phase, weights in _DATA["train_weights"]]
_NUM_VALIDATION_SEQUENCES = dict(_DATA["num_validation_sequences"])
_MIXTURE_BLOCK_SIZE = _DATA["mixture_block_size"]
_ENFORCE_EOS = _DATA["enforce_eos"]
_AUTO_BUILD_CACHES = _DATA["auto_build_caches"]
_CACHE_BATCH_SIZE = 128
"""Cache-read batch size the Exp3 launcher set. Not a release field: recorded from the run's
consumed config (``production51-sft-consumed-config.json``), which matches the release data
config on every field the release carries."""
_STOP_STRATEGY = _DATA["stop_strategy"]

# The 434 components share one packing/format shape; extract it once and hold the mix to it.
_SHAPE_KEYS = ("format", "pack", "split", "flat_cache")
_SHAPES = {json.dumps({key: component[key] for key in _SHAPE_KEYS}, sort_keys=True) for component in _DATA["components"].values()}
if len(_SHAPES) != 1:
    raise ValueError(f"release components do not share one shape: {_SHAPES}")
_SHAPE = json.loads(next(iter(_SHAPES)))
_PACK = _SHAPE["pack"]
_FLAT_CACHE = _SHAPE["flat_cache"]
_SPLIT = _SHAPE["split"]
_FORMAT = dict(_SHAPE["format"], type="prebuilt")
_PACKING_LOOKAHEAD = next(iter(_DATA["components"].values()))["packing_lookahead"]
assert _PACKING_LOOKAHEAD == 128  # bounded-lookahead packing: long-loop fork runtime, recorded but not on main
assert _MANIFEST["geometry"]["cache_dir"] == _CACHE_ROOT
assert _MANIFEST["geometry"]["pack"] == _PACK and _MANIFEST["geometry"]["packing_lookahead"] == _PACKING_LOOKAHEAD

_SOURCE_NAMES = tuple(sorted(_DATA["components"]))
if set(_SOURCE_NAMES) != set(_SEALS) or set(_SOURCE_NAMES) != set(_SCHEDULE[0][1]):
    raise ValueError("release components, seal manifest, and weight schedule name different source sets")


exp3_v5_release = ArtifactStep.adopt(
    "sft-mixes/exp3-v5/release",
    EXP3_V5_VERSION,
    _RELEASE_URI,
    config={"sha256": _RELEASE_SHA256, "members": dict(_RELEASE["artifacts"])},
)
"""The adopted release document: its exact URI, with every member digest recorded."""

exp3_v5_cache_catalog = ArtifactStep.adopt(
    "sft-mixes/exp3-v5/cache-catalog",
    EXP3_V5_VERSION,
    _CATALOG_URI,
    config={"sha256": _CATALOG_SHA256},
)
"""The adopted cache catalog: the authoritative per-source ledgers behind ``cache_catalog``."""


def _seal_config(name: str) -> dict:
    """The fingerprinted seal record for one source: its exact sealed cache, as literals."""
    seal = _SEALS[name]
    return {
        "source_name": name,
        "tokenizer": _TOKENIZER,
        "format": dict(_FORMAT),
        "pack": _PACK,
        "packing_lookahead": _PACKING_LOOKAHEAD,
        "split": _SPLIT,
        "flat_cache": _FLAT_CACHE,
        "tags": None,
        "cache_dir": _CACHE_ROOT,
        "rows": seal["rows"],
        "input_ids_tokens": seal["input_ids_tokens"],
        "loss_weights_tokens": seal["loss_weights_tokens"],
        "loss_mode": seal["loss_mode"],
        "teacher_only": seal["teacher_only"],
        "shards": list(seal["shards"]),
        "release": {"uri": _RELEASE_URI, "sha256": _RELEASE_SHA256},
        "cache_catalog": {"uri": _CATALOG_URI, "sha256": _CATALOG_SHA256},
    }


def _verify_shards(config: dict) -> None:
    """Fail the step unless every content-addressed shard of this seal still exists."""
    missing = [shard for shard in config["shards"] if not StoragePath(prefix_join(config["cache_dir"], shard)).exists()]
    if missing:
        raise FileNotFoundError(
            f"{config['source_name']}: {len(missing)}/{len(config['shards'])} sealed shard caches are "
            f"missing under {config['cache_dir']} (first: {missing[0]})"
        )


@cache
def exp3_v5_sources() -> dict[str, ArtifactStep[TokenizedCache]]:
    """One sealed-cache handle per release source, keyed by the mixture's component name.

    Each handle is a ``TokenizedCache`` step that writes a seal record (its config is pure
    literals: the exact shard paths, counts, and loss mode) and whose function verifies the
    sealed shards exist. The data is never copied or recomputed; the record is what makes the
    source's cache addressable, and ``mixture`` reads it at run time.
    """
    return {
        name: ArtifactStep(
            name=name,
            version=EXP3_V5_VERSION,
            artifact_type=TokenizedCache,
            run=_verify_shards,
            build_config=lambda _ctx, _name=name: _seal_config(_name),
            deps=(exp3_v5_cache_catalog,),
        )
        for name in _SOURCE_NAMES
    }


def exp3_v5_seal_check() -> ArtifactStep[Artifact]:
    """Verify the sealed cache catalog against the pinned release digests and manifest.

    Reads the catalog once, checks its SHA-256, and cross-checks every source's row count,
    shard count, and loss mode against the committed seal manifest plus the release totals.
    Declare this handle among a consumer's dependencies so the seal is verified before the
    mix is used.
    """
    expected = {name: {"rows": s["rows"], "shards": len(s["shards"]), "loss_mode": s["loss_mode"]} for name, s in _SEALS.items()}
    totals = {
        "sources": len(_SEALS),
        "rows": sum(s["rows"] for s in _SEALS.values()),
        "input_ids_tokens": sum(s["input_ids_tokens"] for s in _SEALS.values()),
        "loss_weights_tokens": sum(s["loss_weights_tokens"] for s in _SEALS.values()),
    }

    def build_config(_ctx: StepContext) -> dict:
        return {
            "cache_catalog": {"uri": _CATALOG_URI, "sha256": _CATALOG_SHA256},
            "release": {"uri": _RELEASE_URI, "sha256": _RELEASE_SHA256},
            "cache_dir": _CACHE_ROOT,
            "expected_sources": expected,
            "expected_totals": totals,
        }

    def check(config: dict) -> None:
        raw = StoragePath(config["cache_catalog"]["uri"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != config["cache_catalog"]["sha256"]:
            raise ValueError(f"cache catalog digest differs from the release pin {config['cache_catalog']['sha256']}")
        catalog = json.loads(raw)["splits"]["train"]
        if set(catalog) != set(config["expected_sources"]):
            raise ValueError("cache catalog sources differ from the seal manifest")
        for name, entry in catalog.items():
            ledger = entry["ledger"]
            want = config["expected_sources"][name]
            if not ledger.get("is_finished"):
                raise ValueError(f"{name}: cache ledger is not finished")
            if ledger["total_num_rows"] != want["rows"]:
                raise ValueError(f"{name}: {ledger['total_num_rows']} rows, manifest pins {want['rows']}")
            if len(ledger["finished_shards"]) != want["shards"]:
                raise ValueError(f"{name}: {len(ledger['finished_shards'])} shards, manifest pins {want['shards']}")
            loss_mode = ledger.get("metadata", {}).get("preprocessor_metadata", {}).get("loss_mode")
            if loss_mode != want["loss_mode"]:
                raise ValueError(f"{name}: loss mode {loss_mode!r}, manifest pins {want['loss_mode']!r}")

    return ArtifactStep(
        name="sft-mixes/exp3-v5/seal-check",
        version=EXP3_V5_VERSION,
        artifact_type=Artifact,
        run=check,
        build_config=build_config,
        deps=(exp3_v5_release, exp3_v5_cache_catalog),
    )


def exp3_v5_mix_deps() -> tuple[ArtifactStep, ...]:
    """The dependencies a consumer of :func:`exp3_v5_data_config` must declare."""
    return (exp3_v5_release, exp3_v5_cache_catalog, exp3_v5_seal_check(), *exp3_v5_sources().values())


def exp3_v5_data_config(ctx: StepContext) -> LmDataConfig:
    """The ``LmDataConfig`` the Exp3 SFT run consumed, assembled through ``mixture``.

    Call this inside a consumer's ``build_config`` whose ``deps`` include
    :func:`exp3_v5_mix_deps`. ``mixture`` reads each source's seal record at run time (so the
    components carry the release's exact cache addresses, format, and packing) and the
    schedule, catalog, shuffle, and validation reservations are applied verbatim from the
    release document.
    """
    phase_weights = _SCHEDULE[0][1]
    train = {handle: phase_weights[handle.name] for handle in exp3_v5_sources().values()}
    cfg = mixture(ctx, train, shuffle=_DATA["shuffle"])
    return replace(
        cfg,
        train_weights=[(phase, weights) for phase, weights in _SCHEDULE],
        cache_catalog=ctx.artifact_path(exp3_v5_cache_catalog),
        num_validation_sequences=dict(_NUM_VALIDATION_SEQUENCES),
        mixture_block_size=_MIXTURE_BLOCK_SIZE,
        enforce_eos=_ENFORCE_EOS,
        auto_build_caches=_AUTO_BUILD_CACHES,
        cache_options=CacheOptions(batch_size=_CACHE_BATCH_SIZE),
        stop_strategy=_STOP_STRATEGY,
    )


@click.command()
@click.option("--run", "build", is_flag=True, help="Build the handles (default: only print the plan).")
@click.option("--only", help="Comma-separated subset of source names to select (default: all).")
@click.option("--max-concurrent", type=int, default=8, help="Max handles built concurrently.")
def cli(build: bool, only: str | None, max_concurrent: int) -> None:
    selected = exp3_v5_sources()
    if only is not None:
        keys = only.split(",")
        unknown = [key for key in keys if key not in selected]
        if unknown:
            raise click.BadParameter(f"unknown sources {unknown}; available: {sorted(selected)}", param_hint="--only")
        selected = {key: selected[key] for key in keys}
    handles = [*selected.values(), exp3_v5_seal_check()]
    if not build:
        for handle in handles:
            click.echo(lower(handle))
        return
    run(*handles, max_concurrent=max_concurrent)


if __name__ == "__main__":
    cli()

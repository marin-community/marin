# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The Exp3 SFT training mix — data release V5 — as a Marin artifact graph.

Exp3's production SFT stage (job ``/benfeuer/long-loop-exp3-2026.11.16.51``, coordinator
sign-off ``450bc8a4b24613c6b6f00e07da9b3953``) trained on data release V5 under source
revision ``0f5721bbb4953e98898c847896110dee32a5b293``. This module expresses that mix as
versioned handles so any run's mix is recoverable from its records:

- The release document and its cache catalog are adopted at their exact URIs, with the
  pinned SHA-256 digests recorded in the handles' configs.
- Each of the 434 mixture sources is a ``TokenizedCache`` seal step whose fingerprinted
  config pins its cache-catalog entry digest, row/token/shard counts, and loss mode. The
  record is the deliverable; no data is read or recomputed.
- :func:`exp3_v5_data_config` assembles the ``LmDataConfig`` through
  :func:`marin.experiment.data.mixture` with the release's weights, weight schedule
  (boundaries 0/250/500), feistel shuffle, cache catalog, and validation reservations.

``fixtures/`` carries the consumed ``data.json`` and the per-source seal manifest, each
verified against its pinned SHA-256 at import. The content-addressed shard paths live in
the SHA-pinned cache catalog, not in the repository; :func:`exp3_v5_seal_check` verifies
them at run time. Every seal record also pins the runtime's bounded-lookahead packing
(cap 256, lookahead 128) as provenance.
"""

import hashlib
import json
from dataclasses import replace
from functools import cache
from pathlib import Path

import click
from levanter.data.text.datasets import LmDataConfig
from levanter.store.cache import CacheOptions
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext, lower, run
from marin.experiment.data import mixture
from marin.processing.tokenize.tokenize import TokenizedCache
from rigging.filesystem.storage_path import StoragePath, prefix_join

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


def _load_release_data() -> dict:
    """The release's ``data.json``, byte-exact modulo the fixture's trailing newline byte."""
    raw = (_FIXTURES / "exp3_v5_data.json").read_bytes()
    content = raw[:-1] if raw.endswith(b"\n") else raw
    expected = _RELEASE["artifacts"]["data.json"]["sha256"]
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError(f"exp3_v5_data.json: content does not match the release member digest {expected}")
    return json.loads(content)


_RELEASE = _load_fixture("exp3_v5_release.json", _RELEASE_SHA256)
"""The byte-identical release document. Its ``artifacts`` pins every member's digest."""

_DATA = _load_release_data()
"""The release's ``data.json`` — the ``LmDataConfig`` the Exp3 SFT run consumed."""

_MANIFEST = _load_fixture(
    "exp3_v5_source_seals.json",
    "782a6b8b25b758a4a32f10e9f419841ae5e975890f6f63e21043c9f18e1d3782",
)
"""Per-source seal manifest: each source's catalog-entry digest (sorted-key canonical
JSON), row/token counts, shard count, and loss mode — derived from the two SHA-pinned
release members named in ``derived_from``."""

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
_SHAPES = {
    json.dumps({key: component[key] for key in _SHAPE_KEYS}, sort_keys=True)
    for component in _DATA["components"].values()
}
if len(_SHAPES) != 1:
    raise ValueError(f"release components do not share one shape: {_SHAPES}")
_SHAPE = json.loads(next(iter(_SHAPES)))
_PACK = _SHAPE["pack"]
_FLAT_CACHE = _SHAPE["flat_cache"]
_SPLIT = _SHAPE["split"]
_FORMAT = dict(_SHAPE["format"], type="prebuilt")
_PACKING_LOOKAHEAD = next(iter(_DATA["components"].values()))["packing_lookahead"]
assert _PACKING_LOOKAHEAD == 128  # the Exp3 runtime's bounded-lookahead packing, recorded as provenance
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
    """The fingerprinted seal record for one source, as literals; the entry digest pins
    its catalog entry — the ledger carrying the content-addressed shard paths."""
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
        "shards": seal["shards"],
        "entry_sha256": seal["entry_sha256"],
        "release": {"uri": _RELEASE_URI, "sha256": _RELEASE_SHA256},
        "cache_catalog": {"uri": _CATALOG_URI, "sha256": _CATALOG_SHA256},
    }


def _entry_digest(entry: dict) -> str:
    """The canonical digest of a cache-catalog entry (sorted-key JSON)."""
    return hashlib.sha256(json.dumps(entry, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _register_seal(_config: dict) -> None:
    """No-op: the seal record is the deliverable; ``exp3_v5_seal_check`` verifies content."""


def _verify_shard_paths(cache_dir: str, shards: list[str]) -> None:
    """Fail unless every content-addressed shard cache exists under ``cache_dir``."""
    missing = []
    for shard in shards:
        if not StoragePath(prefix_join(cache_dir, shard)).exists():
            missing.append(shard)
        if len(missing) >= 5:
            break
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} or more sealed shard caches are missing under {cache_dir} (first: {missing[0]})"
        )


@cache
def _exp3_v5_sources() -> dict[str, ArtifactStep[TokenizedCache]]:
    """One sealed-cache handle per release source, keyed by the mixture's component name.

    Private on purpose: each record pins the shared release cache root, so the per-source
    addresses resolve only through the catalog :func:`exp3_v5_data_config` attaches as
    ``cache_catalog``; mixing these handles directly would point every component at the root.
    """
    return {
        name: ArtifactStep(
            name=name,
            version=EXP3_V5_VERSION,
            artifact_type=TokenizedCache,
            run=_register_seal,
            build_config=lambda _ctx, _name=name: _seal_config(_name),
            deps=(exp3_v5_cache_catalog,),
        )
        for name in _SOURCE_NAMES
    }


def _verified_catalog(config: dict) -> dict:
    """The catalog's train split, after its digest and source set match the seal config."""
    raw = StoragePath(config["cache_catalog"]["uri"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != config["cache_catalog"]["sha256"]:
        raise ValueError(f"cache catalog digest differs from the release pin {config['cache_catalog']['sha256']}")
    catalog = json.loads(raw)["splits"]["train"]
    if set(catalog) != set(config["expected_sources"]):
        raise ValueError("cache catalog sources differ from the seal manifest")
    return catalog


def _check_source_ledger(name: str, entry: dict, want: dict) -> list[str]:
    """Check one source's ledger against its seal record; return its finished shards."""
    ledger = entry["ledger"]
    if not ledger.get("is_finished"):
        raise ValueError(f"{name}: cache ledger is not finished")
    if ledger["total_num_rows"] != want["rows"]:
        raise ValueError(f"{name}: {ledger['total_num_rows']} rows, manifest pins {want['rows']}")
    if len(ledger["finished_shards"]) != want["shards"]:
        raise ValueError(f"{name}: {len(ledger['finished_shards'])} shards, manifest pins {want['shards']}")
    if _entry_digest(entry) != want["entry_sha256"]:
        raise ValueError(f"{name}: catalog entry digest differs from the manifest pin")
    loss_mode = ledger.get("metadata", {}).get("preprocessor_metadata", {}).get("loss_mode")
    if loss_mode != want["loss_mode"]:
        raise ValueError(f"{name}: loss mode {loss_mode!r}, manifest pins {want['loss_mode']!r}")
    return ledger["finished_shards"]


@cache
def exp3_v5_seal_check() -> ArtifactStep[Artifact]:
    """Verify the sealed release: catalog SHA-256, per-source entry digests, row/shard
    counts, loss modes, and shard existence. Declare it among a consumer's dependencies
    so the seal is verified before the mix is used.
    """
    expected = {
        name: {
            "rows": s["rows"],
            "shards": s["shards"],
            "loss_mode": s["loss_mode"],
            "entry_sha256": s["entry_sha256"],
        }
        for name, s in _SEALS.items()
    }
    totals = {
        "sources": len(_SEALS),
        "rows": sum(s["rows"] for s in _SEALS.values()),
        "shards": sum(s["shards"] for s in _SEALS.values()),
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
        catalog = _verified_catalog(config)
        shards: list[str] = []
        for name, entry in catalog.items():
            shards.extend(_check_source_ledger(name, entry, config["expected_sources"][name]))
        observed = {
            "sources": len(catalog),
            "rows": sum(entry["ledger"]["total_num_rows"] for entry in catalog.values()),
            "shards": len(shards),
        }
        for key, value in observed.items():
            if value != config["expected_totals"][key]:
                raise ValueError(f"{key}: {value}, manifest pins {config['expected_totals'][key]}")
        _verify_shard_paths(config["cache_dir"], shards)

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
    return (exp3_v5_release, exp3_v5_cache_catalog, exp3_v5_seal_check(), *_exp3_v5_sources().values())


def exp3_v5_data_config(ctx: StepContext) -> LmDataConfig:
    """The ``LmDataConfig`` the Exp3 SFT run consumed.

    Call this inside a consumer's ``build_config`` whose ``deps`` include
    :func:`exp3_v5_mix_deps`, so the seal is verified before the mix resolves.
    """
    # Fails fast unless the consumer declared the seal check; exp3_v5_mix_deps() does.
    ctx.artifact_path(exp3_v5_seal_check())
    phase_weights = _SCHEDULE[0][1]
    train = {handle: phase_weights[handle.name] for handle in _exp3_v5_sources().values()}
    cfg = mixture(ctx, train, shuffle=_DATA["shuffle"])
    return replace(
        cfg,
        train_weights=[(phase, dict(weights)) for phase, weights in _SCHEDULE],
        cache_catalog=ctx.artifact_path(exp3_v5_cache_catalog),
        num_validation_sequences=dict(_NUM_VALIDATION_SEQUENCES),
        mixture_block_size=_MIXTURE_BLOCK_SIZE,
        enforce_eos=_ENFORCE_EOS,
        auto_build_caches=_AUTO_BUILD_CACHES,
        block_cross_document_attention=_DATA["block_cross_document_attention"],
        shuffle_before_trainval_split=_DATA["shuffle_before_trainval_split"],
        cache_options=CacheOptions(batch_size=_CACHE_BATCH_SIZE),
        stop_strategy=_STOP_STRATEGY,
    )


@click.command()
@click.option("--run", "build", is_flag=True, help="Build the handles (default: only print the plan).")
@click.option("--only", help="Comma-separated subset of source names to select (default: all).")
@click.option("--max-concurrent", type=int, default=8, help="Max handles built concurrently.")
def cli(build: bool, only: str | None, max_concurrent: int) -> None:
    selected = _exp3_v5_sources()
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

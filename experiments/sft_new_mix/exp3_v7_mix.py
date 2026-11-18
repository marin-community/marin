# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The Exp3 SFT training mix — data release V7 — as a Marin artifact graph.

Exp3's production SFT (job ``/benfeuer/long-loop-exp3-2026.11.16.51``) trained on data release
V5. Release V7 is a candidate for later long-loop runs: it matches V6 outside the two
comp-prog sources and changes only those two, to the judged-passing option-G rule. This
module expresses that exact mix as versioned handles so any run's mix is recoverable from its
records:

- **Release seal.** The release manifest and its cache catalog are adopted at their exact
  URIs, with the pinned SHA-256 digests recorded in the handles' configs. The manifest is the
  standalone publication seal written by the V7 publication step; its ``artifacts`` map pins
  every member digest.
- **One sealed-cache handle per source.** Each of the 434 mixture sources is a
  ``TokenizedCache`` step whose fingerprinted config pins the digest of the source's
  cache-catalog entry — the ledger that carries the source's exact content-addressed shard
  caches — plus its row and token counts, shard count, and loss mode. The step's record is
  the deliverable; no data is read or recomputed.
- **The mixture.** :func:`exp3_v7_data_config` assembles the ``LmDataConfig`` through
  :func:`marin.experiment.data.mixture`: the 434 sources at their natural weights, the
  three-phase weight schedule (boundaries 0/250/500), the full-permutation feistel shuffle,
  the release's cache catalog, and the per-source IID validation reservation.

The V7 comp-prog caches are served from the cw-us-east-08a mirror of the post-Atlas caches
(object-for-object, SHA-256 verified per object); the other forty-eight sources reuse the
V6 sealed caches byte-identically. The training run's consumed data config will be recorded
in a fixture the same way V5's was; the component-by-component equivalence test covers the
serialization as released. The bounded-lookahead packing the Exp3 runtime used (cap 256,
lookahead 128) is recorded in every seal record as provenance; a consumer packs with its
own policy.

The release is frozen data, not launch approval: the publication records that native consumer
resolve and a training admission are still required. The seal check verifies the sealed
caches exist and match the manifest; it does not qualify the data for a run.
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
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath, prefix_join

EXP3_V7_VERSION = "2026.11.18"
"""The release's creation date; the identity version of every handle in this mix."""

_FIXTURES = Path(__file__).parent / "fixtures"

_RELEASE_URI = (
    "s3://marin-us-east-08a/marin/users/benfeuer/exp3-data-mix/2026.11.18-v7-option-g" "/release-v7/manifest.json"
)
_RELEASE_SHA256 = "0278654468122a87891d1526bd05d95bfb995234e33b9eaa1dbe393e36c84a8f"
"""The publication manifest the V7 release seals its members with."""


def _load_fixture(filename: str, sha256: str) -> dict:
    raw = (_FIXTURES / filename).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError(f"{filename}: fixture does not match the pinned digest {sha256}")
    return json.loads(raw)


def _load_member_fixture(filename: str, expected_sha256: str) -> dict:
    """The release member bytes, byte-exact after the fixture's trailing newline.

    The newline hook requires a terminated last line, so the fixture stores the member
    plus one newline byte; the digest check covers the member's exact bytes.
    """
    raw = (_FIXTURES / filename).read_bytes()
    content = raw[:-1] if raw.endswith(b"\n") else raw
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError(f"{filename}: content does not match the release member digest {expected_sha256}")
    return json.loads(content)


_RELEASE = _load_member_fixture("exp3_v7_release.json", _RELEASE_SHA256)
"""The byte-identical publication manifest. Its ``artifacts`` pins every member's digest."""

_DATA = _load_member_fixture("exp3_v7_data.json", _RELEASE["artifacts"]["data.json"]["sha256"])
"""The release's ``data.json`` — the ``LmDataConfig`` a long-loop run would consume."""

_MANIFEST = _load_fixture(
    "exp3_v7_source_seals.json",
    "996129b289cff44f8eb0a114799fc3efcd8d9aa41a921f44733129eb49125435",
)
"""Per-source seal manifest derived from the release's cache catalog.

Each source pins its catalog-entry digest (the canonical entry JSON, sorted keys), row
and token counts, shard count, and loss mode. The manifest is a pure function of the two
SHA-pinned release members named in its ``derived_from`` block, so regenerating it from
an unchanged catalog is byte-stable."""

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
"""Cache-read batch size the Exp3 launcher pinned. Not a release field; recorded from the
V6/Exp3 consumed config this release inherits unchanged."""
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
_PACKING_LOOKAHEAD = int(_RELEASE["geometry"]["lookahead"])
assert _PACKING_LOOKAHEAD == 128  # the Exp3 runtime's bounded-lookahead packing, recorded as provenance
assert _PACK == int(_RELEASE["geometry"]["packing_cap"]) == 256
assert _MANIFEST["geometry"]["cache_dir"] == _CACHE_ROOT
assert _MANIFEST["geometry"]["pack"] == _PACK and _MANIFEST["geometry"]["packing_lookahead"] == _PACKING_LOOKAHEAD

_SOURCE_NAMES = tuple(sorted(_DATA["components"]))
if set(_SOURCE_NAMES) != set(_SEALS) or set(_SOURCE_NAMES) != set(_SCHEDULE[0][1]):
    raise ValueError("release components, seal manifest, and weight schedule name different source sets")

exp3_v7_release = ArtifactStep.adopt(
    "sft-mixes/exp3-v7/release",
    EXP3_V7_VERSION,
    _RELEASE_URI,
    config={"sha256": _RELEASE_SHA256, "members": dict(_RELEASE["artifacts"])},
)
"""The adopted release manifest: its exact URI, with every member digest recorded."""

exp3_v7_cache_catalog = ArtifactStep.adopt(
    "sft-mixes/exp3-v7/cache-catalog",
    EXP3_V7_VERSION,
    _CATALOG_URI,
    config={"sha256": _CATALOG_SHA256},
)
"""The adopted cache catalog: the authoritative per-source ledgers behind ``cache_catalog``."""


def _seal_config(name: str) -> dict:
    """The fingerprinted seal record for one source: its exact sealed cache, as literals.

    The entry digest pins the source's catalog entry — the ledger that carries the
    content-addressed shard paths, per-shard row counts, and loss metadata.
    """
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
    """Register the sealed cache. The step's record is the deliverable; the data is
    neither read nor moved. ``exp3_v7_seal_check`` verifies the sealed content."""


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
def _exp3_v7_sources() -> dict[str, ArtifactStep[TokenizedCache]]:
    """One sealed-cache handle per release source, keyed by the mixture's component name.

    Private on purpose: each seal record pins the shared release cache root, so the
    per-source cache addresses only resolve through the catalog that
    :func:`exp3_v7_data_config` attaches as ``cache_catalog``. Mixing these handles
    directly (without that catalog) would point every component at the root.

    Each handle is a ``TokenizedCache`` step that writes a seal record; its config is
    pure literals (the entry digest, counts, and loss mode), so the fingerprint pins the
    exact sealed content. The data is never copied or recomputed; the record is what
    makes the source's cache addressable, and ``mixture`` reads it at run time.
    """
    return {
        name: ArtifactStep(
            name=name,
            version=EXP3_V7_VERSION,
            artifact_type=TokenizedCache,
            run=_register_seal,
            build_config=lambda _ctx, _name=name: _seal_config(_name),
            deps=(exp3_v7_cache_catalog,),
        )
        for name in _SOURCE_NAMES
    }


def exp3_v7_seal_check() -> ArtifactStep[Artifact]:
    """Verify the sealed release against the pinned digests and the seal manifest.

    Reads the cache catalog once, checks its SHA-256 and every source's entry digest, and
    cross-checks row counts, shard counts, and loss modes against the manifest and the
    release totals; then verifies that every content-addressed shard cache exists. Declare
    this handle among a consumer's dependencies so the seal is verified before the mix is
    used.
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
        "input_ids_tokens": sum(s["input_ids_tokens"] for s in _SEALS.values()),
        "loss_weights_tokens": sum(s["loss_weights_tokens"] for s in _SEALS.values()),
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
        raw = StoragePath(config["cache_catalog"]["uri"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != config["cache_catalog"]["sha256"]:
            raise ValueError(f"cache catalog digest differs from the release pin {config['cache_catalog']['sha256']}")
        catalog = json.loads(raw)["splits"]["train"]
        if set(catalog) != set(config["expected_sources"]):
            raise ValueError("cache catalog sources differ from the seal manifest")
        shards: list[str] = []
        for name, entry in catalog.items():
            ledger = entry["ledger"]
            want = config["expected_sources"][name]
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
            shards.extend(ledger["finished_shards"])
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
        name="sft-mixes/exp3-v7/seal-check",
        version=EXP3_V7_VERSION,
        artifact_type=Artifact,
        run=check,
        build_config=build_config,
        deps=(exp3_v7_release, exp3_v7_cache_catalog),
    )


def exp3_v7_mix_deps() -> tuple[ArtifactStep, ...]:
    """The dependencies a consumer of :func:`exp3_v7_data_config` must declare."""
    return (exp3_v7_release, exp3_v7_cache_catalog, exp3_v7_seal_check(), *_exp3_v7_sources().values())


def exp3_v7_data_config(ctx: StepContext) -> LmDataConfig:
    """The ``LmDataConfig`` a long-loop run on V7 would consume, assembled through ``mixture``.

    Call this inside a consumer's ``build_config`` whose ``deps`` include
    :func:`exp3_v7_mix_deps`. ``mixture`` reads each source's seal record at run time for
    its format and packing. The release catalog supplies the per-source cache addresses;
    the schedule, shuffle, and validation reservations come from the release document.
    """
    phase_weights = _SCHEDULE[0][1]
    train = {handle: phase_weights[handle.name] for handle in _exp3_v7_sources().values()}
    cfg = mixture(ctx, train, shuffle=_DATA["shuffle"])
    return replace(
        cfg,
        train_weights=[(phase, weights) for phase, weights in _SCHEDULE],
        cache_catalog=ctx.artifact_path(exp3_v7_cache_catalog),
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
    configure_coreweave_s3()
    selected = _exp3_v7_sources()
    if only is not None:
        keys = only.split(",")
        unknown = [key for key in keys if key not in selected]
        if unknown:
            raise click.BadParameter(f"unknown sources {unknown}; available: {sorted(selected)}", param_hint="--only")
        selected = {key: selected[key] for key in keys}
    handles = [*selected.values(), exp3_v7_seal_check()]
    if not build:
        for handle in handles:
            click.echo(lower(handle))
        return
    run(*handles, max_concurrent=max_concurrent)


if __name__ == "__main__":
    cli()

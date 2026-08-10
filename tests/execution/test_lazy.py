# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behaviour tests for the artifact-step runtime (``marin.execution.lazy``).

Drives the hardest path — a config that references its own output path AND a dependency's
output path — end-to-end through the real ``StepRunner``, plus the ``apply`` sugar, the
``resolve`` round-trip, and the ``name``/``version`` grammar.
"""

import os
from dataclasses import dataclass

import pytest
from fray.types import ResourceConfig
from marin.execution.artifact import Artifact, ArtifactTypeMismatchError
from marin.execution.lazy import OUT, ArtifactStep, StepContext, apply, lower, materialized_config, resolve, run
from marin.execution.remote import remote

# --- Toy configs/fns standing in for tokenize + train --------------------------


@dataclass(frozen=True)
class TokenizeCfg:
    out: str  # self-reference: the producer writes here
    source: str
    tokenizer: str


@dataclass(frozen=True)
class TrainCfg:
    out: str  # self-reference
    data: str  # cross-step reference: the dep's output path
    lr: float
    steps: int


class Tokens(Artifact):
    out: str
    source: str
    tokenizer: str
    kind: str = "tokens"


class Ckpt(Artifact):
    out: str
    data: str
    lr: float
    steps: int
    kind: str = "ckpt"


def _make_tokens(config: TokenizeCfg) -> Tokens:
    return Tokens(out=config.out, source=config.source, tokenizer=config.tokenizer)


def _make_ckpt(config: TrainCfg) -> Ckpt:
    return Ckpt(out=config.out, data=config.data, lr=config.lr, steps=config.steps)


# --- How it looks: the user-facing experiment ----------------------------------


def dclm_tokens() -> ArtifactStep[Tokens]:
    return ArtifactStep(
        name="datasets/dclm_tokens",
        version="2026.06.25",
        artifact_type=Tokens,
        run=_make_tokens,
        build_config=lambda ctx: TokenizeCfg(out=ctx.output_path, source="gs://raw/dclm", tokenizer="llama3"),
    )


def dclm_1b(*, lr: float = 3e-3) -> ArtifactStep[Ckpt]:
    data = dclm_tokens()
    return ArtifactStep(
        name="checkpoints/dclm_1b",
        version="2026.06.28",
        artifact_type=Ckpt,
        run=_make_ckpt,
        build_config=lambda ctx: TrainCfg(out=ctx.output_path, data=ctx.artifact_path(data), lr=lr, steps=54931),
        deps=(data,),
    )


# --- Identity vs execution -----------------------------------------------------


def test_explicit_version_paths_and_cross_step_resolution():
    prefix = "gs://marin-spike"
    train_cfg = materialized_config(dclm_1b(), prefix)
    tokens_cfg = materialized_config(dclm_tokens(), prefix)

    # Explicit name@version addressing, no hash.
    assert tokens_cfg.out == "gs://marin-spike/datasets/dclm_tokens/2026.06.25"
    assert train_cfg.out == "gs://marin-spike/checkpoints/dclm_1b/2026.06.28"
    # The hard part: the consumer's config resolves to the producer's output path.
    assert train_cfg.data == tokens_cfg.out


def test_trailing_slash_prefix_writer_and_reader_agree(monkeypatch):
    """Regression for marin-community/marin#6904: with a trailing-slash ``MARIN_PREFIX``
    the lowered step wrote to ``…marin//name/version`` while consumers resolved
    ``…marin/name/version`` — a different object-store key, so the reader found no record."""
    monkeypatch.setenv("MARIN_PREFIX", "s3://marin-na/marin/")
    handle = dclm_tokens()
    spec = lower(handle)
    assert spec.output_path == handle.path()
    assert spec.output_path == "s3://marin-na/marin/datasets/dclm_tokens/2026.06.25"


def test_trailing_slash_prefix_resolves_relative_pin(monkeypatch):
    """Same invariant for a relatively pinned handle (the ``override_path`` branch)."""
    monkeypatch.setenv("MARIN_PREFIX", "s3://marin-na/marin/")
    pinned = apply("data/pinned", _stage, version="2026.06.28", pin="pinned/loc", output_path=OUT)
    assert lower(pinned).output_path == pinned.path()
    assert pinned.path() == "s3://marin-na/marin/pinned/loc"


def test_fingerprint_is_prefix_independent_but_drift_sensitive():
    base = dclm_1b().fingerprint()
    # Region/prefix must not change identity.
    assert base == dclm_1b().fingerprint()
    # A config value change must change the fingerprint (local drift).
    assert dclm_1b(lr=1e-3).fingerprint() != base


def test_dep_version_bump_changes_consumer_fingerprint():
    """Bumping a dependency's version changes the consumer's fingerprint, because
    build_config embeds ctx.artifact_path(dep) = name@version."""
    before = dclm_1b().fingerprint()

    def bumped() -> ArtifactStep[Ckpt]:
        data = ArtifactStep(
            name="datasets/dclm_tokens",
            version="2026.07.01",  # bumped
            artifact_type=Tokens,
            run=_make_tokens,
            build_config=lambda ctx: TokenizeCfg(out=ctx.output_path, source="gs://raw/dclm", tokenizer="llama3"),
        )
        return ArtifactStep(
            name="checkpoints/dclm_1b",
            version="2026.06.28",
            artifact_type=Ckpt,
            run=_make_ckpt,
            build_config=lambda ctx: TrainCfg(out=ctx.output_path, data=ctx.artifact_path(data), lr=3e-3, steps=54931),
            deps=(data,),
        )

    assert bumped().fingerprint() != before


def test_region_is_pulled_live_at_run_time():
    """The region is a runtime-only attribute: ``build_config`` reads it from the
    ``StepContext``, so one step resolves against whatever region it runs in."""

    def build_config(ctx: StepContext) -> TokenizeCfg:
        return TokenizeCfg(out=ctx.output_path, source=f"gs://raw-{ctx.region}/dclm", tokenizer="llama3")

    east = build_config(StepContext.for_run(output_path="o", prefix="p", region="us-east5"))
    central = build_config(StepContext.for_run(output_path="o", prefix="p", region="us-central2"))
    assert (east.source, central.source) == ("gs://raw-us-east5/dclm", "gs://raw-us-central2/dclm")


def test_resources_do_not_affect_identity():
    """Compute rides with the run fn (``remote(fn, resources=…)``), never the config or the
    graph node, so changing the TPU a step runs on must not change its fingerprint."""

    def on(resources: ResourceConfig) -> ArtifactStep[Artifact]:
        return ArtifactStep(
            name="checkpoints/dclm_1b",
            version="2026.06.28",
            artifact_type=Artifact,
            run=remote(_make_ckpt, resources=resources),
            build_config=lambda ctx: TrainCfg(out=ctx.output_path, data="gs://d", lr=3e-3, steps=10),
        )

    assert on(ResourceConfig.with_tpu("v5p-8")).fingerprint() == on(ResourceConfig.with_tpu("v6e-8")).fingerprint()


def test_runtime_arg_is_live_at_run_but_not_in_identity():
    """A runtime arg is declared on the step and pulled via ``ctx.runtime_arg()``: it reaches the
    config at run time but is a ``<key>`` placeholder at fingerprint time."""

    def on(tpu: str) -> ArtifactStep[Artifact]:
        return ArtifactStep(
            name="checkpoints/dclm_1b",
            version="2026.06.28",
            artifact_type=Artifact,
            run=_make_ckpt,
            build_config=lambda ctx: TrainCfg(out=ctx.output_path, data=ctx.runtime_arg("tpu"), lr=3e-3, steps=10),
            runtime_args={"tpu": tpu},
        )

    # Live at run time: the materialized config carries the declared value.
    assert materialized_config(on("v5p-8"), "gs://b").data == "v5p-8"
    # ...but excluded from identity: a different TPU is the same artifact.
    assert on("v5p-8").fingerprint() == on("v6e-8").fingerprint()


def test_pulling_an_undeclared_runtime_arg_fails_loudly():
    art = ArtifactStep(
        name="checkpoints/dclm_1b",
        version="2026.06.28",
        artifact_type=Artifact,
        run=_make_ckpt,
        build_config=lambda ctx: TrainCfg(out=ctx.output_path, data=ctx.runtime_arg("tpu"), lr=3e-3, steps=10),
    )
    with pytest.raises(KeyError, match="tpu"):
        art.fingerprint()


# --- name/version grammar ------------------------------------------------------


@pytest.mark.parametrize("bad", ["", "/leading", "trailing/", "has/../dots", "gs://scheme/x"])
def test_malformed_name_is_rejected(bad):
    with pytest.raises(ValueError):
        ArtifactStep(
            name=bad,
            version="2026.06.28",
            artifact_type=Artifact,
            run=lambda c: None,
            build_config=lambda ctx: None,
        )


@pytest.mark.parametrize("bad", ["", "v1", "llama3", "20260628", "2026-06-28"])
def test_malformed_version_is_rejected(bad):
    """A version must be a calendar version ``YYYY.MM.DD[.N]`` or ``dev``/``<label>-dev``."""
    with pytest.raises(ValueError):
        ArtifactStep(
            name="ok",
            version=bad,
            artifact_type=Artifact,
            run=lambda c: None,
            build_config=lambda ctx: None,
        )


# --- end to end ----------------------------------------------------------------


def test_end_to_end_through_step_runner(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))

    run(dclm_1b())

    tokens_path = f"{tmp_path}/datasets/dclm_tokens/2026.06.25"
    ckpt_path = f"{tmp_path}/checkpoints/dclm_1b/2026.06.28"

    saved_ckpt = Ckpt.raw_load(ckpt_path)
    assert saved_ckpt.kind == "ckpt"
    # The training run actually received the dependency's resolved output path.
    assert saved_ckpt.data == tokens_path

    assert Tokens.raw_load(tokens_path).kind == "tokens"

    # Re-running is a cache hit (no error, idempotent).
    run(dclm_1b())


def test_resolve_round_trips_a_value_artifact(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))

    result = resolve(dclm_tokens())
    assert isinstance(result, Tokens)
    assert (result.kind, result.tokenizer) == ("tokens", "llama3")
    assert result.path == f"{tmp_path}/datasets/dclm_tokens/2026.06.25"


def test_run_returns_loaded_artifacts(tmp_path, monkeypatch):
    """run() builds and hands back each top-level handle's loaded, typed artifact, in order."""
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    results = run(dclm_tokens())
    assert len(results) == 1
    assert isinstance(results[0], Tokens)
    assert (results[0].kind, results[0].tokenizer) == ("tokens", "llama3")


class OtherValue(Artifact):
    note: str = ""


def test_resolve_raises_on_result_type_drift(tmp_path, monkeypatch):
    """A value artifact whose type changed under a reused version is a hard error at load."""
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    resolve(dclm_tokens())  # records artifact type Tokens at this address

    drifted = ArtifactStep(
        name="datasets/dclm_tokens",
        version="2026.06.25",
        artifact_type=OtherValue,
        run=lambda cfg: OtherValue(),
        build_config=lambda ctx: {"out": ctx.output_path},
    )
    with pytest.raises(ArtifactTypeMismatchError):
        resolve(drifted)


# --- ctx.resolved: reading a dependency's value, and the declared-deps gate ----


class TokenizerEcho(Artifact):
    tokenizer: str = ""


def _echo_tokenizer(config: dict) -> TokenizerEcho:
    return TokenizerEcho(tokenizer=config["tokenizer"])


def echo_consumer() -> ArtifactStep[TokenizerEcho]:
    """A consumer that reads its dependency's recorded *value* (its tokenizer), not just its path."""
    tok = dclm_tokens()

    def build_config(ctx: StepContext) -> dict:
        if ctx.is_fingerprint:
            return {"tokenizer": "<tokenizer>"}
        return {"tokenizer": ctx.resolved(tok).tokenizer}

    return ArtifactStep(
        name="checkpoints/echo",
        version="2026.06.28",
        artifact_type=TokenizerEcho,
        run=_echo_tokenizer,
        build_config=build_config,
        deps=(tok,),
    )


def test_consumer_reads_dependency_value_via_ctx_resolved(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    run(echo_consumer())
    saved = TokenizerEcho.raw_load(f"{tmp_path}/checkpoints/echo/2026.06.28")
    # The dep's tokenizer flowed from its record into the consumer's output.
    assert saved.tokenizer == "llama3"


def test_resolved_caches_the_loaded_artifact(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    tok = dclm_tokens()
    run(tok)  # materialize the dep so its record exists
    ctx = StepContext.for_run(output_path="o", prefix=str(tmp_path), deps=(tok,))
    first = ctx.resolved(tok)
    # Resolving the same dep again returns the cached object — one store read, not N.
    assert ctx.resolved(tok) is first
    assert first.tokenizer == "llama3"


def test_resolved_unavailable_at_fingerprint_time():
    tok = dclm_tokens()
    ctx = StepContext.for_fingerprint(deps=(tok,))
    with pytest.raises(ValueError, match="fingerprint time"):
        ctx.resolved(tok)


def test_undeclared_dependency_is_rejected():
    """Resolving a handle the step did not list in ``deps`` is a fail-fast bug, not a silent read."""
    orphan = dclm_tokens()
    forgot_to_declare = ArtifactStep(
        name="checkpoints/bad",
        version="2026.06.28",
        artifact_type=Ckpt,
        run=_make_ckpt,
        build_config=lambda ctx: TrainCfg(out=ctx.output_path, data=ctx.artifact_path(orphan), lr=1e-3, steps=1),
        deps=(),
    )
    with pytest.raises(ValueError, match="not a declared dependency"):
        materialized_config(forgot_to_declare, "gs://b")


# --- apply: the keyword-input single-step builder ------------------------------


def _stage(output_path: str) -> None:
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "data.txt"), "w") as f:
        f.write("staged")


def _transform(input_path: str, output_path: str) -> None:
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(input_path, "data.txt")) as f:
        content = f.read()
    with open(os.path.join(output_path, "data.txt"), "w") as f:
        f.write(content.upper())


def test_apply_chains_steps_by_direct_call(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))

    staged = apply("raw/massive", _stage, version="2026.06.28", output_path=OUT)
    parts = apply("data/massive", _transform, version="2026.06.28", input_path=staged, output_path=OUT)
    run(parts)

    with open(tmp_path / "data" / "massive" / "2026.06.28" / "data.txt") as f:
        assert f.read() == "STAGED"


def test_apply_lazy_dep_and_literal_both_bear_identity():
    a = apply("data/x", _transform, version="2026.06.28", input_path="p", output_path=OUT)
    b = apply("data/x", _transform, version="2026.06.28", input_path="q", output_path=OUT)
    # A literal input change forks identity.
    assert a.fingerprint() != b.fingerprint()


def test_apply_collects_lazy_deps():
    staged = apply("raw/massive", _stage, version="2026.06.28", output_path=OUT)
    parts = apply("data/massive", _transform, version="2026.06.28", input_path=staged, output_path=OUT)
    assert parts.deps == (staged,)


def test_apply_rejects_unbindable_inputs():
    def f(a):
        return a

    with pytest.raises(TypeError):
        apply("x/y", f, version="2026.06.28", a=1, b=2)

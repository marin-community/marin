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
from marin.execution.lazy import OUT, ArtifactStep, StepContext, apply, materialized_config, run
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

    saved_ckpt = Ckpt.load(ckpt_path)
    assert saved_ckpt.kind == "ckpt"
    # The training run actually received the dependency's resolved output path.
    assert saved_ckpt.data == tokens_path

    assert Tokens.load(tokens_path).kind == "tokens"

    # Re-running is a cache hit (no error, idempotent).
    run(dclm_1b())


def test_resolve_round_trips_a_value_artifact(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))

    result = dclm_tokens().resolve()
    assert isinstance(result, Tokens)
    assert (result.kind, result.tokenizer) == ("tokens", "llama3")
    assert result.path == f"{tmp_path}/datasets/dclm_tokens/2026.06.25"


class OtherValue(Artifact):
    note: str = ""


def test_resolve_raises_on_result_type_drift(tmp_path, monkeypatch):
    """A value artifact whose type changed under a reused version is a hard error at load."""
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    dclm_tokens().resolve()  # records artifact type Tokens at this address

    drifted = ArtifactStep(
        name="datasets/dclm_tokens",
        version="2026.06.25",
        artifact_type=OtherValue,
        run=lambda cfg: OtherValue(),
        build_config=lambda ctx: {"out": ctx.output_path},
    )
    with pytest.raises(ArtifactTypeMismatchError):
        drifted.resolve()


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

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behaviour tests for the lazy-artifact runtime (``marin.execution.lazy``).

Drives the hardest path — a config that references its own output path AND a dependency's
output path — end-to-end through the real ``StepRunner``, plus the ``apply`` sugar, the
``resolve`` round-trip, and the ``name``/``version`` grammar.
"""

import os
from dataclasses import dataclass

import pytest
from fray.types import ResourceConfig
from marin.execution.artifact import Artifact, ArtifactTypeMismatchError, Checkpoint, JsonArtifact
from marin.execution.lazy import OUT, Lazy, Recipe, RunContext, apply, lower, materialized_config, resolve, run
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


class Tokens(JsonArtifact):
    out: str
    source: str
    tokenizer: str
    kind: str = "tokens"


class Ckpt(JsonArtifact):
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


def dclm_tokens() -> Lazy[Tokens]:
    return Lazy(
        name="datasets/dclm_tokens",
        version="2026.06.25",
        result_type=Tokens,
        recipe=Recipe(
            fn=_make_tokens,
            build_config=lambda ctx: TokenizeCfg(out=ctx.out, source="gs://raw/dclm", tokenizer="llama3"),
        ),
    )


def dclm_1b(*, lr: float = 3e-3) -> Lazy[Ckpt]:
    data = dclm_tokens()
    return Lazy(
        name="checkpoints/dclm_1b",
        version="v3",
        result_type=Ckpt,
        recipe=Recipe(
            fn=_make_ckpt,
            build_config=lambda ctx: TrainCfg(out=ctx.out, data=ctx.path(data), lr=lr, steps=54931),
            deps=(data,),
        ),
    )


# --- Identity vs execution -----------------------------------------------------


def test_explicit_version_paths_and_cross_step_resolution():
    prefix = "gs://marin-spike"
    train_cfg = materialized_config(dclm_1b(), prefix)
    tokens_cfg = materialized_config(dclm_tokens(), prefix)

    # Explicit name@version addressing, no hash.
    assert tokens_cfg.out == "gs://marin-spike/datasets/dclm_tokens/2026.06.25"
    assert train_cfg.out == "gs://marin-spike/checkpoints/dclm_1b/v3"
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
    build_config embeds ctx.path(dep) = name@version."""
    before = dclm_1b().fingerprint()

    def bumped() -> Lazy[Ckpt]:
        data = Lazy(
            name="datasets/dclm_tokens",
            version="2026.07.01",  # bumped
            result_type=Tokens,
            recipe=Recipe(
                fn=_make_tokens,
                build_config=lambda ctx: TokenizeCfg(out=ctx.out, source="gs://raw/dclm", tokenizer="llama3"),
            ),
        )
        return Lazy(
            name="checkpoints/dclm_1b",
            version="v3",
            result_type=Ckpt,
            recipe=Recipe(
                fn=_make_ckpt,
                build_config=lambda ctx: TrainCfg(out=ctx.out, data=ctx.path(data), lr=3e-3, steps=54931),
                deps=(data,),
            ),
        )

    assert bumped().fingerprint() != before


def test_region_is_pulled_live_at_run_time():
    """The region is a runtime-only attribute: ``build_config`` reads it from the
    ``RunContext``, so one recipe resolves against whatever region the step runs in."""

    def build_config(ctx: RunContext) -> TokenizeCfg:
        return TokenizeCfg(out=ctx.out, source=f"gs://raw-{ctx.region}/dclm", tokenizer="llama3")

    recipe = Recipe(fn=_make_tokens, build_config=build_config)
    east = recipe.build_config(RunContext.for_run(out="o", prefix="p", region="us-east5"))
    central = recipe.build_config(RunContext.for_run(out="o", prefix="p", region="us-central2"))
    assert (east.source, central.source) == ("gs://raw-us-east5/dclm", "gs://raw-us-central2/dclm")


def test_resources_do_not_affect_identity():
    """Compute rides with the fn (``remote(fn, resources=…)``), never the config or the
    graph node, so changing the TPU a step runs on must not change its fingerprint."""

    def on(resources: ResourceConfig) -> Lazy[Checkpoint]:
        return Lazy(
            name="checkpoints/dclm_1b",
            version="v3",
            result_type=Checkpoint,
            recipe=Recipe(
                fn=remote(_make_ckpt, resources=resources),
                build_config=lambda ctx: TrainCfg(out=ctx.out, data="gs://d", lr=3e-3, steps=10),
            ),
        )

    assert on(ResourceConfig.with_tpu("v5p-8")).fingerprint() == on(ResourceConfig.with_tpu("v6e-8")).fingerprint()


def test_run_arg_is_live_at_run_but_not_in_identity():
    """A run-arg is declared on the recipe and pulled via ``ctx.run_arg()``: it reaches the
    config at run time but is a ``<key>`` placeholder at fingerprint time."""

    def on(tpu: str) -> Lazy[Checkpoint]:
        return Lazy(
            name="checkpoints/dclm_1b",
            version="v3",
            result_type=Checkpoint,
            recipe=Recipe(
                fn=_make_ckpt,
                build_config=lambda ctx: TrainCfg(out=ctx.out, data=ctx.run_arg("tpu"), lr=3e-3, steps=10),
                run_args={"tpu": tpu},
            ),
        )

    # Live at run time: the materialized config carries the declared value.
    assert materialized_config(on("v5p-8"), "gs://b").data == "v5p-8"
    # ...but excluded from identity: a different TPU is the same artifact.
    assert on("v5p-8").fingerprint() == on("v6e-8").fingerprint()


def test_pulling_an_undeclared_run_arg_fails_loudly():
    art = Lazy(
        name="checkpoints/dclm_1b",
        version="v3",
        result_type=Checkpoint,
        recipe=Recipe(
            fn=_make_ckpt,
            build_config=lambda ctx: TrainCfg(out=ctx.out, data=ctx.run_arg("tpu"), lr=3e-3, steps=10),
        ),
    )
    with pytest.raises(KeyError, match="tpu"):
        art.fingerprint()


# --- name/version grammar ------------------------------------------------------


@pytest.mark.parametrize("bad", ["", "/leading", "trailing/", "has/../dots", "gs://scheme/x"])
def test_malformed_name_is_rejected(bad):
    with pytest.raises(ValueError):
        Lazy(
            name=bad,
            version="v1",
            result_type=Artifact,
            recipe=Recipe(fn=lambda c: None, build_config=lambda ctx: None),
        )


def test_malformed_version_is_rejected():
    with pytest.raises(ValueError):
        Lazy(
            name="ok",
            version="",
            result_type=Artifact,
            recipe=Recipe(fn=lambda c: None, build_config=lambda ctx: None),
        )


# --- lower-time guards ---------------------------------------------------------


def test_json_artifact_with_remote_fn_is_rejected_at_lower():
    """A value artifact must run inline — a Fray job returns nothing to the caller."""
    handle = Lazy(
        name="datasets/dclm_tokens",
        version="v1",
        result_type=Tokens,
        recipe=Recipe(
            fn=remote(_make_tokens, resources=ResourceConfig.with_cpu()),
            build_config=lambda ctx: TokenizeCfg(out=ctx.out, source="s", tokenizer="t"),
        ),
    )
    with pytest.raises(ValueError, match="remote"):
        lower(handle)


# --- end to end ----------------------------------------------------------------


def test_end_to_end_through_step_runner(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))

    run(dclm_1b())

    tokens_path = f"{tmp_path}/datasets/dclm_tokens/2026.06.25"
    ckpt_path = f"{tmp_path}/checkpoints/dclm_1b/v3"

    saved_ckpt = Ckpt.load(ckpt_path)
    assert saved_ckpt.kind == "ckpt"
    # The training run actually received the dependency's resolved output path.
    assert saved_ckpt.data == tokens_path

    assert Tokens.load(tokens_path).kind == "tokens"

    # Re-running is a cache hit (no error, idempotent).
    run(dclm_1b())


def test_resolve_round_trips_a_json_artifact(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))

    result = resolve(dclm_tokens())
    assert isinstance(result, Tokens)
    assert (result.kind, result.tokenizer) == ("tokens", "llama3")
    assert result.path == f"{tmp_path}/datasets/dclm_tokens/2026.06.25"


class OtherValue(JsonArtifact):
    note: str = ""


def test_resolve_raises_on_result_type_drift(tmp_path, monkeypatch):
    """A value artifact whose type changed under a reused version is a hard error at load."""
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    resolve(dclm_tokens())  # records result_type Tokens at this address

    drifted = Lazy(
        name="datasets/dclm_tokens",
        version="2026.06.25",
        result_type=OtherValue,
        recipe=Recipe(fn=lambda cfg: OtherValue(), build_config=lambda ctx: {"out": ctx.out}),
    )
    with pytest.raises(ArtifactTypeMismatchError):
        resolve(drifted)


# --- apply: the generic single-step builder ------------------------------------


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

    staged = apply("raw/massive", _stage, output_path=OUT)
    parts = apply("data/massive", _transform, input_path=staged, output_path=OUT)
    run(parts)

    with open(tmp_path / "data" / "massive" / "v1" / "data.txt") as f:
        assert f.read() == "STAGED"


def test_apply_lazy_dep_and_literal_both_bear_identity():
    a = apply("data/x", _transform, input_path="p", output_path=OUT)
    b = apply("data/x", _transform, input_path="q", output_path=OUT)
    # A literal input change forks identity.
    assert a.fingerprint() != b.fingerprint()


def test_apply_collects_lazy_deps():
    staged = apply("raw/massive", _stage, output_path=OUT)
    parts = apply("data/massive", _transform, input_path=staged, output_path=OUT)
    assert parts.recipe.deps == (staged,)


def test_apply_rejects_unbindable_inputs():
    def f(a):
        return a

    with pytest.raises(TypeError):
        apply("x/y", f, a=1, b=2)

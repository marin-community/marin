# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SPIKE test for the lazy-artifact prototype (``marin.execution.lazy``).

Exercises the hardest path the redesign must support — a config that references
its own output path AND a dependency's output path — and runs the chain
end-to-end through the existing ``StepRunner`` to confirm the lower → cache →
lock → run pipeline works with explicit ``name@version`` identity.
"""

import dataclasses
from dataclasses import dataclass

from fray.types import ResourceConfig
from marin.execution.artifact import Artifact as ArtifactIO
from marin.execution.lazy import Checkpoint, Dataset, Recipe, RunContext, lower, materialized_config
from marin.execution.remote import remote
from marin.execution.step_runner import StepRunner

# --- Toy configs/fns standing in for tokenize + train --------------------------


@dataclass(frozen=True)
class TokenizeCfg:
    out: str  # self-reference: the producer writes here (was THIS_OUTPUT_PATH)
    source: str
    tokenizer: str


@dataclass(frozen=True)
class TrainCfg:
    out: str  # self-reference
    data: str  # cross-step reference: the dep's output path (was InputName/.cd())
    lr: float
    steps: int


def _run_tokenize(config: TokenizeCfg) -> dict:
    return {**dataclasses.asdict(config), "kind": "tokens"}


def _run_train(config: TrainCfg) -> dict:
    return {**dataclasses.asdict(config), "kind": "ckpt"}


# --- How it looks: the user-facing experiment ----------------------------------


def dclm_tokens() -> Dataset:
    return Dataset(
        name="datasets/dclm_tokens",
        version="2026.06.25",
        recipe=Recipe(
            fn=_run_tokenize,
            build_config=lambda ctx: TokenizeCfg(out=ctx.out, source="gs://raw/dclm", tokenizer="llama3"),
        ),
    )


def dclm_1b(*, lr: float = 3e-3) -> Checkpoint:
    data = dclm_tokens()
    return Checkpoint(
        name="checkpoints/dclm_1b",
        version="v3",
        recipe=Recipe(
            fn=_run_train,
            build_config=lambda ctx: TrainCfg(out=ctx.out, data=ctx.path(data), lr=lr, steps=54931),
            deps=(data,),
        ),
    )


# --- Tests ---------------------------------------------------------------------


def test_explicit_version_paths_and_cross_step_resolution():
    prefix = "gs://marin-spike"
    ckpt = dclm_1b()

    train_cfg = materialized_config(ckpt, prefix)
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
    """Transitive staleness: bumping a dependency's version changes the
    consumer's fingerprint (because build_config embeds ctx.path(dep) = name@version)."""
    before = dclm_1b().fingerprint()

    def bumped_tokens() -> Dataset:
        return Dataset(
            name="datasets/dclm_tokens",
            version="2026.07.01",  # bumped
            recipe=Recipe(
                fn=_run_tokenize,
                build_config=lambda ctx: TokenizeCfg(out=ctx.out, source="gs://raw/dclm", tokenizer="llama3"),
            ),
        )

    def bumped_ckpt() -> Checkpoint:
        data = bumped_tokens()
        return Checkpoint(
            name="checkpoints/dclm_1b",
            version="v3",
            recipe=Recipe(
                fn=_run_train,
                build_config=lambda ctx: TrainCfg(out=ctx.out, data=ctx.path(data), lr=3e-3, steps=54931),
                deps=(data,),
            ),
        )

    assert bumped_ckpt().fingerprint() != before


def test_region_is_pulled_live_at_run_time():
    """The region is a runtime-only attribute: ``build_config`` reads it from the
    ``RunContext``, so one recipe resolves against whatever region the step runs in."""

    def build_config(ctx: RunContext) -> TokenizeCfg:
        return TokenizeCfg(out=ctx.out, source=f"gs://raw-{ctx.region}/dclm", tokenizer="llama3")

    recipe = Recipe(fn=_run_tokenize, build_config=build_config)
    east = recipe.build_config(RunContext.for_run(out="o", prefix="p", region="us-east5"))
    central = recipe.build_config(RunContext.for_run(out="o", prefix="p", region="us-central2"))
    assert (east.source, central.source) == ("gs://raw-us-east5/dclm", "gs://raw-us-central2/dclm")


def test_resources_do_not_affect_identity():
    """Compute rides with the fn (``remote(fn, resources=…)``), never the config or the
    graph node, so changing the TPU a step runs on must not change its fingerprint."""

    def on(resources: ResourceConfig) -> Checkpoint:
        return Checkpoint(
            name="checkpoints/dclm_1b",
            version="v3",
            recipe=Recipe(
                fn=remote(_run_train, resources=resources),
                build_config=lambda ctx: TrainCfg(out=ctx.out, data="gs://d", lr=3e-3, steps=10),
            ),
        )

    assert on(ResourceConfig.with_tpu("v5p-8")).fingerprint() == on(ResourceConfig.with_tpu("v6e-8")).fingerprint()


def test_run_arg_is_live_at_run_but_not_in_identity():
    """A run-arg is declared on the recipe and pulled via ``ctx.run_arg()``: it reaches
    the config at run time, but is a ``<key>`` placeholder at fingerprint time. So an
    execution choice the config must carry (e.g. the TPU a dispatched job uses) reaches
    the step without forking identity when it changes."""

    def on(tpu: str) -> Checkpoint:
        return Checkpoint(
            name="checkpoints/dclm_1b",
            version="v3",
            recipe=Recipe(
                fn=_run_train,
                # `data` here stands in for any config field fed by a run-arg.
                build_config=lambda ctx: TrainCfg(out=ctx.out, data=ctx.run_arg("tpu"), lr=3e-3, steps=10),
                run_args={"tpu": tpu},
            ),
        )

    # Live at run time: the materialized config carries the declared value.
    assert materialized_config(on("v5p-8"), "gs://b").data == "v5p-8"
    # ...but excluded from identity: a different TPU is the same artifact.
    assert on("v5p-8").fingerprint() == on("v6e-8").fingerprint()


def test_pulling_an_undeclared_run_arg_fails_loudly():
    """Pulling a run-arg the recipe never declared is a loud KeyError, not a silent
    miss — the run-args a config reads and the run-args it declares must agree."""
    art = Checkpoint(
        name="checkpoints/dclm_1b",
        version="v3",
        recipe=Recipe(
            fn=_run_train,
            build_config=lambda ctx: TrainCfg(out=ctx.out, data=ctx.run_arg("tpu"), lr=3e-3, steps=10),
            # note: no run_args declared
        ),
    )
    try:
        art.fingerprint()
    except KeyError as e:
        assert "tpu" in str(e)
    else:
        raise AssertionError("expected a KeyError for the undeclared run-arg")


def test_end_to_end_through_step_runner(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))

    ckpt = dclm_1b()
    spec = lower(ckpt)

    StepRunner().run([spec])

    # Both artifacts materialized at their explicit name@version paths.
    tokens_path = f"{tmp_path}/datasets/dclm_tokens/2026.06.25"
    ckpt_path = f"{tmp_path}/checkpoints/dclm_1b/v3"

    saved_ckpt = ArtifactIO.from_path(ckpt_path)
    assert saved_ckpt["kind"] == "ckpt"
    # The training run actually received the dependency's resolved output path.
    assert saved_ckpt["data"] == tokens_path

    saved_tokens = ArtifactIO.from_path(tokens_path)
    assert saved_tokens["kind"] == "tokens"

    # Re-running is a cache hit (no error, idempotent).
    StepRunner().run([lower(dclm_1b())])

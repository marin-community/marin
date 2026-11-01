# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The mechanical checks on an idea's code change (runbook section 1b), made on the weekly branch itself.

Before it is committed to the weekly branch, a change must touch only the fast_track model code and its tests, add
at least one new config field whose default is off, leave the default model bit-for-bit identical to the commit
before it (loss and every gradient of a tiny CPU model), keep the model causal with the flag on, pass its own tests
and pass pre-commit. Losing ideas' flags stay as dead default-off code until a maintainer deletes them.
"""

import dataclasses
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh

from experiments.belay.branch import git
from experiments.grug.fast_track.launch import _apply_settings, _parse_settings
from experiments.grug.fast_track.model import GrugModelConfig, LocalMixer, Transformer
from experiments.grug.fast_track.optimizer import GrugMoeMuonHConfig

ALLOWED_PREFIXES = ("experiments/grug/fast_track/", "tests/grug/test_fast_track_")
TESTS_PREFIX = "tests/grug/test_fast_track_"
OFF_DEFAULTS = ("False", "None", "0", "0.0", "''", "'none'", "'off'")
# Past positions' hidden states may move by at most this much when future tokens change (float32 noise).
LEAK_TOLERANCE = 1e-6
_SEQ = 128  # the banded Inkling bias needs a multiple of 128


@dataclasses.dataclass(frozen=True)
class Probe:
    """What one checkout's code does: config defaults, a digest of the tiny model's loss and gradients, and the
    largest change in past hidden states when only future tokens change."""

    defaults: dict[str, dict[str, str]]
    digest: str
    leak: float


def changed_paths(repo: Path) -> list[str]:
    """Every uncommitted path in ``repo``: staged, unstaged and untracked."""
    tracked = git(repo, "diff", "--name-only", "HEAD").splitlines()
    untracked = git(repo, "ls-files", "--others", "--exclude-standard").splitlines()
    return sorted(set(tracked + untracked))


def refuse_outside_allowed(paths: list[str]) -> None:
    outside = [p for p in paths if not p.startswith(ALLOWED_PREFIXES)]
    if outside:
        raise SystemExit(
            f"refusing: the change touches {outside}; only {', '.join(ALLOWED_PREFIXES)}* may change.\n"
            "next: git checkout -- <those paths> (and delete new files there), then run code-change --finish again"
        )
    if not any(p.startswith(TESTS_PREFIX) for p in paths):
        raise SystemExit(f"refusing: add a test for the flag under {TESTS_PREFIX}*.py.\nnext: code-change --finish")


def new_flags(week: dict[str, dict[str, str]], idea: dict[str, dict[str, str]]) -> list[str]:
    """The idea's new config fields, after checking each defaults to off and no existing default changed."""
    added = []
    for cls, week_fields in week.items():
        idea_fields = idea[cls]
        changed = [k for k, v in week_fields.items() if idea_fields.get(k) != v]
        if changed:
            raise SystemExit(
                f"refusing: {cls} changes or removes existing fields {changed}.\n"
                "next: restore them and add a new default-off field instead, then run code-change --finish again"
            )
        for name, default in idea_fields.items():
            if name in week_fields:
                continue
            if default not in OFF_DEFAULTS:
                raise SystemExit(
                    f"refusing: new flag {cls}.{name} defaults to {default}; its default must be off.\n"
                    "next: make it default off (False, None, 0 or 0.0), then run code-change --finish again"
                )
            added.append(f"{cls}.{name}")
    if not added:
        raise SystemExit(
            "refusing: no new config flag.\nnext: put the change behind a new default-off field, then --finish again"
        )
    return added


def refuse_flag_off_change(week: Probe, idea: Probe) -> None:
    if week.digest != idea.digest:
        raise SystemExit(
            "refusing: with every flag off, the tiny model's loss or gradients differ from the commit before it.\n"
            "next: put every behavior change behind the new flag, then run code-change --finish again"
        )


def refuse_leak(flag_on: Probe) -> None:
    if not flag_on.leak <= LEAK_TOLERANCE:
        raise SystemExit(
            f"refusing: with the flag on, changing future tokens moves past hidden states by {flag_on.leak:.3g} "
            f"(limit {LEAK_TOLERANCE}); the change leaks future information.\n"
            "next: make the flag-on path causal and run code-change --finish again, or give up: git checkout -- . "
            "&& git clean -fd experiments/grug/fast_track tests/grug, then escalate"
        )


def _defaults() -> dict[str, dict[str, str]]:
    out = {}
    for cls in (GrugModelConfig, GrugMoeMuonHConfig):
        fields = {}
        for f in dataclasses.fields(cls):
            if f.default is not dataclasses.MISSING:
                fields[f.name] = repr(f.default)
            elif f.default_factory is not dataclasses.MISSING:
                fields[f.name] = repr(f.default_factory())
        out[cls.__name__] = fields
    return out


def probe_here(model_settings: list[str]) -> Probe:
    """Probe the checkout this process imported ``experiments`` from, with ``model_settings`` applied."""
    hidden = 64
    config = GrugModelConfig(
        vocab_size=128,
        hidden_dim=hidden,
        intermediate_dim=hidden // 2,
        shared_expert_intermediate_dim=hidden // 2,
        num_shared_experts=1,
        num_experts=4,
        num_experts_per_token=2,
        latent_dim=hidden // 2,
        num_layers=4,
        num_heads=2,
        num_kv_heads=2,
        local_kv_heads=None,
        global_kv_heads=None,
        head_dim=16,
        max_seq_len=_SEQ,
        sliding_window=_SEQ,
        global_every=2,
        capacity_factor=8.0,
        pooled_transport_capacity_factor=8.0,
        initializer_std=0.5 / math.sqrt(hidden),
        qk_norm=False,
        sconv_sites=("attn", "mlp"),
        mla=True,
        mla_kv_latent_dim=32,
        inkling_relpos=True,
        rel_dim=4,
        rel_extent=8,
        attn_res=True,
        attn_res_num_blocks=2,
        local_mixer=LocalMixer.KDA,
    )
    config = _apply_settings(config, _parse_settings(tuple(model_settings)))
    mesh = Mesh(
        np.array(jax.devices()[:1], dtype=object).reshape((1, 1, 1, 1)),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    with jax.set_mesh(mesh):
        model = Transformer.init(config, key=jax.random.key(0))
        tokens = jax.random.randint(jax.random.key(1), (2, _SEQ), 0, config.vocab_size)
        cotangent = jax.random.normal(jax.random.key(2), (2, _SEQ, hidden), jnp.float32)

        def loss_fn(m):
            return jnp.sum(m(tokens)[0].astype(jnp.float32) * cotangent)

        loss, grads = eqx.filter_jit(eqx.filter_value_and_grad(loss_fn))(model)
        future = tokens.at[:, _SEQ // 2 :].set(jax.random.randint(jax.random.key(3), (2, _SEQ // 2), 0, 128))
        forward = eqx.filter_jit(lambda m, t: m(t)[0].astype(jnp.float32))
        leak = float(jnp.max(jnp.abs(forward(model, tokens)[:, : _SEQ // 2] - forward(model, future)[:, : _SEQ // 2])))
    digest = hashlib.sha256(np.asarray(loss).tobytes())
    for path, leaf in jax.tree_util.tree_flatten_with_path(eqx.filter(grads, eqx.is_array))[0]:
        digest.update(jax.tree_util.keystr(path).encode())
        digest.update(np.asarray(leaf).tobytes())
    return Probe(_defaults(), digest.hexdigest(), leak)


def probe(checkout: Path, model_settings: list[str]) -> Probe:
    """Run ``probe_here`` on CPU in a subprocess that imports ``experiments`` from ``checkout``."""
    code = (
        "import dataclasses, json, sys; from experiments.belay.code_change import probe_here; "
        "print(json.dumps(dataclasses.asdict(probe_here(sys.argv[1:]))))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code, *model_settings],
        cwd=checkout,
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return Probe(**json.loads(out.strip().splitlines()[-1]))


def run_tests(repo: Path, paths: list[str]) -> None:
    """The change's own tests, on CPU."""
    tests = [p for p in paths if p.startswith(TESTS_PREFIX) and p.endswith(".py")]
    command = [sys.executable, "-m", "pytest", "-q", "-o", "addopts=", "-p", "no:cacheprovider", *tests]
    result = subprocess.run(
        command, cwd=repo, env={**os.environ, "JAX_PLATFORMS": "cpu"}, capture_output=True, text=True
    )
    if result.returncode != 0:
        raise SystemExit(f"refusing: the change's tests fail:\n{result.stdout[-3000:]}\nnext: fix them, then --finish")


def run_precommit(repo: Path, paths: list[str]) -> None:
    files = [p for p in paths if (repo / p).exists()]
    result = subprocess.run(["./infra/pre-commit.py", *files], cwd=repo, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(
            f"refusing: pre-commit fails:\n{result.stdout[-3000:]}\n"
            f"next: ./infra/pre-commit.py --fix {' '.join(files)}, then --finish"
        )


def flag_commits(repo: Path, base_ref: str) -> list[str]:
    """The flag commits on this branch since ``base_ref``: ``X####: <flag> (default off)``."""
    return git(repo, "log", "--format=%h %s", "--grep=(default off)", f"{base_ref}..HEAD").splitlines()

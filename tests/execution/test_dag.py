# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for `marin.execution.dag.upstream_steps` and `materialize`."""

import os
import tempfile
from dataclasses import dataclass, field
from threading import Thread

import pytest

from marin.execution.dag import upstream_steps
from marin.execution.executor import (
    ExecutorStep,
    InputName,
    _resolve_step_output_path,
    materialize,
    mirrored,
    output_path_of,
    this_output_path,
    versioned,
)


@dataclass(frozen=True)
class _Cfg:
    output_path: str
    x: int = 0


def _step(name: str = "step", x: int = 0) -> ExecutorStep:
    """Build a throwaway ExecutorStep. The fn is never invoked in these tests."""
    return ExecutorStep(name=name, fn=lambda c: None, config=_Cfg(output_path="", x=x))


@pytest.mark.parametrize("value", [None, 0, "", "abc", 3.14, True])
def test_primitive_returns_empty(value):
    assert upstream_steps(value) == []


def test_empty_collections_return_empty():
    assert upstream_steps([]) == []
    assert upstream_steps({}) == []
    assert upstream_steps(()) == []
    assert upstream_steps(set()) == []


def test_single_executor_step():
    step = _step()
    assert upstream_steps(step) == [step]


def test_dataclass_with_embedded_step():
    step = _step()

    @dataclass
    class Outer:
        inner: ExecutorStep
        other: int = 0

    result = upstream_steps(Outer(inner=step))
    assert result == [step]


def test_dict_with_embedded_step():
    step = _step()
    assert upstream_steps({"a": step, "b": 1}) == [step]


def test_list_with_embedded_step():
    step = _step()
    assert upstream_steps([1, step, "x"]) == [step]


def test_tuple_with_embedded_step():
    step = _step()
    assert upstream_steps((1, step, "x")) == [step]


def test_set_with_embedded_step():
    step = _step()
    # ExecutorStep is hashable (id-based), so it can live in a set.
    assert upstream_steps({step}) == [step]


def test_duplicate_step_returned_once():
    step = _step()

    @dataclass
    class Cfg:
        a: ExecutorStep
        b: ExecutorStep
        c: list

    result = upstream_steps(Cfg(a=step, b=step, c=[step, step]))
    assert result == [step]


def test_input_name_with_step_included():
    step = _step()
    inp = InputName(step=step, name="ckpt.pt")
    assert upstream_steps(inp) == [step]


def test_input_name_without_step_returns_empty():
    # Hardcoded path: no step reference.
    inp = InputName.hardcoded("some/path")
    assert upstream_steps(inp) == []


def test_input_name_nonblocking_still_included():
    step = _step()
    inp = InputName(step=step, name="ckpt.pt").nonblocking()
    assert upstream_steps(inp) == [step]


def test_versioned_value_is_leaf():
    # versioned() rejects InputName; its inner value is a primitive for dep purposes.
    assert upstream_steps(versioned(42)) == []


def test_mirrored_value_unwrapped():
    # MirroredValue wraps a path string; no embedded steps, but the walker
    # must descend without raising.
    assert upstream_steps(mirrored(versioned("data/v1"))) == []


def test_determinism_repeated_walk():
    a, b, c = _step("a"), _step("b"), _step("c")

    @dataclass
    class Cfg:
        items: list
        extra: dict

    cfg = Cfg(items=[a, b], extra={"k1": c, "k2": a})
    first = upstream_steps(cfg)
    second = upstream_steps(cfg)
    assert first == second
    assert first == [a, b, c]


def test_depth_first_field_declaration_order():
    """Walk must be depth-first in field-declaration order, not BFS or sorted."""
    leaf_a = _step("leaf_a")
    leaf_b = _step("leaf_b")
    leaf_c = _step("leaf_c")

    @dataclass
    class Inner:
        # field order matters: leaf_a before leaf_b
        x: ExecutorStep
        y: ExecutorStep

    @dataclass
    class Outer:
        # depth-first: descend into `first` (yielding leaf_a, leaf_b) BEFORE
        # visiting `second` (leaf_c). BFS would give [leaf_c, leaf_a, leaf_b].
        first: Inner
        second: ExecutorStep

    cfg = Outer(first=Inner(x=leaf_a, y=leaf_b), second=leaf_c)
    assert upstream_steps(cfg) == [leaf_a, leaf_b, leaf_c]


def test_order_distinguishes_from_sorted():
    """Order is traversal-driven, not name-sorted."""
    z = _step("z_first")
    a = _step("a_second")

    @dataclass
    class Cfg:
        head: ExecutorStep  # 'z_first' name but encountered first
        tail: ExecutorStep

    assert upstream_steps(Cfg(head=z, tail=a)) == [z, a]


def test_dict_value_order_preserved():
    # Python dicts preserve insertion order (3.7+); the walker must follow that.
    a, b, c = _step("a"), _step("b"), _step("c")
    d = {"second": b, "first": a, "third": c}
    assert upstream_steps(d) == [b, a, c]


def test_nested_dataclass_dict_list():
    """Mixed-container traversal stays depth-first in declaration order."""
    s1, s2, s3, s4 = _step("s1"), _step("s2"), _step("s3"), _step("s4")

    @dataclass
    class Cfg:
        a: dict = field(default_factory=dict)
        b: list = field(default_factory=list)

    cfg = Cfg(a={"k1": s1, "k2": [s2, s3]}, b=[s4])
    # Expected: dict values in insertion order, then list elements, depth-first.
    # `a` first, descend: s1, then [s2, s3] -> s2, s3; then `b`: s4.
    assert upstream_steps(cfg) == [s1, s2, s3, s4]


def test_input_name_inside_dataclass():
    parent = _step("parent")

    @dataclass
    class Cfg:
        ckpt: InputName

    assert upstream_steps(Cfg(ckpt=InputName(step=parent, name="ckpt.pt"))) == [parent]


def test_does_not_walk_into_step_config():
    """The walker must not descend into a returned step's own config."""
    grandparent = _step("grandparent")

    @dataclass
    class ParentCfg:
        output_path: str
        upstream: ExecutorStep

    parent = ExecutorStep(
        name="parent",
        fn=lambda c: None,
        config=ParentCfg(output_path="", upstream=grandparent),
    )

    # Walking `parent` directly returns only `parent`, not its embedded
    # `grandparent`. Transitive deps are the executor's job.
    assert upstream_steps(parent) == [parent]


# ---------------------------------------------------------------------------
# materialize
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _LeafCfg:
    """Tokenize-style step config — no embedded ExecutorSteps."""

    output_path: str
    n: int = 1


@dataclass(frozen=True)
class _TrainCfg:
    """Training-style config that references an upstream tokenize step."""

    output_path: str
    tokens: str  # holds an InputName placeholder until materialize substitutes it
    n: int = 0


def test_materialize_idempotent_for_placeholder_free_config(tmp_path):
    """A config with no embedded ExecutorSteps round-trips unchanged."""
    cfg = _TrainCfg(output_path=str(tmp_path / "train"), tokens="/already/concrete", n=7)

    # Sanity check the precondition: no embedded steps to materialize.
    assert upstream_steps(cfg) == []

    result = materialize(cfg, prefix=str(tmp_path))

    assert result == cfg
    # And no executor metadata should have been written — `materialize` must
    # short-circuit when there is nothing to do.
    assert not (tmp_path / "experiments").exists()


def test_materialize_substitutes_upstream_step_path(tmp_path):
    """A config with one InputName referencing an upstream step gets a concrete path."""
    call_count = {"n": 0}
    seen_output_paths: list[str] = []

    def tokenize_fn(cfg: _LeafCfg) -> None:
        call_count["n"] += 1
        seen_output_paths.append(cfg.output_path)
        # Simulate writing some output so the directory exists.
        os.makedirs(cfg.output_path, exist_ok=True)

    tok_step = ExecutorStep(
        name="tok",
        fn=tokenize_fn,
        config=_LeafCfg(output_path=this_output_path(), n=versioned(1)),
    )

    # The training config references the upstream step's output.
    train_output_path = str(tmp_path / "train-run")
    train_cfg = _TrainCfg(
        output_path=train_output_path,
        tokens=output_path_of(tok_step, "tokens.bin"),
    )

    result = materialize(train_cfg, prefix=str(tmp_path))

    # The fake fn ran exactly once.
    assert call_count["n"] == 1
    # The placeholder was replaced with a concrete path under tmp_path.
    assert isinstance(result.tokens, str)
    assert result.tokens.startswith(str(tmp_path) + "/tok-")
    assert result.tokens.endswith("/tokens.bin")
    # And it is the same path the fn saw.
    assert result.tokens.startswith(seen_output_paths[0])
    # The training output_path passes through unchanged (already concrete).
    assert result.output_path == train_output_path


def test_materialize_uses_provided_prefix(tmp_path):
    """Sub-step output paths land under the provided prefix, not marin_prefix()."""

    def tokenize_fn(cfg: _LeafCfg) -> None:
        os.makedirs(cfg.output_path, exist_ok=True)

    tok_step = ExecutorStep(
        name="tok",
        fn=tokenize_fn,
        config=_LeafCfg(output_path=this_output_path()),
    )
    train_cfg = _TrainCfg(
        output_path=str(tmp_path / "train"),
        tokens=output_path_of(tok_step, "tokens.bin"),
    )

    result = materialize(train_cfg, prefix=str(tmp_path))

    # The substituted path must be under the explicit prefix we passed.
    assert result.tokens.startswith(str(tmp_path) + "/")


def test_materialize_rejects_unresolved_output_path_placeholder(tmp_path):
    """If the caller forgot to resolve config.output_path, fail loudly."""
    cfg = _TrainCfg(
        output_path=this_output_path(),  # OutputName placeholder, not a concrete str
        tokens="/already/concrete",
    )
    with pytest.raises(TypeError, match="OutputName"):
        materialize(cfg, prefix=str(tmp_path))


def test_materialize_concurrent_callers_run_step_once(tmp_path):
    """Two callers materializing configs that share an upstream step
    must run that step exactly once across them."""

    @dataclass(frozen=True)
    class SlowLeafCfg:
        output_path: str
        marker_dir: str

    call_count = {"n": 0}

    def tokenize_fn(cfg: SlowLeafCfg) -> None:
        # The step_lock + status protocol guarantees only one caller actually
        # invokes fn; sibling callers see STATUS_SUCCESS and skip.
        call_count["n"] += 1
        os.makedirs(cfg.output_path, exist_ok=True)
        # Write a marker so we can verify completion.
        with open(os.path.join(cfg.marker_dir, f"call-{call_count['n']}"), "w") as f:
            f.write("done")

    marker_dir = tempfile.mkdtemp(prefix="materialize-marker-")

    tok_step = ExecutorStep(
        name="shared-tok",
        fn=tokenize_fn,
        config=SlowLeafCfg(output_path=this_output_path(), marker_dir=marker_dir),
    )

    # Both training configs reference the same upstream step.
    train_cfg_a = _TrainCfg(
        output_path=str(tmp_path / "train-a"),
        tokens=output_path_of(tok_step, "tokens.bin"),
    )
    train_cfg_b = _TrainCfg(
        output_path=str(tmp_path / "train-b"),
        tokens=output_path_of(tok_step, "tokens.bin"),
    )

    results: dict[str, _TrainCfg] = {}

    def run(label: str, cfg: _TrainCfg) -> None:
        results[label] = materialize(cfg, prefix=str(tmp_path))

    threads = [
        Thread(target=run, args=("a", train_cfg_a)),
        Thread(target=run, args=("b", train_cfg_b)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Exactly one caller ran the upstream step.
    assert call_count["n"] == 1
    # Both substituted paths point at the same upstream output.
    assert results["a"].tokens == results["b"].tokens
    assert results["a"].tokens.startswith(str(tmp_path) + "/shared-tok-")


# ---------------------------------------------------------------------------
# _resolve_step_output_path
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SubmitCfg:
    """Launcher-style config: own output_path is OutputName, plus an
    upstream InputName placeholder."""

    output_path: object  # str at runtime, but starts as OutputName(None)
    tokenized: object = None  # may hold an InputName referencing upstream
    extra: object = None  # may hold a nested OutputName(name="X")


def _noop_fn(config) -> None:
    return None


def test_resolve_step_output_path_returns_concrete_path(tmp_path):
    """`_resolve_step_output_path` produces a concrete path under the given prefix
    without running the step."""
    step = ExecutorStep(
        name="my-step",
        fn=_noop_fn,
        config=_SubmitCfg(output_path=this_output_path()),
    )

    path = _resolve_step_output_path(step, prefix=str(tmp_path))

    assert isinstance(path, str)
    assert path.startswith(str(tmp_path) + "/my-step-")
    # No GCS / disk side-effects: compute_version only walks in-memory state.
    assert not (tmp_path / "experiments").exists()

# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0


import equinox as eqx
import jax.numpy as jnp
import pytest

import haliax as hax


def test_diagnose_common_issues_repeated():
    class M(eqx.Module):
        a: jnp.ndarray = eqx.field()
        b: jnp.ndarray = eqx.field()

        def __init__(self):
            super().__init__()
            self.a = jnp.zeros(1)
            self.b = self.a

    try:
        hax.debug.diagnose_common_issues(M())
        pytest.fail("Should have raised an exception")
    except hax.debug.ModuleProblems as e:
        assert len(e.reused_arrays) == 1
        assert len(e.static_arrays) == 0


def test_diagnose_common_issues_repeated_nested():
    class M(eqx.Module):
        a: jnp.ndarray = eqx.field()
        b: jnp.ndarray = eqx.field()

        def __init__(self):
            super().__init__()
            self.a = jnp.zeros(1)
            self.b = self.a

    class N(eqx.Module):
        m: M = eqx.field()
        c: jnp.ndarray = eqx.field()

        def __init__(self):
            super().__init__()
            self.m = M()
            self.c = self.m.a

    try:
        hax.debug.diagnose_common_issues(N())
        pytest.fail("Should have raised an exception")
    except hax.debug.ModuleProblems as e:
        assert len(e.reused_arrays) == 1
        assert e.reused_arrays[0][1] == [".m.a", ".m.b", ".c"]
        assert len(e.static_arrays) == 0


def test_diagnose_common_issues_static():
    class M(eqx.Module):
        a: jnp.ndarray = eqx.static_field()
        b: hax.NamedArray = eqx.static_field()

        def __init__(self):
            super().__init__()
            self.a = jnp.zeros(1)
            self.b = hax.named(jnp.zeros(3), "a")

    try:
        hax.debug.diagnose_common_issues(M())
        pytest.fail("Should have raised an exception")
    except hax.debug.ModuleProblems as e:
        assert len(e.reused_arrays) == 0
        assert len(e.static_arrays) == 2


def test_diagnose_common_issues_static_nested():
    class M(eqx.Module):
        a: jnp.ndarray = eqx.static_field()
        b: hax.NamedArray = eqx.static_field()

        def __init__(self):
            super().__init__()
            self.a = jnp.zeros(1)
            self.b = hax.named(jnp.zeros(3), "a")

    class N(eqx.Module):
        m: M = eqx.field()
        c: jnp.ndarray = eqx.field()

        def __init__(self):
            super().__init__()
            self.m = M()
            self.c = self.m.a

    try:
        hax.debug.diagnose_common_issues(N())
        pytest.fail("Should have raised an exception")
    except hax.debug.ModuleProblems as e:
        assert len(e.reused_arrays) == 0
        assert len(e.static_arrays) == 2

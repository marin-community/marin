# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0


from typing import Callable

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

import haliax as hax
from haliax.axis import axis_spec_to_shape_dict

Height = hax.Axis("Height", 4)
Width = hax.Axis("Width", 8)
Digit = hax.Axis("Digit", 10)

HWD = axis_spec_to_shape_dict((Height, Width, Digit))


def test_empty_shape():
    key = jax.random.PRNGKey(0)
    hax.random.uniform(key, shape=())


def test_uniform_with_bounds_scalar():
    check_gen_is_equal(jax.random.uniform, hax.random.uniform)

    key = jax.random.PRNGKey(0)
    u = hax.random.uniform(key, shape=(Height, Width), minval=-3.0, maxval=0.5)

    assert u.axes == (Height, Width)

    assert hax.all(u >= -3.0)
    assert hax.all(u <= 0.5)


def test_uniform_with_bounds_broadcast():
    key = jax.random.PRNGKey(0)
    lb = hax.arange(Height, start=-5.0)
    ub = hax.full(Width, 0.5)
    u = hax.random.uniform(key, shape=(Height, Width), minval=lb, maxval=ub)

    assert u.axes == (Height, Width)

    assert hax.all(u >= lb)
    assert hax.all(u <= 0.5)

    # for lb and ub, broadcast the raw arrays for jax
    lb_raw = lb.array.reshape(-1, 1)
    ub_raw = ub.array.reshape(1, -1)

    lb_raw = jnp.broadcast_to(lb_raw, (Height.size, Width.size))
    ub_raw = jnp.broadcast_to(ub_raw, (Height.size, Width.size))

    check_gen_is_equal(
        lambda k, s: jax.random.uniform(k, shape=s, minval=lb_raw, maxval=ub_raw),
        lambda k, s: hax.random.uniform(k, s, minval=lb, maxval=ub),
    )


def test_uniform_with_bounds_broadcast_and_scalar():
    key = jax.random.PRNGKey(0)
    lb = hax.full(Height, -3.0)
    ub = 0.5
    u = hax.random.uniform(key, shape=(Height, Width), minval=lb, maxval=ub)

    assert u.axes == (Height, Width)

    assert hax.all(u >= -3.0)
    assert hax.all(u <= 0.5)


def test_randint():
    check_gen_is_equal(lambda k, s: jax.random.randint(k, s, 0, 10), lambda k, s: hax.random.randint(k, s, 0, 10))
    # check broadcasting
    minval = hax.arange(Width, step=1)
    check_gen_is_equal(
        lambda k, s: jax.random.randint(k, s, minval.array.reshape(1, -1), 10),
        lambda k, s: hax.random.randint(k, s, minval, 10),
    )

    minval = hax.arange(Height, step=1)
    check_gen_is_equal(
        lambda k, s: jax.random.randint(k, s, minval.array.reshape(-1, 1), 10),
        lambda k, s: hax.random.randint(k, s, minval, 10),
    )


def check_gen_is_equal(
    jax_fn: Callable[[PRNGKeyArray, tuple], jnp.ndarray],
    hax_fn: Callable[[PRNGKeyArray, hax.AxisSpec], hax.NamedArray],
):
    key = jax.random.PRNGKey(0)

    hax_out = hax_fn(key, (Height, Width))
    jax_out = jax_fn(key, (Height.size, Width.size))

    assert hax_out.array.shape == jax_out.shape
    assert hax.all(hax_out.array == jax_out)


def test_normal():
    check_gen_is_equal(jax.random.normal, hax.random.normal)


def test_bernoulli():
    check_gen_is_equal(lambda k, s: jax.random.bernoulli(k, 0.5, s), lambda k, s: hax.random.bernoulli(k, s, 0.5))
    # check broadcasting
    prob = hax.arange(Width, step=0.1)
    check_gen_is_equal(
        lambda k, s: jax.random.bernoulli(k, prob.array.reshape(1, -1), s),
        lambda k, s: hax.random.bernoulli(k, s, prob),
    )
    prob = hax.arange(Height, step=0.1)
    check_gen_is_equal(
        lambda k, s: jax.random.bernoulli(k, prob.array.reshape(-1, 1), s),
        lambda k, s: hax.random.bernoulli(k, s, prob),
    )


def test_poisson():
    check_gen_is_equal(lambda k, s: jax.random.poisson(k, 0.5, s), lambda k, s: hax.random.poisson(k, s, 0.5))
    # check broadcasting
    lam = hax.arange(Width, step=0.1)
    check_gen_is_equal(
        lambda k, s: jax.random.poisson(k, lam.array.reshape(1, -1), s),
        lambda k, s: hax.random.poisson(k, s, lam),
    )
    lam = hax.arange(Height, step=0.1)
    check_gen_is_equal(
        lambda k, s: jax.random.poisson(k, lam.array.reshape(-1, 1), s),
        lambda k, s: hax.random.poisson(k, s, lam),
    )


def test_laplace():
    check_gen_is_equal(lambda k, s: jax.random.laplace(k, s), lambda k, s: hax.random.laplace(k, s))


def test_exponential():
    check_gen_is_equal(lambda k, s: jax.random.exponential(k, s), lambda k, s: hax.random.exponential(k, s))


def test_gamma():
    check_gen_is_equal(lambda k, s: jax.random.gamma(k, 0.5, s), lambda k, s: hax.random.gamma(k, s, 0.5))
    # check broadcasting
    alpha = hax.arange(Width, step=0.1)
    check_gen_is_equal(
        lambda k, s: jax.random.gamma(k, alpha.array.reshape(1, -1), s),
        lambda k, s: hax.random.gamma(k, s, alpha),
    )
    alpha = hax.arange(Height, step=0.1)
    check_gen_is_equal(
        lambda k, s: jax.random.gamma(k, alpha.array.reshape(-1, 1), s),
        lambda k, s: hax.random.gamma(k, s, alpha),
    )


def test_gumbel():
    check_gen_is_equal(lambda k, s: jax.random.gumbel(k, s), lambda k, s: hax.random.gumbel(k, s))


def test_beta():
    check_gen_is_equal(lambda k, s: jax.random.beta(k, 0.6, 0.5, s), lambda k, s: hax.random.beta(k, s, 0.6, 0.5))
    # check broadcasting
    alpha = hax.arange(Width, step=0.1, start=0.01)
    beta = hax.arange(Width, step=0.1, start=0.01)
    check_gen_is_equal(
        lambda k, s: jax.random.beta(k, alpha.array.reshape(1, -1), beta.array.reshape(1, -1), s),
        lambda k, s: hax.random.beta(k, s, alpha, beta),
    )
    alpha = hax.arange(Height, step=0.1, start=0.01)
    beta = hax.arange(Height, step=0.1, start=0.01)
    check_gen_is_equal(
        lambda k, s: jax.random.beta(k, alpha.array.reshape(-1, 1), beta.array.reshape(-1, 1), s),
        lambda k, s: hax.random.beta(k, s, alpha, beta),
    )


def test_rademacher():
    check_gen_is_equal(lambda k, s: jax.random.rademacher(k, s), lambda k, s: hax.random.rademacher(k, s))


def test_ball():
    check_gen_is_equal(lambda k, s: jax.random.ball(k, Digit.size, shape=s), lambda k, s: hax.random.ball(k, s, Digit))


def test_cauchy():
    check_gen_is_equal(lambda k, s: jax.random.cauchy(k, s), lambda k, s: hax.random.cauchy(k, s))


def test_logistic():
    check_gen_is_equal(lambda k, s: jax.random.logistic(k, s), lambda k, s: hax.random.logistic(k, s))


def test_truncated_normal():
    lower = hax.arange(Width, step=0.1, start=0.01)
    upper = hax.arange(Width, step=0.1, start=0.01)
    check_gen_is_equal(
        lambda k, s: jax.random.truncated_normal(k, lower.array.reshape(1, -1), upper.array.reshape(1, -1), s),
        lambda k, s: hax.random.truncated_normal(k, s, lower, upper),
    )

    lower = hax.arange(Height, step=0.1, start=0.01)
    upper = hax.arange(Height, step=0.1, start=0.01)
    check_gen_is_equal(
        lambda k, s: jax.random.truncated_normal(k, lower.array.reshape(-1, 1), upper.array.reshape(-1, 1), s),
        lambda k, s: hax.random.truncated_normal(k, s, lower, upper),
    )

    lower = hax.arange(Width, step=0.1, start=0.01)
    upper = hax.arange(Height, step=0.1, start=0.01)

    check_gen_is_equal(
        lambda k, s: jax.random.truncated_normal(k, lower.array.reshape(1, -1), upper.array.reshape(-1, 1), s),
        lambda k, s: hax.random.truncated_normal(k, s, lower, upper),
    )


def test_choice():
    digits = hax.arange(Digit)
    check_gen_is_equal(
        lambda k, s: jax.random.choice(k, digits.array, shape=s), lambda k, s: hax.random.choice(k, s, digits, Digit)
    )

    weights = hax.arange(Digit, step=0.1, start=0.01)
    check_gen_is_equal(
        lambda k, s: jax.random.choice(k, digits.array, shape=s, p=weights.array),
        lambda k, s: hax.random.choice(k, s, digits, Digit, p=weights),
    )

    # test str selector
    check_gen_is_equal(
        lambda k, s: jax.random.choice(k, digits.array, shape=s, p=weights.array),
        lambda k, s: hax.random.choice(k, s, digits, "Digit", p=weights),
    )


def test_categorical_shape_dict():
    logits = hax.random.uniform(jax.random.PRNGKey(0), HWD)
    check_gen_is_equal(
        lambda k, s: jax.random.categorical(k, logits.array, shape=s, axis=-1),
        lambda k, s: hax.random.categorical(k, logits, Digit, shape=s),
    )

    logits = logits.rearrange((Digit, Height, Width))
    check_gen_is_equal(
        lambda k, s: jax.random.categorical(k, logits.array, shape=s, axis=0),
        lambda k, s: hax.random.categorical(k, logits, Digit, shape=s),
    )

    # check broadcasting
    logits = hax.random.uniform(
        jax.random.PRNGKey(0),
        {"Height": Height.size, "Digit": Digit.size},
    )
    # https://github.com/google/jax/issues/13124 broadcasting is wrong with jax categorical
    raw_logits = jnp.broadcast_to(logits.array.reshape(-1, 1, Digit.size), (Height.size, Width.size, Digit.size))
    check_gen_is_equal(
        lambda k, s: jax.random.categorical(k, raw_logits, shape=s, axis=-1),
        lambda k, s: hax.random.categorical(k, logits, Digit, shape=s),
    )

    # check str arg for selector
    check_gen_is_equal(
        lambda k, s: jax.random.categorical(k, raw_logits, shape=s, axis=-1),
        lambda k, s: hax.random.categorical(k, logits, "Digit", shape=s),
    )


def test_categorical():
    logits = hax.random.uniform(
        jax.random.PRNGKey(0),
        (
            Height,
            Width,
            Digit,
        ),
    )
    check_gen_is_equal(
        lambda k, s: jax.random.categorical(k, logits.array, shape=s, axis=-1),
        lambda k, s: hax.random.categorical(k, logits, Digit, shape=s),
    )

    logits = logits.rearrange((Digit, Height, Width))
    check_gen_is_equal(
        lambda k, s: jax.random.categorical(k, logits.array, shape=s, axis=0),
        lambda k, s: hax.random.categorical(k, logits, Digit, shape=s),
    )

    # check broadcasting
    logits = hax.random.uniform(
        jax.random.PRNGKey(0),
        (
            Height,
            Digit,
        ),
    )
    # https://github.com/google/jax/issues/13124 broadcasting is wrong with jax categorical
    raw_logits = jnp.broadcast_to(logits.array.reshape(-1, 1, Digit.size), (Height.size, Width.size, Digit.size))
    check_gen_is_equal(
        lambda k, s: jax.random.categorical(k, raw_logits, shape=s, axis=-1),
        lambda k, s: hax.random.categorical(k, logits, Digit, shape=s),
    )

    # check str arg for selector
    check_gen_is_equal(
        lambda k, s: jax.random.categorical(k, raw_logits, shape=s, axis=-1),
        lambda k, s: hax.random.categorical(k, logits, "Digit", shape=s),
    )


def test_permutation():
    data = hax.random.uniform(jax.random.PRNGKey(0), (Width, Height))

    hax_perm = hax.random.permutation(jax.random.PRNGKey(0), data, Height)
    jax_perm = jax.random.permutation(jax.random.PRNGKey(0), data.array, 1)
    assert jnp.all(hax_perm.array == jax_perm)

    hax_perm = hax.random.permutation(jax.random.PRNGKey(0), data, Width)
    jax_perm = jax.random.permutation(jax.random.PRNGKey(0), data.array, 0)
    assert jnp.all(hax_perm.array == jax_perm)

    # test str arg for selector
    hax_perm = hax.random.permutation(jax.random.PRNGKey(0), data, "Height")
    jax_perm = jax.random.permutation(jax.random.PRNGKey(0), data.array, 1)
    assert jnp.all(hax_perm.array == jax_perm)


def test_t():
    param = hax.arange(Width, start=0.1)
    check_gen_is_equal(lambda k, s: jax.random.t(k, param.array, shape=s), lambda k, s: hax.random.t(k, s, param))

    check_gen_is_equal(lambda k, s: jax.random.t(k, 0.5, shape=s), lambda k, s: hax.random.t(k, s, 0.5))


def test_weibull_min():
    scale = hax.arange(Width, start=0.1)
    concentration = hax.arange(Height, start=0.1)

    check_gen_is_equal(
        lambda k, s: jax.random.weibull_min(
            k, scale.array.reshape(1, -1), concentration.array.reshape(-1, 1), shape=s
        ),
        lambda k, s: hax.random.weibull_min(k, s, scale, concentration),
    )


def test_pareto():
    b = hax.arange(Width, start=0.1)

    check_gen_is_equal(
        lambda k, s: jax.random.pareto(k, b.array.reshape(1, -1), shape=s), lambda k, s: hax.random.pareto(k, s, b)
    )


def test_loggamma():
    a = hax.arange(Width, start=0.1)

    check_gen_is_equal(
        lambda k, s: jax.random.loggamma(k, a.array.reshape(1, -1), shape=s), lambda k, s: hax.random.loggamma(k, s, a)
    )

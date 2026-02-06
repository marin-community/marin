# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for Kelp noise schedules and diffusion utilities."""

import jax.numpy as jnp
from jax import random

from experiments.kelp.model.noise import (
    cosine_schedule,
    corrupt_tokens,
    create_initial_tokens,
    linear_schedule,
    sample_timesteps,
)


class TestNoiseSchedules:
    def test_cosine_schedule_shape(self):
        schedule = cosine_schedule(100)
        assert schedule.alphas.shape == (100,)
        assert schedule.alphas_cumprod.shape == (100,)
        assert schedule.betas.shape == (100,)

    def test_cosine_schedule_values(self):
        schedule = cosine_schedule(100)
        assert jnp.all(schedule.alphas >= 0)
        assert jnp.all(schedule.alphas <= 1)
        assert jnp.all(schedule.alphas_cumprod >= 0)
        assert jnp.all(schedule.alphas_cumprod <= 1)
        assert schedule.alphas_cumprod[0] > schedule.alphas_cumprod[-1]

    def test_linear_schedule_shape(self):
        schedule = linear_schedule(50)
        assert schedule.alphas.shape == (50,)
        assert schedule.num_steps == 50

    def test_linear_schedule_values(self):
        schedule = linear_schedule(50, beta_start=0.0001, beta_end=0.02)
        assert schedule.betas[0] < schedule.betas[-1]


class TestCorruptTokens:
    def test_corrupt_at_t0(self):
        schedule = cosine_schedule(100)
        tokens = jnp.array([[1, 2, 3, 4]])
        key = random.PRNGKey(0)

        corrupted = corrupt_tokens(tokens, 0, schedule, mask_token_id=99, key=key)
        assert jnp.allclose(corrupted, tokens)

    def test_corrupt_at_max_t(self):
        schedule = cosine_schedule(100)
        tokens = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8]])
        key = random.PRNGKey(0)

        corrupted = corrupt_tokens(tokens, 99, schedule, mask_token_id=99, key=key)
        mask_count = jnp.sum(corrupted == 99)
        assert mask_count > 0

    def test_corrupt_preserves_prefix(self):
        schedule = cosine_schedule(100)
        tokens = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8]])
        key = random.PRNGKey(0)

        corrupted = corrupt_tokens(tokens, 99, schedule, mask_token_id=99, key=key, prefix_len=4)
        assert jnp.allclose(corrupted[:, :4], tokens[:, :4])


class TestSampleTimesteps:
    def test_sample_range(self):
        key = random.PRNGKey(0)
        timesteps = sample_timesteps(100, num_steps=50, key=key)
        assert timesteps.shape == (100,)
        assert jnp.all(timesteps >= 0)
        assert jnp.all(timesteps < 50)


class TestCreateInitialTokens:
    def test_create_initial_tokens(self):
        prefix = jnp.array([[1, 2, 3]])
        initial = create_initial_tokens(prefix, target_len=10, mask_token_id=99, pad_token_id=0)

        assert initial.shape == (1, 10)
        assert jnp.allclose(initial[:, :3], prefix)
        assert jnp.all(initial[:, 3:] == 99)

    def test_prefix_longer_than_target(self):
        prefix = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        initial = create_initial_tokens(prefix, target_len=5, mask_token_id=99, pad_token_id=0)

        assert initial.shape == (1, 5)
        assert jnp.allclose(initial, prefix[:, :5])

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping

import jax
import numpy as np
from haliax import Axis
from jax import random
from levanter.models.qwen import QwenConfig, QwenLMHeadModel
from levanter.testing.helpers import use_test_mesh
from marin.inference.dashboard_server import bind_serving_socket, serve_app_background
from marin.rl.levanter_policy import (
    LevanterPolicy,
    LevanterPolicyClient,
    PolicyBatch,
    build_levanter_policy_app,
    decode_policy_batch,
    encode_policy_batch,
)
from transformers import Qwen3Config


class RecordingPublisher:
    def __init__(self) -> None:
        self.step = -1
        self.weights: dict[str, np.ndarray] = {}

    def publish(self, weights: Mapping[str, jax.Array], *, step: int) -> None:
        self.step = step
        self.weights = {name: np.asarray(value) for name, value in weights.items()}


def _tiny_qwen() -> QwenLMHeadModel:
    hf_config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=16,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
    )
    config = QwenConfig.from_hf_config(hf_config)
    return QwenLMHeadModel.init(Axis("vocab", hf_config.vocab_size), config, key=random.PRNGKey(0))


def test_levanter_policy_forward_train_and_publish_changes_policy() -> None:
    sequences = np.asarray([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=np.int32)
    publisher = RecordingPublisher()

    with use_test_mesh():
        policy = LevanterPolicy(_tiny_qwen(), learning_rate=0.05, weight_publisher=publisher)
        initial = policy.forward(PolicyBatch(sequences, action_count=2)).action_log_probs
        result = policy.ppo_train(
            PolicyBatch(
                sequences,
                action_count=2,
                old_action_log_probs=initial,
                advantages=np.ones_like(initial),
                loss_mask=np.ones_like(initial),
            )
        )
        updated = policy.forward(PolicyBatch(sequences, action_count=2)).action_log_probs
        published_step = policy.broadcast_weights()

    assert result.step == 1
    assert published_step == 1
    assert result.action_log_probs.shape == (2, 2)
    assert result.action_log_probs.dtype == np.float32
    assert not np.allclose(initial, updated)
    assert publisher.step == 1
    assert publisher.weights
    assert all(np.isfinite(weight).all() for weight in publisher.weights.values())


def test_policy_batch_codec_preserves_training_inputs() -> None:
    batch = PolicyBatch(
        sequences=np.asarray([[1, 2, 3]], dtype=np.int32),
        action_count=2,
        old_action_log_probs=np.asarray([[-0.5, -0.25]], dtype=np.float32),
        advantages=np.asarray([[1.0, -1.0]], dtype=np.float32),
        loss_mask=np.asarray([[1.0, 0.0]], dtype=np.float32),
    )

    decoded = decode_policy_batch(encode_policy_batch(batch))

    np.testing.assert_array_equal(decoded.sequences, batch.sequences)
    np.testing.assert_array_equal(decoded.old_action_log_probs, batch.old_action_log_probs)
    np.testing.assert_array_equal(decoded.advantages, batch.advantages)
    np.testing.assert_array_equal(decoded.loss_mask, batch.loss_mask)


def test_levanter_policy_http_client_runs_forward_and_training() -> None:
    sequences = np.asarray([[1, 2, 3, 4]], dtype=np.int32)
    publisher = RecordingPublisher()

    with use_test_mesh():
        policy = LevanterPolicy(_tiny_qwen(), learning_rate=0.05, weight_publisher=publisher)
        socket = bind_serving_socket("127.0.0.1", 0)
        port = socket.getsockname()[1]
        with serve_app_background(build_levanter_policy_app(policy), socket, name="test-levanter-policy"):
            client = LevanterPolicyClient(f"http://127.0.0.1:{port}")
            initial = client.forward(PolicyBatch(sequences, action_count=2)).action_log_probs
            trained = client.ppo_train(
                PolicyBatch(
                    sequences,
                    action_count=2,
                    old_action_log_probs=initial,
                    advantages=np.ones_like(initial),
                )
            )
            updated = client.forward(PolicyBatch(sequences, action_count=2)).action_log_probs
            published_step = client.broadcast_weights()

    assert trained.step == 1
    assert published_step == 1
    assert publisher.step == 1
    assert not np.allclose(initial, updated)
    assert updated.shape == (1, 2)

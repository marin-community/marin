# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Install a frozen native learner checkpoint on a Snowball inference server."""

import dataclasses

import equinox as eqx
import jax

from levanter.checkpoint import load_checkpoint, load_checkpoint_metadata
from levanter.inference.openai import InferenceServer
from levanter.models.lm_model import LmHeadModel
from levanter.models.snowball import SnowballLMHeadModel


def publish_snowball_checkpoint(
    server: InferenceServer, checkpoint_path: str, *, policy_version: int, expected_serving_version: int
) -> int:
    """Install the checkpoint's model subtree and publish its immutable policy ID."""
    metadata = load_checkpoint_metadata(checkpoint_path)
    policy_id = metadata.get("policy_id")
    if metadata.get("policy_version") != policy_version or not isinstance(policy_id, str) or not policy_id:
        raise ValueError("Checkpoint metadata does not identify the requested Snowball policy version")
    context = server.inference_context

    def restore(active: LmHeadModel) -> SnowballLMHeadModel:
        if not isinstance(active, SnowballLMHeadModel):
            raise ValueError("Native Snowball checkpoints require a Snowball serving model")
        packed = active.transformer.blocks[0].mlp.expert_mlp.w_up is None
        template = active
        if packed:
            template = eqx.filter_eval_shape(
                SnowballLMHeadModel.init, active.Vocab, active.config, key=jax.random.key(0)
            )
        restored = load_checkpoint(
            template,
            checkpoint_path,
            subpath="model",
            mesh=context.mesh,
            axis_mapping=context.config.trainer.parameter_axis_mapping,
        )
        if packed:
            restored = restored.with_packed_experts()
        candidate = dataclasses.replace(active, transformer=restored.transformer)

        def match_array(source, destination):
            if isinstance(destination, jax.Array):
                if source.shape != destination.shape:
                    raise ValueError("Checkpoint model dimensions differ from the active Snowball policy")
                return jax.device_put(source.astype(destination.dtype), destination.sharding)
            return source

        candidate = jax.tree.map(match_array, candidate, active)
        return candidate

    staged = context.stage_policy(restore, expected_version=expected_serving_version, policy_id=policy_id)
    return context.install_policy(staged)

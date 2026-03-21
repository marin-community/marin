# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""HF/safetensors policy export for alternating RL.

Exports the trained model to HF format so vLLM can bootstrap from it
in the next sampling phase.
"""

import logging
from datetime import datetime, timezone

from levanter.compat.hf_checkpoints import HFCheckpointConverter, HFCompatConfig, RepoRef
from levanter.models.lm_model import LmHeadModel
from transformers import PreTrainedTokenizer

from marin.rl.alternating.state import PolicyManifest, write_json_to_path

logger = logging.getLogger(__name__)


def export_policy(
    model: LmHeadModel,
    model_config_class: type[HFCompatConfig],
    tokenizer: PreTrainedTokenizer,
    output_dir: str,
    policy_version: int,
    phase_id: int,
    source_global_step: int,
    model_name: str,
    levanter_checkpoint_path: str | None,
) -> PolicyManifest:
    """Export a Levanter model to HF/safetensors format and write a policy manifest.

    This should be called while the full training mesh is still alive,
    because the model may require the full pod mesh to materialize correctly.

    Args:
        model: The trained Levanter model.
        model_config_class: The HFCompatConfig class for the model architecture.
        tokenizer: The tokenizer to save alongside the model.
        output_dir: GCS or local directory for the export.
        policy_version: The policy version number.
        phase_id: The training phase that produced this policy.
        source_global_step: The global trainer step.
        model_name: HF model name (e.g., "meta-llama/Llama-3.1-8B-Instruct").
        levanter_checkpoint_path: Path to the Levanter checkpoint this was exported from.

    Returns:
        PolicyManifest for the exported policy.
    """
    logger.info(
        "Exporting policy version %d from phase %d (step %d) to %s",
        policy_version,
        phase_id,
        source_global_step,
        output_dir,
    )

    converter = HFCheckpointConverter(
        model_config_class,
        reference_checkpoint=RepoRef(model_name),
        tokenizer=tokenizer,
    )

    converter.save_pretrained(
        model,
        output_dir,
        save_tokenizer=True,
    )

    manifest = PolicyManifest(
        policy_version=policy_version,
        phase_id=phase_id,
        source_global_step=source_global_step,
        hf_export_path=output_dir,
        levanter_checkpoint_path=levanter_checkpoint_path,
        model_name=model_name,
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    manifest_path = f"{output_dir}/manifest.json"
    write_json_to_path(manifest_path, manifest.to_json())
    logger.info("Policy manifest written to %s", manifest_path)

    return manifest

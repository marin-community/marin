# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SFT step builder for the Delphi ladder + midtrain variants.

`build_sft_step(slug, rel_path, tokenize_step)` returns a single SFT
`ExecutorStep` initialized from the latest HF checkpoint under
`{MARIN_PREFIX}/{rel_path}` and pinned to a stable output path so
preemption-driven restarts resume the same checkpoint stream.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import re

from fray.cluster import ResourceConfig
from levanter.compat.hf_checkpoints import HFCheckpointConverter
from marin.evaluation.utils import discover_hf_checkpoints
from marin.execution.executor import ExecutorStep
from marin.processing.tokenize import lm_data_config

from experiments.defaults import default_sft
from experiments.downstream_scaling.sft.tokenize import SFT_RESOURCES
from experiments.simple_sft_config import SimpleSFTConfig

logger = logging.getLogger(__name__)


SFT_OUTPUT_PREFIX = "checkpoints/downstream_scaling/sft/delphi/gsm8k_qa_nopack_1ep"


# 1-epoch SFT on GSM8K train (7,473 examples), batch 64, no sequence packing
# -> ceil(7473 / 64) = 117 steps. With pack=False each sequence is one Q+A pair,
# so num_train_steps * batch_size maps 1:1 to pair-views.
DEFAULT_SFT_CONFIG = SimpleSFTConfig(
    resources=ResourceConfig.with_tpu("v5p-8"),
    train_batch_size=64,
    max_seq_len=1024,
    num_train_steps=117,
    learning_rate=5e-6,
    warmup=0.03,
    weight_decay=0.0,
    lr_schedule="linear",
    decay=0.9,
    min_lr_ratio=0.0,
    steps_per_hf_export=117,
    steps_per_eval=30,
    steps_per_checkpoint=None,
    pad_tokenizer_to_match_model=True,
    seed=0,
)


_MARIN_BUCKET_RE = re.compile(r"^gs://marin-[^/]+/(.+)$")


def _resolve_latest_hf_checkpoint(rel_path: str) -> str:
    """Resolve `{MARIN_PREFIX}/{rel_path}` and return the latest `step-N` subdir.

    `rel_path` may also be an absolute `gs://` URI, in which case `MARIN_PREFIX`
    is ignored (so we can pin specific bases to specific regions when the HF
    checkpoint only exists in one bucket).
    """
    if "://" in rel_path:
        base_path = rel_path
    else:
        prefix = os.environ.get("MARIN_PREFIX")
        if not prefix:
            raise RuntimeError(
                "MARIN_PREFIX must be set before building SFT steps; "
                "the launcher resolves base-checkpoint paths at plan time."
            )
        base_path = os.path.join(prefix, rel_path)
    step_dirs = discover_hf_checkpoints(base_path=base_path)
    if not step_dirs:
        raise FileNotFoundError(f"No HF checkpoints (config.json + tokenizer_config.json) found under {base_path}")

    def step_int(path: str) -> int:
        return int(path.rsplit("step-", 1)[-1])

    return sorted(step_dirs, key=step_int)[-1]


def _to_mirror_uri(gs_url: str) -> str:
    """Rewrite `gs://marin-<region>/<path>` → `mirror://<path>` so Levanter's
    cross-region check (which only looks at `gs://` paths) doesn't reject
    out-of-region bases. The MirrorFileSystem resolves to the local bucket if
    the file exists there; otherwise it copies on first access. Non-marin
    paths pass through unchanged.
    """
    match = _MARIN_BUCKET_RE.match(gs_url)
    if match:
        return f"mirror://{match.group(1)}"
    return gs_url


def build_sft_step(slug: str, rel_path: str, tokenize_step: ExecutorStep) -> ExecutorStep:
    resolved_gs = _resolve_latest_hf_checkpoint(rel_path)
    base_checkpoint = _to_mirror_uri(resolved_gs)
    logger.info("SFT[%s] init_from=%s (resolved=%s)", slug, base_checkpoint, resolved_gs)

    converter = HFCheckpointConverter.from_hf(base_checkpoint)
    model_config = converter.config_from_hf_checkpoint(base_checkpoint)

    tokenize_set_name = os.path.basename(tokenize_step.name)
    data_config = lm_data_config(
        training_set=tokenize_step,
        num_validation_sequences={tokenize_set_name: 100},
    )

    per_slug_sft_config = dataclasses.replace(
        DEFAULT_SFT_CONFIG,
        resources=ResourceConfig.with_tpu(SFT_RESOURCES[slug]),
        initialize_from_hf=base_checkpoint,
    )

    sft_step = default_sft(
        name=f"downstream_scaling/sft/delphi/gsm8k_qa/{slug}",
        tokenized=data_config,
        model_config=model_config,
        sft_config=per_slug_sft_config,
        tags=["sft", "downstream_scaling", "delphi", "gsm8k_qa", slug],
    )
    return sft_step.with_output_path(f"{SFT_OUTPUT_PREFIX}/{slug}")

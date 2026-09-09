# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded native TPU scoring check against the frozen Snowball GPU smoke."""

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import jax
import jmp
import levanter.trainer
from huggingface_hub import snapshot_download
from levanter.grug.sharding import compact_grug_mesh
from levanter.tokenizers import load_tokenizer
from levanter.tracker import NoopConfig
from levanter.trainer import TrainerConfig
from marin.testing.inference.snowball import read_prompt_fixture, read_representative_goldens
from rigging.filesystem.storage_path import StoragePath

from experiments.evaluation.native_snowball_losses import load_native_runner


@dataclass
class SmokeConfig:
    checkpoint_path: str
    executor_info_path: str
    output_path: str
    max_eval_length: int = 256
    trainer: TrainerConfig = field(default_factory=lambda: TrainerConfig(tracker=NoopConfig()))


def main(config: SmokeConfig):
    levanter.trainer.initialize(config)
    fixture = read_prompt_fixture(read_representative_goldens())
    tokenizer_path = snapshot_download(
        fixture.tokenizer,
        revision=fixture.tokenizer_revision,
        allow_patterns=["tokenizer*", "special_tokens*", "added_tokens*", "chat_template*"],
    )
    tokenizer = load_tokenizer(tokenizer_path).as_hf_tokenizer()
    cases = sorted(fixture.cases, key=lambda case: case.id)[:8]
    documents = [(case.id, tokenizer.decode(case.prompt_token_ids[:96], skip_special_tokens=True)) for case in cases]
    documents.extend(
        [
            ("long-document", "The snowy mountains reflect sunlight across the valley. " * 100 + "雪山"),
            ("duplicate-content", documents[0][1]),
            ("empty-document", ""),
        ]
    )
    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)
    with jax.set_mesh(mesh):
        runner = load_native_runner(
            checkpoint_path=config.checkpoint_path,
            executor_info_path=config.executor_info_path,
            tokenizer_path=tokenizer_path,
            eval_batch_size=jax.device_count(),
            max_eval_length=config.max_eval_length,
            mp=jmp.get_policy("params=bfloat16,compute=bfloat16,output=float32"),
            mesh=mesh,
        )
        logging.info("NATIVE_SMOKE_MODEL_LOADED")
        start = time.perf_counter()
        totals, counts = runner.score_token_totals([text for _, text in documents])
        elapsed = time.perf_counter() - start
        memory_stats = [device.memory_stats() for device in jax.local_devices()]
    rows = [
        dict(
            doc_id=doc_id,
            total_nll=float(total),
            scored_tokens=int(count),
            loss=float(total / count) if count else None,
            bits_per_byte=float(total / (math.log(2) * len(text.encode("utf-8")))) if count else None,
        )
        for (doc_id, text), total, count in zip(documents, totals, counts, strict=True)
    ]
    summary = dict(
        checkpoint=config.checkpoint_path,
        tokenizer_revision=fixture.tokenizer_revision,
        max_eval_length=config.max_eval_length,
        elapsed_seconds=elapsed,
        eval_batch_size=runner.eval_batch_size,
        local_device_memory_stats=memory_stats,
        results=rows,
    )
    if jax.process_index() == 0:
        payload = json.dumps(summary, indent=2)
        StoragePath(config.output_path).write_text(payload)
        Path(os.environ["IRIS_OUTPUT_DIR"], "summary.json").write_text(payload)
        logging.info("NATIVE_SMOKE_COMPLETE %s", json.dumps(summary))


if __name__ == "__main__":
    levanter.config.main(main)()

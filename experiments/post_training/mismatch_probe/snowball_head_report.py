# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare the Snowball head-precision arms and the BF16 repeat on frozen tokens."""

import hashlib
import json

import click

from experiments.post_training.mismatch_probe.metrics import comparison_metrics
from experiments.post_training.mismatch_probe.report import (
    ArchiveData,
    _comparison_rows,
    _finite_json,
    _route_counts,
    load_archive,
)
from experiments.post_training.mismatch_probe.snowball_head import HEAD_PRECISION

REPLAY = "trainer@0:router_replay"
SERVING = "vllm.rescore@0"
METRICS = ("tokens", "abs_mean", "abs_p99", "k3", "chosen_token_binary_kl")


def _frozen_identity(archive: ArchiveData) -> tuple:
    manifest = archive.manifest
    batch = json.loads(manifest.batch_layout_json)
    return (
        manifest.checkpoint_path,
        manifest.runtime_commit,
        manifest.probe_hash,
        manifest.tokenizer_fingerprint,
        manifest.starting_global_step,
        manifest.hardware_json,
        batch["sample_ids"],
        batch["native_order"],
        batch["padded_rows"],
        batch["native_micro_batch_size"],
    )


def _route_hash(archive: ArchiveData) -> str:
    digest = hashlib.sha256()
    for row in archive.probes:
        if row.routed_experts is None:
            raise ValueError(f"Missing captured routes for {row.sample_id}")
        metadata = (row.sample_id, row.routed_experts_shape, row.routed_experts_dtype, row.route_valid_mask)
        digest.update(json.dumps(metadata).encode())
        digest.update(row.routed_experts)
    return digest.hexdigest()


def summarize(archives: dict[str, ArchiveData]) -> dict:
    baseline = archives["bf16"]
    route_hash = _route_hash(baseline)
    result = {"probe_hash": baseline.manifest.probe_hash, "route_hash": route_hash, "arms": {}}
    for arm, archive in archives.items():
        manifest = archive.manifest
        if _frozen_identity(archive) != _frozen_identity(baseline) or _route_hash(archive) != route_hash:
            raise ValueError(f"{arm} does not use the same checkpoint, runtime, layout, tokens and captured routes")
        if arm != "bf16" and manifest.source_probe_archive != baseline.manifest.archive:
            raise ValueError(f"{arm} did not reuse the BF16 collector archive")
        if manifest.scored_updates != [0] or manifest.optimizer_steps_per_update != 0:
            raise ValueError(f"{arm} is not a scoring-only run")
        if any(
            row.prompt_token_ids != row.trainer_prompt_ids or row.vllm_output_ids != row.trainer_input_ids
            for row in archive.probes
        ):
            raise ValueError(f"{arm} has different token IDs in the trainer and vLLM")
        config = json.loads(manifest.config_json)
        serving_head = config["generator"]["engine_init_kwargs"].get("hf_overrides", {}).get("head_dtype", "bfloat16")
        learner_head = config["trainer"]["policy"]["megatron_config"].get("fp32_lm_head", False)
        serving_fp32, learner_fp32 = HEAD_PRECISION[arm]
        if serving_head != ("float32" if serving_fp32 else "bfloat16") or learner_head != learner_fp32:
            raise ValueError(f"{arm} recorded the wrong head precision flags")
        counts, matches = _route_counts(archive.probes, archive.scores[REPLAY])
        if counts.sum() == 0 or counts.sum() != matches.sum():
            raise ValueError(f"{arm} did not replay all captured routes on valid loss tokens")
        timing = json.loads(manifest.timing_json)
        scores = _comparison_rows(archive.probes, archive.scores, REPLAY, SERVING).metrics(
            list(range(len(archive.probes)))
        )
        result["arms"][arm] = {
            "archive": manifest.archive,
            "config_sha256": hashlib.sha256(manifest.config_json.encode()).hexdigest(),
            "replayed_route_rows": int(counts.sum()),
            **{metric: scores[metric] for metric in METRICS},
            "serve_scored_tokens_per_second": scores["tokens"] / timing[f"{SERVING}/seconds"],
            "trainer_scored_tokens_per_second": scores["tokens"] / timing[f"{REPLAY}/seconds"],
        }
    repeat = archives["bf16-repeat"]
    result["bf16_repeat"] = {}
    for role, scoring in (("serve", SERVING), ("trainer", REPLAY)):
        scores = comparison_metrics(
            [repeat.scores[scoring][row.sample_id].logprobs for row in baseline.probes],
            [baseline.scores[scoring][row.sample_id].logprobs for row in baseline.probes],
            [row.loss_mask for row in baseline.probes],
        )
        result["bf16_repeat"][role] = {metric: scores[metric] for metric in METRICS}
    return _finite_json(result)


@click.command(help=__doc__)
@click.argument("archives", nargs=len(HEAD_PRECISION))
def main(archives: tuple[str, ...]) -> None:
    """Read archives in BF16, BF16 repeat, vLLM FP32, Megatron FP32, both FP32 order."""
    loaded = {arm: load_archive(uri) for arm, uri in zip(HEAD_PRECISION, archives, strict=True)}
    click.echo(json.dumps(summarize(loaded), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reproduce the frozen 100-response Hero sanity check on a supplied checkpoint.

Prepare on CPU, then launch ``run request.json`` on an Iris 16-task/four-process
GPU gang. Generation uses two waves, restores once, and retains raw probabilities
remotely alongside small derived transcripts. It measures no serving throughput.
"""

import argparse
import hashlib
import json
import logging
import os
import unicodedata
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Self

import draccus
import jax
import jax.numpy as jnp
import numpy as np
from haliax.jax_utils import broadcast_one_to_all
from jax.experimental.multihost_utils import sync_global_devices
from levanter.cutlass_kernel_cache import cutlass_kernel_cache, install
from levanter.distributed import DistributedConfig
from levanter.grug.sharding import compact_grug_mesh
from levanter.tokenizers import MarinTokenizer, load_tokenizer
from pydantic import Field, model_validator
from rigging.filesystem.conditional_object import conditional_object
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.log_setup import configure_logging

from experiments.grug.moe_hero_ep.format_eval import EOT_ID, Panel, Serialization, grade_panel, render_panel
from experiments.grug.moe_hero_ep.format_sample import (
    NativePanelRequest,
    native_evaluation_model,
    native_logits,
    sample_panel,
)
from experiments.grug.moe_hero_ep.model import GrugModelConfig
from experiments.grug.moe_hero_ep.ops.vibe_check.completions import (
    Checkpoint,
    Record,
    SampleRequest,
    SampleResult,
    StopReason,
    digest,
)
from experiments.grug.moe_hero_ep.train import _apply_hero_ep_runtime_defaults
from experiments.grug.moe_hero_ep.weights import cast_weights, restore_weights

PANEL_DIRECTORY = Path(__file__).with_name("rollout_sanity_data")


class RolloutSanityRequest(Record):
    native_request: NativePanelRequest
    source_model: dict
    destination: str
    batch_size: int = Field(default=64, gt=0)
    physical_padding: int = Field(default=4096, gt=0)
    overlap_evidence: dict

    @model_validator(mode="after")
    def validate_model(self) -> Self:
        native = self.native_request
        source = draccus.decode(GrugModelConfig, self.source_model)
        actual = draccus.decode(GrugModelConfig, native.model)
        if actual != native_evaluation_model(source):
            raise ValueError("Evaluation must retain source forward settings with explicit dropless routing")
        if native.serialization != Serialization.MARIN or native.replica_axis_size != 1:
            raise ValueError("Sanity sampling requires Marin serialization and replica1/data mesh")
        if native.panel.context_length != source.max_seq_len or self.physical_padding > source.max_seq_len:
            raise ValueError("Sampling context or physical padding differs from the source architecture")
        return self


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def write_json(path: StoragePath, value: dict) -> str:
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    conditional_object(str(path)).write(raw, expected_version=None)
    return sha256(raw)


def frozen_panel(source_revision: str) -> tuple[Panel, list[dict]]:
    """Load the exact public questions, order and seeds, rebinding implementation identity."""
    value = json.loads((PANEL_DIRECTORY / "panel.json").read_text())
    value["source_revision"] = source_revision
    panel = Panel.model_validate(value)
    references = json.loads((PANEL_DIRECTORY / "references.json").read_text())
    return panel, references


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def check_disjoint(panel: Panel, references: list[dict], splits: dict[str, Path]) -> dict:
    """Check exact normalized user turns and conversations against supplied SFT splits."""
    by_id = {row["id"]: row for row in references}
    evidence = {}
    for split, path in splits.items():
        raw = path.read_bytes()
        rows = [json.loads(line) for line in raw.decode().splitlines()]
        users = {
            normalized(message["content"]) for row in rows for message in row["messages"] if message["role"] == "user"
        }
        conversations = {normalized("\n".join(message["content"] for message in row["messages"])) for row in rows}
        matches = []
        for case in panel.cases:
            probes = [message["content"] for message in case.messages if message["role"] == "user"]
            reference = by_id[case.id]
            if "raw_question" in reference:
                probes.append(reference["raw_question"])
            conversation = normalized("\n".join(message["content"] for message in case.messages))
            if any(normalized(probe) in users for probe in probes) or conversation in conversations:
                matches.append(case.id)
        if matches:
            raise ValueError(f"Panel overlaps supplied {split} split: {matches}")
        evidence[split] = {"sha256": sha256(raw), "conversations": len(rows), "exact_matches": matches}
    return {
        "algorithm": "NFKC, casefold, whitespace collapse; exact user-turn or full-conversation matches only",
        "splits": evidence,
    }


def prepare_request(
    checkpoint: Checkpoint,
    source_model: dict,
    source_revision: str,
    target_cluster: str,
    destination: str,
    splits: dict[str, Path],
    output: Path,
) -> RolloutSanityRequest:
    """Freeze the exact panel and request without model-array reads or submission."""
    if set(splits) != {"train", "heldout"}:
        raise ValueError("Supply both actual prepared train and heldout JSONL splits")
    panel, references = frozen_panel(source_revision)
    evidence = check_disjoint(panel, references, splits)
    source = draccus.decode(GrugModelConfig, source_model)
    request = RolloutSanityRequest(
        native_request=NativePanelRequest(
            panel=panel,
            checkpoint=checkpoint,
            model=draccus.encode(native_evaluation_model(source)),
            serialization=Serialization.MARIN,
            target_cluster=target_cluster,
            expert_axis_size=1,
            replica_axis_size=1,
        ),
        source_model=source_model,
        destination=destination,
        overlap_evidence=evidence,
    )
    tokenizer = load_tokenizer(f"{panel.tokenizer}@{panel.tokenizer_revision}")
    validate_rendered_window(request, tokenizer)
    output.mkdir(parents=True, exist_ok=False)
    root = StoragePath(str(output))
    for name, value in (
        ("panel.json", panel.model_dump(mode="json")),
        ("references.json", {"cases": references}),
        ("request.json", request.model_dump(mode="json")),
    ):
        write_json(root / name, value)
    return request


def validate_rendered_window(request: RolloutSanityRequest, tokenizer: MarinTokenizer) -> None:
    spec = render_panel(request.native_request.panel, tokenizer, Serialization.MARIN)
    if any(
        len(tokenizer.encode(prompt.text, add_special_tokens=False)) + spec.max_new_tokens > request.physical_padding
        for prompt in spec.prompts
    ):
        raise ValueError("Rendered prompt plus full generation cap exceeds physical padding")


def transcript_view(samples: SampleResult, source_sha256: str) -> dict:
    """Omit only probability payloads; keep all text, token IDs, seeds and stopping."""
    payload = samples.model_dump(mode="json")
    for row in payload["completions"]:
        row["expected_scores"] = None
        for sample in row["samples"]:
            sample["token_scores"] = None
    return {
        "kind": "derived_slim_transcripts",
        "derivation": "Only token_scores/expected_scores omitted; original samples.json retained unchanged",
        "source_samples_sha256": source_sha256,
        "source_sample_id": samples.request.sample_id,
        "samples": payload,
    }


def completion_summary(samples: SampleResult, report: dict) -> dict:
    failures = {row["id"]: row["failures"] for row in report["generations"]}
    rows = [
        {
            "prompt_id": row.prompt_id,
            "prompt_tokens": len(row.prompt_token_ids),
            "generated_tokens": len(row.samples[0].token_ids),
            "stop_reason": row.samples[0].stop_reason.value,
            "cap_hit": row.samples[0].stop_reason == StopReason.MAX_NEW_TOKENS,
            "seed": row.samples[0].seed,
            "completion_status": (
                "eot_completed" if row.samples[0].stop_reason == StopReason.EOS else "unfinished_censored"
            ),
            "formal_structural_failures": failures[row.prompt_id],
            "role_leak_flag": "role_or_observation_continuation" in failures[row.prompt_id],
        }
        for row in samples.completions
    ]
    return {
        "sample_id": samples.request.sample_id,
        "cases": len(rows),
        "rollouts": rows,
        "generated_tokens": sum(row["generated_tokens"] for row in rows),
        "capped_unfinished_cases": sum(row["cap_hit"] for row in rows),
        "stop_counts": {reason.value: sum(row["stop_reason"] == reason.value for row in rows) for reason in StopReason},
        "censoring_note": (
            "A cap hit is unfinished/censored, not evidence of role leakage or content failure; "
            "inspect the raw response."
        ),
    }


def sample_waves(
    request: RolloutSanityRequest,
    tokenizer: MarinTokenizer,
    logits: Callable[[np.ndarray, np.ndarray], np.ndarray],
    destination: str,
) -> dict:
    """Generate ordered fixed-size waves and publish a hash-bound complete result."""
    validate_rendered_window(request, tokenizer)
    native = request.native_request
    root = StoragePath(destination)
    completions = []
    waves = []
    for wave, start in enumerate(range(0, len(native.panel.cases), request.batch_size)):
        wave_panel = native.panel.model_copy(update={"cases": native.panel.cases[start : start + request.batch_size]})
        wave_request = native.model_copy(update={"panel": wave_panel})
        durations = []

        def cropped_logits(tokens: np.ndarray, positions: np.ndarray, durations: list[float] = durations) -> np.ndarray:
            if np.any(positions >= request.physical_padding):
                raise ValueError("Generation position escaped physical causal prefix")
            begin = perf_counter()
            scores = logits(tokens[:, : request.physical_padding], positions)
            durations.append(perf_counter() - begin)
            return scores

        begin = perf_counter()
        samples, report = sample_panel(wave_request, tokenizer, cropped_logits, batch_size=request.batch_size)
        completions.extend(samples.completions)
        timing = {
            "wave": wave,
            "actual_prompts": len(wave_panel.cases),
            "physical_batch": request.batch_size,
            "physical_sequence_length": request.physical_padding,
            "model_calls": len(durations),
            "model_seconds": sum(durations),
            "wave_seconds": perf_counter() - begin,
            "timing_scope": "Full-prefix recompute; no KV cache; not serving/RL throughput",
        }
        files = {}
        if jax.process_index() == 0:
            wave_root = root / "waves" / str(wave)
            for name, value in (
                ("request.json", wave_request.model_dump(mode="json")),
                ("samples.json", samples.model_dump(mode="json")),
                ("report.json", report),
                ("timing.json", timing),
            ):
                files[name] = write_json(wave_root / name, value)
            files["transcripts.json"] = write_json(
                wave_root / "transcripts.json", transcript_view(samples, files["samples.json"])
            )
            files["summary.json"] = write_json(wave_root / "summary.json", completion_summary(samples, report))
        waves.append(
            {"wave": wave, "start": start, "cases": len(wave_panel.cases), "file_sha256": files, "timing": timing}
        )
        sync_global_devices(f"sanity-wave-{wave}-saved")
    spec = render_panel(native.panel, tokenizer, Serialization.MARIN)
    result = SampleResult(
        request=SampleRequest(
            checkpoint=native.checkpoint,
            spec=spec,
            model=native.model,
            source_revision=native.panel.source_revision,
            target_cluster=native.target_cluster,
        ),
        completions=tuple(completions),
        completed_at=datetime.now(UTC).isoformat(),
        eos_token_id=EOT_ID,
    )
    report = grade_panel(native.panel, result, Serialization.MARIN)
    manifest = {
        "request_digest": digest(request.model_dump(mode="json")),
        "checkpoint": native.checkpoint.model_dump(),
        "panel_id": native.panel.panel_id,
        "sample_id": result.request.sample_id,
        "cases": len(result.completions),
        "waves": waves,
    }
    if jax.process_index() == 0:
        files = {}
        for name, value in (
            ("request.json", request.model_dump(mode="json")),
            ("samples.json", result.model_dump(mode="json")),
            ("report.json", report),
        ):
            files[name] = write_json(root / name, value)
        files["transcripts.json"] = write_json(root / "transcripts.json", transcript_view(result, files["samples.json"]))
        files["summary.json"] = write_json(root / "summary.json", completion_summary(result, report))
        files["runtime.json"] = write_json(
            root / "runtime.json",
            {
                "source_model": request.source_model,
                "actual_evaluation_model": native.model,
                "processes": jax.process_count(),
                "physical_padding": request.physical_padding,
                "waves": waves,
                "scope": "Dropless BF16 evaluation; no serving/RL throughput or heldout quality acceptance",
            },
        )
        manifest["file_sha256"] = files
        write_json(root / "manifest.json", manifest)
    return manifest


def run(request_path: Path) -> None:
    """Run one immutable Iris gang attempt, restoring the supplied model once."""
    raw = request_path.read_bytes()
    request = RolloutSanityRequest.model_validate_json(raw)
    native = request.native_request
    config = draccus.decode(GrugModelConfig, native.model)
    _apply_hero_ep_runtime_defaults(
        inline_watch_enabled=False,
        processes_per_task=4,
        moe_implementation=config.moe_implementation,
        remat_mode=config.remat_mode,
    )
    configure_coreweave_s3()
    DistributedConfig().initialize()
    configure_logging(logging.INFO if jax.process_index() == 0 else logging.WARNING)
    if (
        jax.default_backend() != "gpu"
        or jax.process_count() != 64
        or jax.device_count() != 64
        or len(jax.local_devices()) != 1
        or request.batch_size != 64
    ):
        raise ValueError("Launch on 16 tasks with four GPU processes each")
    metadata = json.loads((StoragePath(native.checkpoint.uri) / "metadata.json").read_text())
    if (
        digest(metadata) != native.checkpoint.metadata_digest
        or metadata.get("is_temporary") is not False
        or metadata["step"] != native.checkpoint.step
    ):
        raise ValueError("Supplied native checkpoint metadata changed or is incomplete")
    uid = os.environ["IRIS_ATTEMPT_UID"].encode("ascii")
    if not uid.isalnum() or len(uid) > 64:
        raise ValueError("Unexpected Iris attempt UID")
    buffer = np.zeros(64, np.uint8)
    buffer[: len(uid)] = np.frombuffer(uid, dtype=np.uint8)
    leader = np.asarray(broadcast_one_to_all(buffer)).tobytes().rstrip(b"\0").decode("ascii")
    root = StoragePath(request.destination)
    destination = str(root / "attempts" / leader)
    record = {
        "request_sha256": sha256(raw),
        "request_digest": digest(request.model_dump(mode="json")),
        "leader_attempt_uid": leader,
        "destination": destination,
        "resume_policy": "Restart interrupted computation in a fresh attempt; preserve completed prior waves",
    }
    if jax.process_index() == 0:
        write_json(root / "attempt-records" / f"{leader}.json", record)
    sync_global_devices("sanity-attempt-recorded")
    tokenizer = load_tokenizer(f"{native.panel.tokenizer}@{native.panel.tokenizer_revision}")
    validate_rendered_window(request, tokenizer)
    install(cutlass_kernel_cache())
    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)
    with jax.set_mesh(mesh):
        model = cast_weights(
            restore_weights(native.checkpoint.uri, native.checkpoint.metadata_digest, config, mesh), jnp.bfloat16
        )
        jax.block_until_ready(model)
        manifest = sample_waves(request, tokenizer, native_logits(model, mesh), destination)
    if jax.process_index() == 0:
        raw_manifest = (StoragePath(destination) / "manifest.json").read_bytes()
        write_json(
            root / "completions" / f"{leader}.json",
            {**record, "manifest_sha256": sha256(raw_manifest), "cases": manifest["cases"]},
        )
    sync_global_devices("sanity-attempt-finished")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint JSON including URI, metadata digest, step, run ID and timestamp",
    )
    prepare.add_argument("--source-model", type=Path, required=True, help="Saved GrugModelConfig JSON")
    prepare.add_argument("--source-revision", required=True)
    prepare.add_argument("--target-cluster", required=True)
    prepare.add_argument("--destination", required=True)
    prepare.add_argument("--train-jsonl", type=Path, required=True)
    prepare.add_argument("--heldout-jsonl", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    worker = commands.add_parser("run")
    worker.add_argument("request", type=Path)
    args = parser.parse_args()
    if args.command == "run":
        run(args.request)
        return
    prepare_request(
        Checkpoint.model_validate_json(args.checkpoint.read_text()),
        json.loads(args.source_model.read_text()),
        args.source_revision,
        args.target_cluster,
        args.destination,
        {"train": args.train_jsonl, "heldout": args.heldout_jsonl},
        args.output,
    )


if __name__ == "__main__":
    main()

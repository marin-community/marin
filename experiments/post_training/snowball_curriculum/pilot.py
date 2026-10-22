# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze and annotate a source-stratified pre-admission Snowball curriculum pilot."""

import argparse
import json
import os
from collections import Counter
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa
import pyarrow.parquet as pq
from marin.inference.openai_chat import OpenAIChatClient
from marin.processing.embeddings.harrier import HARRIER_DIM, HARRIER_REPO, HARRIER_REVISION, HarrierConfig, load_harrier
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.chat_requests import chat_output
from taskcompendium.pipeline.curriculum.embedding import embed_texts, task_text_records
from taskcompendium.pipeline.curriculum.models import EmbeddingSpec
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

from experiments.post_training.snowball_curriculum.annotations import (
    PROMPT_VERSION,
    AnnotationModel,
    annotate_inputs,
    annotation_request,
    public_input,
    quality_input,
)
from experiments.post_training.snowball_curriculum.selection import load_sources, read_candidates, select_candidates


def freeze_pilot(
    manifest_uri: str,
    *,
    output: Path,
    target: int,
    seed: int,
    heldout_contract_ids: list[str],
    heldout_public_content_ids: list[str],
) -> dict:
    """Freeze the original pilot denominator before any model quality decisions."""
    source_manifest = json.loads(StoragePath(manifest_uri).read_bytes())
    sources = load_sources(manifest_uri, population="converted")
    selected = select_candidates(
        partial(read_candidates, sources),
        target=target,
        seed=seed,
        heldout_contract_ids=heldout_contract_ids,
        heldout_public_content_ids=heldout_public_content_ids,
    )
    output.mkdir(parents=True, exist_ok=False)
    rows = [
        {
            **asdict(candidate),
            "aliases": [asdict(alias) for alias in selected.selected_aliases[candidate.contract_id]],
            "inclusion_probability": (
                selected.difficulty_quotas[candidate.source_id][candidate.difficulty]
                / selected.eligible_difficulty_counts[candidate.source_id][candidate.difficulty]
            ),
        }
        for candidate in selected.selected
    ]
    (output / "tasks.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    manifest = {
        "format": 1,
        "population": "converted",
        "sampling": "half-equal-source-half-sqrt-eligible-population",
        "corpus_manifest": manifest_uri,
        "corpus_manifest_sha256": canonical_sha256(source_manifest),
        "sources": [asdict(source) for source in sources],
        "target": target,
        "seed": seed,
        "selected": len(rows),
        "selection_id": canonical_sha256(rows),
        "candidate_counts": selected.candidate_counts,
        "eligible_counts": selected.eligible_counts,
        "source_quotas": selected.source_quotas,
        "eligible_difficulty_counts": selected.eligible_difficulty_counts,
        "difficulty_quotas": selected.difficulty_quotas,
        "difficulty_counts": selected.difficulty_counts,
        "rejected_counts": selected.rejected_counts,
        "shortfall": selected.shortfall,
        "heldout_contract_ids": sorted(heldout_contract_ids),
        "heldout_public_content_ids": sorted(heldout_public_content_ids),
        "refill_after_quality": False,
        "prior_admission": "unknown unless supplied by source lineage; never inferred from static quality",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def load_frozen(path: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads((path / "manifest.json").read_text())
    rows = [json.loads(line) for line in (path / "tasks.jsonl").read_text().splitlines() if line.strip()]
    if len(rows) != manifest["selected"] or canonical_sha256(rows) != manifest["selection_id"]:
        raise ValueError("Frozen task selection does not match its manifest")
    return manifest, rows


def annotate_pilot(
    frozen: Path,
    *,
    output: Path,
    quality_model: AnnotationModel,
    annotation_model: AnnotationModel,
    quality_submit,
    annotation_submit,
    cache_root: str,
    execute: bool,
) -> dict:
    """Emit reviewable requests, or execute cached quality and annotation stages."""
    manifest, rows = load_frozen(frozen)
    output.mkdir(parents=True, exist_ok=False)
    tasks = {row["contract_id"]: TaskSpec.model_validate(row["task"]) for row in rows}
    public = {row["contract_id"]: public_input(tasks[row["contract_id"]], row["source_id"]) for row in rows}
    private = {identifier: quality_input(task) for identifier, task in tasks.items()}
    plan = {
        "selection_id": manifest["selection_id"],
        "original_denominator": len(rows),
        "prompt_version": PROMPT_VERSION,
        "quality_model": asdict(quality_model),
        "annotation_model": asdict(annotation_model),
        "executed": execute,
        "quality_evidence": "inline TaskSpec contracts; external resources unresolved",
        "prior_admission_counts": dict(Counter(row.get("prior_admission_status", "unknown") for row in rows)),
        "refill_after_quality": False,
    }
    requests = [annotation_request(key, value, model=quality_model, stage="quality") for key, value in private.items()]
    (output / "quality-requests.jsonl").write_text("".join(json.dumps(row) + "\n" for row in requests))
    if execute:
        quality, quality_traces = annotate_inputs(
            private, stage="quality", model=quality_model, submit=quality_submit, cache_root=cache_root
        )
        kept = {
            row.content_hash: row.public_input for key, row in public.items() if quality[key].get("verdict") == "keep"
        }
        annotations, annotation_traces = annotate_inputs(
            kept, stage="annotation", model=annotation_model, submit=annotation_submit, cache_root=cache_root
        )
        sidecars = []
        descriptions = {}
        for row in rows:
            key = row["contract_id"]
            content_hash = public[key].content_hash
            annotation = annotations.get(content_hash) if quality[key].get("verdict") == "keep" else None
            sidecars.append(
                {
                    "source_id": row["source_id"],
                    "task_id": row["task_id"],
                    "contract_id": key,
                    "aliases": row["aliases"],
                    "public_content_id": row["public_content_id"],
                    "content_hash": content_hash,
                    "prior_admission_status": row.get("prior_admission_status", "unknown"),
                    "inclusion_probability": row["inclusion_probability"],
                    "quality": quality[key],
                    "annotation": annotation,
                }
            )
            if annotation and "summary" in annotation:
                descriptions[content_hash] = {
                    "content_hash": content_hash,
                    "status": "ready",
                    "description": annotation["summary"],
                    "reason": "",
                }
        (output / "sidecars.jsonl").write_text("".join(json.dumps(row) + "\n" for row in sidecars))
        (output / "traces.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in quality_traces + annotation_traces)
        )
        schema = pa.schema(
            [
                ("content_hash", pa.string()),
                ("status", pa.string()),
                ("description", pa.string()),
                ("reason", pa.string()),
            ]
        )
        pq.write_table(pa.Table.from_pylist(list(descriptions.values()), schema=schema), output / "descriptions.parquet")
        plan["quality_counts"] = dict(Counter(row.get("verdict", "failed") for row in quality.values()))
        plan["annotated"] = len(descriptions)
        plan["by_source"] = {
            source: dict(
                Counter(
                    quality[row["contract_id"]].get("verdict", "failed") for row in rows if row["source_id"] == source
                )
            )
            for source in sorted({row["source_id"] for row in rows})
        }
    (output / "manifest.json").write_text(json.dumps(plan, sort_keys=True, indent=2) + "\n")
    return plan


def embed_pilot(
    descriptions: Path, *, output: Path, cache_root: str, execute: bool, device: Literal["cpu", "cuda"]
) -> dict:
    """Prepare or run the existing content-addressed Harrier embedding pass."""
    spec = EmbeddingSpec(
        model=HARRIER_REPO,
        revision=HARRIER_REVISION,
        tokenizer_revision=HARRIER_REVISION,
        dimensions=HARRIER_DIM,
        max_tokens=512,
        compute_dtype="float32",
    )
    description_rows = pq.read_table(descriptions).to_pylist()
    embedding_inputs = [record for row in description_rows for record in task_text_records(row, spec)]
    plan: dict[str, Any] = {
        "descriptions": str(descriptions),
        "descriptions_sha256": canonical_sha256(description_rows),
        "embedding_inputs_sha256": canonical_sha256(embedding_inputs),
        "embedding_spec": spec.model_dump(mode="json"),
        "executed": execute,
    }
    output.mkdir(parents=True, exist_ok=False)
    inputs_path = output / "embedding_inputs.parquet"
    input_schema = pa.schema([("content_hash", pa.string()), ("embedding_key", pa.string()), ("text", pa.string())])
    pq.write_table(pa.Table.from_pylist(embedding_inputs, schema=input_schema), inputs_path)
    plan["embedding_inputs"] = str(inputs_path)
    if execute:
        config = HarrierConfig(
            model=spec.model,
            revision=spec.revision,
            tokenizer_revision=spec.tokenizer_revision,
            dimensions=spec.dimensions,
            max_tokens=spec.max_tokens,
            compute_dtype=spec.compute_dtype,
            device=device,
        )
        texts = Dataset.from_list([str(inputs_path)]).load_parquet()
        with ZephyrContext(max_workers=1, chunk_storage_prefix=str(output / "chunks")) as context:
            plan["paths"] = embed_texts(
                context,
                texts=texts,
                spec=spec,
                embedder_factory=partial(load_harrier, config),
                cache_root=cache_root,
                output_root=str(output),
            )
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--manifest", required=True)
    freeze.add_argument("--target", type=int, default=10000)
    freeze.add_argument("--seed", type=int, default=0)
    freeze.add_argument(
        "--holdouts",
        type=Path,
        required=True,
        help="JSON with contract_ids/public_content_ids, explicit empty arrays if no known holdouts",
    )
    freeze.add_argument("--output", type=Path, required=True)
    annotate = commands.add_parser("annotate")
    annotate.add_argument("--frozen", type=Path, required=True)
    annotate.add_argument("--output", type=Path, required=True)
    annotate.add_argument("--cache-root", required=True)
    annotate.add_argument(
        "--bindings", type=Path, required=True, help="JSON quality/annotation endpoint, model, revision and token_env"
    )
    annotate.add_argument("--execute", action="store_true")
    embed = commands.add_parser("embed")
    embed.add_argument("--descriptions", type=Path, required=True)
    embed.add_argument("--output", type=Path, required=True)
    embed.add_argument("--cache-root", required=True)
    embed.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    embed.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.command == "freeze":
        holdouts = json.loads(args.holdouts.read_text())
        result = freeze_pilot(
            args.manifest,
            output=args.output,
            target=args.target,
            seed=args.seed,
            heldout_contract_ids=holdouts["contract_ids"],
            heldout_public_content_ids=holdouts["public_content_ids"],
        )
    elif args.command == "annotate":
        bindings = json.loads(args.bindings.read_text())
        models, submitters = {}, {}
        for stage in ("quality", "annotation"):
            binding = bindings[stage]
            models[stage] = AnnotationModel(
                model=binding["model"], revision=binding["revision"], endpoint=binding["endpoint"]
            )
            submitters[stage] = None
            if args.execute:
                submitters[stage] = partial(
                    chat_output,
                    OpenAIChatClient(binding["endpoint"], os.environ[binding["token_env"]], timeout=180),
                    max_concurrent=8,
                )
        result = annotate_pilot(
            args.frozen,
            output=args.output,
            quality_model=models["quality"],
            annotation_model=models["annotation"],
            quality_submit=submitters["quality"],
            annotation_submit=submitters["annotation"],
            cache_root=args.cache_root,
            execute=args.execute,
        )
    else:
        result = embed_pilot(
            args.descriptions, output=args.output, cache_root=args.cache_root, execute=args.execute, device=args.device
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

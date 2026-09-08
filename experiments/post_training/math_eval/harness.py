# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prove finalized dumps, bind frozen pool membership, and materialize scored rows."""

import hashlib
import json
import math
import posixpath
from collections import Counter, defaultdict
from dataclasses import asdict

import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training import async_rl_audit as audit
from experiments.post_training.math_eval.audit_overlay import validated_statuses
from experiments.post_training.math_eval.contract import CONTRACT_IDS, PromptTemplate, render_prompt
from experiments.post_training.math_eval.scoring import score_row


def summarize(records):
    """Sequence means retain their own raw, binary, completion and diagnostic channels."""
    populations = defaultdict(list)
    for record in records:
        populations["all"].append(record)
        populations[record["bin"]].append(record)
    result = {}
    for name, rows in populations.items():
        count = len(rows)
        result[name] = {"sequences": count, "questions": len({row["prompt_sha256"] for row in rows})}
        for key in ("score_contract", "contract_correct", "score_contract_completed", "truncated", "thinking_closed"):
            result[name][key] = sum(row[key] for row in rows) / count
        result[name]["semantic_resolved"] = sum(row["score_semantic"] is not None for row in rows)
        result[name]["semantic_status"] = dict(Counter(row["semantic_status"] for row in rows))
        result[name]["legacy_positive_reward"] = sum(row["legacy_positive_reward"] for row in rows) / count
    return dict(result)


def _read_and_score(
    root,
    step,
    *,
    decoder,
    model,
    thinking,
    expected_rows,
    samples,
    wandb_metrics,
    bind_row,
    metric_reference="wandb",
):
    provenance = {"wandb": "finalized_trajectory", "serving_score_receipt": "raw_engine_response"}
    if metric_reference not in provenance:
        raise ValueError("Unknown generation metric reference")
    proof, audited = audit.audit_eval_dump(
        root,
        step,
        expected_rows,
        samples,
        wandb_metrics,
        True,
        expected_token_provenance=provenance[metric_reference],
        metric_reference=metric_reference,
    )
    uri = posixpath.join(root, "dumped_evals", f"global_step_{step}_evals")
    raw_rows = []
    for entry in audit.list_entries(uri):
        if posixpath.basename(entry["name"]) == "aggregated_results.jsonl":
            continue
        fs, path = audit.fs_path(posixpath.join(uri, posixpath.basename(entry["name"])))
        with fs.open(path, "rb") as stream:
            raw_rows.extend(json.loads(line) for line in audit.bounded_lines(stream))
    raw_rows.sort(key=lambda row: row["row_ordinal"])
    if len(raw_rows) != expected_rows:
        raise ValueError("Dump membership changed after its audit")
    records = []
    for ordinal, (row, proven) in enumerate(zip(raw_rows, audited, strict=True)):
        score = sum(row["score"]) if isinstance(row["score"], list) else row["score"]
        identity = [
            row["uid"],
            audit.canonical_sha(row["prompt_token_ids"]),
            audit.canonical_sha(row["response_ids"]),
            score,
            row["stop_reason"],
            row.get("generator_engine_index"),
        ]
        if identity != proven or row["row_ordinal"] != ordinal:
            raise ValueError("Dump tokens, scores or order changed after its audit")
        if decoder.decode(row["response_ids"], skip_special_tokens=False) != row["output_response"]:
            raise ValueError("Dump response text does not decode from its proven tokens")
        binding = bind_row(row)
        scored = asdict(score_row(row, decoder, model=model, thinking=thinking))
        outcome = (row["score"][-1] if row["score"] else 0) if isinstance(row["score"], list) else score
        records.append(
            scored
            | binding
            | {
                "row_ordinal": ordinal,
                "step": step,
                "prompt_token_ids_sha256": identity[1],
                "response_ids_sha256": identity[2],
                "generator_engine_index": identity[5],
                "legacy_positive_reward": float(outcome > 0),
            }
        )
    summary = summarize(records)
    for name, metrics in summary.items():
        key = f"eval/{name.replace('/', '_')}/avg_score"
        if not math.isclose(metrics["score_contract"], proof["metrics"][key], rel_tol=0, abs_tol=1e-12):
            raise ValueError("Scored native aggregate differs from dump audit")
    return records, {"audit": proof, "summary": summary}


def build_records(
    root,
    step,
    *,
    manifest,
    selection,
    overlay,
    expected_ids,
    template: PromptTemplate,
    decoder,
    model,
    tokenizer_sha256,
    samples,
    wandb_metrics,
    output_uri,
    metric_reference="wandb",
):
    """Bind every response to accepted immutable pool membership and the frozen template.

    Caller supplies the pinned tokenizer loaded from its verified artifact. The returned
    proof records its exact hash. All expected questions must occur exactly samples times;
    heldout membership cannot silently disappear or be replaced by another pool row.
    """
    statuses, overlay_sha = validated_statuses(manifest, selection, overlay)
    lookup = {row["prompt_sha256"]: row for row in manifest}
    if len(expected_ids) != len(set(expected_ids)) or not expected_ids or not set(expected_ids) <= lookup.keys():
        raise ValueError("Invalid frozen expected question membership")
    seen = Counter()

    def bind(row):
        extra = row["env_extras"]["extra_info"]
        digest = extra["prompt_sha256"]
        if digest not in expected_ids or statuses[digest] != "accept":
            raise ValueError("Dump includes a question outside accepted frozen membership")
        item = lookup[digest]
        if (
            row["env_class"] != item["env_class"]
            or row["data_source"] != item["bin"]
            or extra["prompt_template_id"] != template.template_id
            or extra["contract"] != CONTRACT_IDS[item["env_class"]]
            or any(
                row["env_extras"][channel]["ground_truth"] != item["gold"] for channel in ("reward_model", "reward_spec")
            )
        ):
            raise ValueError("Dump metadata differs from the frozen manifest")
        expected_tokens = decoder.encode(
            render_prompt(item["problem"], item["env_class"], template), add_special_tokens=False
        ).ids
        if row["prompt_token_ids"] != expected_tokens:
            raise ValueError("Prompt tokens differ from the frozen answer template")
        seen[digest] += 1
        return {"bin": item["bin"], "split": item["split"], "prompt_template_id": template.template_id}

    records, receipt = _read_and_score(
        root,
        step,
        decoder=decoder,
        model=model,
        thinking=template.enable_thinking,
        expected_rows=len(expected_ids) * samples,
        samples=samples,
        wandb_metrics=wandb_metrics,
        bind_row=bind,
        metric_reference=metric_reference,
    )
    if seen != Counter({digest: samples for digest in expected_ids}):
        raise ValueError("Frozen question sample coverage differs")
    aggregate = audit.read_json(
        posixpath.join(root, "dumped_evals", f"global_step_{step}_evals", "aggregated_results.jsonl")
    )
    for source, metrics in receipt["summary"].items():
        for name, column in (
            ("contract_correct", "contract_correct"),
            ("contract_completed", "score_contract_completed"),
        ):
            key = f"eval/{source.replace('/', '_')}/{name}"
            for label, values in (("dump", aggregate), (metric_reference, wandb_metrics)):
                if key not in values or not math.isclose(values[key], metrics[column], rel_tol=0, abs_tol=1e-7):
                    raise ValueError(f"Frozen contract metric missing or differs in {label}: {key}")
    if metric_reference != "wandb":
        receipt["metric_reference"] = metric_reference
    receipt["contract_metric_parity_verified"] = True
    receipt.update(
        {
            "manifest_sha256": selection["manifest_sha256"],
            "audit_overlay_sha256": overlay_sha,
            "tokenizer_sha256": tokenizer_sha256,
            "prompt_template_id": template.template_id,
            "expected_ids_sha256": audit.canonical_sha(sorted(expected_ids)),
            "scope": "frozen_pool",
        }
    )
    return _write(records, receipt, output_uri)


def build_legacy_records(
    root, step, *, decoder, model, thinking, tokenizer_sha256, expected_proof, wandb_metrics, output_uri
):
    """Qualify historical dump parity only; token hashes cannot certify v2 pool membership."""

    def bind(row):
        row["env_extras"]["extra_info"]["prompt_sha256"] = row["prompt_token_ids_sha256"]
        return {"bin": row["data_source"], "split": "legacy", "prompt_template_id": None}

    records, receipt = _read_and_score(
        root,
        step,
        decoder=decoder,
        model=model,
        thinking=thinking,
        expected_rows=expected_proof["rows"],
        samples=1,
        wandb_metrics=wandb_metrics,
        bind_row=bind,
    )
    for key in ("ordered_prompt_sha256", "ordered_response_sha256", "ordered_result_sha256"):
        if receipt["audit"][key] != expected_proof[key]:
            raise ValueError("Historical qualification fixture changed")
    receipt.update({"scope": "legacy_token_hash_qualification_only", "tokenizer_sha256": tokenizer_sha256})
    return _write(records, receipt, output_uri)


def _write(records, receipt, output_uri):
    buffer = pa.BufferOutputStream()
    pq.write_table(pa.Table.from_pylist(records), buffer)
    content = buffer.getvalue().to_pybytes()

    receipt["records_sha256"] = hashlib.sha256(content).hexdigest()
    receipt["records"] = len(records)
    StoragePath(output_uri + "/records.parquet").write_bytes(content)
    StoragePath(output_uri + "/summary.json").write_bytes(json.dumps(receipt, sort_keys=True).encode())
    return receipt

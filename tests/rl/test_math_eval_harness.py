# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Real dump proof plus frozen membership, repeated samples, and scorer integration."""

import hashlib
import inspect
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from tokenizers import Regex, Tokenizer
from tokenizers.decoders import Fuse
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split

from experiments.post_training import async_rl_audit as audit
from experiments.post_training.curriculum_rl.pool import GSM8K_BIN, _pool_record
from experiments.post_training.math_eval import harness
from experiments.post_training.math_eval.audit_overlay import VERIFIER_REVISION, VERIFIER_SOURCES_SHA256
from experiments.post_training.math_eval.contract import QWEN, render_prompt
from experiments.post_training.math_eval.pool import SourceRows, build_pool
from experiments.post_training.math_eval.scoring import semantic_answer
from experiments.post_training.math_eval.semantic_worker import SemanticWorker


def Decoder():
    vocab = {chr(index): index for index in range(128)}
    decoder = Tokenizer(WordLevel(vocab, unk_token="?"))
    decoder.pre_tokenizer = Split(Regex(""), behavior="isolated")
    decoder.decoder = Fuse()
    return decoder


@pytest.fixture
def fixture(tmp_path):
    source = [
        _pool_record(question=f"What is {i} plus 3?", answer="5", pool_bin=GSM8K_BIN, split="test", index=i)
        for i in range(2)
    ]
    pool = build_pool(
        [SourceRows("fixture", "a" * 40, "MIT", source, "heldout")],
        {"qwen": lambda _text: [0], "snowball": lambda _text: [0]},
        version="fixture",
        code_sha="b" * 40,
        tokenizer_hashes={"qwen": "c" * 64, "snowball": "d" * 64},
    )
    overlay = {
        "manifest_sha256": pool.selection["manifest_sha256"],
        "verifier_revision": VERIFIER_REVISION,
        "verifier_sources_sha256": VERIFIER_SOURCES_SHA256,
        "audit_source_sha256": "e" * 64,
        "statuses": {row["prompt_sha256"]: "accept" for row in pool.manifest},
    }
    rows = []
    for index, source in enumerate(pool.records["qwen"]["heldout"]):
        for sample in range(2):
            response = "#### 5" if sample == 0 else "#### 9"
            tokens = list(map(ord, response))
            item = pool.manifest[index]
            prompt = list(map(ord, render_prompt(item["problem"], item["env_class"], QWEN)))
            rows.append(
                {
                    "uid": str(index),
                    "row_ordinal": len(rows),
                    "token_provenance": "finalized_trajectory",
                    "generator_engine_index": 0,
                    "prompt_token_ids": prompt,
                    "response_ids": tokens,
                    "prompt_token_ids_sha256": audit.canonical_sha(prompt),
                    "response_ids_sha256": audit.canonical_sha(tokens),
                    "response_length": len(tokens),
                    "score": [0, float(sample == 0)],
                    "stop_reason": "stop",
                    "data_source": source["data_source"],
                    "env_class": source["env_class"],
                    "env_extras": source,
                    "output_response": response,
                }
            )
    metrics = {
        f"eval/{name}/{metric}": value
        for name in ("all", GSM8K_BIN.name)
        for metric, value in (
            ("avg_score", 0.5),
            ("pass_at_2", 1.0),
            ("contract_correct", 0.5),
            ("contract_completed", 0.5),
        )
    }
    path = tmp_path / "dumped_evals/global_step_0_evals"
    path.mkdir(parents=True)
    (path / "aggregated_results.jsonl").write_text(json.dumps(metrics))
    raw = path / "rows.jsonl"
    raw.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return (
        dict(
            root=str(tmp_path),
            step=0,
            manifest=pool.manifest,
            selection=pool.selection,
            overlay=overlay,
            expected_ids=[row["prompt_sha256"] for row in pool.manifest],
            template=QWEN,
            decoder=Decoder(),
            model="qwen",
            tokenizer_sha256="c" * 64,
            samples=2,
            wandb_metrics=metrics,
            output_uri=str(tmp_path / "out"),
        ),
        raw,
        rows,
    )


def test_harness_proves_and_materializes_repeated_samples(fixture):
    kwargs, _raw, _rows = fixture
    receipt = harness.build_records(**kwargs)
    assert receipt["records"] == 4
    assert receipt["audit"]["hashes_verified"] == 8
    assert receipt["summary"]["all"]["score_contract"] == 0.5
    assert receipt["summary"]["all"]["score_contract_completed"] == 0.5
    assert receipt["summary"]["all"]["questions"] == 2
    table = pq.read_table(kwargs["output_uri"] + "/records.parquet").to_pylist()
    assert len(table) == 4 and all(row["thinking_closed"] for row in table)
    assert all(row["native_reward_reduction"] == "token_reward_sum" for row in table)
    with open(kwargs["output_uri"] + "/records.parquet", "rb") as stream:
        assert hashlib.sha256(stream.read()).hexdigest() == receipt["records_sha256"]


def test_harness_semantic_process_preserves_scores_and_binds_execution_receipt(fixture):
    kwargs, _raw, _rows = fixture
    original = harness.build_records(**kwargs)
    original_rows = pq.read_table(kwargs["output_uri"] + "/records.parquet").to_pylist()
    kwargs["output_uri"] += "-isolated"
    with SemanticWorker(
        source_sha256=hashlib.sha256(inspect.getsource(semantic_answer).encode()).hexdigest(),
        startup_timeout=60,
        row_timeout=30,
        cleanup_timeout=1,
    ) as worker:
        receipt = harness.build_records(**kwargs, semantic_worker=worker)
    assert receipt["summary"] == original["summary"]
    assert pq.read_table(kwargs["output_uri"] + "/records.parquet").to_pylist() == original_rows
    execution = json.loads(Path(kwargs["output_uri"] + "/semantic-execution.json").read_text())
    assert audit.canonical_sha(execution) == receipt["semantic_execution_sha256"]
    assert len(execution["rows"]) == 4
    assert [row["status"] for row in execution["rows"]] == [row["semantic_status"] for row in original_rows]


@pytest.mark.parametrize("field", ["gold", "template", "prompt", "text", "membership", "reject"])
def test_harness_refuses_auditable_but_wrong_contract_rows(fixture, field):
    kwargs, raw, rows = fixture
    if field == "gold":
        rows[0]["env_extras"]["reward_model"]["ground_truth"] = "9"
    elif field == "template":
        rows[0]["env_extras"]["extra_info"]["prompt_template_id"] = "wrong"
    elif field == "text":
        rows[0]["output_response"] = "#### 9"
    elif field == "prompt":
        rows[0]["prompt_token_ids"].append(42)
        rows[0]["prompt_token_ids_sha256"] = audit.canonical_sha(rows[0]["prompt_token_ids"])
    elif field == "membership":
        rows[0]["env_extras"]["extra_info"]["prompt_sha256"] = "unknown"
    else:
        kwargs["overlay"]["statuses"][kwargs["expected_ids"][0]] = "reject"
    raw.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match=r"frozen|decode|accepted"):
        harness.build_records(**kwargs)


def test_harness_checks_new_contract_metric_parity_independently_of_legacy_scores(fixture):
    kwargs, raw, _rows = fixture
    aggregate = raw.parent / "aggregated_results.jsonl"
    metrics = json.loads(aggregate.read_text())
    metrics["eval/all/contract_completed"] = 1.0
    aggregate.write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match="Frozen contract metric"):
        harness.build_records(**kwargs)


def test_raw_serving_dump_requires_explicit_reference_and_preserves_forensic_provenance(fixture):
    kwargs, raw, rows = fixture
    for row in rows:
        row["token_provenance"] = "raw_engine_response"
        row["generator_engine_index"] = None
        row["score"] = sum(row["score"])
    raw.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="provenance"):
        harness.build_records(**kwargs)
    receipt = harness.build_records(**kwargs, metric_reference="serving_score_receipt")
    assert receipt["metric_reference"] == "serving_score_receipt"
    assert receipt["audit"]["token_provenance"] == "raw_engine_response"
    assert receipt["summary"]["all"]["contract_correct"] == 0.5
    table = pq.read_table(kwargs["output_uri"] + "/records.parquet").to_pylist()
    assert all(row["native_reward_reduction"] == "scalar_identity" for row in table)

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regenerate the passage-bearing reading-comprehension eval shards with the
cluster-B policy (marin#6852), bypassing lm-eval (whose pinned fork is
incompatible with marin's current transformers/datasets/hf_hub).

Loads each RC dataset straight from HF, applies ``_lmh_doc_text`` (drops the
public passage, keeps question+answer+options), and overwrites that task's
``lmh/<task>/eval.parquet`` under the active MARIN_PREFIX + writes the
extraction-version sidecar. Only the passage-bearing tasks change; the rest of
the corpus is untouched (B doesn't affect them). Run where HF + the store creds
are ambient:

    MARIN_PREFIX=s3://marin-us-east-02a/marin HF_TOKEN=… python .../regen_rc_evals.py
"""

import logging

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from rigging.filesystem import StoragePath, marin_prefix

from experiments.datakit.decontam.prepare_eval_corpus import (
    _EVALS_RELATIVE,
    _LMH_EXTRACTION_VERSION,
    _LMH_VERSION_SIDECAR,
    _lmh_doc_text,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)
_SCHEMA = pa.schema([("id", pa.string()), ("text", pa.string())])


def _answers_text(d: dict) -> str:
    a = d.get("answers")
    if isinstance(a, dict):  # squad: {text: [...]}, coqa: {input_text: [...]}
        return " ".join(a.get("text") or a.get("input_text") or [])
    return str(a or "")


# (cw_task_dir, split_label_in_id, hf_id, hf_config, hf_split, target_fn)
_TASKS = [
    ("anli_r1", "test", "facebook/anli", None, "test_r1", lambda d: str(d.get("label", ""))),
    ("anli_r2", "test", "facebook/anli", None, "test_r2", lambda d: str(d.get("label", ""))),
    ("anli_r3", "test", "facebook/anli", None, "test_r3", lambda d: str(d.get("label", ""))),
    ("race", "test", "ehovy/race", "all", "test", lambda d: str(d.get("answer", ""))),
    ("boolq", "validation", "google/boolq", None, "validation", lambda d: str(d.get("answer", ""))),
    ("squadv2", "validation", "rajpurkar/squad_v2", None, "validation", _answers_text),
    ("sciq", "test", "allenai/sciq", None, "test", lambda d: str(d.get("correct_answer", ""))),
    ("coqa", "validation", "stanfordnlp/coqa", None, "validation", _answers_text),
    (
        "drop",
        "validation",
        "ucinlp/drop",
        None,
        "validation",
        lambda d: " ".join((d.get("answers_spans") or {}).get("spans", [])),
    ),
]


def _stringify(doc: dict) -> dict:
    return {k: str(v) if isinstance(v, (int, float, bool)) else v for k, v in doc.items()}


def main() -> None:
    root = f"{marin_prefix()}/{_EVALS_RELATIVE}/lmh"
    for task, split_label, hf_id, cfg, hf_split, target_fn in _TASKS:
        try:
            ds = load_dataset(hf_id, cfg, split=hf_split)
        except Exception as e:
            logger.warning("SKIP %s: load failed: %s %s", task, type(e).__name__, str(e)[:80])
            continue
        ids, texts = [], []
        for i, raw in enumerate(ds):
            doc = _stringify(raw)
            text = _lmh_doc_text(doc, lambda _d: "", target_fn)  # passage doc -> prompt_fn skipped
            if not text.strip():
                continue
            ids.append(f"{task}-{split_label}-{i}")
            texts.append(text)
        out = f"{root}/{task}/eval.parquet"
        with StoragePath(out).open("wb") as fh:
            pq.write_table(pa.table({"id": ids, "text": texts}, schema=_SCHEMA), fh, compression="zstd")
        with StoragePath(f"{root}/{task}/{_LMH_VERSION_SIDECAR}").open("w") as vf:
            vf.write(_LMH_EXTRACTION_VERSION)
        logger.info("regenerated %s: %d records -> %s", task, len(ids), out)
    logger.info("DONE")


if __name__ == "__main__":
    main()

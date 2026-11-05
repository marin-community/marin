# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Eval-family grouping and eval-text lookup in the decontam viewer."""

from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq

from experiments.datakit.decontam.viewer import export_run


def test_uncheatable_records_group_by_subset_and_lm_eval_records_by_task():
    eval_ids = ["uncheatable-bbc_news-0", "uncheatable-bbc_news-17", "uncheatable-github_cpp-3", "gsm8k-test-12"]

    families = Counter(export_run.eval_id_to_family(eval_id) for eval_id in eval_ids)

    assert families == {"uncheatable_bbc_news": 2, "uncheatable_github_cpp": 1, "gsm8k": 1}


def test_uncheatable_eval_text_loads_from_its_subset_artifact(tmp_path, monkeypatch):
    artifact = tmp_path / "datakit/decontam/evals/aa-index-v4.1.1-v5/uncheatable/bbc_news/test.parquet"
    artifact.parent.mkdir(parents=True)
    pq.write_table(
        pa.table({"id": ["uncheatable-bbc_news-0", "uncheatable-bbc_news-1"], "text": ["First.", "Second."]}),
        str(artifact),
    )
    monkeypatch.setattr(export_run, "marin_prefix", lambda: str(tmp_path))

    texts = export_run._load_eval_texts({"uncheatable-bbc_news-1"})

    assert texts == {"uncheatable-bbc_news-1": "Second."}

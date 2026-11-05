# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Eval-family grouping and eval-text lookup in the decontam viewer."""

from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq

from experiments.datakit.decontam.viewer import export_reference_run, export_run


def test_uncheatable_records_group_by_subset_and_lm_eval_records_by_task():
    eval_ids = [
        "uncheatable-bbc_news-0",
        "uncheatable-bbc_news-17",
        "uncheatable-github_cpp-3",
        "gsm8k-test-12",
        "gsm8k-test-12-fields",
    ]

    families = Counter(export_run.eval_id_to_family(eval_id) for eval_id in eval_ids)

    assert families == {"uncheatable_bbc_news": 2, "uncheatable_github_cpp": 1, "gsm8k": 2}


def test_uncheatable_eval_text_loads_from_its_subset_artifact(tmp_path, monkeypatch):
    artifact = tmp_path / "datakit/decontam/evals/aa-index-v4.1.1-v10/uncheatable/bbc_news/test.parquet"
    artifact.parent.mkdir(parents=True)
    pq.write_table(
        pa.table({"id": ["uncheatable-bbc_news-0", "uncheatable-bbc_news-1"], "text": ["First.", "Second."]}),
        str(artifact),
    )
    monkeypatch.setattr(export_run, "marin_prefix", lambda: str(tmp_path))

    texts = export_run._load_eval_texts({"uncheatable-bbc_news-1"})

    assert texts == {"uncheatable-bbc_news-1": "Second."}


def test_reference_run_groups_pinned_suite_records_and_views_by_eval():
    eval_ids = [
        "swebench_verified-django__django-11099",
        "swebench_verified-django__django-11099-instruction",
        "terminal_bench_4-build-pmars-solution",
        "hle_diamond-6702b8",
        "hle-6702b8",
        "math500-algebra/1",
    ]

    families = Counter(export_reference_run._eval_family(eval_id) for eval_id in eval_ids)

    assert families == {
        "SWE-bench Verified": 2,
        "Terminal-Bench 4.0": 1,
        "HLE-Diamond": 1,
        "Humanity's Last Exam": 1,
        "MATH-500": 1,
    }

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pandas as pd
import pytest

from experiments.domain_phase_mix.release import verify_wandb_ids


@pytest.fixture(autouse=True)
def release_sources():
    """Overrides the package's source guard: these tests read only the files they write, so they run anywhere."""


def test_build_ids_reads_every_id_column_of_every_table(tmp_path):
    # Ids sit in CSV and Parquet tables, under observation_id and role-prefixed id columns, some of them null; content
    # ids, other `_id` columns and tables without ids are not runs.
    tables = {
        "data/starcoder/starcoder_ramp_designs.csv": pd.DataFrame(
            {
                "observation_id": ["tramp_s0821", "tramp_s0821", "gcfv9_gcf_p3_s1"],
                "source_observation_id": ["dsopt_s0821", None, None],
                "mixture_id": ["mix_0123456789abcdef"] * 3,
            }
        ),
        "data/fits/fits_swarm_predictions.parquet": pd.DataFrame(
            {"observation_id": ["singleavg_fit_002_run_00002-fc9cd7"], "task_id": ["arc_easy"]}
        ),
        "data/partition/partition.csv": pd.DataFrame({"bucket_id": ["arxiv"]}),
    }
    entries = []
    for path, frame in tables.items():
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        if path.endswith(".parquet"):
            frame.to_parquet(tmp_path / path, index=False)
        else:
            frame.to_csv(tmp_path / path, index=False)
        table, form = path.rsplit("/", 1)[1].split(".")
        entries.append({"table": table, "path": path, "format": form, "columns": list(frame.columns)})
    (tmp_path / "MANIFEST.json").write_text(json.dumps({"tables": entries}))
    assert verify_wandb_ids.build_ids(tmp_path) == {
        ("starcoder_ramp_designs", "observation_id"): {"tramp_s0821", "gcfv9_gcf_p3_s1"},
        ("starcoder_ramp_designs", "source_observation_id"): {"dsopt_s0821"},
        ("fits_swarm_predictions", "observation_id"): {"singleavg_fit_002_run_00002-fc9cd7"},
    }

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from experiments.domain_phase_mix.release import scan_upload

LEAK = "gs://bucket/x"
# One planted string per needle and the one pattern it must trip. Ids and hashes also fire when an identifier or a
# file name continues them with `_`.
PLANTS = [
    ("gs://bucket/x", "gcs_path"),
    ("bucket marin-eu-west4", "gcs_path"),
    ("wandb.ai/marin/runs/abc", "wandb"),
    ("wandb_run_id", "wandb"),
    ("wandb_url", "wandb"),
    ("wandb:marin/abc", "wandb"),
    ("/Users/someone/Projects/x", "home_path"),
    ("/home/someone/x", "home_path"),
    ("exp_01m4wyyvz7b90mkbr7c8y1fwgb", "fieldbook"),
    ("run_01m4wyyvz7b90mkbr7c8y1fwgb_notes.json", "fieldbook"),
    ("see the Fieldbook entry", "fieldbook"),
    ("/checkpoints/step-1000", "checkpoint"),
    ("hf/step-1000", "checkpoint"),
    ("lwspu_t9_snc_cap08_3e18-6c15f1", "run_hash"),
    ("lwspu_t9_snc_cap08_3e18-6c15f1_eval_metrics.csv", "run_hash"),
    ("a" * 64, "long_hash"),
    ("coordinate_" + "b" * 64, "long_hash"),
    ("pinlin_calvin_xu", "personal"),
    ("CalvinXu", "personal"),
    ("someone@example.com", "email"),
    ("http://example.org", "url"),
]
DIGEST = hashlib.sha256(b"table").hexdigest()  # 64 hex digits, like a file's SHA-256
REVISION = hashlib.sha1(b"commit").hexdigest()  # 40 hex digits, like a git commit


@pytest.fixture(autouse=True)
def release_sources():
    """Overrides the package's source guard: these tests read only the files they write, so they run anywhere."""


def _write_clean_tree(root: Path) -> dict:
    """A build tree holding only public strings: the public id forms, an exposure id that begins `exp_01`, a source
    file name ending in `_wandb.csv`, and hex digests in the fields that hold them by design. Returns the manifest."""
    (root / "data/heldout").mkdir(parents=True)
    (root / "data/tasks").mkdir(parents=True)
    pd.DataFrame(
        {
            "observation_id": ["obs_0123456789ab"],
            "mixture_id": ["mix_0123456789abcdef"],
            "exposure_id": ["exp_01585550f65360b9"],
            "schedule_id": ["sch_0123456789abcdef"],
            "bpb__arc_challenge": [0.81],
        }
    ).to_csv(root / "data/heldout/heldout.csv", index=False)
    pd.DataFrame(
        {"task_id": ["arc_challenge"], "source_revision": [f"OLMo-Eval {REVISION}; HF revision unpinned"]}
    ).to_csv(root / "data/tasks/tasks.csv", index=False)
    manifest = {
        "commit": REVISION,
        "tables": [
            {
                "table": "heldout",
                "sha256": DIGEST,
                "sources": ["experiments/reference_outputs/delphi_scaling_completed_wandb.csv"],
            }
        ],
    }
    (root / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def test_scan_covers_every_file(tmp_path):
    (tmp_path / "data/x").mkdir(parents=True)
    pd.DataFrame({"a": ["ok", LEAK]}).to_csv(tmp_path / "data/x/cell.csv", index=False)
    pd.DataFrame({LEAK: [1.0]}).to_csv(tmp_path / "data/x/header.csv", index=False)
    pd.DataFrame({"a": [0.5, 0.25], "b": ["ok", LEAK]}).to_parquet(tmp_path / "data/x/cell.parquet", index=False)
    pd.DataFrame({LEAK: [0.5]}).to_parquet(tmp_path / "data/x/name.parquet", index=False)
    (tmp_path / "MANIFEST.json").write_text(json.dumps({"tables": [{"note": LEAK}], LEAK: 1}))
    (tmp_path / "README.md").write_text(f"# Card\n\nsee {LEAK}\n")
    (tmp_path / "data/x/notes-6c15f1.md").write_text("ok\n")  # a run name in the file's own name
    hits = scan_upload.scan(tmp_path)
    # Each hit names its file, its line (text), data row (tables; 0 for the header) or 0 (JSON), and its field.
    assert {(h.path.name, h.line_or_row, h.field) for h in hits} == {
        ("cell.csv", 2, "a"),
        ("header.csv", 0, LEAK),
        ("cell.parquet", 2, "b"),
        ("name.parquet", 0, LEAK),
        ("MANIFEST.json", 0, "$.tables[0].note"),
        ("MANIFEST.json", 0, f"$.{LEAK}"),
        ("README.md", 3, ""),
        ("notes-6c15f1.md", 0, ""),
    }


def test_every_pattern_fires_and_clean_text_passes(tmp_path):
    assert {pattern for _, pattern in PLANTS} == set(scan_upload.PATTERNS)
    plants = tmp_path / "plants"
    plants.mkdir()
    for i, (plant, _) in enumerate(PLANTS):
        (plants / f"f{i}.md").write_text(f"ok {plant} ok\n")
    hits = scan_upload.scan(plants)
    for i, (plant, pattern) in enumerate(PLANTS):
        assert {h.pattern for h in hits if h.path.name == f"f{i}.md"} == {pattern}, plant
    clean = tmp_path / "clean"
    _write_clean_tree(clean)
    assert scan_upload.scan(clean) == []


def test_allowance_exempts_only_its_field_and_pattern(tmp_path):
    # An allowance names a file, a field and the hash pattern: the same digest under any other key, column or file is a
    # hit, and so is any other pattern inside an allowed field.
    manifest = _write_clean_tree(tmp_path)
    manifest["tables"][0]["policy"] = DIGEST
    manifest["notes"] = {"sha256": DIGEST}
    (tmp_path / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    pd.DataFrame(
        {
            "task_id": ["arc_challenge"],
            "source_revision": [f"{REVISION} via someone@example.com"],
            "upstream": [REVISION],
        }
    ).to_csv(tmp_path / "data/tasks/tasks.csv", index=False)
    pd.DataFrame({"observation_id": ["obs_0123456789ab"], "source_revision": [REVISION]}).to_csv(
        tmp_path / "data/heldout/heldout.csv", index=False
    )
    assert {(h.path.name, h.field, h.pattern) for h in scan_upload.scan(tmp_path)} == {
        ("MANIFEST.json", "$.tables[0].policy", "long_hash"),
        ("MANIFEST.json", "$.notes.sha256", "long_hash"),
        ("tasks.csv", "source_revision", "email"),
        ("tasks.csv", "upstream", "long_hash"),
        ("heldout.csv", "source_revision", "long_hash"),
    }


@pytest.mark.parametrize(
    "name, content",
    [
        ("table.csv.gz", gzip.compress(f"a\n{LEAK}\n".encode())),
        ("table.csv", gzip.compress(f"a\n{LEAK}\n".encode())),
        ("ragged.csv", f"a,b\n{LEAK},y,z\n".encode()),
    ],
    ids=["compressed", "compressed-named-csv", "ragged-csv"],
)
def test_scan_refuses_a_file_it_cannot_read(tmp_path, name, content):
    # Compression hides every string from the patterns, and pandas' default reader would take a ragged row's extra
    # first field for the index, outside every column; such a file must stop the scan rather than pass it.
    (tmp_path / name).write_bytes(content)
    with pytest.raises(ValueError):
        scan_upload.scan(tmp_path)


def test_cli_exits_nonzero_on_any_hit(tmp_path):
    (tmp_path / "README.md").write_text("A public card.\n")
    with pytest.raises(SystemExit) as clean:
        scan_upload.main([str(tmp_path)])
    (tmp_path / "notes.md").write_text(f"see {LEAK}\n")
    with pytest.raises(SystemExit) as leaked:
        scan_upload.main([str(tmp_path)])
    assert (clean.value.code, leaked.value.code) == (0, 1)

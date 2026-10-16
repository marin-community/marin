# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import hashlib
import io
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from experiments.domain_phase_mix.release import scan_upload

LEAK = "gs://bucket/x"
# One planted string per needle and a pattern it must trip. Ids and hashes also fire when an identifier or a file name
# continues them with `_`.
PLANTS = [
    ("gs://bucket/x", "bucket"),
    ("bucket marin-eu-west4", "bucket"),
    ("s3://marin-us-east-02a/x", "bucket"),
    ("s3://marin-na/x", "bucket"),
    ("marin-us-west-04a", "bucket"),
    ("/Users/someone/Projects/x", "home_path"),
    ("/home/someone/x", "home_path"),
    ("exp_01m4wyyvz7b90mkbr7c8y1fwgb", "fieldbook"),
    ("run_01m4wyyvz7b90mkbr7c8y1fwgb_notes.json", "fieldbook"),
    ("note_01h2xk9m3p4q5r6s7t8v9w0y1z", "fieldbook"),
    ("see the Fieldbook entry", "fieldbook"),
    ("/checkpoints/step-1000", "checkpoint"),
    ("hf/step-1000", "checkpoint"),
    ("a" * 64, "long_hash"),
    ("coordinate_" + "b" * 64, "long_hash"),
    ("pinlin_calvin_xu", "personal"),
    ("CalvinXu", "personal"),
    ("someone@example.com", "email"),
    ("http://example.org", "url"),
    ("gcs://bucket/x", "url"),
    ("source_https://example.org/private", "url"),
    ("x_s3://private-bucket/x", "url"),
    ("wandb.ai/someone/project/runs/abc", "wandb"),
    ("wandb.ai/marin-community/../someone/project", "wandb"),
]
DIGEST = hashlib.sha256(b"table").hexdigest()  # 64 hex digits, like a file's SHA-256
# W&B run ids of the public project marin-community/marin, the observation ids: a training run name and its hash, one
# that W&B shortened from the front, and two that a launch prefixed.
RUN_IDS = [
    "lwspu_t9_snc_cap08_3e18-6c15f1",
    "trl_tilt_domain_dolma3_cc_education_and_jobs_high_plus-f9b90f",
    "gcfv9_gcf_b_h5d28160_m100a_boundary_beta_0p60_s2026084000",
    "c2v4c_lrcd_ds_m100_0p60_c000_s0711",
]
REVISION = hashlib.sha1(b"commit").hexdigest()  # 40 hex digits, like a git commit


def _parquet_bytes(table: pa.Table) -> bytes:
    sink = io.BytesIO()
    pq.write_table(table, sink)
    return sink.getvalue()


# Files whose strings the scan cannot fully see, each with the file name it is written under.
UNREADABLE = {
    "compressed": ("table.csv.gz", gzip.compress(f"a\n{LEAK}\n".encode())),
    "compressed-named-csv": ("table.csv", gzip.compress(f"a\n{LEAK}\n".encode())),
    "ragged-csv": ("ragged.csv", f"a,b\n{LEAK},y,z\n".encode()),
    "repeated-json-key": ("MANIFEST.json", f'{{"note": "{LEAK}", "note": "ok"}}'.encode()),
    "parquet-list": ("t.parquet", _parquet_bytes(pa.table({"a": [["ok", LEAK]]}))),
    "parquet-struct": ("t.parquet", _parquet_bytes(pa.table({"a": [{"b": LEAK}]}))),
    "parquet-map": (
        "t.parquet",
        _parquet_bytes(pa.table({"a": pa.array([[("k", LEAK)]], type=pa.map_(pa.string(), pa.string()))})),
    ),
    "parquet-binary": ("t.parquet", _parquet_bytes(pa.table({"a": pa.array([LEAK.encode()], type=pa.binary())}))),
    "parquet-dictionary": ("t.parquet", _parquet_bytes(pa.table({"a": pa.array([LEAK]).dictionary_encode()}))),
}


@pytest.fixture(autouse=True)
def release_sources():
    """Overrides the package's source guard: these tests read only the files they write, so they run anywhere."""


def _write_clean_tree(root: Path) -> dict:
    """A build tree holding only public strings: W&B run ids, the content id forms (each continuing `_01`, like a
    Fieldbook id), an exposure id that begins `exp_01`, `marin-` names that are no bucket, a source file name ending in
    `_wandb.csv`, and hex digests in the fields that hold them by design. Returns the manifest."""
    (root / "data/heldout").mkdir(parents=True)
    (root / "data/tasks").mkdir(parents=True)
    pd.DataFrame(
        {
            "observation_id": RUN_IDS,
            "mixture_id": "mix_0123456789abcdef",
            "exposure_id": "exp_01585550f65360b9",
            "schedule_id": "sch_0123456789abcdef",
            "bpb__arc_challenge": 0.81,
        }
    ).to_csv(root / "data/heldout/heldout.csv", index=False)
    pd.DataFrame(
        {
            "task_id": ["arc_challenge"],
            "source_revision": [f"OLMo-Eval {REVISION}; HF revision unpinned"],
            "tokenizer": ["meta-llama/Meta-Llama-3.1-8B (same vocabulary as marin-community/marin-tokenizer)"],
        }
    ).to_csv(root / "data/tasks/tasks.csv", index=False)
    manifest = {
        "commit": REVISION,
        "notes": ["Each run's logs are at wandb.ai/marin-community/marin/runs/<observation_id>."],
        "tables": [
            {
                "table": "heldout",
                "sha256": DIGEST,
                "keying": f"Each run's W&B run id, such as {RUN_IDS[0]}.",
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
    pd.DataFrame({"a": ["ok", f"first line\nthen {LEAK}"]}).to_csv(tmp_path / "data/x/multiline.csv", index=False)
    pd.DataFrame({"a": [0.5, 0.25], "b": ["ok", LEAK]}).to_parquet(tmp_path / "data/x/cell.parquet", index=False)
    pd.DataFrame({LEAK: [0.5]}).to_parquet(tmp_path / "data/x/name.parquet", index=False)
    (tmp_path / "MANIFEST.json").write_text(json.dumps({"tables": [{"note": LEAK}], LEAK: 1}))
    (tmp_path / "README.md").write_text(f"# Card\n\nsee {LEAK}\n")
    (tmp_path / "data/x/notes-pinlin_calvin_xu.md").write_text("ok\n")  # a user name in the file's own name
    hits = scan_upload.scan(tmp_path)
    # Each hit names its file, its line (text), row (tables: 0 for the header, a quoted multi-line cell is one row) or
    # 0 (JSON), and its field. pandas' own Parquet metadata repeats the column names.
    assert {(h.path.name, h.line_or_row, h.field) for h in hits} == {
        ("cell.csv", 2, "a"),
        ("header.csv", 0, LEAK),
        ("multiline.csv", 2, "a"),
        ("cell.parquet", 2, "b"),
        ("name.parquet", 0, LEAK),
        ("name.parquet", 0, "schema metadata pandas"),
        ("MANIFEST.json", 0, "$.tables[0].note"),
        ("MANIFEST.json", 0, f"$.{LEAK}"),
        ("README.md", 3, ""),
        ("notes-pinlin_calvin_xu.md", 0, ""),
    }


def test_scan_reads_parquet_index_and_metadata(tmp_path):
    # pandas stores a string index as a column, and `attrs` in the schema metadata twice (under PANDAS_ATTRS and in
    # its own `pandas` entry); Arrow fields carry metadata too.
    pd.DataFrame({"a": [0.5]}, index=pd.Index([LEAK], name="run")).to_parquet(tmp_path / "index.parquet")
    frame = pd.DataFrame({"a": [0.5]})
    frame.attrs["source"] = LEAK
    frame.to_parquet(tmp_path / "attrs.parquet", index=False)
    schema = pa.schema([pa.field("a", pa.float64(), metadata={"note": LEAK})], metadata={LEAK: "key"})
    pq.write_table(pa.table({"a": [0.5]}, schema=schema), tmp_path / "metadata.parquet")
    assert {(h.path.name, h.line_or_row, h.field) for h in scan_upload.scan(tmp_path)} == {
        ("index.parquet", 1, "run"),
        ("attrs.parquet", 0, "schema metadata PANDAS_ATTRS"),
        ("attrs.parquet", 0, "schema metadata pandas"),
        ("metadata.parquet", 0, f"schema metadata {LEAK}"),
        ("metadata.parquet", 0, "a metadata note"),
    }


def test_parquet_under_a_key_value_directory_gains_no_column(tmp_path):
    # pyarrow's default reader would take `run=abc` above the build for a partition column (a dictionary column, which
    # the scan refuses); the scan reads each Parquet file alone.
    root = tmp_path / "run=abc/build"
    root.mkdir(parents=True)
    pd.DataFrame({"a": ["ok", LEAK]}).to_parquet(root / "t.parquet", index=False)
    assert {(h.line_or_row, h.field) for h in scan_upload.scan(root)} == {(2, "a")}


def test_every_pattern_fires_and_clean_text_passes(tmp_path):
    assert {pattern for _, pattern in PLANTS} == set(scan_upload.PATTERNS)
    plants = tmp_path / "plants"
    plants.mkdir()
    for i, (plant, _) in enumerate(PLANTS):
        (plants / f"f{i}.md").write_text(f"ok {plant} ok\n")
    hits = scan_upload.scan(plants)
    for i, (plant, pattern) in enumerate(PLANTS):
        assert pattern in {h.pattern for h in hits if h.path.name == f"f{i}.md"}, plant
    clean = tmp_path / "clean"
    _write_clean_tree(clean)
    assert scan_upload.scan(clean) == []


def test_run_ids_pass_but_the_runs_storage_paths_do_not(tmp_path):
    # W&B run ids are the public observation ids, in a table, the manifest or a W&B link's path; the same run's
    # checkpoint path is still a hit.
    (tmp_path / "data/validation_3e18").mkdir(parents=True)
    checkpoints = [f"gs://marin-us-central1/checkpoints/{run}/hf/step-3006" for run in RUN_IDS]
    pd.DataFrame({"observation_id": RUN_IDS, "checkpoint": checkpoints}).to_csv(
        tmp_path / "data/validation_3e18/validation_3e18.csv", index=False
    )
    notes = [f"wandb.ai/marin-community/marin/runs/{run}" for run in RUN_IDS]
    (tmp_path / "MANIFEST.json").write_text(json.dumps({"notes": notes}))
    hits = scan_upload.scan(tmp_path)
    assert {(h.path.name, h.field, h.pattern) for h in hits} == {
        ("validation_3e18.csv", "checkpoint", pattern) for pattern in ("bucket", "checkpoint", "url")
    }
    assert sorted(h.line_or_row for h in hits if h.pattern == "bucket") == [1, 2, 3, 4]


def test_wandb_references_must_stay_in_the_public_entity(tmp_path):
    # Outside the card's links too, a W&B reference may name marin-community only; another entity, or a dot segment,
    # plain or percent-encoded, that may lead out of it is a hit.
    notes = [
        "wandb.ai/marin-community/marin/runs/lwspu_t9_snc_cap08_3e18-6c15f1",
        "wandb.ai/someone/private/runs/x",
        "wandb.ai/marin-community/../someone/private",
        "wandb.ai/marin-community/%2E%2E/someone/private",
    ]
    (tmp_path / "MANIFEST.json").write_text(json.dumps({"notes": notes}))
    hits = scan_upload.scan(tmp_path)
    assert {(h.field, h.pattern) for h in hits} == {(f"$.notes[{i}]", "wandb") for i in (1, 2, 3)}


def test_allowance_exempts_only_its_field_and_pattern(tmp_path):
    # An allowance names a file relative to the root, a field and the hash pattern: the same digest under any other key,
    # column or file (a nested MANIFEST.json included) is a hit, and so is any other pattern inside an allowed field.
    manifest = _write_clean_tree(tmp_path)
    manifest["tables"][0]["policy"] = DIGEST
    manifest["notes"] = {"sha256": DIGEST}
    (tmp_path / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    (tmp_path / "data/x").mkdir()
    (tmp_path / "data/x/MANIFEST.json").write_text(json.dumps({"commit": REVISION, "tables": [{"sha256": DIGEST}]}))
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
    hits = scan_upload.scan(tmp_path)
    assert {(h.path.relative_to(tmp_path).as_posix(), h.field, h.pattern) for h in hits} == {
        ("MANIFEST.json", "$.tables[0].policy", "long_hash"),
        ("MANIFEST.json", "$.notes.sha256", "long_hash"),
        ("data/x/MANIFEST.json", "$.commit", "long_hash"),
        ("data/x/MANIFEST.json", "$.tables[0].sha256", "long_hash"),
        ("data/tasks/tasks.csv", "source_revision", "email"),
        ("data/tasks/tasks.csv", "upstream", "long_hash"),
        ("data/heldout/heldout.csv", "source_revision", "long_hash"),
    }


@pytest.mark.parametrize("name, content", list(UNREADABLE.values()), ids=list(UNREADABLE))
def test_scan_refuses_a_file_it_cannot_read(tmp_path, name, content):
    # Compression hides every string from the patterns; pandas' default reader would take a ragged row's extra first
    # field for the index, outside every column; `json.loads` keeps only a repeated key's last value; and a nested,
    # binary or dictionary Parquet column holds strings outside any string column. Each must stop the scan.
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


# Lines of a dataset card and the patterns each trips: a Markdown link to an allowed host trips nothing, every other URL
# trips `url`, and an allowed link still trips every other pattern.
CARD_LINES = [
    ("The paper is [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX).", set()),
    ("<https://huggingface.co/datasets/allenai/dolma3_pool>", set()),
    ("| [Olmo 3](https://allenai.org/olmo) | [Marin](https://marin.community) |", set()),
    ("[code](https://github.com/marin-community/marin)", set()),
    ("see https://github.com/marin-community/marin", {"url"}),  # outside a Markdown link
    ("[data](https://example.org/data)", {"url"}),  # a host outside the allowlist
    ("[code](http://github.com/marin-community/marin)", {"url"}),  # not https
    ("[x](https://github.com.example.org/x)", {"url"}),  # a host that only begins with an allowed one
    ("[x](https://github.com:8443/x)", {"url"}),  # a port
    ("[x](https://someone@github.com/x)", {"url", "email"}),  # credentials
    ("[x](https://github.com/x?next=https://example.org)", {"url"}),  # a second URL inside an allowed link
    ("[https://example.org](https://github.com/x)", {"url"}),  # a URL in a link's text
    ("[x]: https://github.com/x", {"url"}),  # a reference definition, outside an inline link
    ("[x](https://huggingface.co/pinlin_calvin_xu)", {"personal"}),  # other patterns inside an allowed link
    ("[logs](https://wandb.ai/marin-community/marin/runs/lwspu_t9_snc_cap08_3e18-6c15f1)", set()),
    ("[x](https://wandb.ai/someone/marin/runs/x)", {"url", "wandb"}),  # another W&B entity than the public one
    ("[x](https://wandb.ai/marin-community-x/marin)", {"url", "wandb"}),  # an entity that only begins with it
    ("[x](https://wandb.ai/marin-community/../someone/private)", {"url", "wandb"}),  # a dot segment out of it
    ("[x](https://wandb.ai/marin-community/%2e%2e/someone/private)", {"url", "wandb"}),  # the same, percent-encoded
]


def test_card_links_only_to_allowed_hosts(tmp_path):
    (tmp_path / "README.md").write_text("\n".join(line for line, _ in CARD_LINES) + "\n")
    tripped: dict[int, set[str]] = {}
    for hit in scan_upload.scan(tmp_path):
        tripped.setdefault(hit.line_or_row, set()).add(hit.pattern)
    assert [tripped.get(line, set()) for line in range(1, len(CARD_LINES) + 1)] == [p for _, p in CARD_LINES]
    # The allowance covers the card at the root only: the same link in any other file is a hit.
    other = tmp_path / "other"
    (other / "data").mkdir(parents=True)
    for name in ("notes.md", "data/README.md"):
        (other / name).write_text(CARD_LINES[0][0] + "\n")
    assert {(h.path.relative_to(other).as_posix(), h.pattern) for h in scan_upload.scan(other)} == {
        ("notes.md", "url"),
        ("data/README.md", "url"),
    }


def test_scan_reads_only_the_included_files(tmp_path):
    for name in ("data/x/table.md", "fits/inputs.md", "README.md"):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(f"{LEAK}\n")
    hits = scan_upload.scan(tmp_path, include=["data/x/table.md", "README.md"])
    assert {h.path.relative_to(tmp_path).as_posix() for h in hits} == {"data/x/table.md", "README.md"}
    with pytest.raises(FileNotFoundError):  # a file to scan that the tree lacks is never silently skipped
        scan_upload.scan(tmp_path, include=["README.md", "MANIFEST.json"])

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import re

import pandas as pd
import pytest

from experiments.domain_phase_mix.release import context, export_dataset, paths, tables

pytestmark = pytest.mark.skipif(not paths.MIXSEL.exists(), reason="sources absent")
CONFIGS = {
    "partition",
    "tasks",
    "swarm_qwen3_360m_1p6b",
    "swarm_llama_160m_1p2b",
    "swarm_llama_200m_6b",
    "heldout",
    "validation_3e18",
    "ladder",
    "ladder_accuracy",
    "starcoder",
    "domain_sweeps",
    "fits",
    "run_sets",
}
# What must never reach the upload: GCS and W&B references, user paths, e-mail addresses, 64-hex policy hashes,
# Fieldbook ids and run names, which end in a six-hex-digit hash.
INTERNAL = [
    re.compile(r"gs://|wandb|/Users/|pinlin_calvin_xu"),
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    re.compile(r"[0-9a-f]{64}"),
    re.compile(r"(?:exp|run|job)_01[a-z0-9]{24}"),
    re.compile(r"\w-[0-9a-f]{6}(?![0-9a-z])"),
]
SPECIAL_CONFIGS = {"deferred", "constant", "untraced"}  # figures_tables.csv entries that name no config


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    root = tmp_path_factory.mktemp("release")
    export_dataset.main(["--out", str(root / "build"), "--private-dir", str(root / "private")])
    return root


@pytest.fixture(scope="module")
def manifest(build):
    return json.loads((build / "build/MANIFEST.json").read_text())


@pytest.mark.timeout(1800)
def test_full_export_writes_every_config(build, manifest):
    out = build / "build"
    assert {t["config"] for t in manifest["tables"]} >= CONFIGS
    assert manifest["consistency_problems"] == []
    for t in manifest["tables"]:
        path = out / t["path"]
        assert t["path"] == f"data/{t['config']}/{t['table']}.{t['format']}"
        assert path.stat().st_size == t["bytes"] and hashlib.sha256(path.read_bytes()).hexdigest() == t["sha256"]
        assert t["rows"] > 0 and t["keying"] and all(not source.startswith("/") for source in t["sources"])
    layout = {t["table"]: (t["config"], t["format"]) for t in manifest["tables"]}
    assert layout["starcoder_dense_grid"] == ("starcoder", "parquet")
    assert layout["starcoder_observations"] == ("starcoder", "csv")
    assert layout["domain_sweeps_olmix_fits"] == ("domain_sweeps", "csv")
    assert layout["ladder_accuracy"] == ("ladder_accuracy", "csv")
    parquet = {name for name, (_, form) in layout.items() if form == "parquet"}
    assert parquet == {name for name in layout if name.startswith("fits_")} | {
        "starcoder_dense_grid",
        "starcoder_trajectories",
    }
    assert (out / "figures_tables.csv").read_bytes() == export_dataset.FIGURES_TABLES.read_bytes()
    # The run-name map and the absolute source paths stay in the private directory, never in the upload.
    assert (build / "private/observation_map.csv").exists()
    sources = json.loads((build / "private/SOURCES.json").read_text())
    assert set(sources["tables"]) == set(layout) and all(p.startswith("/") for p in sources["context"])
    assert not [p for p in out.rglob("*") if p.name in ("observation_map.csv", "SOURCES.json", "salt")]


@pytest.mark.timeout(1800)
def test_written_files_hold_the_exported_values_exactly(build, manifest):
    # The same salt and run-name map give the same ids; every value survives its file bit for bit.
    built = export_dataset.build_tables(context.build(private_dir=build / "private"))
    assert set(built) == {t["table"] for t in manifest["tables"]}
    for t in manifest["tables"]:
        path = build / "build" / t["path"]
        written = pd.read_parquet(path) if t["format"] == "parquet" else tables.read_csv(path)
        assert list(written.columns) == t["columns"] and len(written) == t["rows"]
        pd.testing.assert_frame_equal(written, built[t["table"]], check_dtype=False, check_exact=True)


def test_manifest_and_figure_index_name_nothing_internal(build, manifest):
    # File digests and the commit are hex by design, and source paths name files; every other string must pass.
    public = {key: value for key, value in manifest.items() if key != "commit"}
    public["tables"] = [{k: v for k, v in t.items() if k not in ("sha256", "sources")} for t in manifest["tables"]]
    text = json.dumps(public) + (build / "build/figures_tables.csv").read_text()
    for pattern in INTERNAL:
        assert not pattern.search(text), pattern.search(text)
    # Source paths are relative to a repository and replace the run names that begin per-run file names by `*`.
    for source in [s for t in manifest["tables"] for s in t["sources"]]:
        assert not source.startswith(("/", "~")) and "calvinxu" not in source and not INTERNAL[-1].search(source)


def test_figures_tables_lists_every_figure_and_table_of_the_paper(manifest):
    index = tables.read_csv(export_dataset.FIGURES_TABLES, keep_default_na=False)
    assert index.label.is_unique and index.kind.value_counts().to_dict() == {"figure": 26, "table": 20}
    configs = {t["config"] for t in manifest["tables"]}
    assert {c for cell in index.configs for c in cell.split(";")} <= configs | SPECIAL_CONFIGS
    sections = paths.DRIVE / "sections"
    if not sections.exists():
        pytest.skip("paper folder absent")
    tex = "\n".join(path.read_text() for path in sorted(sections.glob("*.tex")))
    assert set(index.label) == set(re.findall(r"\\label\{((?:fig|tab):[^}]+)\}", tex))
    for stem in index.file[index.kind == "figure"]:
        assert re.search(rf"\\includegraphics(?:\[[^]]*\])?\{{{stem}(?:\.pdf)?\}}", tex), stem


def test_write_refuses_to_replace_a_data_folder_it_did_not_write(tmp_path):
    stray = tmp_path / "data/notes.txt"
    stray.parent.mkdir()
    stray.write_text("not an export")
    with pytest.raises(ValueError, match=r"MANIFEST\.json"):
        export_dataset.write({}, tmp_path, tmp_path / "private")
    assert stray.read_text() == "not an export"

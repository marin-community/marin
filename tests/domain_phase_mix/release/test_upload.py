# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import hashlib
import json
import re
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml
from datasets import get_dataset_config_names, load_dataset
from huggingface_hub import CommitOperationAdd, CommitOperationDelete

from experiments.domain_phase_mix.release import upload, verify_paper_numbers

LEAK = "gs://bucket/x"
# A card draft's prose: a table, whose `---` rules open no YAML header, and a link the card may carry.
PROSE = """# MARINER data

| Config | Rows |
| --- | ---: |
| `heldout` | 2 |

The paper is [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX).
"""
TABLES = {
    "data/heldout/heldout.csv": pd.DataFrame(
        {"observation_id": ["obs_0123456789ab", "obs_123456789abc"], "bpb__arc_easy": [0.81, 0.79]}
    ),
    "data/fits/fits_task_parameters.parquet": pd.DataFrame({"task_id": ["arc_easy"], "rate": [0.25]}),
}
# One paper table whose numbers the audit finds differing (0.80), matching (0.79), registered as a constant (7) and
# unregistered (8).
TEX = r"""
\begin{table}
\label{tab:x}
\begin{tabular}{lrr}
\toprule
Run & BPB & Layers \\
\midrule
a & 0.80 & 7 \\
b & 0.79 & 8 \\
\bottomrule
\end{tabular}
\end{table}
"""
REGISTRY = pd.DataFrame(
    {
        "label": ["tab:x"] * 3,
        "row": [0, 1, 0],
        "col": [1, 1, 2],
        "part": [0, 0, 0],
        "query": [
            "heldout[observation_id=='obs_0123456789ab'].bpb__arc_easy.value",
            "heldout[observation_id=='obs_123456789abc'].bpb__arc_easy.value",
            "",
        ],
        "decimals": [2, 2, 0],
        "note": ["", "", "constant"],
    }
)
FILES = (  # what an upload of the tree below sends
    "data/fits/fits_task_parameters.parquet",
    "data/heldout/heldout.csv",
    "MANIFEST.json",
    "figures_tables.csv",
    "README.md",
)


@pytest.fixture(autouse=True)
def release_sources():
    """Overrides the package's source guard: these tests read only the files they write, so they run anywhere."""


@dataclasses.dataclass(frozen=True)
class Tree:
    build: Path
    card: Path
    sections: Path
    registry: Path

    @property
    def argv(self) -> list[str]:
        args = ("--build", self.build, "--card", self.card, "--sections", self.sections, "--registry", self.registry)
        return [str(a) for a in args]


def _write_build(build: Path, tables: dict[str, pd.DataFrame], **manifest: object) -> None:
    """Write the tables and a manifest describing them, as export_dataset does."""
    entries = []
    for path, df in tables.items():
        file = build / path
        file.parent.mkdir(parents=True, exist_ok=True)
        if file.suffix == ".parquet":
            df.to_parquet(file, index=False)
        else:
            df.to_csv(file, index=False)
        content = file.read_bytes()
        entries.append(
            {
                "config": file.parent.name,
                "table": file.stem,
                "path": path,
                "format": file.suffix.removeprefix("."),
                "rows": len(df),
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    commit = hashlib.sha1(b"commit").hexdigest()
    record = {"commit": commit, "built_at": "2026-10-15T00:00:00+00:00", "tables": entries, "consistency_problems": []}
    (build / "MANIFEST.json").write_text(json.dumps({**record, **manifest}, indent=1))


@pytest.fixture
def tree(tmp_path) -> Tree:
    """A build beside its generated fit inputs (fits/, which may hold internal names), a card draft, and a paper with
    one table and its audit registry."""
    build = tmp_path / "build"
    _write_build(build, TABLES)
    (build / "figures_tables.csv").write_text("label,kind,file,builder,configs\nfig:x,figure,x,build_x.py,heldout\n")
    (build / "fits/llama_160m_1p2b").mkdir(parents=True)
    (build / "fits/llama_160m_1p2b/swarm_outcomes.csv").write_text(f"run,source\nrun_00125,{LEAK}\n")
    (tmp_path / "card.md").write_text(PROSE)
    (tmp_path / "sections").mkdir()
    (tmp_path / "sections/results.tex").write_text(TEX)
    REGISTRY.to_csv(tmp_path / "paper_cells.csv", index=False)
    return Tree(build, tmp_path / "card.md", tmp_path / "sections", tmp_path / "paper_cells.csv")


class Hub:
    """The Hub as an upload's calls see it: one dataset repository, absent while `private` is None."""

    def __init__(self, private: bool | None, files: dict[str, bytes]):
        self.private = private
        self.files = files

    def create_repo(self, repo_id: str, *, repo_type: str, private: bool, exist_ok: bool) -> None:
        assert repo_type == "dataset" and exist_ok
        if self.private is None:  # an existing repository keeps its visibility
            self.private = private

    def repo_info(self, repo_id: str, *, repo_type: str) -> SimpleNamespace:
        return SimpleNamespace(private=self.private)

    def list_repo_files(self, repo_id: str, *, repo_type: str) -> list[str]:
        return sorted(self.files)

    def create_commit(
        self,
        repo_id: str,
        operations: list[CommitOperationAdd | CommitOperationDelete],
        *,
        commit_message: str,
        repo_type: str,
    ) -> None:
        for op in operations:
            if isinstance(op, CommitOperationDelete):
                del self.files[op.path_in_repo]
            else:
                self.files[op.path_in_repo] = Path(str(op.path_or_fileobj)).read_bytes()


def _offline() -> None:
    raise AssertionError("a dry run must not reach the Hub")


def test_card_configs_cover_exactly_the_csv_tables():
    manifest = {
        "tables": [
            {"config": "partition", "table": "partition", "path": "data/partition/partition.csv", "format": "csv"},
            {
                "config": "fits",
                "table": "fits_learning_curves",
                "path": "data/fits/fits_learning_curves.parquet",
                "format": "parquet",
            },
            {"config": "heldout", "table": "heldout", "path": "data/heldout/heldout.csv", "format": "csv"},
            {
                "config": "fits",
                "table": "fits_task_parameters",
                "path": "data/fits/fits_task_parameters.parquet",
                "format": "parquet",
            },
        ]
    }
    configs = yaml.safe_load(upload.card_configs(manifest))["configs"]
    assert [c["config_name"] for c in configs] == ["partition", "heldout"]
    assert not any(f["path"].endswith(".parquet") for c in configs for f in c["data_files"])
    heldout = next(c for c in configs if c["config_name"] == "heldout")
    assert heldout["data_files"] == [{"split": "train", "path": "data/heldout/heldout.csv"}]


@pytest.mark.parametrize("draft", [PROSE, "---\nlicense: other\n---\n" + PROSE], ids=["prose", "own-header"])
def test_build_card_generates_the_header_and_keeps_the_prose(draft: str):
    manifest = {
        "tables": [
            {"table": "heldout", "path": "data/heldout/heldout.csv", "format": "csv", "rows": 929},
            {
                "table": "starcoder_trajectories",
                "path": "data/starcoder/starcoder_trajectories.parquet",
                "format": "parquet",
                "rows": 1409700,
            },
        ]
    }
    header, prose = re.fullmatch(r"---\n(.*?)---\n(.*)", upload.build_card(manifest, draft), re.S).groups()
    meta = yaml.safe_load(header)
    assert meta["license"] == "apache-2.0"
    assert meta["size_categories"] == ["1M<n<10M"]  # 1,410,629 rows, the Parquet table's included
    assert [c["config_name"] for c in meta["configs"]] == ["heldout"]
    assert prose.strip() == PROSE.strip()  # the draft's own header is dropped


def test_prepare_scans_what_an_upload_sends_and_reports_the_audit(tree, capsys):
    upload_plan = upload.prepare(tree.build, tree.card, tree.sections, tree.registry)
    # fits/ holds a leak; the scan reads only the files an upload sends, so the build passes.
    assert upload_plan.files == FILES
    assert upload_plan.skipped == ("fits",)
    assert (tree.build / "README.md").read_text().startswith("---\nlicense: apache-2.0\n")
    # The summary carries the audit's own summary line.
    verify_paper_numbers.main(
        ["--build", str(tree.build), "--sections", str(tree.sections), "--registry", str(tree.registry)]
    )
    assert upload_plan.audit == capsys.readouterr().out.splitlines()[-1]
    assert upload_plan.audit == "1 differences, 1 untraced cells, 1 registered skips"


def test_every_listed_config_loads_with_datasets(tree, tmp_path):
    # datasets reads all of a repository's configs with the one builder that the first config's files select, so a
    # Parquet table listed beside CSV tables fails to load; the card lists the CSV tables only.
    upload.prepare(tree.build, tree.card, tree.sections, tree.registry)
    assert get_dataset_config_names(str(tree.build)) == ["heldout"]
    heldout = load_dataset(str(tree.build), "heldout", split="train", cache_dir=str(tmp_path / "cache"))
    assert heldout.num_rows == len(TABLES["data/heldout/heldout.csv"])


def _add_top_level_file(tree: Tree) -> None:
    (tree.build / "notes.txt").write_text("draft\n")


def _leak_into_a_table(tree: Tree) -> None:
    heldout = TABLES["data/heldout/heldout.csv"].assign(source=[LEAK, "public"])
    _write_build(tree.build, {**TABLES, "data/heldout/heldout.csv": heldout})


def _link_the_card_elsewhere(tree: Tree) -> None:
    tree.card.write_text(PROSE + "\nRuns: [W&B](https://example.org/marin).\n")


def _add_an_unlisted_data_file(tree: Tree) -> None:
    (tree.build / "data/heldout/heldout_old.csv").write_text("observation_id\nobs_0123456789ab\n")


def _edit_a_table_after_the_build(tree: Tree) -> None:
    with (tree.build / "data/heldout/heldout.csv").open("a") as f:
        f.write("obs_23456789abcd,0.8\n")


def _record_a_consistency_problem(tree: Tree) -> None:
    _write_build(tree.build, TABLES, consistency_problems=["heldout and ladder disagree about obs_0123456789ab"])


@pytest.mark.parametrize(
    "damage",
    [
        _add_top_level_file,
        _leak_into_a_table,
        _link_the_card_elsewhere,
        _add_an_unlisted_data_file,
        _edit_a_table_after_the_build,
        _record_a_consistency_problem,
    ],
)
def test_upload_refuses_a_build_it_should_not_send(tree, damage: Callable[[Tree], None], monkeypatch):
    damage(tree)
    monkeypatch.setattr(upload, "HfApi", _offline)
    with pytest.raises(ValueError):
        upload.main([*tree.argv, "--confirm"])


def test_dry_run_is_the_default_and_stays_offline(tree, monkeypatch):
    # The contract is the absence of a network call: creating the Hub client fails the test.
    monkeypatch.setattr(upload, "HfApi", _offline)
    upload.main(tree.argv)
    upload.main([*tree.argv, "--dry-run"])


def test_upload_mirrors_the_build_into_a_private_repository(tree, monkeypatch):
    argv = [*tree.argv, "--repo", "someone/mariner-data-test", "--confirm"]
    stale = {".gitattributes": b"*.parquet filter=lfs\n", "data/old/old.csv": b"a\n1\n"}
    public = Hub(private=False, files=dict(stale))
    monkeypatch.setattr(upload, "HfApi", lambda: public)
    with pytest.raises(ValueError):
        upload.main(argv)
    assert public.files == stale  # a public repository receives nothing

    new = Hub(private=None, files={})
    monkeypatch.setattr(upload, "HfApi", lambda: new)
    upload.main(argv)
    assert new.private is True
    assert sorted(new.files) == sorted(FILES)
    assert all(new.files[name] == (tree.build / name).read_bytes() for name in FILES)

    # A later upload deletes the data files the build no longer holds and keeps the repository's other files.
    existing = Hub(private=True, files=dict(stale))
    monkeypatch.setattr(upload, "HfApi", lambda: existing)
    upload.main(argv)
    assert sorted(existing.files) == sorted([".gitattributes", *FILES])

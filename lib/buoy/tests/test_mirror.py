# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Mirror-layer behavior: cache layout, commit marker, idempotency, schema union."""

from __future__ import annotations

import tarfile
import threading

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from buoy import cache
from buoy.mirror import MirrorManager, RunRef, mirror_run
from fakes import FakeArtifact, FakeRun

REF = RunRef("marin-community", "marin_moe", "run-1")


def _rows(n: int, keys: tuple[str, ...]) -> list[dict]:
    return [{"_step": i, **{k: float(i) for k in keys}} for i in range(n)]


def test_layout_and_manifest(cfg, patch_wandb, profile_logdir):
    run = FakeRun(
        name="r1",
        state="finished",
        summary_dict={"train/loss": 0.1, "optim/learning_rate": 3e-4, "_runtime": 5, "notes": "x"},
        config={"lr": 3e-4},
        rows=_rows(10, ("train/loss", "optim/learning_rate")),
        artifacts=[FakeArtifact("jax_profile", "prof:v0", profile_logdir)],
    )
    patch_wandb(run)
    manifest = mirror_run(cfg, REF)

    prefix = cache.run_prefix(cfg.cache_root, *REF.key.split("/"))
    assert cache.read_manifest(prefix) == manifest
    assert manifest["state"] == "finished"
    assert manifest["history"]["rows"] == 10
    # "notes" (str) and "_runtime" (internal) are excluded; numeric metrics kept.
    assert set(manifest["history"]["columns"]) == {"train/loss", "optim/learning_rate"}
    assert cache.read_json(f"{prefix}/config.json") == {"lr": 3e-4}
    # profile artifact mirrored into the cache as an xprof logdir
    assert manifest["profile"]["artifact_name"] == "prof:v0"
    assert cache.exists(manifest["profile"]["logdir"])


def test_manifest_written_last(cfg, patch_wandb, monkeypatch):
    run = FakeRun(summary_dict={"train/loss": 1.0}, rows=_rows(5, ("train/loss",)))
    patch_wandb(run)

    def boom(*a, **k):
        raise RuntimeError("disk full mid-history")

    monkeypatch.setattr(cache, "write_parquet", boom)
    with pytest.raises(RuntimeError):
        mirror_run(cfg, REF)

    prefix = cache.run_prefix(cfg.cache_root, *REF.key.split("/"))
    # No manifest => a reader correctly sees "not cached".
    assert cache.read_manifest(prefix) is None


def test_idempotent_finished(cfg, patch_wandb):
    run = FakeRun(state="finished", summary_dict={"train/loss": 1.0}, rows=_rows(3, ("train/loss",)))
    api = patch_wandb(run)
    mirror_run(cfg, REF)
    mirror_run(cfg, REF)  # second call should hit the cached manifest
    assert api.run_calls == 1


def test_running_run_refetches(cfg, patch_wandb):
    run = FakeRun(state="running", summary_dict={"train/loss": 1.0}, rows=_rows(3, ("train/loss",)))
    api = patch_wandb(run)
    mirror_run(cfg, REF)
    mirror_run(cfg, REF)  # a still-running run is always re-fetched
    assert api.run_calls == 2


def test_divergent_schema_union(cfg, patch_wandb, monkeypatch):
    # Two pages with different key sets; summary carries the union so every shard
    # gets the same columns and read_history concats cleanly.
    monkeypatch.setattr("buoy.mirror.HISTORY_PAGE_ROWS", 2)
    rows = [
        {"_step": 0, "train/loss": 1.0},
        {"_step": 1, "train/loss": 0.9},
        {"_step": 2, "optim/learning_rate": 3e-4},
        {"_step": 3, "optim/learning_rate": 2e-4},
    ]
    run = FakeRun(summary_dict={"train/loss": 0.9, "optim/learning_rate": 2e-4}, rows=rows)
    patch_wandb(run)
    manifest = mirror_run(cfg, REF)
    assert manifest["history"]["parts"] == 2

    prefix = cache.run_prefix(cfg.cache_root, *REF.key.split("/"))
    frame = cache.read_history(prefix, ["_step", "train/loss", "optim/learning_rate"])
    assert len(frame) == 4
    loss = frame[["_step", "train/loss"]].dropna()
    assert list(loss["_step"]) == [0, 1]
    lr = frame[["_step", "optim/learning_rate"]].dropna()
    assert list(lr["_step"]) == [2, 3]


def test_profile_reuse_skips_redownload(cfg, patch_wandb, profile_logdir):
    art = FakeArtifact("jax_profile", "prof:v0", profile_logdir)
    run = FakeRun(state="running", summary_dict={"train/loss": 1.0}, rows=_rows(2, ("train/loss",)), artifacts=[art])
    patch_wandb(run)
    mirror_run(cfg, REF)
    assert art.download_calls == 1
    mirror_run(cfg, REF, refresh=True)  # same artifact version → reuse, no re-download
    assert art.download_calls == 1


def test_finished_run_uses_history_artifact(cfg, patch_wandb, tmp_path):
    # A finished run's `wandb-history` parquet is used directly (fast bulk path);
    # scan_history is NOT used (its rows here are a single non-numeric record).
    art_dir = tmp_path / "histart"
    art_dir.mkdir()
    pq.write_table(
        pa.table({"_step": [0, 1, 2], "train/loss": [3.0, 2.0, 1.0], "note": ["a", "b", "c"]}), art_dir / "0000.parquet"
    )
    art = FakeArtifact("wandb-history", "run-x-history:v5", str(art_dir))
    run = FakeRun(state="finished", summary_dict={"train/loss": 1.0}, rows=[{"nope": 1}], artifacts=[art])
    patch_wandb(run)
    manifest = mirror_run(cfg, REF)

    assert manifest["history"]["rows"] == 3
    assert manifest["history"]["columns"] == ["train/loss"]  # numeric, non-"_" only ("note"/"_step" excluded)
    prefix = cache.run_prefix(cfg.cache_root, *REF.key.split("/"))
    frame = cache.read_history(prefix, ["_step", "train/loss"])
    assert list(frame["train/loss"]) == [3.0, 2.0, 1.0]


def test_pending_run_is_refetched(cfg, patch_wandb):
    # Only terminal states are immutable; a nonterminal 'pending' run must re-fetch.
    run = FakeRun(state="pending", summary_dict={"train/loss": 1.0}, rows=_rows(2, ("train/loss",)))
    api = patch_wandb(run)
    mirror_run(cfg, REF)
    mirror_run(cfg, REF)
    assert api.run_calls == 2


def test_write_json_sanitizes_non_finite(tmp_path):
    # NaN/±Inf (common in wandb summaries) must be cached as strict JSON (null),
    # not the bare NaN/Infinity tokens that break the browser's JSON.parse.
    path = str(tmp_path / "summary.json")
    cache.write_json(path, {"a": float("nan"), "b": float("inf"), "c": 1.5, "d": {"e": float("-inf")}})
    text = (tmp_path / "summary.json").read_text()
    assert "NaN" not in text and "Infinity" not in text
    assert cache.read_json(path) == {"a": None, "b": None, "c": 1.5, "d": {"e": None}}


def test_running_run_incremental_history(cfg, patch_wandb):
    # A running run's refresh fetches only new steps and appends a shard.
    run = FakeRun(state="running", summary_dict={"train/loss": 1.0}, rows=_rows(3, ("train/loss",)))
    api = patch_wandb(run)
    m1 = mirror_run(cfg, REF)
    assert m1["history"]["rows"] == 3 and m1["history"]["last_step"] == 2 and m1["history"]["parts"] == 1

    run.rows = _rows(5, ("train/loss",))  # run advanced to steps 0..4
    m2 = mirror_run(cfg, REF, refresh=True)
    assert m2["history"]["rows"] == 5  # 3 kept + 2 appended
    assert m2["history"]["last_step"] == 4
    assert m2["history"]["parts"] == 2  # a new shard for steps 3,4 (old shard untouched)
    assert api.run_calls == 2

    prefix = cache.run_prefix(cfg.cache_root, *REF.key.split("/"))
    frame = cache.read_history(prefix, ["_step", "train/loss"])
    assert sorted(frame["_step"].tolist()) == [0, 1, 2, 3, 4]


def test_read_history_tolerates_column_absent_from_all_shards(cfg, patch_wandb):
    # A metric can be advertised in the manifest (summary) yet appear in no shard —
    # e.g. it turns numeric on a refresh that adds no new rows. read_history must
    # still return it (all-NaN), not omit it (which would KeyError get_metrics).
    run = FakeRun(state="running", summary_dict={"train/loss": 1.0}, rows=_rows(3, ("train/loss",)))
    patch_wandb(run)
    mirror_run(cfg, REF)
    run.summary_dict = {"train/loss": 1.0, "eval/acc": 0.5}  # newly advertised, no new rows
    manifest = mirror_run(cfg, REF, refresh=True)
    assert "eval/acc" in manifest["history"]["columns"]

    prefix = cache.run_prefix(cfg.cache_root, *REF.key.split("/"))
    frame = cache.read_history(prefix, ["_step", "eval/acc"])
    assert "eval/acc" in frame.columns and frame["eval/acc"].isna().all()


def test_running_run_scans_despite_history_artifact(cfg, patch_wandb, tmp_path):
    # A running run's history artifact is a stale snapshot — mirror must scan for freshness.
    art_dir = tmp_path / "h"
    art_dir.mkdir()
    pq.write_table(pa.table({"_step": [0], "train/loss": [9.0]}), art_dir / "0000.parquet")
    art = FakeArtifact("wandb-history", "run-x-history:v1", str(art_dir))
    run = FakeRun(state="running", summary_dict={"train/loss": 1.0}, rows=_rows(4, ("train/loss",)), artifacts=[art])
    patch_wandb(run)
    assert mirror_run(cfg, REF)["history"]["rows"] == 4  # from scan (4 rows), not the 1-row artifact


def test_profiler_tarball_extracted(cfg, patch_wandb, tmp_path):
    # A `profiler`-type artifact ships a single .tgz of the xprof logdir; the mirror
    # must extract it so xprof can read plugins/profile.
    logdir = tmp_path / "profiler" / "plugins" / "profile" / "2026_01_01"
    logdir.mkdir(parents=True)
    (logdir / "host.xplane.pb").write_bytes(b"\x00xplane")
    download = tmp_path / "download"
    download.mkdir()
    with tarfile.open(download / "prof.tgz", "w:gz") as tar:
        tar.add(tmp_path / "profiler", arcname="profiler")

    art = FakeArtifact("profiler", "myprof:v0", str(download))
    run = FakeRun(summary_dict={"train/loss": 1.0}, rows=_rows(2, ("train/loss",)), artifacts=[art])
    patch_wandb(run)
    manifest = mirror_run(cfg, REF)

    assert manifest["profile"]["artifact_name"] == "myprof:v0"
    gcs_logdir = manifest["profile"]["logdir"]
    assert cache.exists(f"{gcs_logdir}/plugins/profile/2026_01_01/host.xplane.pb")


def test_touch_running_noop_when_finished(cfg):
    mgr = MirrorManager(cfg)
    assert mgr.touch_running(REF, {"state": "finished"}) is None


def test_watcher_refreshes_until_terminal(cfg, patch_wandb, monkeypatch):
    monkeypatch.setattr("buoy.mirror.WATCH_INTERVAL", 0.001)
    run = FakeRun(state="running", summary_dict={"train/loss": 1.0}, rows=_rows(3, ("train/loss",)))
    patch_wandb(run)
    mgr = MirrorManager(cfg)
    mirror_run(cfg, REF)  # initial running manifest
    run.state = "finished"  # the watcher's next refresh observes a terminal state
    thread = mgr.touch_running(REF, {"state": "running"})
    assert thread is not None
    thread.join(5)
    assert not thread.is_alive()
    prefix = cache.run_prefix(cfg.cache_root, *REF.key.split("/"))
    assert cache.read_manifest(prefix)["state"] == "finished"


def test_manager_coalesces_concurrent(cfg, patch_wandb, monkeypatch):
    run = FakeRun(summary_dict={"train/loss": 1.0}, rows=_rows(3, ("train/loss",)))
    patch_wandb(run)
    real = mirror_run
    entered = threading.Event()
    release = threading.Event()
    calls: list[int] = []

    def gated(cfg_, ref, *, refresh=False, on_progress=None):
        calls.append(1)
        entered.set()
        assert release.wait(5)
        return real(cfg_, ref, refresh=refresh, on_progress=on_progress)

    monkeypatch.setattr("buoy.mirror.mirror_run", gated)
    mgr = MirrorManager(cfg)

    first = mgr.start(REF)
    assert entered.wait(5)  # the first worker is now inside the gated mirror
    second = mgr.start(REF)  # coalesced: a mirror is already running
    assert second is None
    assert calls == [1]

    release.set()
    first.join(5)
    assert not first.is_alive()
    assert mgr.status(REF)["state"] == "done"
    assert calls == [1]

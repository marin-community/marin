"""Tests for traces_utils.iter_trial_dirs trial-directory discovery.

Focus: the pruning behavior added to avoid stat-ing every file inside every
trial dir (an O(all-files) GPFS stall on large jobs). Coverage must be
identical to a naive recursive walk while never descending into a trial's
contents.
"""

from pathlib import Path

from upath import UPath

from harbor.utils.traces_utils import is_trial_dir, iter_trial_dirs


def _make_trial(parent: Path, name: str) -> Path:
    """Create a directory that looks like a trial (has an ``agent/`` subdir)
    plus the usual content subdirs that the old rglob path would have stat-ed."""
    trial = parent / name
    (trial / "agent").mkdir(parents=True)
    (trial / "agent" / "trajectory.json").write_text("{}")
    (trial / "verifier").mkdir()
    (trial / "verifier" / "reward.txt").write_text("1.0")
    (trial / "artifacts").mkdir()
    return trial


def test_is_trial_dir(tmp_path):
    assert not is_trial_dir(tmp_path)
    (tmp_path / "agent").mkdir()
    assert is_trial_dir(tmp_path)


def test_root_is_trial_dir_yields_self(tmp_path):
    trial = _make_trial(tmp_path, "solo")
    assert list(iter_trial_dirs(trial)) == [trial]


def test_datagen_single_nest(tmp_path):
    # <run>/trace_jobs/<inner>/<trial>
    inner = tmp_path / "trace_jobs" / "inner_run"
    inner.mkdir(parents=True)
    trials = {_make_trial(inner, f"task{i}__{i}") for i in range(3)}
    assert set(iter_trial_dirs(tmp_path)) == trials


def test_rl_double_nest(tmp_path):
    # <job>/<job>/trace_jobs/<inner>/<trial> — variable depth still discovered
    inner = tmp_path / "job" / "job" / "trace_jobs" / "inner_run"
    inner.mkdir(parents=True)
    trials = {_make_trial(inner, f"task{i}__{i}") for i in range(3)}
    assert set(iter_trial_dirs(tmp_path)) == trials


def test_pruning_does_not_descend_into_trial(tmp_path):
    """A trial dir's contents must NOT be searched: a directory that happens to
    look like a trial dir *inside* an already-matched trial is not yielded."""
    inner = tmp_path / "trace_jobs" / "inner_run"
    inner.mkdir(parents=True)
    outer = _make_trial(inner, "task__1")
    # Plant a decoy "trial-looking" dir under the matched trial's contents.
    decoy = outer / "agent" / "nested_decoy"
    (decoy / "agent").mkdir(parents=True)

    found = set(iter_trial_dirs(tmp_path))
    assert found == {outer}
    assert decoy not in found


def test_non_recursive_returns_nothing_when_root_not_trial(tmp_path):
    inner = tmp_path / "trace_jobs" / "inner_run"
    inner.mkdir(parents=True)
    _make_trial(inner, "task__1")
    assert list(iter_trial_dirs(tmp_path, recursive=False)) == []


def test_discovers_trials_on_non_local_filesystem():
    """Job dirs are commonly remote URIs (``gs://``). Discovery must work on a
    non-local fsspec filesystem; the old ``os.walk`` path raised ``TypeError``
    on a remote ``UPath`` (its ``__fspath__`` is not a local str). ``memory://``
    is a stand-in for any object store with the same non-local semantics."""
    root = UPath("memory://job/job")
    inner = root / "trace_jobs" / "inner_run"
    trials = set()
    for i in range(3):
        trial = inner / f"task{i}__{i}"
        (trial / "agent").mkdir(parents=True)
        (trial / "agent" / "trajectory.json").write_text("{}")
        trials.add(trial)

    assert set(iter_trial_dirs(root)) == trials

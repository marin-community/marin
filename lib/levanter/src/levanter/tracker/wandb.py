# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import logging
import os
import subprocess
import tempfile
import typing
import warnings
from dataclasses import dataclass
from typing import Any, List, Optional, Union

import fsspec
import jax
import numpy as np
import wandb
from draccus import field
from git import InvalidGitRepositoryError, NoSuchPathError, Repo

from levanter.tracker.background import maybe_wrap_background
from levanter.tracker.helpers import generate_pip_freeze, infer_experiment_git_root
from levanter.tracker.histogram import SummaryStats
from levanter.tracker.tracker import Tracker, TrackerConfig
from levanter.utils import jax_utils

if typing.TYPE_CHECKING:
    import wandb.sdk.lib.disabled


logger = logging.getLogger(__name__)

WandbRun = Union["wandb.sdk.wandb_run.Run", "wandb.sdk.lib.disabled.RunDisabled"]


_WANDB_ARTIFACT_NAME_MAX_LENGTH = 128


class WandbTracker(Tracker):
    name: str = "wandb"
    run: WandbRun

    def __init__(
        self,
        run: Optional[WandbRun],
        replicate_path: Optional[str] = None,
        suppress_logging: bool = False,
        minimum_log_step: int = 0,
    ):
        if run is None:
            if wandb.run is None:
                logger.warning("Wandb run is not initialized. Initializing a new run.")
                runx = wandb.init()
                if runx is None:
                    raise RuntimeError("Wandb run is not initialized.")
                self.run = runx
            else:
                self.run = wandb.run
        else:
            self.run = run

        self._last_warning_step = -500
        self._replicate_path = replicate_path
        self._suppress_logging = suppress_logging
        self._minimum_log_step = minimum_log_step

    # The prepare hooks do the full wandb-side conversion (SummaryStats expansion,
    # wandb.Histogram construction) so the background worker only uploads. They are
    # idempotent, so re-running them on an already-prepared payload is a no-op.

    def _prepare_log(self, metrics):
        return _convert_metrics_to_wandb_loggable(metrics)

    def _prepare_summary(self, metrics):
        return _convert_value_to_loggable_rec(metrics)

    def _prepare_hyperparameters(self, hparams):
        return _convert_value_to_loggable_rec(hparams)

    def log_hyperparameters(self, hparams: dict[str, Any]):
        if self._suppress_logging:
            return
        self.run.config.update(self._prepare_hyperparameters(hparams), allow_val_change=True)

    def log(self, metrics: typing.Mapping[str, Any], *, step, commit=None):
        if step is None and not commit:
            step = self.run.step

        if step < self._minimum_log_step:
            if step - self._last_warning_step > 500:
                logger.warning(
                    f"Step {step} is less than the current step {self._minimum_log_step}. "
                    "Cowardly refusing to log metrics."
                )
                self._last_warning_step = step
            return

        step = int(step)

        to_log = self._prepare_log(metrics)

        if self._suppress_logging:
            return

        if step < self.run.step:
            if step - self._last_warning_step > 500:
                logger.warning(
                    f"Step {step} is less than the current step {self.run.step}. Cowardly refusing to log metrics."
                )
                self._last_warning_step = step
            return

        self.run.log(to_log, step=step, commit=commit)

    def log_summary(self, metrics: typing.Mapping[str, Any]):
        if self._suppress_logging:
            return
        self.run.summary.update(self._prepare_summary(metrics))

    def log_artifact(self, artifact_path, *, name: Optional[str] = None, type: Optional[str] = None):
        if self._suppress_logging:
            return
        artifact_name = name if name is not None else _default_wandb_artifact_name(artifact_path)
        self.run.log_artifact(
            artifact_path,
            name=_truncate_wandb_artifact_name(artifact_name),
            type=type,
        )

    def log_html(self, key: str, html_path, *, step: Optional[int], commit: Optional[bool] = None):
        if step is None and not commit:
            step = self.run.step
        if step is not None and step < self.run.step:
            if step - self._last_warning_step > 500:
                logger.warning(
                    f"Step {step} is less than the current step {self.run.step}. Cowardly refusing to log HTML."
                )
                self._last_warning_step = step
            return

        wandb_step = None if step is None else int(step)
        self.run.log({key: wandb.Html(str(html_path))}, step=wandb_step, commit=commit)

    def finish(self):
        if self._suppress_logging:
            return

        # Write the replicate file before touching wandb: run.finish() can wedge
        # past the background finish timeout, and by the time it raises the
        # process is tearing down and fsspec can no longer schedule writes.
        # This write is best-effort insurance — its failure must not keep
        # run.finish() from running — and its summary may miss the final
        # commit, so a successful finish rewrites the file with the flushed
        # values below.
        try:
            self._write_replicate_file()
        except Exception:
            logger.exception("Pre-finish replicate write failed; retrying after wandb finish.")

        logger.info("Finishing wandb run...")
        self.run.finish()
        self._write_replicate_file()

    def _write_replicate_file(self):
        if self._replicate_path is None:
            return

        metrics_file = f"{self._replicate_path}/tracker_metrics.jsonl"
        fs, _, _ = fsspec.get_fs_token_paths(metrics_file)
        fs.makedirs(self._replicate_path, exist_ok=True)

        record = {
            "config": _convert_value_to_loggable_rec(dict(self.run.config)),
            "summary": _convert_value_to_loggable_rec(_summary_for_replicate(self.run)),
        }
        # Write to a temp name and rename into place so a failed write can never
        # truncate an existing tracker_metrics.jsonl (e.g. the pre-finish copy).
        tmp_file = f"{metrics_file}.tmp"
        with fs.open(tmp_file, "w") as f:
            f.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        fs.mv(tmp_file, metrics_file)


def _summary_for_replicate(run: WandbRun) -> dict[str, Any]:
    """Read final W&B summary in a way that survives `run.finish()`."""
    # run.summary can be stale before finish(); _final_summary has the fully flushed values
    final_summary = getattr(run, "_final_summary", None)
    if final_summary is None:
        return dict(run.summary)

    summary: dict[str, Any] = {}
    for item in final_summary.item:
        path: list[str] = list(item.nested_key)
        if item.key:
            path.append(item.key)
        if not path:
            continue

        try:
            value = json.loads(item.value_json)
        except (TypeError, json.JSONDecodeError):
            value = item.value_json

        _set_nested(summary, path, value)

    return summary


def _set_nested(target: dict[str, Any], path: list[str], value: Any) -> None:
    cur = target
    for key in path[:-1]:
        next_val = cur.get(key)
        if not isinstance(next_val, dict):
            next_val = {}
            cur[key] = next_val
        cur = next_val
    cur[path[-1]] = value


def _convert_metrics_to_wandb_loggable(metrics: typing.Mapping[str, Any]) -> dict[str, Any]:
    """Flatten metrics into a wandb-ready dict.

    Expands every :class:`SummaryStats` value into its individual loggable keys
    (computing ``mean``/``variance``/``rms`` and building ``wandb.Histogram`` as
    needed) and passes every other value through
    :func:`_convert_value_to_loggable_rec`.

    Pure conversion: no wandb run state is touched. Safe to call on the producer
    thread before handing off to a :class:`BackgroundTracker` worker, and
    idempotent when called a second time on an already-flat dict.
    """
    to_log: dict[str, Any] = {}
    for k, v in metrics.items():
        if isinstance(v, SummaryStats):
            to_log.update(_convert_summary_stats_to_loggable(k, v))
        else:
            to_log[k] = _convert_value_to_loggable_rec(v)
    return to_log


def _convert_value_to_loggable_rec(value: Any):
    if isinstance(value, (list, tuple)):
        return [_convert_value_to_loggable_rec(v) for v in value]
    elif isinstance(value, typing.Mapping):
        return {k: _convert_value_to_loggable_rec(v) for k, v in value.items()}
    elif isinstance(value, jax.Array):
        if value.ndim == 0:
            return value.item()
        else:
            return np.array(value)
    elif isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        else:
            return value.tolist()
    elif isinstance(value, np.generic):
        return value.item()
    elif isinstance(value, SummaryStats):
        return _convert_summary_stats_to_loggable("", value, include_prefix=False)
    else:
        return value


def _convert_summary_stats_to_loggable(prefix: str, value: SummaryStats, *, include_prefix: bool = True):
    out: dict[str, Any] = {}
    base = f"{prefix}/" if include_prefix and prefix else ""
    out[f"{base}min"] = _convert_value_to_loggable_rec(value.min)
    out[f"{base}max"] = _convert_value_to_loggable_rec(value.max)
    out[f"{base}num"] = _convert_value_to_loggable_rec(value.num)
    out[f"{base}nonzero_count"] = _convert_value_to_loggable_rec(value.nonzero_count)
    out[f"{base}sum"] = _convert_value_to_loggable_rec(value.sum)
    out[f"{base}sum_squares"] = _convert_value_to_loggable_rec(value.sum_squares)
    out[f"{base}mean"] = _convert_value_to_loggable_rec(value.mean)
    out[f"{base}variance"] = _convert_value_to_loggable_rec(value.variance)
    out[f"{base}rms"] = _convert_value_to_loggable_rec(value.rms)
    if value.histogram is not None:
        counts, limits = value.histogram.to_numpy_histogram()
        out[f"{base}histogram"] = wandb.Histogram(np_histogram=(counts.tolist(), limits.tolist()))
    return out


def is_wandb_available():
    return wandb.run is not None


@TrackerConfig.register_subclass("wandb")
@dataclass
class WandbConfig(TrackerConfig):
    """
    Configuration for wandb.
    """

    entity: Optional[str] = None  # An entity is a username or team name where you send runs
    project: Optional[str] = "levanter"  # The name of the project where you are sending the enw run.
    name: Optional[str] = None  # A short display name for this run, which is how you'll identify this run in the UI.
    tags: List[str] = field(default_factory=list)  # Will populate the list of tags on this run in the UI.
    id: Optional[str] = None  # A unique ID for this run, used for resuming. It must be unique in the project
    group: Optional[str] = None  # Specify a group to organize individual runs into a larger experiment.
    mode: Optional[str] = None  # Can be "online", "offline" or "disabled". If None, it will be whatever W&B decides.
    resume: Optional[Union[bool, str]] = "allow"
    """
    Set the resume behavior. Options: "allow", "must", "never", "auto" or None.
    By default, if the new run has the same ID as a previous run, this run overwrites that data.
    Please refer to [init](https://docs.wandb.ai/ref/python/init) and [resume](https://docs.wandb.ai/guides/runs/resuming)
    document for more details.
    """

    save_code: Union[bool, str] = True
    """If string, will save code from that directory. If True, will attempt to sniff out the main directory (since we
    typically don't run from the root of the repo)."""

    save_xla_dumps: bool = False
    """If True, will save the XLA code to wandb (as configured by XLA_FLAGS). This is useful for debugging."""

    replicate_path: Optional[str] = None
    """If set, write config and summary to this path (local or GCS) on finish()."""

    background: bool = True
    """If True (default), forward all log calls through a background thread that catches
    exceptions from W&B. This keeps long-running training jobs alive when W&B is
    unreachable, runs out of storage quota, or returns transient errors. Set to False
    only if you need synchronous, fail-fast behavior (e.g. from tests)."""

    background_max_queue_size: int = 10000
    """Max number of pending tracker calls. If exceeded, additional calls are dropped
    with a rate-limited warning rather than blocking the trainer."""

    background_finish_timeout: float = 120.0
    """Maximum seconds to wait for the background thread to drain on finish()."""

    def init(self, run_id: Optional[str]) -> Tracker:
        if run_id is not None and self.id is not None and run_id != self.id:
            warnings.warn(
                f"Both trainer's id {run_id} and WandB's id {self.id} are set. WandB will use the id set in its"
                " config."
            )

        id = self.id
        if id is None:
            id = run_id

        hparams_to_save = {}

        # for distributed runs, we only want the primary worker to use wandb, so we make everyone else be disabled
        # however, we do share information about the run id, so that we can link to it from the other workers
        is_primary_process = jax.process_index() == 0
        if is_primary_process:
            mode = self.mode
        else:
            mode = "disabled"

        git_settings = self._git_settings()

        if "git_commit" in git_settings:
            hparams_to_save["git_commit"] = git_settings["git_commit"]

        r = wandb.init(
            entity=self.entity,
            project=self.project,
            name=self.name,
            tags=self.tags,
            id=id,
            group=self.group,
            resume=self.resume,
            mode=mode,
            config=hparams_to_save,
            settings=git_settings,
            allow_val_change=True,
        )

        assert r is not None

        if r.step != 0:
            logger.info("Resuming wandb run. Attempting to mitigate issues.")

        minimum_log_step = int(r.step)
        if jax.process_count() > 1:
            # we need to share wandb run information across all hosts, because we use it for checkpoint paths and things
            metadata_to_share = dict(
                # entity=r.entity,
                project=r.project,
                name=r.name,
                tags=r.tags,
                id=r.id,
                group=r.group,
                minimum_log_step=minimum_log_step,
            )
            metadata_to_share = jax_utils.multihost_broadcast_sync(
                metadata_to_share, is_source=jax.process_index() == 0
            )
            minimum_log_step = int(metadata_to_share["minimum_log_step"])

            # if jax.process_index() != 0:
            # assert r.mode == "disabled", f"Only the primary worker should be using wandb. Got {r.mode}"
            # for k, v in metadata_to_share.items():
            #     setattr(r, k, v)

            logger.info(f"Synced wandb run information from process 0: {r.name} {r.id}")

        # generate a pip freeze
        if is_primary_process:
            with tempfile.TemporaryDirectory() as tmpdir:
                requirements_path = os.path.join(tmpdir, "requirements.txt")
                requirements = generate_pip_freeze()
                with open(requirements_path, "w") as f:
                    f.write(requirements)
                if wandb.run is not None:
                    wandb.run.log_artifact(str(requirements_path), name="requirements.txt", type="requirements")

            wandb.summary["num_devices"] = jax.device_count()  # type: ignore
            wandb.summary["num_hosts"] = jax.process_count()  # type: ignore
            wandb.summary["backend"] = jax.default_backend()  # type: ignore

        return maybe_wrap_background(
            WandbTracker(
                r,
                replicate_path=self.replicate_path,
                suppress_logging=not is_primary_process,
                minimum_log_step=minimum_log_step,
            ),
            # Only the primary process actually logs; a suppressed tracker no-ops every
            # call, so wrapping it in a background thread is pure overhead — and would
            # make non-primary hosts stage (copy) large profile artifacts they discard.
            enabled=self.background and is_primary_process,
            max_queue_size=self.background_max_queue_size,
            finish_timeout=self.background_finish_timeout,
        )

    def _git_settings(self):
        other_settings = dict()
        if isinstance(self.save_code, str):
            code_dir = self.save_code
        elif self.save_code:
            code_dir = infer_experiment_git_root() or "."  # type: ignore
        else:
            code_dir = None
        if code_dir is not None:
            logger.info(f"Setting wandb code_dir to {code_dir}")
            other_settings["code_dir"] = code_dir
            other_settings["git_root"] = code_dir
            # for some reason, wandb isn't populating the git commit, so we do it here
            try:
                sha = self._get_git_sha(code_dir)
            except:  # noqa: E722
                logger.warning(f"Could not get git sha for {code_dir}. Will not log git commit.")
                sha = None
            if sha is not None:
                other_settings["git_commit"] = sha

        return other_settings

    def _get_git_sha(self, code_dir) -> Optional[str]:
        if "GIT_COMMIT" in os.environ:
            return os.environ["GIT_COMMIT"]

        try:
            repo = Repo(code_dir)
            git_sha = repo.head.commit.hexsha
        except (NoSuchPathError, InvalidGitRepositoryError):
            logger.warning(f"Could not find git repo at {code_dir}")
            return None
        except ValueError as e:
            if "SHA is empty" in str(e):
                # we have another workaround, which is to use the git command line
                # git --git-dir={code_dir}/.git rev-parse HEAD
                try:
                    out = subprocess.run(
                        ["git", "--git-dir", f"{code_dir}/.git", "rev-parse", "HEAD"], check=True, capture_output=True
                    )
                    git_sha = out.stdout.decode().strip()
                except subprocess.CalledProcessError:
                    return None
            else:
                raise e

        return git_sha


def _truncate_wandb_artifact_name(name: Optional[str]) -> Optional[str]:
    """Truncate artifact names to keep within WandB's artifact-name limit."""
    if name is None:
        return None
    if len(name) <= _WANDB_ARTIFACT_NAME_MAX_LENGTH:
        return name
    # Keep names stable and unique across different long inputs by keeping a short hash suffix.
    hash_suffix = hashlib.sha256(name.encode("utf-8")).hexdigest()[:7]
    max_truncated_prefix_len = _WANDB_ARTIFACT_NAME_MAX_LENGTH - len(hash_suffix) - 1
    truncated = f"{name[:max_truncated_prefix_len]}-{hash_suffix}"
    logger.warning(
        "Wandb artifact name exceeds %d characters and will be truncated: %s -> %s",
        _WANDB_ARTIFACT_NAME_MAX_LENGTH,
        name,
        truncated,
    )
    return truncated


_WANDB_RUN_NAME_MAX_LENGTH = 64
_WANDB_RUN_NAME_MIN_PREFIX_LENGTH = 24
_WANDB_RUN_NAME_AGGRESSIVE_TRUNCATION_CHARS = 16
_WANDB_RUN_NAME_SUFFIX_MARKERS = ("_seed", "-seed", "_step", "-step")


def _preferred_wandb_suffix_start(name: str) -> int | None:
    """Return the preferred suffix start for a truncated W&B run name.

    We prefer underscore-delimited semantic tails like ``lr7.5e-7_seed2`` or
    ``foo_step400``. This avoids splitting on the ``-`` inside scientific
    notation, which would corrupt names like ``lr7.5e-7``.
    """
    for marker in _WANDB_RUN_NAME_SUFFIX_MARKERS:
        marker_start = name.rfind(marker)
        if marker_start == -1:
            continue

        prior_slash = name.rfind("/", 0, marker_start)
        prior_underscore = name.rfind("_", 0, marker_start)
        if prior_underscore > prior_slash:
            return prior_underscore
        return marker_start

    last_underscore = name.rfind("_")
    if last_underscore == -1:
        return None

    prior_slash = name.rfind("/", 0, last_underscore)
    second_last_underscore = name.rfind("_", 0, last_underscore)
    if second_last_underscore > prior_slash:
        return second_last_underscore

    return last_underscore


def truncate_wandb_run_name(name: str) -> str:
    """Truncate a run name to fit within W&B's run-name length limit.

    W&B rejects run names longer than 64 characters. This trims an over-long name
    while preserving a readable prefix and the trailing semantic suffix (e.g.
    ``lr7.5e-7_seed2``), avoiding splits inside scientific-notation tails, and logs
    a warning so the truncation is visible. Exposed as a public helper so callers
    (e.g. experiment configs) share one run-name policy.
    """
    if len(name) <= _WANDB_RUN_NAME_MAX_LENGTH:
        return name

    old_name = name
    suffix_start = _preferred_wandb_suffix_start(name)

    if suffix_start is None:
        name = name[:_WANDB_RUN_NAME_MAX_LENGTH]
        preserved_suffix = ""
    else:
        suffix = name[suffix_start:]
        preserved_suffix = suffix
        if len(suffix) >= _WANDB_RUN_NAME_MAX_LENGTH:
            name = name[:_WANDB_RUN_NAME_MAX_LENGTH]
            preserved_suffix = ""
        else:
            prefix_budget = _WANDB_RUN_NAME_MAX_LENGTH - len(suffix)
            prefix = name[:prefix_budget]

            # Prefer trimming at a token boundary so the retained prefix stays readable.
            boundary = max(prefix.rfind("_"), prefix.rfind("/"))
            if boundary >= _WANDB_RUN_NAME_MIN_PREFIX_LENGTH:
                prefix = prefix[:boundary]

            name = prefix + suffix

    logger.warning(f"Truncated name from {old_name} to {name} to fit within WANDB limits.")

    removed_chars = len(old_name) - len(name)
    retained_prefix_len = len(name) - len(preserved_suffix)
    if (
        removed_chars >= _WANDB_RUN_NAME_AGGRESSIVE_TRUNCATION_CHARS
        or retained_prefix_len < _WANDB_RUN_NAME_MIN_PREFIX_LENGTH
    ):
        logger.warning(
            "W&B run name %r required aggressive truncation to %r. Consider shortening the explicit name.",
            old_name,
            name,
        )

    return name


def _default_wandb_artifact_name(artifact_path: Any) -> str:
    path = os.fspath(artifact_path)
    basename = os.path.basename(path.rstrip("/\\"))
    return basename or "artifact"

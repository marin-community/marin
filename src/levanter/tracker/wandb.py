# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import tempfile
import typing
import warnings
from dataclasses import dataclass
from typing import Any, List, Optional, Union

import jax
import numpy as np
from draccus import field
from git import InvalidGitRepositoryError, NoSuchPathError, Repo

from levanter.tracker import Tracker
from levanter.tracker.helpers import generate_pip_freeze, infer_experiment_git_root
from levanter.tracker.histogram import Histogram
from levanter.tracker.tracker import TrackerConfig
from levanter.utils import jax_utils


if typing.TYPE_CHECKING:
    import wandb.sdk.lib.disabled

    import wandb


logger = logging.getLogger(__name__)

WandbRun = Union["wandb.sdk.wandb_run.Run", "wandb.sdk.lib.disabled.RunDisabled"]


class WandbTracker(Tracker):
    name: str = "wandb"
    run: WandbRun

    def __init__(self, run: Optional[WandbRun]):
        import wandb

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

    def log_hyperparameters(self, hparams: dict[str, Any]):
        self.run.config.update(_convert_value_to_loggable_rec(hparams), allow_val_change=True)

    def log(self, metrics: typing.Mapping[str, Any], *, step, commit=None):
        if step is None and not commit:
            step = self.run.step

        if step < self.run.step:
            if step - self._last_warning_step > 500:
                logger.warning(
                    f"Step {step} is less than the current step {self.run.step}. Cowardly refusing to log metrics."
                )
                self._last_warning_step = step
            return

        step = int(step)

        # wandb histograms are pretty limited: they log only the counts and the bin edges.
        # Our histograms have the same set of things Tensorboard. we log those as separate values.
        to_log = {}
        for k, v in metrics.items():
            if isinstance(v, Histogram):
                # if the value is a Histogram, convert it to a wandb Histogram
                # this will log the histogram counts and bin edges
                import wandb

                counts, limits = v.to_numpy_histogram()
                wandb_hist = wandb.Histogram(np_histogram=(counts.tolist(), limits.tolist()))
                to_log[f"{k}/histogram"] = wandb_hist
                to_log[f"{k}/min"] = v.min
                to_log[f"{k}/max"] = v.max
                to_log[f"{k}/mean"] = v.mean
                to_log[f"{k}/variance"] = v.variance
            else:
                # otherwise, just log the value normally
                to_log[k] = _convert_value_to_loggable_rec(v)

        self.run.log(to_log, step=step, commit=commit)

    def log_summary(self, metrics: typing.Mapping[str, Any]):
        self.run.summary.update(_convert_value_to_loggable_rec(metrics))

    def log_artifact(self, artifact_path, *, name: Optional[str] = None, type: Optional[str] = None):
        self.run.log_artifact(artifact_path, name=name, type=type)

    def finish(self):
        logger.info("Finishing wandb run...")
        self.run.finish()


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
    elif isinstance(value, Histogram):
        import wandb

        counts, limits = value.to_numpy_histogram()

        return wandb.Histogram(np_histogram=(counts.tolist(), limits.tolist()))
    else:
        return value


def is_wandb_available():
    try:
        import wandb
    except ImportError:
        return False
    return wandb is not None and wandb.run is not None


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

    def init(self, run_id: Optional[str]) -> WandbTracker:
        import wandb

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
        if jax.process_index() == 0:
            mode = self.mode
        else:
            mode = "offline"

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

        if jax.process_count() > 1:
            # we need to share wandb run information across all hosts, because we use it for checkpoint paths and things
            metadata_to_share = dict(
                # entity=r.entity,
                project=r.project,
                name=r.name,
                tags=r.tags,
                id=r.id,
                group=r.group,
            )
            metadata_to_share = jax_utils.multihost_broadcast_sync(
                metadata_to_share, is_source=jax.process_index() == 0
            )

            # if jax.process_index() != 0:
            # assert r.mode == "disabled", f"Only the primary worker should be using wandb. Got {r.mode}"
            # for k, v in metadata_to_share.items():
            #     setattr(r, k, v)

            logger.info(f"Synced wandb run information from process 0: {r.name} {r.id}")

        # generate a pip freeze
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

        return WandbTracker(r)

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
                import subprocess

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

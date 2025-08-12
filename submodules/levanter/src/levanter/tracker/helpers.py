import contextlib
import dataclasses
import logging
import os
import time
from typing import Optional

from git import InvalidGitRepositoryError, NoSuchPathError, Repo

import levanter.tracker
from levanter.utils.jax_utils import jnp_to_python


logger = logging.getLogger(__name__)


def log_optimizer_hyperparams(opt_state, prefix: Optional[str] = None, *, step=None):
    try:
        from optax._src.wrappers import MultiStepsState

        if isinstance(opt_state, MultiStepsState):
            opt_state = opt_state.inner_opt_state
    except ImportError:
        pass

    def wrap_key(key):
        if prefix:
            return f"{prefix}/{key}"
        return key

    if hasattr(opt_state, "hyperparams"):
        params = {wrap_key(k): jnp_to_python(v) for k, v in opt_state.hyperparams.items()}
        levanter.tracker.log(params, step=step)


def hparams_to_dict(hparams, **extra_hparams):
    if hparams is None:
        hparams_to_save = {}
    elif dataclasses.is_dataclass(hparams):
        hparams_to_save = dataclasses.asdict(hparams)
    else:
        hparams_to_save = dict(hparams)
    if extra_hparams:
        hparams_to_save.update(extra_hparams)
    return hparams_to_save


def infer_experiment_git_root() -> Optional[str | os.PathLike[str]]:
    # sniff out the main directory (since we typically don't run from the root of the repo)
    # we'll walk the stack and directories for the files in the stack the until we're at a git root
    import os
    import traceback

    stack = traceback.extract_stack()
    # start from the top of the stack and work our way down since we want to hit the main file first
    top_git_root = None
    for frame in stack:
        dirname = os.path.dirname(frame.filename)
        # bit hacky but we want to skip anything that's in the python env
        if any(x in dirname for x in ["site-packages", "dist-packages", "venv", "opt/homebrew", "conda", "pyenv"]):
            continue
        # see if it's under a git root
        try:
            repo = Repo(dirname, search_parent_directories=True)
            top_git_root = repo.working_dir
            break
        except (NoSuchPathError, InvalidGitRepositoryError):
            logger.debug(f"Skipping {dirname} since it's not a git root")
            pass
    return top_git_root


def generate_pip_freeze():
    from importlib.metadata import distributions

    dists = distributions()
    return "\n".join(f"{dist.name}=={dist.version}" for dist in dists)


@contextlib.contextmanager
def capture_time():
    start = time.perf_counter()
    done = False

    def fn():
        if done:
            return end - start
        else:
            return time.perf_counter() - start

    yield fn
    end = time.perf_counter()
    done = True

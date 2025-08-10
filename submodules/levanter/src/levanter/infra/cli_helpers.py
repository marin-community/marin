import argparse
import base64
import dataclasses
import os
import subprocess
import warnings
from dataclasses import dataclass
from functools import cached_property
from typing import Optional

import draccus
import yaml
from google.cloud import storage


@dataclass(frozen=True)
class CliConfig:
    project: str | None = None
    zone: str | None = None
    tpu: str | None = None
    repository: str | None = None
    image: str | None = None
    tag: str | None = None
    github_user: str | None = None
    github_token: str | None = None
    docker_file: str | None = None
    extra_context: str | None = None
    docker_target: str | None = None
    docker_repository: str | None = None
    subnetwork: str | None = None

    env: dict[str, str] = dataclasses.field(default_factory=dict)

    accel_env: dict[str, dict[str, str]] = dataclasses.field(default_factory=dict)
    """
    Environment variables specific to a type of accelerator. The keys are the accelerator type (e.g. v5litepod-256) or
    generation (e.g. v5litepod), with priority given to the more specific key. The values are dictionaries of environment
    variables to set. These take priority over the general `env` field.
    """

    def env_for_accel(self, accel_type: str) -> dict[str, str]:

        base_env = self.env.copy()

        if "-" in accel_type:
            base_env.update(self.accel_env.get(accel_type.split("-")[0], {}))

        if accel_type in self.accel_env:
            base_env.update(self.accel_env[accel_type])

        return base_env

    @cached_property
    def as_dict(self):
        dict = dataclasses.asdict(self)
        # remove Nones
        return {k: v for k, v in dict.items() if v is not None}


# Oddly enough, there's no API to simply fetch the current gcloud configuration...
def gcloud_config():
    client = storage.Client()
    out: dict[str, str | None] = {
        "project": client.project,
    }
    try:
        out["zone"] = get_default_zone()
    except subprocess.CalledProcessError:
        out["zone"] = None

    return out


def get_default_zone() -> Optional[str]:
    try:
        result = subprocess.run(["gcloud", "config", "get-value", "compute/zone"], stdout=subprocess.PIPE, text=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None


def add_arg(parser: argparse.ArgumentParser, config: CliConfig, flags: list[str], required=False, default=None, **kw):
    """Add an argument to the parser, using `config` or the environment to resolve default values."""
    key = flags[0].lstrip("-").replace("-", "_")
    if key in config.as_dict:
        default = config.as_dict[key]

    if key.upper() in os.environ:
        default = os.environ[key.upper()]

    if default is not None:
        kw["default"] = default
    elif required:
        kw["required"] = True

    parser.add_argument(*flags, **kw)


def load_config() -> CliConfig:
    if os.path.exists(".levanter.yaml"):
        d = yaml.load(open(".levanter.yaml", "r"), Loader=yaml.SafeLoader)
    elif os.path.exists(".config"):
        warnings.warn("Using deprecated .config file. Please rename to .levanter.yaml")
        d = yaml.load(open(".config", "r"), Loader=yaml.SafeLoader)
    else:
        d = {}

    return draccus.decode(CliConfig, d)


def get_git_commit():
    """Get the current git commit hash."""
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()


def default_run_id():
    """Generate a run ID for wandb and continuation.

    Wandb expects a base36 encoded ID of exactly 8 lowercase characters
    or it won't generate a display name."""
    rng_bytes = os.urandom(16)
    run_id = base64.b32encode(rng_bytes)[:8].lower()
    run_id = run_id.decode("utf-8")
    assert len(run_id) == 8
    for char in run_id:
        assert char in "abcdefghijklmnopqrstuvwxyz0123456789"
    return run_id


def add_capacity_type_args(parser, config):
    """
    Add capacity type arguments to the parser. This emulates the behavior of Google's `gcloud` CLI.
    The capacity type will be stored in the `capacity_type` attribute of the parsed arguments.

    Args:
        parser: The argparse parser to add arguments to.
        config: The configuration dictionary to use for defaults.


    """
    add_arg(
        parser,
        config,
        ["--capacity_type"],
        default=None,
        choices=["preemptible", "spot", "reserved", "on-demand", "best-effort"],
    )
    add_arg(
        parser,
        config,
        ["--preemptible"],
        required=False,
        action="store_const",
        const="preemptible",
        dest="capacity_type",
    )
    add_arg(parser, config, ["--spot"], required=False, action="store_const", const="spot", dest="capacity_type")
    add_arg(
        parser, config, ["--reserved"], required=False, action="store_const", const="reserved", dest="capacity_type"
    )
    add_arg(
        parser, config, ["--on-demand"], required=False, action="store_const", const="on-demand", dest="capacity_type"
    )

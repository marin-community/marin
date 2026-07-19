# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The Nsight Systems profiling hook: client-side spec, its install script, and its CLI flags.

Everything the ``nsys`` backend contributes to *submission* lives here:

- ``NsysHook`` — the :class:`~iris.cluster.hooks.TaskHook` that wraps the run command with
  ``iris.cluster.hooks.nsys_main`` and installs the CLI via ``nsys_setup_script``.
- ``nsys_setup_script`` / ``nsys_bin_glob`` — the build-phase install and the glob both the
  script and the run-phase wrapper resolve the binary through.
- ``profile_cli_options`` / ``build_profile_hook`` — the ``--profile*`` flags and the builder
  that turns them into a hook, so the CLI never has to know an ``nsys`` flag by name.

The run-phase half is :mod:`iris.cluster.hooks.nsys_main`; it is imported only in-task
(via ``python -m``) so this module — and the package ``__init__`` that pulls it into the
client — stays free of the profiler's runtime dependencies.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import click

# Module path of the in-task wrapper this hook prepends (``python -m <module> -- <cmd>``).
_NSYS_MAIN_MODULE = "iris.cluster.hooks.nsys_main"

# nsys ``--trace`` default: CUDA kernels + NVTX ranges + cuBLAS. NCCL shows up as CUDA
# kernels plus its own NVTX ranges. CPU sampling and GPU metrics need privileges an
# unprivileged task container lacks, so they are never enabled.
NSYS_DEFAULT_TRACE = "cuda,nvtx,cublas"

# Nsight Systems is not in the task image and has no PyPI distribution, so a profiled
# job fetches it. Pinned: a report is read by a GUI of the same or newer version, and a
# floating version would silently change what a rerun produces.
NSYS_VERSION = "2026.1.3"
_NSYS_BUILD = "2026.1.3.425-1"
# Install root, relative to the task workdir; living there needs no new mount.
NSYS_INSTALL_DIR = ".iris-nsys"
_NSYS_DEB_REPO = "https://developer.download.nvidia.com/compute/cuda/repos/debian12"

# The one profiling backend today; ``--profile`` selects it. A new backend adds its value
# here and a branch in ``build_profile_hook``.
_NSYS_BACKEND = "nsys"


def nsys_bin_glob(install_root: str) -> str:
    """Return the glob matching the ``nsys`` binary under *install_root*.

    The deb lays the target CLI out per-architecture, so the leaf directory differs
    between Grace (``sbsa-armv8``) and x86 nodes; the glob resolves either.

    *install_root* is interpolated verbatim, so the caller decides who expands it:
    the setup script passes a shell expression for bash, while the run-phase wrapper
    passes an already-resolved path (it globs in Python, which expands nothing).
    """
    return f"{install_root}/opt/nvidia/nsight-systems/{NSYS_VERSION}/target-linux-*/nsys"


def nsys_setup_script() -> str:
    """Return a setup script that installs the Nsight Systems CLI into ``NSYS_HOME``.

    Appended to a job's setup only when Nsight profiling is requested. The deb is
    *extracted*, not installed: ``apt install`` would pull the Qt/GUI dependency
    chain, while the target CLI binary is self-contained. The arm64 CLI-only deb is
    too old for Blackwell, so the full package is the only option on Grace.

    Re-running is a no-op once the binary is present, so a retry on a warm workdir
    skips the ~450 MB fetch.
    """
    nsys_home = f'"$IRIS_WORKDIR"/{NSYS_INSTALL_DIR}'
    bin_glob = nsys_bin_glob(f"$IRIS_WORKDIR/{NSYS_INSTALL_DIR}")
    return rf"""set -e
_nsys_bin=$(ls {bin_glob} 2>/dev/null | head -1 || true)
if [ -n "$_nsys_bin" ]; then echo 'nsight-systems already installed'; exit 0; fi
case "$(uname -m)" in
  aarch64) _nsys_arch=sbsa; _nsys_pkg_arch=arm64 ;;
  x86_64)  _nsys_arch=x86_64; _nsys_pkg_arch=amd64 ;;
  *) echo "[iris setup] no nsight-systems build for $(uname -m)" >&2; exit 1 ;;
esac
echo "installing nsight-systems {NSYS_VERSION} ($_nsys_arch)"
mkdir -p {nsys_home}
_nsys_deb={nsys_home}/nsight-systems.deb
curl -fsSL -o "$_nsys_deb" \
  "{_NSYS_DEB_REPO}/$_nsys_arch/nsight-systems-{NSYS_VERSION}_{_NSYS_BUILD}_$_nsys_pkg_arch.deb"
dpkg-deb -x "$_nsys_deb" {nsys_home}
rm -f "$_nsys_deb"
_nsys_bin=$(ls {bin_glob} 2>/dev/null | head -1 || true)
if [ -z "$_nsys_bin" ]; then echo '[iris setup] nsight-systems extract produced no binary' >&2; exit 1; fi
"$_nsys_bin" --version
"""


@dataclass(frozen=True)
class NsysHook:
    """Run the command under ``nsys profile`` on the selected tasks (see ``iris.cluster.hooks.nsys_main``).

    A launch wrapper by necessity — nsys injects CUDA tracing at ``cuInit`` — so it cannot
    profile a job that is already running, unlike the attach-based py-spy/memray profiler.
    Sitting outside the multi-process GPU supervisor, one report covers every rank a
    selected task runs (nsys traces children): the better artifact for intra-node
    collectives, at the cost of no sub-node GPU selection. Selection, report upload, and
    signal forwarding all happen inside the wrapper module; this only builds its argv and
    names the install script.

    Attributes:
        output_uri: Report directory URI. ``None`` lets the task resolve its cluster's temp
            bucket from its own env (``iris.cluster.hooks.nsys_main.default_output_uri``) —
            correct even under ``--target-cluster``, where the launcher's cluster is the
            wrong store.
        tasks: Which tasks write a report, by index: ``first``, ``all``, or a list (``0,7``).
        trace: The nsys ``--trace`` value; CPU sampling and GPU metrics need privileges the
            task container lacks and are never enabled.
        capture_range: Collect only between ``cuProfilerStart``/``cuProfilerStop`` — keeps
            compilation out and aligns multi-task captures on the same step; the app must
            call the API or nothing is collected.
    """

    output_uri: str | None = None
    tasks: str = "first"
    trace: str = NSYS_DEFAULT_TRACE
    capture_range: bool = False

    def setup(self) -> str | None:
        return nsys_setup_script()

    def wrap(self, command: Sequence[str]) -> list[str]:
        argv = ["python", "-m", _NSYS_MAIN_MODULE, "--tasks", self.tasks, "--trace", self.trace]
        # Omitted entirely when unset: the wrapper then defaults it from the task's own
        # cluster env, which is the correct store even under federation.
        if self.output_uri is not None:
            argv += ["--output-uri", self.output_uri]
        if self.capture_range:
            argv.append("--capture-range")
        argv.append("--")
        return [*argv, *command]


def profile_cli_options(command):
    """Attach the ``--profile`` flag group to a click command.

    The nsys hook owns these so the flags that configure it live beside it: the CLI
    applies this decorator and calls :func:`build_profile_hook`, never naming an nsys
    flag itself. A new backend joins ``--profile``'s choices and grows a branch in the
    builder; the CLI stays a pass-through.
    """
    options = [
        click.option(
            "--profile",
            type=click.Choice([_NSYS_BACKEND], case_sensitive=False),
            default=None,
            help=(
                "Profile the run with the named backend, uploading one report per profiled task to "
                "--profile-output. 'nsys' (Nsight Systems) requires --gpu and yields a CUDA/NVTX/NCCL "
                "timeline; CPU sampling and GPU metrics are unavailable in an unprivileged task container."
            ),
        ),
        click.option(
            "--profile-output",
            default=None,
            help=(
                "Directory URI for the reports. Defaults, on the task, to that cluster's "
                "lifecycle-cleaned temp bucket (tmp/ttl=30d/iris-profiles/<job>) — correct even under "
                "--target-cluster. Pass one to override; it must be storage the job's cluster can write, "
                "and must outlive the pod (the task workdir does not)."
            ),
        ),
        click.option(
            "--profile-tasks",
            default=NsysHook.tasks,
            show_default=True,
            help=(
                "Which tasks to profile, by task index: 'first', 'all', or a comma-separated list "
                "(e.g. 0,7). One report per selected task covers every GPU it runs — the minimum "
                "granularity is a whole task/node. Reports are never merged, so prefer a subset."
            ),
        ),
        click.option(
            "--profile-trace",
            default=NsysHook.trace,
            show_default=True,
            help="nsys --trace value (backend-specific). NCCL appears as CUDA kernels plus its own NVTX ranges.",
        ),
        click.option(
            "--profile-capture-range",
            is_flag=True,
            default=False,
            help=(
                "Collect only between cuProfilerStart/Stop instead of the whole run. Keeps compile "
                "out of the report; the app must call the API or nothing is collected."
            ),
        ),
    ]
    for option in reversed(options):
        command = option(command)
    return command


def build_profile_hook(
    profile: str | None,
    *,
    output_uri: str | None,
    tasks: str,
    trace: str,
    capture_range: bool,
) -> NsysHook | None:
    """Build the profiling hook the ``--profile*`` flags select, or ``None`` if unrequested.

    An omitted ``--profile-output`` is fine: the task defaults it to its own cluster's
    temp bucket.
    """
    if profile is None:
        return None
    if profile == _NSYS_BACKEND:
        return NsysHook(output_uri=output_uri, tasks=tasks, trace=trace, capture_range=capture_range)
    raise click.UsageError(f"unknown --profile backend {profile!r}")

"""
Apptainer Bridge Worker: runs on Jusuf/JUWELS CPU nodes, polls the bridge
server for environment lifecycle jobs, manages Apptainer instances locally.

Per-node dispatcher architecture:
  - 1 dispatcher thread polls the bridge server (reduces SSH tunnel traffic)
  - N worker threads consume jobs from a local in-process queue
  - Result posting goes directly from workers to bridge via HTTP

Usage:
    python -m harbor.environments.apptainer.worker \
        --bridge-url http://localhost:9910 \
        --sif-cache /p/scratch/transfernetx/nezhurina1/sif_cache \
        --num-workers 16 \
        --max-exec-output-chars 500000
"""

import argparse
import base64
import fcntl
import json
import os
import queue
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.request
import urllib.error

DEFAULT_MAX_EXEC_OUTPUT_CHARS = 500_000

# Cap concurrent `singularity instance start` calls per node. The suid-starter
# overloads when too many fakeroot+overlay.img creations run in parallel (seen
# at >~16 concurrent starts: "starter-suid exit 255"). Exec/stop ops aren't
# throttled — only the heavy-weight start path.
_INSTANCE_START_SEM = threading.Semaphore(
    int(os.environ.get("BRIDGE_START_CONCURRENCY", "8"))
)


def http_post(url, data, timeout=30):
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"error": str(e)}


def http_get(url, timeout=30):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"error": str(e)}


def detect_apptainer():
    """Detect apptainer or singularity binary."""
    for binary in ("apptainer", "singularity"):
        if shutil.which(binary):
            return binary
    raise FileNotFoundError("Neither apptainer nor singularity found in PATH")


APPTAINER = None


def _patch_test_sh_for_offline_pip(workspace):
    """Inject offline-pip env-var exports at the top of any test.sh under
    `workspace`. Idempotent — won't double-inject if already patched.

    Sets UV_NO_INDEX / UV_FIND_LINKS / PIP_NO_INDEX / PIP_FIND_LINKS so
    subsequent `uv pip install <pkg>` and `pip install <pkg>` commands
    look at a local wheel cache instead of PyPI.

    No-op when the wheel cache dir is missing — preserves existing
    behavior on clusters with internet (e.g. Jupiter).
    """
    agent_tools = os.environ.get("BRIDGE_AGENT_TOOLS", "")
    wheels_dir = os.environ.get("BRIDGE_WHEELS_DIR", "")
    if not wheels_dir and agent_tools:
        wheels_dir = os.path.join(agent_tools, "wheels")
    if not wheels_dir or not os.path.isdir(wheels_dir):
        return

    _MARKER = "# === BRIDGE OFFLINE-PIP PATCH ==="
    patched_count = 0
    for root, _, files in os.walk(workspace):
        for fn in files:
            if fn != "test.sh":
                continue
            p = os.path.join(root, fn)
            try:
                with open(p, "rb") as f:
                    raw = f.read()
                try:
                    orig = raw.decode("utf-8")
                except UnicodeDecodeError:
                    continue
                if _MARKER in orig:
                    continue  # already patched
                shebang_end = 0
                if orig.startswith("#!"):
                    nl = orig.find("\n")
                    shebang_end = (nl + 1) if nl != -1 else 0
                preamble = (
                    f"{_MARKER}\n"
                    "# Force uv / pip to use a local wheel cache because\n"
                    "# the compute node has no PyPI access.\n"
                    f"export PIP_NO_INDEX=1\n"
                    f"export PIP_FIND_LINKS={shlex.quote(wheels_dir)}\n"
                    f"export UV_NO_INDEX=1\n"
                    f"export UV_FIND_LINKS={shlex.quote(wheels_dir)}\n"
                    "# === END BRIDGE OFFLINE-PIP PATCH ===\n"
                )
                with open(p, "w") as f:
                    f.write(orig[:shebang_end] + preamble + orig[shebang_end:])
                patched_count += 1
            except Exception as e:
                print(f"[offline-pip] patch failed for {p}: {e}", flush=True)
    if patched_count:
        print(
            f"[offline-pip] patched {patched_count} test.sh file(s) "
            f"under {workspace} (wheels={wheels_dir})",
            flush=True,
        )


# --- Images whose build steps cannot run rootless -----------------------------
# Some base images (old Debian, e.g. mcr.microsoft.com/dotnet/sdk:6.0) cannot
# run `%post` under apptainer's fakeroot at all: the injected helper cannot
# load against the image's old glibc ("exec /.singularity.d/libs/fakeroot
# failed"), and without fakeroot apt/dpkg refuse. For those, _build_sif
# imports the BASE image from the registry (needs no root) and defers the
# Dockerfile's RUN steps: they are baked once into a persistent overlay next
# to the SIF, inside a root-mapped user namespace (`unshare -r`, no helper
# preload), and every instance mounts that overlay read-only. Sidecar and
# overlay share the local-mode fork's naming, so caches are interchangeable.


def _dockerfile_logical_lines(dockerfile_path):
    """Return Dockerfile instructions with backslash continuations joined."""
    logical_lines = []
    pending = ""
    with open(dockerfile_path) as f:
        for raw in f:
            stripped = raw.strip()
            if pending:
                stripped = pending + " " + stripped
                pending = ""
            if stripped.endswith("\\"):
                pending = stripped[:-1].rstrip()
                continue
            logical_lines.append(stripped)
    if pending:
        logical_lines.append(pending)
    return logical_lines


def _parse_copies(dockerfile_path):
    """COPY/ADD steps of a Dockerfile as (sources, dest) tuples (flags dropped)."""
    copies = []
    for line in _dockerfile_logical_lines(dockerfile_path):
        upper = line.upper()
        if not (upper.startswith("COPY ") or upper.startswith("ADD ")):
            continue
        parts = shlex.split(line)[1:]
        while parts and parts[0].startswith("--"):
            parts = parts[1:]
        if len(parts) >= 2:
            copies.append((parts[:-1], parts[-1]))
    return copies


def _deferred_sidecar(sif):
    real = os.path.realpath(sif)
    return (real[:-4] if real.endswith(".sif") else real) + ".deferred.json"


def _deferred_overlay(sif):
    real = os.path.realpath(sif)
    return (real[:-4] if real.endswith(".sif") else real) + ".overlay.img"


def _fakeroot_attempts(mode):
    normalized = mode.lower()
    if normalized in ("1", "true", "yes"):
        return [True]
    if normalized in ("0", "false", "no"):
        return [False]
    if normalized == "auto":
        return [True, False]
    raise ValueError(f"BRIDGE_USE_FAKEROOT must be auto, 1, or 0 (got {normalized!r})")


def _run_unshared(cmd, timeout=3600):
    r = subprocess.run(
        ["unshare", "-r", *cmd], capture_output=True, text=True, timeout=timeout
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"unshare -r {' '.join(cmd[:3])} ... failed (exit {r.returncode}): "
            f"{r.stderr[-1500:]}{r.stdout[-300:]}"
        )


def _bake_deferred_overlay(sif):
    """Return the overlay holding the deferred RUN steps, baking it on first use."""
    overlay = _deferred_overlay(sif)
    if os.path.exists(overlay):
        return overlay
    with open(_deferred_sidecar(sif)) as f:
        spec = json.load(f)
    run_cmds = spec.get("run") or []
    lock_path = overlay + ".lock"
    with open(lock_path, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)  # several workers may hit the same image
        if os.path.exists(overlay):
            return overlay
        tmp = f"{overlay}.{os.getpid()}.tmp"
        size_mb = os.environ.get("BRIDGE_DEFERRED_OVERLAY_MB", "2048")
        prelude = (
            "set -e\n"
            "mkdir -p /opt/aptcache/archives/partial /opt/aptcache/lists/partial\n"
            "if [ -d /etc/apt ]; then printf '%s\\n' 'APT::Sandbox::User \"root\";' "
            "'Dir::Cache::archives \"/opt/aptcache/archives/\";' "
            "'Dir::State::lists \"/opt/aptcache/lists/\";' "
            "> /etc/apt/apt.conf.d/00-harbor-rootless; fi\n"
            # Docker images ship a post-invoke hook that rm's the original
            # archives dir, which is owned by the unmapped _apt uid: apt then
            # exits 100 after installing. Drop the hook.
            "rm -f /etc/apt/apt.conf.d/docker-clean\n"
        )
        # What _build_sif's %post always adds for the terminus agent — but
        # AFTER the Dockerfile's steps and non-fatal: tmux pulls libutempter0,
        # whose package chowns a file to a non-root uid, which a single-uid
        # namespace cannot do; dpkg then stays broken for every later apt call.
        epilogue = (
            "\n(apt-get install -y -qq tmux || apt-get -f install -y -qq) "
            ">/dev/null 2>&1 || true\n"
        )
        for k, v in (spec.get("env") or {}).items():
            prelude += f"export {k}={shlex.quote(str(v))}\n"
        workdir = spec.get("workdir") or "/"
        script = (
            prelude
            + f"mkdir -p {shlex.quote(workdir)}; cd {shlex.quote(workdir)}\n"
            + "\n".join(run_cmds)
            + "\n"
            + epilogue
        )
        print(
            f"[build] baking {len(run_cmds)} deferred RUN step(s) into "
            f"{os.path.basename(overlay)} ({size_mb} MB sparse overlay, unshare -r)",
            flush=True,
        )
        try:
            _run_unshared(
                [
                    APPTAINER,
                    "overlay",
                    "create",
                    "--sparse",
                    "--size",
                    str(size_mb),
                    tmp,
                ],
                timeout=300,
            )
            _run_unshared(
                [
                    APPTAINER,
                    "exec",
                    "--overlay",
                    tmp,
                    "--cleanenv",
                    "--pwd",
                    "/",
                    os.path.realpath(sif),
                    "bash",
                    "-lc",
                    script,
                ]
            )
        except Exception:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
        os.replace(tmp, overlay)
    return overlay


class ApptainerInstance:
    """Manages a single Apptainer instance lifecycle."""

    def __init__(
        self,
        env_id,
        sif_cache_dir,
        staging_base,
        max_exec_output_chars=DEFAULT_MAX_EXEC_OUTPUT_CHARS,
    ):
        if max_exec_output_chars <= 0:
            raise ValueError("max_exec_output_chars must be positive")
        self.env_id = env_id
        self.sif_cache = sif_cache_dir
        self.staging_base = staging_base
        self.max_exec_output_chars = max_exec_output_chars
        self.instance_name = f"hb_{env_id.replace('-', '_')}"[:50]
        self.staging_dir = None
        self.workdir = "/workspace"
        self.started = False

    def start(self, payload):
        """Build/pull SIF and start instance."""
        global APPTAINER
        if not APPTAINER:
            APPTAINER = detect_apptainer()

        sif_path = payload.get("sif_path", "")
        task_name = payload.get("task_name", "unknown")
        files_b64 = payload.get("files_b64", {})

        # Create staging directory
        self.staging_dir = tempfile.mkdtemp(
            prefix=f"apt_{self.env_id}_", dir=self.staging_base
        )
        workspace = os.path.join(self.staging_dir, "workspace")
        os.makedirs(workspace, exist_ok=True)
        logs_dir = os.path.join(self.staging_dir, "logs")
        os.makedirs(os.path.join(logs_dir, "verifier"), exist_ok=True)
        os.makedirs(os.path.join(logs_dir, "agent"), exist_ok=True)

        # Write any files sent with the start request (task files)
        for rel_path, content_b64 in files_b64.items():
            target = os.path.join(workspace, rel_path)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "wb") as f:
                f.write(base64.b64decode(content_b64))

        # Replicate COPY instructions from Dockerfile in workspace.
        # Docker's COPY puts files at specific paths inside the image.
        # Since the SIF build ignores COPY instructions, we replicate them
        # by copying files within the workspace to match where Docker would
        # place them. Without this, files like seeds/ end up at
        # /workspace/seeds/ instead of being copied to /workspace/ as
        # specified by "COPY seeds/ /workspace/".
        self._apply_dockerfile_copies(workspace)

        # Patch any test.sh under workspace to use an offline wheel cache
        # for `uv pip install` and `pip install` lines. JUWELS Booster
        # compute nodes have no internet — without this, uv (Rust, bypasses
        # LD_PRELOAD proxychains and glibc DNS) fails with
        # "Address family not supported by protocol" on PyPI lookup, the
        # test.sh exits non-zero, and reward.txt is never written.
        # Wheel cache path = $BRIDGE_AGENT_TOOLS/wheels (bind-mounted in via
        # agent_tools bind, so same path inside and outside container).
        # One-time setup (on a node with internet, e.g. JUWELS login):
        #   pip download chardet -d $BRIDGE_AGENT_TOOLS/wheels/
        _patch_test_sh_for_offline_pip(workspace)

        # Resolve SIF image
        resolved_sif = None
        df_hash = payload.get("dockerfile_hash", "")

        # 1. Exact path from payload
        print(
            f"[{self.env_id}] SIF resolution: sif_path={sif_path}, exists={os.path.isfile(sif_path) if sif_path else 'N/A'}, base_image={payload.get('base_image', '')[:50]}"
        )
        if sif_path and os.path.isfile(sif_path):
            resolved_sif = sif_path

        # 2. Cache by task name
        if not resolved_sif:
            cached = os.path.join(self.sif_cache, f"{task_name}.sif")
            if os.path.isfile(cached):
                resolved_sif = cached

        # 3. Cache by task name + dockerfile hash
        if not resolved_sif and df_hash:
            hash_sif = os.path.join(self.sif_cache, f"build_{task_name}-{df_hash}.sif")
            if os.path.isfile(hash_sif):
                resolved_sif = hash_sif

        # 4. Any SIF with matching dockerfile hash (for shared Dockerfiles)
        if not resolved_sif and df_hash:
            import glob

            matches = glob.glob(os.path.join(self.sif_cache, f"*-{df_hash}.sif"))
            if matches:
                # Use the first match (they're all the same image)
                target = matches[0]
                # Follow symlinks to get real file
                resolved_sif = os.path.realpath(target)

        # 5. Base image SIF fallback (for SweSmith-style tasks with shared base images)
        #    If no task-specific SIF found, check for a prebuilt base image SIF.
        #    The remaining RUN commands (e.g. git checkout) will be executed after start.
        if not resolved_sif:
            base_image = payload.get("base_image", "")
            if base_image:
                import hashlib

                img_hash = hashlib.sha256(base_image.encode()).hexdigest()[:12]
                base_sif = os.path.join(self.sif_cache, f"swesmith_base_{img_hash}.sif")
                if os.path.isfile(base_sif):
                    resolved_sif = base_sif
                    # Mark that we need to run Dockerfile RUN commands after start
                    self._pending_run_commands = self._extract_run_commands(workspace)
                    self._using_base_sif = True
                    print(
                        f"[{self.env_id}] Using base image SIF: {base_sif} "
                        f"(+{len(self._pending_run_commands)} RUN commands)"
                    )

        # 6. Build from Dockerfile (use transmitted files from workspace, not remote path)
        if not resolved_sif and df_hash:
            # Check workspace for Dockerfile (sent via files_b64)
            dockerfile = os.path.join(workspace, "Dockerfile")
            if not os.path.isfile(dockerfile):
                dockerfile = os.path.join(workspace, "environment", "Dockerfile")
            if os.path.isfile(dockerfile):
                hash_sif = os.path.join(
                    self.sif_cache, f"build_{task_name}-{df_hash}.sif"
                )
                resolved_sif = self._build_sif(dockerfile, hash_sif)

        # 6. Fallback: base.sif
        if not resolved_sif:
            base_sif = os.path.join(self.sif_cache, "base.sif")
            if os.path.isfile(base_sif):
                resolved_sif = base_sif

        if not resolved_sif:
            raise FileNotFoundError(
                f"No SIF found for task {task_name}. "
                f"Checked cache dir: {self.sif_cache}, hash: {df_hash}"
            )

        # Start instance
        cmd = [APPTAINER, "instance", "start"]

        # Bind mounts
        cmd.extend(["--bind", f"{workspace}:/workspace:rw"])
        cmd.extend(
            ["--bind", f"{os.path.join(logs_dir, 'verifier')}:/logs/verifier:rw"]
        )
        cmd.extend(["--bind", f"{os.path.join(logs_dir, 'agent')}:/logs/agent:rw"])

        # /tests bind mount for verifier uploads
        tests_dir = os.path.join(self.staging_dir, "tests")
        os.makedirs(tests_dir, exist_ok=True)
        cmd.extend(["--bind", f"{tests_dir}:/tests:rw"])

        # Registry-imported image with deferred RUN steps (see module top):
        # its overlay is mounted read-only under the writable layer, and its
        # COPY payload is staged from the workspace (the environment dir the
        # driver sent) — the overlay was baked as namespace-root and cannot
        # hold files the (possibly non-root) task must own.
        self._deferred_overlay = None
        self._deferred_spec = None
        if resolved_sif and os.path.exists(_deferred_sidecar(resolved_sif)):
            with open(_deferred_sidecar(resolved_sif)) as f:
                self._deferred_spec = json.load(f)
            self._deferred_overlay = _bake_deferred_overlay(resolved_sif)
            if not getattr(self, "_dockerfile_workdir", None):
                self._dockerfile_workdir = self._deferred_spec.get("workdir")

        # Bind a writable per-instance dir over the Dockerfile WORKDIR.
        # In apptainer setuid mode (MN5: no /etc/subuid mapping → no fakeroot),
        # the SIF's WORKDIR is owned by root:root and the calling user cannot
        # write to it (chmod also rejected, even with overlay). The verifier's
        # test.sh fails on `python3 -m venv /app/.venv` with EACCES.
        # Fix: pre-copy the SIF's workdir contents to a host dir, then bind
        # that host dir over the workdir so the container sees the same files
        # but writable as the calling user.
        # NOTE: /testbed intentionally NOT in this list. On clusters without
        # subuid mappings (MN5), the SIF's /testbed contents are root-owned
        # and not writable by the calling user. The writable workdir bind
        # below is the only way to make /testbed editable for SWE-bench tasks.
        # On Jupiter fakeroot makes this unnecessary but the bind is still safe.
        _SYSTEM_PATHS = {
            "/",
            "/tmp",
            "/usr",
            "/var",
            "/etc",
            "/bin",
            "/sbin",
            "/lib",
            "/lib64",
            "/proc",
            "/sys",
            "/dev",
            "/root",
            "/home",
            "/workspace",
            "/logs",
            "/tests",
        }
        dockerfile_workdir_for_bind = getattr(self, "_dockerfile_workdir", None)
        if (
            dockerfile_workdir_for_bind
            and dockerfile_workdir_for_bind not in _SYSTEM_PATHS
        ):
            workdir_host = os.path.join(self.staging_dir, "workdir")
            os.makedirs(workdir_host, exist_ok=True)
            os.chmod(workdir_host, 0o777)
            # Deferred image: replay the Dockerfile's COPY steps into the
            # workdir (destinations under it) or as writable binds (others).
            for copy_cmd in (self._deferred_spec or {}).get("copy") or []:
                parts = shlex.split(copy_cmd)
                if len(parts) < 2:
                    continue
                dest = parts[-1]
                wd = dockerfile_workdir_for_bind.rstrip("/")
                for src in parts[:-1]:
                    src_abs = os.path.normpath(os.path.join(workspace, src))
                    if not os.path.exists(src_abs):
                        print(
                            f"[{self.env_id}] COPY source not found: {src}", flush=True
                        )
                        continue
                    if dest == wd or dest.startswith(wd + "/"):
                        rel = dest[len(wd) :].lstrip("/")
                        target = (
                            os.path.join(workdir_host, rel) if rel else workdir_host
                        )
                        if os.path.isdir(src_abs):
                            shutil.copytree(src_abs, target, dirs_exist_ok=True)
                        else:
                            os.makedirs(
                                os.path.dirname(target) or target, exist_ok=True
                            )
                            shutil.copy2(src_abs, target)
                    else:
                        stage = os.path.join(
                            self.staging_dir,
                            "deferred_copy",
                            dest.strip("/").replace("/", "_"),
                        )
                        if os.path.isdir(src_abs):
                            shutil.copytree(src_abs, stage, dirs_exist_ok=True)
                        else:
                            os.makedirs(os.path.dirname(stage), exist_ok=True)
                            shutil.copy2(src_abs, stage)
                        cmd.extend(["--bind", f"{stage}:{dest}:rw"])
            # Pre-populate from SIF's workdir if non-empty. Bind workdir_host
            # at a separate container path and cp inside; the real bind below
            # will then expose those files at the workdir, writable.
            init_cmd = [
                APPTAINER,
                "exec",
                "--bind",
                f"{workdir_host}:/_workdir_init:rw",
                resolved_sif,
                "sh",
                "-c",
                f"if [ -d {dockerfile_workdir_for_bind} ] && "
                f'[ -n "$(ls -A {dockerfile_workdir_for_bind} 2>/dev/null)" ]; then '
                f"cp -a {dockerfile_workdir_for_bind}/. /_workdir_init/ 2>/dev/null || true; "
                f"fi",
            ]
            try:
                subprocess.run(init_cmd, capture_output=True, text=True, timeout=120)
            except Exception as e:
                print(f"[{self.env_id}] Workdir pre-copy failed: {e}", flush=True)
            cmd.extend(["--bind", f"{workdir_host}:{dockerfile_workdir_for_bind}:rw"])

        # Per-instance /tmp to prevent concurrent containers on the same node
        # from racing on shared host /tmp (apptainer binds host /tmp by default).
        tmp_dir = os.path.join(self.staging_dir, "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        cmd.extend(["--bind", f"{tmp_dir}:/tmp:rw"])

        # Note: --memory/--cpus/--net require cgroups which needs root on HPC
        # Resource isolation is handled by SLURM instead

        # Use an overlay image file (ext3) instead of a directory.
        # Singularity 4.1.5 in setuid mode forbids non-root users from using
        # a directory overlay ("only root user can use sandbox as overlay in
        # setuid mode"). An overlay image file is allowed and gives the same
        # writable layer without size limits of --writable-tmpfs.
        # The overlay is chosen per start attempt (see the fakeroot loop
        # below): with --fakeroot a DIRECTORY overlay on node-local disk is the
        # one that works (kernel overlayfs inside the user namespace; an ext3
        # image mounted there is owned by the real uid and namespace-root
        # cannot write its upper dir: "overlay upper dir ... is not writable").
        # Without fakeroot the ext3 image is kept, as before.
        overlay_img = os.path.join(self.staging_dir, "overlay.img")
        overlay_dir = os.path.join(self.staging_dir, "overlay")

        def _overlay_args(use_fakeroot):
            if use_fakeroot:
                os.makedirs(overlay_dir, exist_ok=True)
                return ["--overlay", overlay_dir]
            if not os.path.isfile(overlay_img):
                mk = subprocess.run(
                    [APPTAINER, "overlay", "create", "--size", "4096", overlay_img],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                if mk.returncode != 0:
                    # Fallback: directory overlay (works for fakeroot/userns)
                    os.makedirs(overlay_dir, exist_ok=True)
                    return ["--overlay", overlay_dir]
            return ["--overlay", overlay_img]

        cmd.extend(["--cleanenv"])
        # Optional host-side agent_tools dir bind-mounted into container at the
        # SAME path so venv shebangs resolve. Provides uv, pytest, swebench,
        # datasets, swesmith without needing fakeroot to install in the SIF.
        agent_tools = os.environ.get("BRIDGE_AGENT_TOOLS", "")
        agent_tools_path_prefix = ""
        if agent_tools and os.path.isdir(agent_tools):
            cmd.extend(["--bind", f"{agent_tools}:{agent_tools}:ro"])
            agent_tools_path_prefix = (
                f"{agent_tools}/bin:{agent_tools}/uv_env/.venv/bin:"
            )
            # Bundled libs (tmux + deps) — prepend so libevent/libtinfo from agent_tools wins
            cmd.extend(["--env", f"LD_LIBRARY_PATH={agent_tools}/lib"])
        # Ensure PATH includes standard dirs (--cleanenv can strip it)
        cmd.extend(
            [
                "--env",
                f"PATH={agent_tools_path_prefix}/root/.local/bin:/testbed/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            ]
        )
        # Override apptainer's "Apptainer> " PS1 with a plain shell prompt so
        # the agent's tmux session shows a normal prompt and doesn't get
        # confused by container-specific markers. Apptainer sets this via
        # /.singularity.d/env/91-custom-env.sh; this --env override takes
        # precedence for all shells inside the instance.
        cmd.extend(["--env", "PS1=\\$ "])
        cmd.extend(["--env", "PROMPT_COMMAND="])
        # Prevent host home mount — it causes quota issues and overwrites
        # container's /root/.bashrc with the host's (which has broken conda).
        cmd.extend(["--no-home"])

        if getattr(self, "_using_base_sif", False):
            # SweSmith: redirect HOME to /tmp/fakehome to avoid writes to /root/
            cmd.extend(["--home", "/tmp/fakehome"])
        else:
            # R2EGym/NL2Bash: keep HOME=/root so test scripts find /root/.bashrc
            cmd.extend(["--home", "/root"])

        # A deferred build imports the base SIF directly, so its Dockerfile ENV
        # values are not present in the image's runtime environment.
        for key, value in (self._deferred_spec or {}).get("env", {}).items():
            cmd.extend(["--env", f"{key}={value}"])

        # Bind-mount proxychains lib and config for internet access inside container.
        # Only for base-image SIF tasks (SweSmith) that need internet for test.sh.
        # R2EGym/NL2Bash don't need internet and proxychains LD_PRELOAD can
        # interfere with tools like tmux.
        proxy_host = os.environ.get("BRIDGE_PROXY_HOST", "")
        proxy_port = os.environ.get("BRIDGE_PROXY_PORT", "")
        proxychains_lib = os.path.expanduser("~/.local/lib/libproxychains4.so")
        if proxy_host and proxy_port and os.path.isfile(proxychains_lib):
            pc_conf = os.path.join(self.staging_dir, "proxychains.conf")
            with open(pc_conf, "w") as f:
                f.write(
                    f"strict_chain\nquiet_mode\n"
                    f"tcp_read_time_out 30000\ntcp_connect_time_out 15000\n"
                    f"localnet 127.0.0.0/255.0.0.0\nlocalnet 10.0.0.0/255.0.0.0\n"
                    f"localnet 172.16.0.0/255.240.0.0\nlocalnet 192.168.0.0/255.255.0.0\n"
                    f"[ProxyList]\nsocks5 {proxy_host} {proxy_port}\n"
                )
            cmd.extend(
                ["--bind", f"{proxychains_lib}:/usr/local/lib/libproxychains4.so:ro"]
            )
            cmd.extend(["--bind", f"{pc_conf}:/etc/proxychains.conf:ro"])

        ro_overlay = (
            ["--overlay", f"{self._deferred_overlay}:ro"]
            if self._deferred_overlay
            else []
        )

        cmd.extend([resolved_sif, self.instance_name])

        # Throttle concurrent singularity instance start calls (see module top).
        # On clusters without subuid mappings (MN5), apptainer uses SUID starter
        # which has shared kernel state and breaks at high concurrency.
        #
        # Root inside the instance: Docker runs tasks as root, so verifiers and
        # solutions call apt-get/dpkg/pip freely. Rootless apptainer runs them
        # as the calling user and `dpkg` refuses ("requested operation requires
        # superuser privilege") — every such task scores 0 even though the
        # command "ran". `--fakeroot` on instance start restores root-like
        # behaviour for everything exec'd into the instance (execs join its
        # namespace, so they need no flag). BRIDGE_USE_FAKEROOT: "1" always,
        # "0" never, "auto" (default) try with and fall back without, for
        # images whose old libraries cannot load apptainer's fakeroot helper.
        fakeroot_mode = os.environ.get("BRIDGE_USE_FAKEROOT", "auto")
        attempts = _fakeroot_attempts(fakeroot_mode)
        with _INSTANCE_START_SEM:
            for i, use_fakeroot in enumerate(attempts):
                start_cmd = list(cmd)
                if not use_fakeroot and "--home" in start_cmd:
                    # Not root inside: /root in the image is unwritable for us,
                    # but tasks/verifiers write there ($HOME, uv, pip, ...).
                    # Bind a per-instance host dir at /root, seeded with the
                    # image's /root contents, so HOME stays /root and writable.
                    hi = start_cmd.index("--home")
                    if start_cmd[hi + 1] == "/root":
                        home_host = os.path.join(self.staging_dir, "home")
                        if not os.path.isdir(home_host):
                            os.makedirs(home_host, exist_ok=True)
                            subprocess.run(
                                [
                                    APPTAINER,
                                    "exec",
                                    "--bind",
                                    f"{home_host}:/_home_init:rw",
                                    resolved_sif,
                                    "sh",
                                    "-c",
                                    "cp -a /root/. /_home_init/ 2>/dev/null || true",
                                ],
                                capture_output=True,
                                text=True,
                                timeout=120,
                            )
                        start_cmd[hi + 1] = f"{home_host}:/root"
                # overlay goes right before the image + instance name
                start_cmd[-2:-2] = ro_overlay + _overlay_args(use_fakeroot)
                if use_fakeroot:
                    start_cmd.insert(3, "--fakeroot")
                result = subprocess.run(
                    start_cmd, capture_output=True, text=True, timeout=300
                )
                if result.returncode == 0:
                    self._fakeroot = use_fakeroot
                    break
                if i + 1 < len(attempts):
                    print(
                        f"[{self.env_id}] WARNING: instance start with --fakeroot "
                        f"failed, retrying WITHOUT root (tasks needing root at task "
                        f"time will fail): {result.stderr.strip()[-300:]}",
                        flush=True,
                    )
        if result.returncode != 0:
            raise RuntimeError(f"Instance start failed: {result.stderr}")

        self.started = True
        # Record SIF + pool key + uses count so the instance can be reused later.
        self._resolved_sif = resolved_sif
        self._pool_key = _pool_key_from_payload(payload)
        self._uses_count = 1

        # Ensure tmux is accessible: create symlinks in /usr/local/bin/ if needed.
        # --cleanenv strips PATH and apptainer exec may not find tmux even with PATH prefix.
        # This is a reliable workaround: /usr/local/bin/ is always in the default PATH.
        # Also force IPv4 preference via gai.conf and pre-install chardet —
        # proxychains doesn't support IPv6, and uv (Rust) bypasses glibc DNS,
        # so R2EGym's test.sh "uv pip install chardet" fails. Pre-installing
        # with pip (which uses glibc, respects gai.conf) avoids the issue.
        proxy_prefix = self._proxy_command_prefix()
        # Ensure Dockerfile WORKDIR exists (Docker auto-creates it, apptainer doesn't)
        dockerfile_workdir = getattr(self, "_dockerfile_workdir", None)
        ensure_workdir = (
            f"mkdir -p {dockerfile_workdir}; " if dockerfile_workdir else ""
        )
        agent_tools = os.environ.get("BRIDGE_AGENT_TOOLS", "")
        have_agent_tools = bool(agent_tools and os.path.isdir(agent_tools))
        ensure_script = (
            ensure_workdir
            + "echo 'precedence ::ffff:0:0/96 100' > /etc/gai.conf 2>/dev/null; "
        )
        if getattr(self, "_fakeroot", False):
            # Single-uid user namespace: apt's privilege drop to _apt fails and
            # its _apt-owned mode-700 partial/ dirs are unwritable even for
            # namespace-root. Run apt as root with fresh download dirs.
            ensure_script += (
                "if command -v apt-get >/dev/null 2>&1; then "
                "mkdir -p /apt-rootless/archives/partial /apt-rootless/lists/partial; "
                'printf \'APT::Sandbox::User "root";\nDir::Cache::Archives '
                '"/apt-rootless/archives/";\nDir::State::Lists '
                '"/apt-rootless/lists/";\n\' > /etc/apt/apt.conf.d/90-rootless 2>/dev/null; '
                "fi; "
            )
        if have_agent_tools:
            # agent_tools provides tmux + uv + pytest via bind mount,
            # no internet-dependent install needed. Just symlink tmux so
            # callers on vanilla $PATH find it.
            ensure_script += (
                f"ln -sf {agent_tools}/bin/tmux /usr/local/bin/tmux 2>/dev/null; "
                f"ln -sf {agent_tools}/bin/uv /usr/local/bin/uv 2>/dev/null; "
            )
        else:
            # Legacy path: try apt/pip installs (requires internet or proxy)
            ensure_script += (
                "if ! command -v tmux >/dev/null 2>&1; then "
                "  apt-get update -qq && apt-get install -y -qq tmux 2>/dev/null || true; "
                "fi; "
                "test -x /usr/bin/tmux && ln -sf /usr/bin/tmux /usr/local/bin/tmux 2>/dev/null; "
            )
            if proxy_prefix:
                ensure_script += (
                    "pip install chardet 2>/dev/null || "
                    "python3 -m pip install chardet 2>/dev/null || true; "
                    "if ! command -v uv >/dev/null 2>&1; then "
                    "  pip install uv 2>/dev/null || "
                    "  python3 -m pip install uv 2>/dev/null || true; "
                    "fi; "
                )
        ensure_script += "true"
        ensure_cmd = [
            APPTAINER,
            "exec",
            f"instance://{self.instance_name}",
            "/bin/bash",
            "-c",
            (proxy_prefix if proxy_prefix else "") + ensure_script,
        ]
        subprocess.run(ensure_cmd, capture_output=True, text=True, timeout=120)

        # Execute pending RUN commands from Dockerfile (for base-image SIF fallback)
        pending_cmds = getattr(self, "_pending_run_commands", [])
        if pending_cmds:
            print(
                f"[{self.env_id}] Executing {len(pending_cmds)} Dockerfile RUN commands..."
            )
            for i, cmd_str in enumerate(pending_cmds):
                exec_cmd = [
                    APPTAINER,
                    "exec",
                    "--pwd",
                    "/testbed",
                    "instance://" + self.instance_name,
                    "bash",
                    "-c",
                    cmd_str,
                ]
                try:
                    r = subprocess.run(
                        exec_cmd, capture_output=True, text=True, timeout=300
                    )
                    if r.returncode != 0:
                        print(
                            f"[{self.env_id}] RUN cmd {i} failed (rc={r.returncode}): {cmd_str[:80]}"
                        )
                except subprocess.TimeoutExpired:
                    print(f"[{self.env_id}] RUN cmd {i} timed out: {cmd_str[:80]}")
            self._pending_run_commands = []

        return {
            "sif": resolved_sif,
            "instance": self.instance_name,
            "fakeroot": bool(getattr(self, "_fakeroot", False)),
            "deferred_overlay": bool(self._deferred_overlay),
        }

    def _extract_run_commands(self, workspace):
        """Extract RUN commands from Dockerfile that aren't baked into the base SIF.

        Skips commands that are already in the base SIF (apt-get, uv install, mkdir).
        Returns only task-specific commands like git checkout.
        """
        dockerfile = os.path.join(workspace, "Dockerfile")
        if not os.path.isfile(dockerfile):
            dockerfile = os.path.join(workspace, "environment", "Dockerfile")
        if not os.path.isfile(dockerfile):
            return []

        # Commands already in the base SIF — skip these
        base_cmds = {"apt-get", "curl -LsSf", "mkdir -p /output", "mkdir -p /logs"}

        run_cmds = []
        for line in _dockerfile_logical_lines(dockerfile):
            if not line or line.startswith("#"):
                continue
            if line.upper().startswith("RUN "):
                cmd = line[4:].strip()
                # Skip commands already in the base SIF
                if any(cmd.startswith(bc) for bc in base_cmds):
                    continue
                run_cmds.append(cmd)
        return run_cmds

    def _build_sif(self, dockerfile_path, output_sif):
        """Build SIF from Dockerfile using apptainer build."""
        # Convert Dockerfile to .def format (simplified)
        base_image = "ubuntu:22.04"
        run_cmds = []
        env_vars = {}
        workdir = "/workspace"

        for line in _dockerfile_logical_lines(dockerfile_path):
            if not line or line.startswith("#"):
                continue
            upper = line.upper()
            if upper.startswith("FROM "):
                base_image = line[5:].strip().split(" AS ")[0].strip()
                if base_image.startswith("--platform"):
                    parts = base_image.split(None, 2)
                    base_image = parts[2] if len(parts) > 2 else parts[-1]
            elif upper.startswith("RUN "):
                run_cmds.append(line[4:].strip())
            elif upper.startswith("ENV "):
                env_part = line[4:].strip()
                if "=" in env_part:
                    for part in shlex.split(env_part):
                        if "=" in part:
                            k, _, v = part.partition("=")
                            env_vars[k] = v
            elif upper.startswith("WORKDIR "):
                workdir = line[8:].strip()

        def_content = f"Bootstrap: docker\nFrom: {base_image}\n\n"
        # Docker's COPY steps: put the build-context files into the image
        # (%files). Without this every COPY is silently dropped and tasks
        # whose files live outside /workspace (e.g. "COPY task_file
        # /app/task_file") start without their inputs.
        context = os.path.dirname(os.path.abspath(dockerfile_path))
        copies = _parse_copies(dockerfile_path)
        files_lines = []
        for sources, dest in copies:
            for src in sources:
                src_abs = os.path.normpath(os.path.join(context, src))
                if not os.path.exists(src_abs):
                    raise FileNotFoundError(
                        f"Dockerfile COPY source does not exist: {src}"
                    )
                files_lines.append(f"    {src_abs} {dest}")
        if files_lines:
            def_content += "%files\n" + "\n".join(files_lines) + "\n\n"
        if env_vars:
            def_content += "%environment\n"
            for k, v in env_vars.items():
                def_content += f"    export {k}={shlex.quote(v)}\n"
            def_content += "\n"
        def_content += "%post\n"
        def_content += f"    mkdir -p {workdir}\n"
        # Always install tmux (needed by Terminus-2 and Terminus-KIRA).
        def_content += "    apt-get update -qq && apt-get install -y -qq tmux > /dev/null 2>&1 || true\n"
        for k, v in env_vars.items():
            def_content += f"    export {k}={shlex.quote(v)}\n"
        if run_cmds:
            for cmd in run_cmds:
                def_content += f"    {cmd}\n"
        def_content += "\n"
        def_content += "%runscript\n"
        def_content += f'    cd {workdir}\n    exec "$@"\n'

        with tempfile.NamedTemporaryFile(suffix=".def", delete=False, mode="w") as f:
            f.write(def_content)
            def_path = f.name

        try:
            os.makedirs(os.path.dirname(output_sif), exist_ok=True)
            tmp_sif = output_sif + ".tmp"
            # Try with fakeroot first
            for fakeroot in [True, False]:
                cmd = [APPTAINER, "build"]
                if fakeroot:
                    cmd.append("--fakeroot")
                cmd.extend([tmp_sif, def_path])
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=1800
                )
                if result.returncode == 0:
                    os.rename(tmp_sif, output_sif)
                    return output_sif
                if os.path.exists(tmp_sif):
                    os.unlink(tmp_sif)
            build_err = result.stderr.strip()[-300:]
            print(
                f"[build] Dockerfile build failed for {os.path.basename(output_sif)} "
                f"({build_err}); importing docker://{base_image} and deferring "
                f"{len(run_cmds)} RUN step(s) to a baked overlay",
                flush=True,
            )
            imp = subprocess.run(
                [APPTAINER, "build", tmp_sif, f"docker://{base_image}"],
                capture_output=True,
                text=True,
                timeout=1800,
            )
            if imp.returncode != 0:
                if os.path.exists(tmp_sif):
                    os.unlink(tmp_sif)
                raise RuntimeError(
                    f"SIF build failed: {build_err}; base image import "
                    f"also failed: {imp.stderr[-300:]}"
                )
            os.rename(tmp_sif, output_sif)
            with open(_deferred_sidecar(output_sif), "w") as f:
                json.dump(
                    {
                        "run": run_cmds,
                        "env": env_vars,
                        "workdir": workdir,
                        "base_image": base_image,
                        "copy": [" ".join(src + [dest]) for src, dest in copies],
                    },
                    f,
                    indent=2,
                )
            _bake_deferred_overlay(output_sif)
            return output_sif
        finally:
            os.unlink(def_path)

    def _apply_dockerfile_copies(self, workspace):
        """Parse Dockerfile for COPY instructions and replicate them in workspace.

        Docker's COPY copies files from the build context into the image at
        specific paths. Since the Apptainer SIF build ignores COPY, we
        replicate the file layout here so the workspace matches what Docker
        would produce.

        Example: "COPY seeds/ /workspace/" copies the CONTENTS of seeds/
        into /workspace/, so seeds/dir1 → /workspace/dir1.
        """
        # Find the Dockerfile in workspace
        dockerfile = os.path.join(workspace, "Dockerfile")
        if not os.path.isfile(dockerfile):
            dockerfile = os.path.join(workspace, "environment", "Dockerfile")
        if not os.path.isfile(dockerfile):
            return

        lines = _dockerfile_logical_lines(dockerfile)

        # Extract WORKDIR for default cwd in exec
        for line in lines:
            stripped = line.strip()
            if stripped.upper().startswith("WORKDIR "):
                self._dockerfile_workdir = stripped[8:].strip()

        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            upper = stripped.upper()
            if not upper.startswith("COPY ") and not upper.startswith("ADD "):
                continue

            # Parse COPY/ADD instruction: COPY [--flags] <src...> <dest>
            parts = shlex.split(stripped)
            parts = parts[1:]  # remove COPY/ADD keyword

            # Skip flags like --from=builder, --chown=..., --chmod=...
            while parts and parts[0].startswith("--"):
                parts = parts[1:]

            if len(parts) < 2:
                continue

            dest = parts[-1]
            sources = parts[:-1]

            # Only process COPY instructions that target paths inside /workspace
            # (since /workspace is our bind mount)
            if not dest.startswith("/workspace"):
                continue

            # Map dest to workspace-relative path
            dest_rel = dest[len("/workspace") :].lstrip("/")
            dest_abs = os.path.join(workspace, dest_rel) if dest_rel else workspace

            for src in sources:
                src_path = os.path.join(workspace, src.rstrip("/"))
                if not os.path.exists(src_path):
                    continue

                if os.path.isdir(src_path):
                    # COPY dir/ /workspace/ → copy CONTENTS of dir into dest
                    # (Docker behavior: trailing / on src copies contents)
                    if src.endswith("/"):
                        # Copy contents of directory
                        for item in os.listdir(src_path):
                            s = os.path.join(src_path, item)
                            d = os.path.join(dest_abs, item)
                            if os.path.isdir(s):
                                if os.path.exists(d):
                                    # Merge into existing dir
                                    for sub_root, sub_dirs, sub_files in os.walk(s):
                                        sub_rel = os.path.relpath(sub_root, s)
                                        sub_dest = os.path.join(d, sub_rel)
                                        os.makedirs(sub_dest, exist_ok=True)
                                        for sf in sub_files:
                                            shutil.copy2(
                                                os.path.join(sub_root, sf),
                                                os.path.join(sub_dest, sf),
                                            )
                                else:
                                    shutil.copytree(s, d)
                            else:
                                shutil.copy2(s, d)
                    else:
                        # Docker COPY behavior: if dest is an existing directory,
                        # copy CONTENTS of src into dest (not the dir itself).
                        # This handles "COPY workspace /workspace" correctly.
                        if os.path.isdir(dest_abs) and os.path.realpath(
                            src_path
                        ) != os.path.realpath(dest_abs):
                            for item in os.listdir(src_path):
                                s = os.path.join(src_path, item)
                                d = os.path.join(dest_abs, item)
                                if os.path.isdir(s):
                                    if os.path.exists(d):
                                        for sub_root, sub_dirs, sub_files in os.walk(s):
                                            sub_rel = os.path.relpath(sub_root, s)
                                            sub_dest = os.path.join(d, sub_rel)
                                            os.makedirs(sub_dest, exist_ok=True)
                                            for sf in sub_files:
                                                shutil.copy2(
                                                    os.path.join(sub_root, sf),
                                                    os.path.join(sub_dest, sf),
                                                )
                                    else:
                                        shutil.copytree(s, d)
                                else:
                                    shutil.copy2(s, d)
                        else:
                            d = os.path.join(dest_abs, os.path.basename(src_path))
                            if not os.path.exists(d):
                                shutil.copytree(src_path, d)
                else:
                    # Single file
                    os.makedirs(dest_abs, exist_ok=True)
                    shutil.copy2(
                        src_path, os.path.join(dest_abs, os.path.basename(src_path))
                    )

    def _proxychains_prefix(self):
        """Return proxychains command prefix if proxy is configured."""
        conf = os.environ.get("PROXYCHAINS_CONF_FILE", "")
        pc_bin = os.path.expanduser("~/.local/bin/proxychains4")
        if conf and os.path.isfile(conf) and os.path.isfile(pc_bin):
            return [pc_bin, "-q", "-f", conf]
        return []

    def _proxy_command_prefix(self):
        """Return shell prefix to enable proxychains inside container.

        Apptainer strips LD_PRELOAD for security, so we set it inside the
        bash -c command string instead.
        """
        proxy_host = os.environ.get("BRIDGE_PROXY_HOST", "")
        proxy_port = os.environ.get("BRIDGE_PROXY_PORT", "")
        agent_tools = os.environ.get("BRIDGE_AGENT_TOOLS", "")
        at_paths = (
            f"{agent_tools}/bin:{agent_tools}/uv_env/.venv/bin:"
            if agent_tools and os.path.isdir(agent_tools)
            else ""
        )
        at_libs = (
            f"export LD_LIBRARY_PATH={agent_tools}/lib${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}; "
            if agent_tools and os.path.isdir(agent_tools)
            else ""
        )
        if proxy_host and proxy_port:
            return (
                at_libs + "export LD_PRELOAD=/usr/local/lib/libproxychains4.so "
                "PROXYCHAINS_CONF_FILE=/etc/proxychains.conf "
                "PROXYCHAINS_QUIET_MODE=1 "
                f"PATH={at_paths}/root/.local/bin:/testbed/.venv/bin:$PATH 2>/dev/null; "
            )
        if at_paths:
            return at_libs + f"export PATH={at_paths}$PATH; "
        return ""

    def _base_sif_command_prefix(self):
        """Return shell prefix for base-image SIF tasks (SweSmith).

        Sets PATH to include tools installed in the SIF at /root/.local/bin
        since --no-home --home /tmp/fakehome changes $HOME away from /root.
        Also prepends host-side agent_tools (uv, pytest, swebench, ...) when bound.
        """
        agent_tools = os.environ.get("BRIDGE_AGENT_TOOLS", "")
        at_paths = (
            f"{agent_tools}/bin:{agent_tools}/uv_env/.venv/bin:"
            if agent_tools and os.path.isdir(agent_tools)
            else ""
        )
        at_libs = (
            f"export LD_LIBRARY_PATH={agent_tools}/lib${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}; "
            if agent_tools and os.path.isdir(agent_tools)
            else ""
        )
        if getattr(self, "_using_base_sif", False):
            return (
                at_libs
                + f"export PATH={at_paths}/root/.local/bin:/testbed/.venv/bin:$PATH; "
            )
        if at_paths:
            return at_libs + f"export PATH={at_paths}$PATH; "
        return ""

    def exec(self, payload):
        """Execute a command in the running instance."""
        command = payload.get("command", "echo ok")
        # Patch: replace "uv pip install" with "pip install" in commands.
        # uv (Rust) bypasses proxychains LD_PRELOAD (uses its own DNS/network
        # stack), causing IPv6 "Address family not supported" errors on HPC
        # nodes without direct internet. pip uses glibc and works with proxychains.
        if "uv pip install" in command:
            command = command.replace("uv pip install", "pip install")
        default_cwd = getattr(self, "_dockerfile_workdir", None) or (
            "/testbed" if getattr(self, "_using_base_sif", False) else "/workspace"
        )
        cwd = payload.get("cwd") or default_cwd
        env_vars = payload.get("env_vars", {})
        timeout_sec = payload.get("timeout_sec", 600)

        # Enable proxychains for all commands so tmux sessions inherit LD_PRELOAD
        # and commands typed by the model (pip install, etc.) can access the internet.
        # The [proxychains] output noise is filtered from results below.
        proxy_prefix = self._proxy_command_prefix()
        if proxy_prefix:
            command = proxy_prefix + command
        else:
            # No proxy available, just fix PATH
            command = (
                "export PATH=/root/.local/bin:/testbed/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; "
                + command
            )

        cmd = [APPTAINER, "exec"]
        # Always set PATH on exec (--cleanenv strips it from the instance)
        cmd.extend(
            [
                "--env",
                "PATH=/root/.local/bin:/testbed/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            ]
        )
        # Plain shell prompt (override apptainer default "Apptainer> ")
        cmd.extend(["--env", "PS1=\\$ ", "--env", "PROMPT_COMMAND="])
        if cwd:
            cmd.extend(["--pwd", cwd])
        for k, v in env_vars.items():
            cmd.extend(["--env", f"{k}={shlex.quote(v)}"])
        cmd.append(f"instance://{self.instance_name}")
        cmd.extend(["/bin/bash", "-lc", command])

        if "tmux" in command and "-V" in command:
            dbg_cmd = [
                APPTAINER,
                "exec",
                f"instance://{self.instance_name}",
                "/bin/bash",
                "-lc",
                "ls -la /usr/bin/tmux /usr/local/bin/tmux 2>&1; file /usr/bin/tmux 2>&1; /usr/bin/tmux -V 2>&1",
            ]
            dbg_result = subprocess.run(
                dbg_cmd, capture_output=True, text=True, timeout=30
            )
            print(
                f"[{self.env_id}] TMUX DEBUG: stdout={repr(dbg_result.stdout[:300])} stderr={repr(dbg_result.stderr[:200])}",
                flush=True,
            )

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_sec if timeout_sec else 600,
            )
            if result.returncode == 255:
                # apptainer itself failed to run the exec (instance gone,
                # bad --pwd, mount error, ...) — log it, the caller only sees rc.
                print(
                    f"[{self.env_id}] exec rc=255: {' '.join(cmd[:-1])[:400]} | "
                    f"stderr: {result.stderr.strip()[-600:]}",
                    flush=True,
                )
            # Filter proxychains noise from output — these lines confuse
            # the model and cause it to waste turns trying to suppress them.
            stdout = result.stdout or ""
            stderr = result.stderr or ""
            if (
                "[proxychains]" in stdout
                or "[proxychains]" in stderr
                or "DLL init" in stderr
            ):
                stdout = "\n".join(
                    line
                    for line in stdout.split("\n")
                    if not line.startswith("[proxychains]") and "DLL init" not in line
                )
                stderr = "\n".join(
                    line
                    for line in stderr.split("\n")
                    if not line.startswith("[proxychains]") and "DLL init" not in line
                )
            stdout_truncated = len(stdout) > self.max_exec_output_chars
            stderr_truncated = len(stderr) > self.max_exec_output_chars
            return {
                "stdout": stdout[-self.max_exec_output_chars :],
                "stderr": stderr[-self.max_exec_output_chars :],
                "return_code": result.returncode,
                "stdout_truncated": stdout_truncated,
                "stderr_truncated": stderr_truncated,
            }
        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": f"Command timed out after {timeout_sec}s",
                "return_code": -1,
            }

    def upload(self, payload):
        """Upload a file into the instance via staging."""
        file_b64 = payload.get("file_b64", "")
        target_path = payload.get("target_path", "/workspace/upload")
        is_dir = payload.get("is_dir", False)

        staging_file = os.path.join(self.staging_dir, "upload_tmp")
        with open(staging_file, "wb") as f:
            f.write(base64.b64decode(file_b64))

        if is_dir:
            # It's a tar archive — extract into target
            cmd = [
                APPTAINER,
                "exec",
                f"instance://{self.instance_name}",
                "bash",
                "-c",
                f"mkdir -p {shlex.quote(target_path)}",
            ]
            subprocess.run(cmd, capture_output=True, timeout=30)
            # Copy via bind mount (staging_dir is accessible as /staging isn't bound, use workspace)
            shutil.copy2(
                staging_file, os.path.join(self.staging_dir, "workspace", ".upload_tmp")
            )
            cmd = [
                APPTAINER,
                "exec",
                f"instance://{self.instance_name}",
                "bash",
                "-c",
                f"tar xf /workspace/.upload_tmp -C {shlex.quote(target_path)} && rm /workspace/.upload_tmp",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            if result.returncode == 0:
                if target_path == "/tests":
                    verify_script = (
                        "if [ -f /tests/test.sh ]; then "
                        "  echo '__BRIDGE_UPLOAD_OK__'; "
                        "  find /tests -maxdepth 2 -type f | sort | sed -n '1,50p'; "
                        "else "
                        "  echo '__BRIDGE_UPLOAD_MISSING__'; "
                        "  ls -la /tests; "
                        "  find /tests -maxdepth 3 | sort | sed -n '1,80p'; "
                        "  exit 97; "
                        "fi"
                    )
                else:
                    verify_script = (
                        f"if [ -e {shlex.quote(target_path)} ]; then "
                        f"  find {shlex.quote(target_path)} -maxdepth 2 | sort | sed -n '1,40p'; "
                        "else "
                        f"  echo 'upload target missing: {target_path}'; exit 98; "
                        "fi"
                    )
                verify_cmd = [
                    APPTAINER,
                    "exec",
                    f"instance://{self.instance_name}",
                    "bash",
                    "-lc",
                    verify_script,
                ]
                verify = subprocess.run(
                    verify_cmd, capture_output=True, text=True, timeout=60
                )
                result = verify
        else:
            shutil.copy2(
                staging_file, os.path.join(self.staging_dir, "workspace", ".upload_tmp")
            )
            cmd = [
                APPTAINER,
                "exec",
                f"instance://{self.instance_name}",
                "bash",
                "-c",
                f"mkdir -p $(dirname {shlex.quote(target_path)}); "
                f"rm -f {shlex.quote(target_path)}; "
                f"cp /workspace/.upload_tmp {shlex.quote(target_path)} && rm -f /workspace/.upload_tmp",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        os.unlink(staging_file)
        return {
            "return_code": result.returncode,
            "stdout": result.stdout[:4000] if result.stdout else "",
            "stderr": result.stderr[:2000] if result.stderr else "",
        }

    def download(self, payload):
        """Download a file from the instance."""
        source_path = payload.get("source_path", "")
        is_dir = payload.get("is_dir", False)

        download_tmp = os.path.join(self.staging_dir, "workspace", ".download_tmp")
        if is_dir:
            cmd = [
                APPTAINER,
                "exec",
                f"instance://{self.instance_name}",
                "bash",
                "-c",
                f"tar cf /workspace/.download_tmp -C {shlex.quote(source_path)} .",
            ]
        else:
            cmd = [
                APPTAINER,
                "exec",
                f"instance://{self.instance_name}",
                "bash",
                "-c",
                f"cp {shlex.quote(source_path)} /workspace/.download_tmp",
            ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            return {"error": f"Download failed: {result.stderr[:500]}"}

        if os.path.isfile(download_tmp):
            with open(download_tmp, "rb") as f:
                data = base64.b64encode(f.read()).decode()
            os.unlink(download_tmp)
            return {"file_b64": data, "is_dir": is_dir}
        return {"error": "Downloaded file not found"}

    def reset_for_env(self, new_env_id, payload):
        """Rebind a pooled, already-running instance to a new env_id.

        Clears /workspace, /tmp, /logs/{verifier,agent}, /tests, and the
        dockerfile-workdir bind contents in-place (bind mounts cannot be
        changed on a running apptainer instance). Then writes the new task's
        files into workspace and re-applies dockerfile COPY logic so the
        instance is ready for `exec` calls under the new env_id.

        Returns the same dict shape as `start()` so the worker handler can
        treat reused-start identically to fresh-start at the bridge level.
        """
        # Update env_id BEFORE the kill, so log lines from the kill are
        # tagged with the new id.
        self.env_id = new_env_id
        self._uses_count = getattr(self, "_uses_count", 0) + 1

        # 1. Kill all user-owned processes inside the container except init
        #    (PID 1). The apptainer instance's PID 1 is a long-lived sleep;
        #    pkill -P 1 hits its direct children. The pgrep fallback also
        #    catches grandchildren that re-parented away.
        kill_cmd = [
            APPTAINER,
            "exec",
            f"instance://{self.instance_name}",
            "/bin/bash",
            "-c",
            "pkill -9 -P 1 2>/dev/null; "
            "MYUID=$(id -u); "
            "for pid in $(pgrep -U $MYUID 2>/dev/null); do "
            '  if [ "$pid" != "1" ]; then kill -9 $pid 2>/dev/null; fi; '
            "done; true",
        ]
        try:
            subprocess.run(kill_cmd, capture_output=True, timeout=15)
        except Exception as e:
            print(f"[{self.env_id}] reset: kill step failed: {e}", flush=True)

        # 2. Clear host-side contents of bound dirs. This is faster than
        #    running `rm -rf` inside the container because we can do it
        #    in parallel from the host without exec overhead.
        for sub in ("workspace", "tmp", "logs/verifier", "logs/agent", "tests"):
            d = os.path.join(self.staging_dir, sub)
            if not os.path.isdir(d):
                continue
            for name in os.listdir(d):
                p = os.path.join(d, name)
                try:
                    if os.path.isdir(p) and not os.path.islink(p):
                        shutil.rmtree(p, ignore_errors=True)
                    else:
                        os.unlink(p)
                except Exception:
                    pass

        # Clear the dockerfile-workdir bind dir too if present
        workdir_host = os.path.join(self.staging_dir, "workdir")
        if os.path.isdir(workdir_host):
            for name in os.listdir(workdir_host):
                p = os.path.join(workdir_host, name)
                try:
                    if os.path.isdir(p) and not os.path.islink(p):
                        shutil.rmtree(p, ignore_errors=True)
                    else:
                        os.unlink(p)
                except Exception:
                    pass

        # 3. Write the new trial's task files into workspace
        workspace = os.path.join(self.staging_dir, "workspace")
        files_b64 = payload.get("files_b64", {})
        for rel_path, content_b64 in files_b64.items():
            target = os.path.join(workspace, rel_path)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "wb") as f:
                f.write(base64.b64decode(content_b64))

        # 4. Re-apply Dockerfile COPY logic into workspace
        try:
            self._apply_dockerfile_copies(workspace)
        except Exception as e:
            print(f"[{self.env_id}] reset: dockerfile copies failed: {e}", flush=True)

        # 5. Re-seed the dockerfile-workdir bind from the SIF (if present).
        #    The first start() copied SIF's workdir contents into workdir_host;
        #    we just cleared it, so we need to repeat that seed.
        dockerfile_workdir = getattr(self, "_dockerfile_workdir", None)
        resolved_sif = getattr(self, "_resolved_sif", None)
        if (
            dockerfile_workdir
            and resolved_sif
            and dockerfile_workdir not in _SYSTEM_PATHS_FOR_BIND
        ):
            try:
                init_cmd = [
                    APPTAINER,
                    "exec",
                    "--bind",
                    f"{workdir_host}:/_workdir_init:rw",
                    resolved_sif,
                    "sh",
                    "-c",
                    f"if [ -d {dockerfile_workdir} ] && "
                    f'[ -n "$(ls -A {dockerfile_workdir} 2>/dev/null)" ]; then '
                    f"cp -a {dockerfile_workdir}/. /_workdir_init/ 2>/dev/null || true; "
                    f"fi",
                ]
                subprocess.run(init_cmd, capture_output=True, text=True, timeout=120)
            except Exception as e:
                print(f"[{self.env_id}] reset: workdir reseed failed: {e}", flush=True)

        # 6. If this is a base-image SIF, re-extract and re-run the per-task
        #    Dockerfile RUN commands. Trials of DIFFERENT tasks sharing a
        #    base-image SIF will have different RUN commands (e.g. git
        #    checkout of different commits) — we must re-execute them.
        if getattr(self, "_using_base_sif", False):
            try:
                self._pending_run_commands = self._extract_run_commands(workspace)
                if self._pending_run_commands:
                    print(
                        f"[{self.env_id}] reset: running "
                        f"{len(self._pending_run_commands)} Dockerfile RUN commands",
                        flush=True,
                    )
                    for i, cmd_str in enumerate(self._pending_run_commands):
                        exec_cmd = [
                            APPTAINER,
                            "exec",
                            "--pwd",
                            "/testbed",
                            f"instance://{self.instance_name}",
                            "bash",
                            "-c",
                            cmd_str,
                        ]
                        try:
                            r = subprocess.run(
                                exec_cmd, capture_output=True, text=True, timeout=300
                            )
                            if r.returncode != 0:
                                print(
                                    f"[{self.env_id}] reset: RUN cmd {i} "
                                    f"rc={r.returncode}: {cmd_str[:80]}",
                                    flush=True,
                                )
                        except subprocess.TimeoutExpired:
                            print(
                                f"[{self.env_id}] reset: RUN cmd {i} timed out",
                                flush=True,
                            )
                    self._pending_run_commands = []
            except Exception as e:
                print(
                    f"[{self.env_id}] reset: RUN-cmd extraction failed: {e}", flush=True
                )

        print(
            f"[{self.env_id}] Reused instance {self.instance_name} "
            f"(uses={self._uses_count})",
            flush=True,
        )
        return {
            "sif": resolved_sif or "",
            "instance": self.instance_name,
            "reused": True,
        }

    def stop(self, payload):
        """Stop instance and cleanup."""
        if self.started:
            try:
                subprocess.run(
                    [APPTAINER, "instance", "stop", self.instance_name],
                    capture_output=True,
                    timeout=30,
                )
            except Exception:
                pass
            self.started = False

        # Cleanup staging
        if self.staging_dir and os.path.exists(self.staging_dir):
            shutil.rmtree(self.staging_dir, ignore_errors=True)

        return {"stopped": True}


# Active instances managed by this worker
_instances: dict = {}  # env_id -> ApptainerInstance
_instances_lock = threading.Lock()


# ============================================================================
# Instance reuse pool (opt-in via BRIDGE_INSTANCE_REUSE=1)
# ============================================================================
# When enabled, STOP jobs return the apptainer instance to a pool keyed by
# (sif_path, dockerfile_hash, base_image, task_name) instead of calling
# `apptainer instance stop`. Subsequent START jobs with a matching payload
# grab an idle pool instance and clear/repopulate its workspace in-place
# rather than spinning up a fresh apptainer instance.
#
# Why: stop_env is the bridge's most failure-prone op (suid-starter races,
# timeouts under load). Failed stops leak zombie envs that eat worker slots.
# Bypassing the stop/start cycle eliminates both the leak AND the ~5-10s
# start overhead per trial.
#
# Constraints:
# - Apptainer bind mounts are fixed at start time. The pool reuses the same
#   /workspace bind (a host staging dir); we only clear/repopulate its contents.
# - Only safe when the next trial has the same SIF + WORKDIR + base_image flag.
#   The pool key encodes this — different keys never share an instance.
# - State leakage between trials is possible (env vars in init's env, modified
#   /etc files, kernel-side caches). For SWE-bench-style tasks where all
#   trial state lives under /workspace + /tmp + /logs, this is acceptable.
# - Instances are recycled (hard-stopped + replaced) after BRIDGE_POOL_MAX_USES
#   trials to bound state drift.
_instance_pool: dict = {}  # pool_key -> list of (ApptainerInstance, idle_since_ts)
_pool_lock = threading.Lock()

# Reserved paths that the dockerfile-workdir bind logic refuses to bind over;
# kept in sync with the same set in ApptainerInstance.start().
_SYSTEM_PATHS_FOR_BIND = {
    "/",
    "/tmp",
    "/usr",
    "/var",
    "/etc",
    "/bin",
    "/sbin",
    "/lib",
    "/lib64",
    "/proc",
    "/sys",
    "/dev",
    "/root",
    "/home",
    "/workspace",
    "/logs",
    "/tests",
}


def _pool_enabled():
    return os.environ.get("BRIDGE_INSTANCE_REUSE", "0") == "1"


def _pool_max_per_key():
    return int(os.environ.get("BRIDGE_POOL_MAX_PER_KEY", "4"))


def _pool_max_uses():
    return int(os.environ.get("BRIDGE_POOL_MAX_USES", "50"))


def _pool_idle_timeout_sec():
    return int(os.environ.get("BRIDGE_POOL_IDLE_TIMEOUT_SEC", "1800"))


def _pool_key_from_payload(payload):
    """Key for pool lookup. Two payloads with the same key are assumed to
    resolve to the same SIF + WORKDIR + RUN-command setup, so they can
    share an instance."""
    return (
        payload.get("sif_path", "") or "",
        payload.get("dockerfile_hash", "") or "",
        payload.get("base_image", "") or "",
        payload.get("task_name", "") or "",
    )


def _pool_acquire(payload):
    """Try to grab an idle pooled instance matching `payload`. None if no hit."""
    if not _pool_enabled():
        return None
    key = _pool_key_from_payload(payload)
    with _pool_lock:
        bucket = _instance_pool.get(key)
        if bucket:
            inst, _ = bucket.pop(0)
            if not bucket:
                del _instance_pool[key]
            return inst
    return None


def _pool_release(instance):
    """Return `instance` to the pool. Returns True if pooled (caller should
    NOT hard-stop), False if the caller should hard-stop the instance."""
    if not _pool_enabled():
        return False
    key = getattr(instance, "_pool_key", None)
    if key is None:
        return False
    if getattr(instance, "_uses_count", 0) >= _pool_max_uses():
        return False  # recycle: caller hard-stops, fresh instance next time
    with _pool_lock:
        bucket = _instance_pool.setdefault(key, [])
        if len(bucket) >= _pool_max_per_key():
            return False
        bucket.append((instance, time.time()))
    return True


def _pool_evictor_loop():
    """Background thread: hard-stop pool instances idle longer than the timeout."""
    while True:
        time.sleep(60)
        if not _pool_enabled():
            continue
        now = time.time()
        timeout = _pool_idle_timeout_sec()
        to_stop = []
        with _pool_lock:
            for key, bucket in list(_instance_pool.items()):
                fresh = []
                for inst, idle_since in bucket:
                    if now - idle_since > timeout:
                        to_stop.append(inst)
                    else:
                        fresh.append((inst, idle_since))
                if fresh:
                    _instance_pool[key] = fresh
                else:
                    del _instance_pool[key]
        for inst in to_stop:
            try:
                inst.stop({"_recycle": True})
            except Exception as e:
                print(
                    f"[pool-evict] stop failed for {inst.instance_name}: {e}",
                    flush=True,
                )


def dispatcher_loop(bridge_url, node_id, job_queue, num_workers):
    """Single dispatcher thread per node: polls bridge, distributes jobs locally.

    This reduces SSH tunnel traffic from N workers polling to 1 dispatcher polling.
    Uses batch polling: requests up to num_workers jobs at once.
    """
    print(f"[Dispatcher {node_id}] Starting, bridge={bridge_url}")
    min_interval = 0.05  # Fast poll when jobs are flowing
    max_interval = 2.0  # Slow poll when idle
    idle_interval = 0.1

    while True:
        try:
            # Request a batch of jobs (up to number of idle workers)
            idle_count = max(1, num_workers - job_queue.qsize())
            resp = http_get(
                f"{bridge_url}/worker/get_jobs?worker_id={node_id}&count={idle_count}",
                timeout=15,
            )

            if resp.get("error"):
                # Fallback to single-job endpoint if batch not supported
                resp = http_get(
                    f"{bridge_url}/worker/get_job?worker_id={node_id}",
                    timeout=15,
                )
                if resp.get("error"):
                    time.sleep(max_interval)
                    continue
                # Wrap single response as a list
                if resp.get("job_id") is not None:
                    jobs = [resp]
                else:
                    jobs = []
            else:
                jobs = resp.get("jobs", [])

            if not jobs:
                time.sleep(idle_interval)
                idle_interval = min(idle_interval * 1.3, max_interval)
                continue

            # Distribute jobs to worker threads via local queue
            idle_interval = min_interval
            for job in jobs:
                job_queue.put(job)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[Dispatcher {node_id}] Error: {e}")
            time.sleep(2.0)


def worker_loop(
    bridge_url,
    worker_id,
    sif_cache,
    staging_base,
    job_queue,
    max_exec_output_chars,
):
    """Worker thread: consumes jobs from local queue, executes, posts results.

    No HTTP polling — jobs come from the dispatcher via in-process queue.
    """
    print(f"[Worker {worker_id}] Starting (queue-based)")

    while True:
        try:
            # Block on local queue (no HTTP overhead)
            try:
                resp = job_queue.get(timeout=5.0)
            except queue.Empty:
                continue

            job_id = resp.get("job_id")
            if job_id is None:
                continue

            env_id = resp["env_id"]
            job_type = resp["type"]
            payload = resp.get("payload", {})

            print(f"[Worker {worker_id}] Job {job_id}: {job_type} env={env_id}")

            result_data = {"job_id": job_id}

            try:
                if job_type == "start":
                    # Try the reuse pool first; fall back to fresh start.
                    pooled = _pool_acquire(payload)
                    if pooled is not None:
                        try:
                            result = pooled.reset_for_env(env_id, payload)
                            with _instances_lock:
                                _instances[env_id] = pooled
                            result_data["result"] = result
                        except Exception as e:
                            # Reset failed — hard-stop the pooled instance and
                            # fall through to fresh start so the trial still runs.
                            print(
                                f"[Worker {worker_id}] pool reset failed "
                                f"for {pooled.instance_name}: {e}",
                                flush=True,
                            )
                            try:
                                pooled.stop({"_recycle": True})
                            except Exception:
                                pass
                            pooled = None
                    if pooled is None:
                        instance = ApptainerInstance(
                            env_id,
                            sif_cache,
                            staging_base,
                            max_exec_output_chars,
                        )
                        result = instance.start(payload)
                        with _instances_lock:
                            _instances[env_id] = instance
                        result_data["result"] = result

                elif job_type in ("exec", "upload", "download"):
                    with _instances_lock:
                        instance = _instances.get(env_id)
                    if not instance:
                        result_data["error"] = f"No instance for env {env_id}"
                    else:
                        method = getattr(instance, job_type)
                        result = method(payload)
                        if isinstance(result, dict) and "error" in result:
                            result_data["error"] = result["error"]
                        else:
                            result_data["result"] = result

                elif job_type == "stop":
                    with _instances_lock:
                        instance = _instances.pop(env_id, None)
                    if instance:
                        # Try to release to pool; if pool declines (full or
                        # uses cap exceeded) fall back to hard stop.
                        if _pool_release(instance):
                            result_data["result"] = {"stopped": True, "pooled": True}
                        else:
                            result = instance.stop(payload)
                            result_data["result"] = result
                    else:
                        result_data["result"] = {"stopped": True, "note": "no instance"}

                print(f"[Worker {worker_id}] Job {job_id} done")

            except Exception as e:
                result_data["error"] = str(e)
                print(f"[Worker {worker_id}] Job {job_id} failed: {e}")

            # Result posting still goes directly to bridge via HTTP
            http_post(f"{bridge_url}/worker/result", result_data)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[Worker {worker_id}] Error: {e}")
            time.sleep(1.0)


def _cleanup_stale_instances(hostname, staging_base):
    """Stop any leftover hb_env_* instances and clean staging dirs from previous runs."""
    global APPTAINER
    if not APPTAINER:
        try:
            APPTAINER = detect_apptainer()
        except FileNotFoundError:
            return

    # List running instances
    try:
        result = subprocess.run(
            [APPTAINER, "instance", "list", "--json"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout)
            instances = data.get("instances", [])
            for inst in instances:
                name = inst.get("instance", "")
                if name.startswith("hb_env_"):
                    print(f"[{hostname}] Cleaning stale instance: {name}")
                    subprocess.run(
                        [APPTAINER, "instance", "stop", name],
                        capture_output=True,
                        timeout=30,
                    )
    except Exception as e:
        print(f"[{hostname}] Warning: failed to list/stop stale instances: {e}")

    # Clean stale staging dirs
    if os.path.isdir(staging_base):
        for entry in os.listdir(staging_base):
            if entry.startswith("apt_env-"):
                path = os.path.join(staging_base, entry)
                try:
                    shutil.rmtree(path, ignore_errors=True)
                    print(f"[{hostname}] Cleaned stale staging: {entry}")
                except Exception:
                    pass


def main():
    parser = argparse.ArgumentParser(description="Apptainer Bridge Worker")
    parser.add_argument("--bridge-url", required=True)
    parser.add_argument(
        "--sif-cache", required=True, help="Directory with pre-built .sif files"
    )
    parser.add_argument(
        "--staging-base", default="/tmp", help="Base dir for instance staging"
    )
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--poll-interval", type=float, default=0.1)
    parser.add_argument(
        "--max-exec-output-chars",
        type=int,
        default=DEFAULT_MAX_EXEC_OUTPUT_CHARS,
        help="Maximum stdout and stderr characters retained per exec",
    )
    args = parser.parse_args()

    if args.max_exec_output_chars <= 0:
        parser.error("--max-exec-output-chars must be positive")

    os.makedirs(args.sif_cache, exist_ok=True)
    os.makedirs(args.staging_base, exist_ok=True)

    hostname = os.uname().nodename
    print(f"Starting {args.num_workers} workers + 1 dispatcher on {hostname}")
    print(f"  Bridge: {args.bridge_url}")
    print(f"  SIF cache: {args.sif_cache}")
    print(f"  Staging: {args.staging_base}")
    print(
        f"  Architecture: 1 dispatcher polls bridge, {args.num_workers} workers consume locally"
    )

    # Cleanup stale instances from previous runs on this node
    _cleanup_stale_instances(hostname, args.staging_base)

    # Shared job queue: dispatcher produces, workers consume
    job_queue = queue.Queue(maxsize=args.num_workers * 2)

    # Start dispatcher thread (1 per node — the only thread that polls bridge)
    node_id = hostname
    dispatcher = threading.Thread(
        target=dispatcher_loop,
        args=(args.bridge_url, node_id, job_queue, args.num_workers),
        daemon=True,
    )
    dispatcher.start()

    # Pool evictor: drops pooled instances idle longer than the timeout.
    # No-op when BRIDGE_INSTANCE_REUSE!=1; cheap to always start.
    evictor = threading.Thread(target=_pool_evictor_loop, daemon=True)
    evictor.start()
    if _pool_enabled():
        print(
            f"[{hostname}] Instance reuse pool ENABLED "
            f"(max_per_key={_pool_max_per_key()}, "
            f"max_uses={_pool_max_uses()}, "
            f"idle_timeout={_pool_idle_timeout_sec()}s)",
            flush=True,
        )

    # Start worker threads (consume from local queue, no HTTP polling)
    threads = []
    for i in range(args.num_workers):
        wid = f"{hostname}-{i}"
        t = threading.Thread(
            target=worker_loop,
            args=(
                args.bridge_url,
                wid,
                args.sif_cache,
                args.staging_base,
                job_queue,
                args.max_exec_output_chars,
            ),
            daemon=True,
        )
        t.start()
        threads.append(t)

    try:
        while True:
            time.sleep(10)
            try:
                http_get(f"{args.bridge_url}/heartbeat")
            except Exception:
                pass
    except KeyboardInterrupt:
        print("Shutting down workers.")
        # Cleanup all in-use instances
        with _instances_lock:
            for env_id, instance in _instances.items():
                try:
                    instance.stop({})
                except Exception:
                    pass
        # Cleanup pooled (idle) instances
        with _pool_lock:
            pooled_instances = [
                inst for bucket in _instance_pool.values() for inst, _ in bucket
            ]
            _instance_pool.clear()
        for inst in pooled_instances:
            try:
                inst.stop({"_recycle": True})
            except Exception:
                pass


if __name__ == "__main__":
    main()

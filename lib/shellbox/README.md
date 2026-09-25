# Marin Shellbox (0.1)

`marin-shellbox` provides three Harbor import paths:

- `shellbox.agent:BashAgent`: an **external** agent. It runs in the Harbor process and calls an OpenAI-compatible endpoint from that process. Its `Bash` tool uses the selected environment's persistent shell.
- `shellbox.backends.qemu.environment:QemuEnvironment`: one persistent QEMU system guest per Harbor trial. It uses KVM when the process can access `/dev/kvm` and QEMU can initialize it, then falls back to software emulation (TCG). Running a prebuilt bundle needs no Docker daemon, user namespace, or guest network.
- `shellbox.backends.shellsim.environment:ShellSimEnvironment`: one in-memory [ShellSim](https://pypi.org/project/shellsim/) instance per trial. It uses ShellSim's built-in commands and ignores the task's Docker image or Dockerfile.

Install the backend dependencies you need:

```sh
uv pip install 'marin-shellbox[shellsim]'
uv pip install 'marin-shellbox[qemu]'
```

The base wheel contains the Harbor adapter, machine API, and guest source. Harbor is the host application: install this wheel into an environment that already has the [Marin Harbor fork](../../config/external/harbor/pyproject.toml), whose `harbor==0.8.1` distribution is not published on PyPI. The machine API can be used without Harbor. The `shellsim` extra requires ShellSim 0.1.17 or newer within the 0.1 series. The `qemu` extra requires [quicksand-qemu](https://pypi.org/project/quicksand-qemu/) 0.5.12 or newer within the 0.5 series; it bundles QEMU and its shared libraries in a platform wheel. A QEMU guest bundle still needs a Linux amd64 kernel, static BusyBox, and `bios-microvm.bin`; OCI staging also needs Skopeo, `umoci`, `mkfs.ext4`, and `cpio`. These inputs are explicit until we have a portable, licensed guest-runtime wheel. The tested quicksand-qemu Linux wheel requires glibc 2.38 or newer; use a compatible host QEMU on older clusters.

## Machine API

The package provides a Harbor-independent machine interface. QEMU and Docker factories accept a registry reference, a local Dockerfile, or a `PreparedImage`. QEMU also accepts a prebuilt guest bundle; Docker accepts a local image. `ShellSimMachineFactory` accepts only `ShellSimBuiltins()`. Each `create` returns a fresh machine with a persistent writable filesystem. `run` returns bytes, exit status, and output truncation flags. `upload`, `download`, and `close` complete the common interface.

Shared contracts and OCI image preparation live at the package root. Backend machines and their Harbor adapters live under `shellbox.backends.qemu`, `shellbox.backends.shellsim`, and `shellbox.backends.docker`. The Docker backend has no Harbor adapter yet.

```python
from shellbox.machine import Command, MachineSpec, QemuBundle
from shellbox.backends.qemu.machine import QemuMachineFactory

machine = await QemuMachineFactory().create(MachineSpec(source=QemuBundle(bundle_path)))
try:
    result = await machine.run(Command(("/bin/sh", "-c", "echo hello")))
finally:
    await machine.close()
```

For Docker, use `DockerMachineFactory()` and `DockerImage("image:tag")` in the same `MachineSpec`. This form requires a locally available image. Backends reject unsupported source types; QEMU also rejects guest networking.

To prepare a registry image, use a standard image reference without `https://`:

```python
from pathlib import Path
from shellbox.backends.qemu.image import QemuAssets
from shellbox.image import RegistryImage
from shellbox.machine import MachineSpec
from shellbox.backends.qemu.machine import QemuMachineFactory

spec = MachineSpec(source=RegistryImage("ghcr.io/astral-sh/uv:alpine3.21"))
machine = await QemuMachineFactory(
    assets=QemuAssets(
        qemu=Path("/opt/qemu/qemu-system-x86_64"),
        kernel=Path("/opt/qemu/vmlinuz"),
        busybox=Path("/opt/qemu/static-busybox"),
        firmware=Path("/opt/qemu/firmware"),
        libraries=Path("/opt/qemu/lib"),
        umoci=Path("/usr/bin/umoci"),
        disk_size_mb=512,
        runtime_id="pinned-runtime-v1",
    ),
    image_cache=Path("/var/cache/marin-shellbox/oci"),
    bundle_cache=Path("/var/cache/marin-shellbox/bundles"),
    skopeo=Path("/usr/bin/skopeo"),
).create(spec)
```

The factory uses a process-local `ImageCache` to call `prepare_image` once per source. Preparation selects `linux/amd64`, copies it to a local OCI layout, records its selected manifest digest, and caches the layout by that digest. It accepts Docker Hub references such as `docker.io/library/ubuntu:24.04` and GHCR references. A registry reference is resolved once per Python process; a later process resolves it again. Use `@sha256:...` to fix the source version across processes. For a private registry, pass a Skopeo `authfile` path to the factory. The default Skopeo policy accepts unsigned images over verified TLS; pass `policy` to require a signature policy. Credentials are used only during preparation.

`DockerfileSource(context=..., dockerfile=...)` is the other source type. It requires a local Docker builder, builds for `linux/amd64`, and copies the result into the OCI cache. The cache fingerprints the Dockerfile and every path in its build context on each call. An unchanged context builds once per Python process; a changed context triggers a new build. A base image referenced by a mutable tag can change without a context edit, so pin `FROM` by digest for repeatable builds. `DockerMachineFactory` loads a prepared layout under a digest-derived local tag when needed. `QemuMachineFactory` needs `QemuAssets` and a `bundle_cache` to stage the same prepared layout into a guest bundle on first use. `tests/manual/check_prepared_image.py` checks both backends against one public GHCR image. The QEMU bundle cache key includes the prepared image digest, disk size, and caller-supplied `runtime_id`; change `runtime_id` when QEMU, kernel, BusyBox, firmware, or libraries change.

`run` remains noninteractive for setup and verification. It starts a new process for each call, while files persist until `close`. Its serial protocol buffers command output in the guest and streams it to the host in bounded chunks; a timeout destroys the machine. `QemuMachine.open_shell()` opens a separate PTY-backed Bash session for agent commands. `tests/manual/check_machine.py` checks one-shot command results, persistent files, and file transfer. `tests/manual/check_shell.py` checks shell state, input, interruption, jobs, bounded output, and reset.

The wheel contains Python integration, guest init code, and the source for a small PTY helper. Staging compiles that helper with `cc -static` and `-lutil`; the runtime host does not need a compiler when it uses a prebuilt bundle. The runtime host needs QEMU, its libraries, firmware, a Linux kernel, and a **static** BusyBox staged into a bundle. A minimal guest can be staged with:

```sh
shellbox-stage \
  --qemu /path/to/qemu-system-x86_64 \
  --kernel /path/to/vmlinuz \
  --busybox /path/to/static-busybox \
  --firmware /path/to/qemu-firmware \
  --libraries /path/to/qemu-libraries \
  --output /opt/marin-shellbox/guest
```

The `firmware` directory must contain `bios-microvm.bin` from SeaBIOS. The `libraries` directory must contain every shared library required by the QEMU executable on the runtime host. The bundle can be prepared on a build machine and copied to the runtime host.
Re-stage bundles built with an earlier prototype when updating this package; the guest and host serial protocols must match.

With the `qemu` extra, `quicksand_qemu.get_bin_dir()` gives the QEMU executable at `bin/qemu-system-x86_64` and its libraries at `bin/lib`. Pass those paths as `--qemu` and `--libraries` when staging. The current quicksand-qemu wheel does not supply `bios-microvm.bin`; pass a firmware directory that does. Its QEMU modules are loaded from the staged `lib/qemu` directory.

## ShellSim backend

ShellSim requires no QEMU assets, Docker daemon, image pull, or build step. The `shellsim` extra accepts ShellSim releases from 0.1.17 to before 0.2. Select it in a Harbor job:

```yaml
environment:
  import_path: shellbox.backends.shellsim.environment:ShellSimEnvironment
agents:
  - import_path: shellbox.agent:BashAgent
    model_name: openai/local-model
    kwargs:
      base_url: http://127.0.0.1:8000/v1
```

The task's `docker_image` and `environment/Dockerfile` are ignored. Files already in `environment/` are copied to the simulated workdir, which defaults to `/workspace`; Dockerfile instructions are never applied. ShellSim's trusted import accepts regular files and rejects symlinks and special files. ShellSim provides its own command set and in-memory filesystem. Use this backend only for tasks whose setup, agent commands, and verifier use [supported ShellSim facilities](https://pypi.org/project/shellsim/). Unsupported commands fail visibly. For direct machine use, pass `MachineSpec(source=ShellSimBuiltins())` to `ShellSimMachineFactory.create()`.

`BashAgent` offers only `Bash` with ShellSim. Each action completes before returning; interactive input and interruption are unavailable. Cwd, exports, functions, and virtual files persist across Bash actions. ShellSim currently also carries shell functions and exports into Harbor setup and verifier actions on the same instance. This differs from QEMU's separate verifier shell and can affect rewards. [ShellSim issue #89](https://github.com/rjpower/shellsim/issues/89) tracks an isolated shell action that shares the virtual filesystem. Treat ShellSim verifier results as exploratory until that isolation is available. The backend has no host filesystem or network access from simulated commands; the trusted harness copies task files into the virtual filesystem.

The default ShellSim limits per trial are 10 billion CPU units, 256 MiB modeled memory, 256 MiB virtual disk, and 128 MiB cumulative output. `cpu_limit`, `memory_mb`, `disk_limit_bytes`, and `output_limit_bytes` configure them. An action that exhausts a ShellSim resource terminates the trial. These are ShellSim fuel and working-set limits, not host CPU or RSS limits. Tasks that request internet are rejected unless `network_policy: deny` explicitly runs them offline.

Run the local Harbor smoke without QEMU:

```sh
uv run --no-project --python config/external/harbor/.venv/bin/python \
  lib/shellbox/tests/harbor_smoke.py shellsim /tmp/shellsim-smoke-jobs
```

## OCI image ingestion

On a build machine, copy a prebuilt image into an OCI layout, then stage it. For example:

```sh
skopeo copy docker://docker.io/library/ubuntu:24.04 oci:/tmp/ubuntu-oci:ubuntu
shellbox-stage \
  --qemu /path/to/qemu-system-x86_64 \
  --kernel /path/to/vmlinuz \
  --busybox /path/to/static-busybox \
  --firmware /path/to/qemu-firmware \
  --libraries /path/to/qemu-libraries \
  --oci-layout /tmp/ubuntu-oci \
  --oci-tag ubuntu \
  --umoci /path/to/umoci \
  --disk-size-mb 512 \
  --output /opt/marin-shellbox/ubuntu
```

Staging uses `umoci unpack --rootless` to apply OCI layers and whiteouts, then builds an ext4 guest disk. It restores file UID/GID values from the OCI layers, which rootless unpacking cannot retain on the host. Each Harbor trial gets its own writable disk copy. The bundle records the unpacked manifest digest, image environment, and working directory. The guest uses GNU Bash when the image supplies `/bin/bash`; otherwise it uses `/bin/sh`. OCI images must be Linux amd64, provide `/bin/sh`, and specify root as their default user. Entrypoint and CMD are not run because Harbor drives commands directly.

Harbor can prepare a task's `environment/Dockerfile` or `docker_image` automatically when configured as shown below. For prebuilt bundles, pass `--task-dockerfile` when staging a Dockerfile image or `--image-reference` when staging a registry image. Harbor checks the recorded Dockerfile SHA-256 or image reference before starting QEMU. These checks catch mismatched task inputs; they do not prove which Dockerfile built an OCI image. Compose tasks remain unsupported.

This staging step requires `umoci`, `mkfs.ext4` built with libarchive support, and `cpio` on the build machine. `mkfs.ext4 -d` reads a tarball so the guest filesystem retains OCI file ownership. A runtime host using only prebuilt bundles needs none of these tools. For repeatable builds, pin the source image by digest before copying it to the OCI layout. See the [OCI image layout](https://specs.opencontainers.org/image-spec/image-layout/) and [umoci unpack](https://umo.ci/quick-start/workflow/) documentation for the source format and layer unpacking.

Use the import paths in a Harbor job config:

```yaml
environment:
  import_path: shellbox.backends.qemu.environment:QemuEnvironment
  kwargs:
    guest_bundle: /opt/marin-shellbox/bash-image
agents:
  - import_path: shellbox.agent:BashAgent
    model_name: openai/local-model
    kwargs:
      base_url: http://127.0.0.1:8000/v1
```

To use the image named by a task's `docker_image` or build its `environment/Dockerfile`, omit `guest_bundle` and configure image preparation:

```yaml
environment:
  import_path: shellbox.backends.qemu.environment:QemuEnvironment
  kwargs:
    image_cache: /var/cache/marin-shellbox
    skopeo: /usr/bin/skopeo
    qemu_assets:
      qemu: /opt/qemu/qemu-system-x86_64
      kernel: /opt/qemu/vmlinuz
      busybox: /opt/qemu/static-busybox
      firmware: /opt/qemu/firmware
      libraries: /opt/qemu/lib
      umoci: /usr/bin/umoci
      disk_size_mb: 512
      runtime_id: pinned-runtime-v1
agents:
  - import_path: shellbox.agent:BashAgent
    model_name: openai/local-model
    kwargs:
      base_url: http://127.0.0.1:8000/v1
```

The task must specify exactly one source: `docker_image` or `environment/Dockerfile`. The first trial in a Harbor process prepares the image. Later trials with the same source reuse it; concurrent trials wait for the same preparation. A new Harbor process resolves registry references and builds Dockerfiles again, while the OCI and QEMU bundle caches on disk reuse matching digests. For fully offline trials, prepare the image ahead of time and use a prebuilt guest bundle. Image preparation requires Skopeo, `umoci`, `mkfs.ext4`, and `cpio` on the host. Dockerfile preparation also requires a working Docker builder. `registry_authfile` and `signature_policy` are optional environment kwargs for private registries and signed images; private registry authentication has not been tested end to end.

The 0.1 QEMU extra supplies the emulator but does not supply the guest kernel, BusyBox, firmware, or OCI staging tools, so task-image preparation needs the explicit paths above. A prebuilt `guest_bundle` needs no staging tools on the trial host. Skopeo is available through Linux distribution packages; its upstream project does not publish a portable standalone release binary and recommends distribution packages or its container image. We therefore leave Skopeo outside the wheel for now.

`BashAgent` runs in Harbor even though Harbor's config `mode` stays at its default `container`: that field selects a factory path and does not move a custom `BaseAgent` into the guest. The endpoint above is contacted from Harbor, not from QEMU. `BashAgent` requires an image with executable `/bin/bash` and a newly staged PTY-enabled bundle; a minimal BusyBox bundle still supports direct one-shot machine commands but cannot run this agent.

The guest has no network. By default, the environment rejects tasks that declare internet access. Set `environment.kwargs.network_policy: deny` to explicitly run such a task offline; commands that actually require internet will still fail. The TaskTrove fixture below uses this override because its task config leaves `allow_internet` at Harbor's default.
The guest defaults to 512 MiB RAM; set `environment.kwargs.guest_memory_mb` to change that limit.

Acceleration defaults to `auto`. This selects KVM when `/dev/kvm` is readable and writable by the Harbor process; QEMU falls back to TCG if KVM initialization fails. Set `environment.kwargs.acceleration: kvm` to require KVM or `tcg` to force software emulation. The environment logs the accelerator QEMU actually selected after startup. A host may have `/dev/kvm` but deny access to the current user; `auto` uses TCG in that case. The Python package cannot grant device access.

The local smoke test uses a deterministic OpenAI-compatible server and a Harbor task. Supply a PTY-enabled OCI bundle with `/bin/bash`:

```sh
uv run --no-project --python config/external/harbor/.venv/bin/python \
  lib/shellbox/tests/harbor_smoke.py /opt/marin-shellbox/bash-image /tmp/qemu-smoke-jobs
```

For a task with `docker_image` or `environment/Dockerfile`, use `task-image` and supply the QEMU asset paths in a JSON file with the same keys as `qemu_assets` above:

```sh
uv run --no-project --python config/external/harbor/.venv/bin/python \
  lib/shellbox/tests/harbor_smoke.py task-image /tmp/qemu-image-smoke-jobs \
  --task-path /path/to/task \
  --image-cache /var/cache/marin-shellbox \
  --skopeo /usr/bin/skopeo \
  --qemu-assets-json /path/to/qemu-assets.json
```

The TaskTrove `nl2bash` fixture can be built on a Docker-capable build machine, copied to an OCI layout, and staged with `--task-dockerfile` pointing to its extracted `environment/Dockerfile`. Its unchanged Harbor task passes with:

```sh
uv run --no-project --python config/external/harbor/.venv/bin/python \
  lib/shellbox/tests/harbor_smoke.py /opt/marin-shellbox/nl2bash /tmp/nl2bash-jobs \
  --task-path /path/to/extracted/nl2bash \
  --command-file lib/shellbox/tests/manual/tasktrove_nl2bash_command.sh \
  --network-policy deny
```

This example uses a deterministic fake model to issue one Bash tool call. It exercises Harbor's external agent, task setup upload, QEMU command execution, and the real TaskTrove verifier; it does not evaluate model quality.

`tests/harbor_smoke.py --commands-json /path/to/commands.json` sends successive Bash calls from a JSON list. For example, the first call can run `cd /tmp; export ANSWER='hello from qemu'`, and the second can use `$ANSWER` and `$PWD`. `tests/manual/check_shell.py /opt/marin-shellbox/bash-image` checks interactive behavior without Harbor.

`benchmarks/tasktrove/sample.py` extracts a pinned 100-task TaskTrove Clean sample into a local directory. It needs `fsspec` and `pyarrow`:

```sh
uv run --no-project python lib/shellbox/benchmarks/tasktrove/sample.py /tmp/marin-shellbox-benchmark
```

Build one Docker image per `dockerfile_id` in `sample.json`, copy each image to an OCI layout, and stage it with its `environment/Dockerfile`. Name the images `tasktrove-clean-qemu:<dockerfile_id>` and the bundles `/tmp/marin-shellbox-benchmark/bundles/<dockerfile_id>`. Then run:

```sh
uv run --no-project --python config/external/harbor/.venv/bin/python \
  lib/shellbox/benchmarks/tasktrove/compare.py /tmp/marin-shellbox-benchmark --concurrency 2
```

The comparison runs the same empty and available oracle checks in offline Docker containers and fresh QEMU guests. It records one JSON result per check and fails on a verdict mismatch.

## Current support boundary

The guest has persistent files for a trial. A minimal bundle uses BusyBox `ash` for one-shot commands; an OCI bundle uses its image shell. Harbor setup and verifier commands still run as root in separate shell processes. The agent's `Bash` calls use one interactive `/bin/bash` per trial, so cwd, exports, functions, and jobs persist across calls. The verifier does not inherit that shell state. The guest has no network interface.

The QEMU image must contain the task tools and verifier dependencies needed without guest internet. `Bash` runs through a guest PTY on a separate virtio-serial channel. One `Bash` tool returns JSON with `output`, `status`, `exit_code`, and `truncated`. Pass `command` to start a command; omit it to read a running command, pass `input` to send text, or set `signal` to `interrupt` to send Ctrl-C. Use at most one of `command`, `input`, and `signal` per call. `Bash` waits at most 30 seconds per call; a longer command remains active for later reads or interruption. PTY stdout and stderr are combined. A tool result retains at most 128 KiB and marks excess output as truncated; excess output cannot be retrieved later. Agents can redirect large output to a file and inspect a smaller excerpt. `exit` or shell failure resets Bash while files remain. The one-shot serial protocol still buffers setup and verifier output in guest files. The bundle copies a full ext4 disk for each trial; this can be costly for large images. Extended attributes, Linux capabilities, device nodes outside `/dev`, and volume semantics have not been checked against an OCI runtime.

The guest is isolated by QEMU's emulated machine boundary, but this prototype has not been security audited. It is intended for nonhostile agents on an already controlled compute node.

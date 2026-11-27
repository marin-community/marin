# Tasks and containers

`shellbox.task.TaskSpec` describes one or more named `ContainerSpec` members and
names the primary container. Every member has a `MachineSpec`, optional named
endpoints and readiness probe. A single-container task uses this same model:
`LocalMachineFactory.create(machine_spec)` normalizes to a one-container
`TaskSpec` and calls `create_task`. The returned `Task` exposes all members through `containers` and
its default execution handle through `primary`.

The primary stays alive for machine commands. Supporting containers use their
image's ENTRYPOINT and CMD unless their container spec overrides them. Images
must already contain their dependencies. Startup overrides on the primary are
currently unsupported by these Shellbox backends. All members use the same
network policy; backend admission applies the sandbox restrictions separately.

`LocalMachineFactory.create_task` starts each member in bubblewrap. Members
share a private loopback network and keep separate filesystems, PID, IPC and
UTS namespaces. Container names resolve to `127.0.0.1` inside the group.
`group.endpoint("runtime", "http")` returns the declared private `runtime:8000`
address. Concurrent groups may use the same ports. Every member must request
`NetworkPolicy.DENY`; the group never falls back to host networking.

The Iris adapter supports single-container machine commands and file transfer;
it provides neither `create_task` nor process sessions.

Creation starts the containers and returns handles without waiting for health.
Start host capabilities before readiness when a container needs them:

```python
import asyncio

from shellbox.http_service import serve_http
from shellbox.task import ContainerSpec, ServiceEndpoint, TaskSpec

spec = TaskSpec(
    containers=(
        ContainerSpec("agent", task_machine_spec),
        ContainerSpec(
            name="runtime",
            machine=runtime_machine_spec,
            endpoints=(ServiceEndpoint("http", 8000),),
            readiness=runtime_readiness_probe,
        ),
    ),
    primary="agent",
)
group = await factory.create_task(spec)
try:
    async with asyncio.TaskGroup() as service_tasks, serve_http(
        group.primary, model_handler, task_group=service_tasks, port=18080
    ):
        await group.wait_ready()
        result = await group.primary.run(agent_command)
finally:
    await group.close()
```

`serve_http` starts one Python HTTP listener through a persistent process session.
The required `task_group` owns the listener pump and reports late helper failures to
its owner, even when startup ran in a completed phase task. Close the service
context before leaving that group.
The host handler receives bounded requests on that process's stdout and performs only the
operations it authorizes. Model URLs and authentication stay with the host
handler; a runtime service uses the group-local listener URL and a dummy API key.
The group needs no route to a model server or cloud storage. The listener requires
Python 3.8 or newer in its member image. It supports bounded nonstreaming GET and POST
requests, with no caller headers forwarded to the host handler. Request IDs allow
concurrent calls; responses remain whole messages, without SSE or token streaming.
There is no per-request exec or guest control endpoint.
Host-side Pydantic models validate and encode the wire messages, including
base64 bodies. The guest listener uses only the Python standard library.

Local, Docker, gVisor and Daytona expose `ProcessMachine.open_process`.
Its `ProcessSession` provides incremental binary `read`/`write`, bounded stderr,
exit status and explicit close. `Machine.run` still executes complete commands.
Pipe and SDK adapters share `StreamProcessSession` for read/write serialization,
bounded stderr and owned teardown. The adapter supplies its reader, writer, wait
and stop operations. Local and Docker use raw binary pipes; Daytona uses the
shared `process_framing` codec in its host and guest bridge because its SDK
transports text. HTTP request/response messages sit above this byte-stream API.
QEMU, ShellSim and Iris do not expose process sessions; online relays require a supported
backend and fail explicitly otherwise.

The same API works with `DockerMachineFactory` and `GvisorMachineFactory`.
Create the machine with `NetworkPolicy.DENY`, then enter `serve_http` before
running the application and close the relay before closing the machine:

```python
import asyncio

from shellbox.backends.docker.machine import DockerMachineFactory
from shellbox.http_service import serve_http
from shellbox.image import RegistryImage
from shellbox.machine import Command, MachineSpec, NetworkPolicy

machine = await DockerMachineFactory().create(
    MachineSpec(RegistryImage("python:3.12-slim"), network=NetworkPolicy.DENY)
)
try:
    async with asyncio.TaskGroup() as service_tasks, serve_http(
        machine, authorized_handler, task_group=service_tasks, port=8787
    ) as url:
        result = await machine.run(
            Command(("python3", "/app/worker.py"), env={"SERVICE_URL": url})
        )
finally:
    await machine.close()
```

`authorized_handler` is a trusted application function accepting `HttpRequest`
and returning `HttpResponse`. It must authorize each operation and select the
upstream destination before attaching any credentials. Treat guest methods,
paths and bodies as untrusted input. Credentials belong to that host function,
never to the machine environment or command arguments.

Docker retains `--network=none`: the listener uses guest loopback, while its
request and response frames cross one `docker exec` stdin/stdout connection. No published port,
host gateway or guest Docker socket is needed. The guest image needs Python 3
in addition to the Docker backend's normal command and transfer utilities.
Responses are buffered, with a 4 MiB body limit and at most 16 pending requests;
streaming and arbitrary caller headers are unsupported. A shared-loopback group
can share one listener; separate network namespaces need their own listeners.
This relay does not add Docker task-group execution or change plain Docker's
host-kernel isolation boundary.

`group.wait_ready()` runs each container's own readiness command with its declared
timeout, interval, start period and optional attempt limit. Timed readiness
requires Python 3 in each container image: a guest supervisor kills the probe's
process group when its attempt times out, allowing the next attempt without
stopping the runtime service. Exhausted attempts, the overall readiness timeout
or cancelled startup propagate to the task owner, which must close relay contexts
before closing the group. Closing any member, a member exiting,
or the job lifetime ending also tears down the group; containers are not reused
between trials. Download outputs through machine handles before closing them.

Without delegated cgroups, an ordinary command timeout or caller cancellation closes the whole local
machine or group; successful commands preserve background processes.
With delegation, interruption reclaims only the current command and descendants.
Before grading, stop agent command submission and call `LocalMachine.quiesce_commands()`
to reclaim work left by earlier successful commands, including tmux servers.
Process sessions, relay helpers and peer containers remain alive.

For agent commands that need a separate grading boundary, use
`ProcessScopeMachine` and call
`scope = await machine.open_process_scope()`, run those commands through
`scope.run(command)`, then `await scope.terminate()` before collection or
verification. Scope termination closes admission, kills detached descendants,
and waits for confirmed exit; it raises if confirmation fails. A scoped command
timeout or cancellation stops that command and its descendants; the scope and
background work from earlier commands remain available. Setup, sidecar,
and verifier commands use `machine.run` outside the scope. Local machines support
this operation when the worker has delegated cgroup v2. Docker, gVisor, Iris,
Daytona, QEMU, and ShellSim currently reject scope creation, so callers must
select a separate verifier machine or fail before the agent turn on those
backends. Declare `process_scope` in `ExecutionRequirements.capabilities` during
execution preflight to reject unsupported hosts before allocating a machine.
Ordinary `Machine.run` behavior is unchanged.

Docker and Daytona stop the command's process group on a timeout. A successful stop preserves the machine for later commands.
A failed stop without confirmed completion raises an infrastructure error and closes the machine.
Docker and Daytona also stop the process group on caller cancellation. Daytona preserves the machine only after confirming the stop and removing private command files; otherwise it closes the machine.

Daytona repeats a request that failed in transit: one that lost its connection or got a 502, 503, or 504 status.
A command whose request failed in transit runs once: the command script claims its private directory before it runs,
and the exit trap records the exit code that the machine then reads. The machine waits for that exit code until the command's timeout.
If the command shell was killed before it recorded an exit code, or the failure persists, `Machine.run` raises an infrastructure error.

Docker keeps the process-group ID on the host and stops commands as the execution user or image default user.
Before Docker machine reuse, a bounded root probe confirms that the group has no live members. Zombies do not prevent reuse.
Daytona records process-group leader PIDs as root in a root-only directory before the command changes users.
Daytona images require `setsid`. Non-root commands require an account in the task machine and util-linux `su --session-command`.
Model commands that run as root can stop other root-owned processes.

For Daytona, set `DAYTONA_API_KEY`. The API URL and target region are constants
(`https://app.daytona.io/api` and `us`).
Alternatively, supply a function that creates a configured `AsyncDaytona` client to `DaytonaMachineFactory`.
Each machine owns and closes its client. In the default build mode, the factory
creates sandboxes from shared snapshots. Snapshot names derive from the registry
reference or build-context contents and the requested resources. Use immutable
image references. A mutable tag can select a snapshot built from an earlier image.
Snapshot creation requires available organization quota.
Machine cleanup deletes the sandbox and retains the snapshot for reuse.
`DaytonaMachineFactory(auto_stop_interval=30)` sets the idle-stop interval in
minutes and configures Daytona to delete sandboxes immediately on stop. An open
machine renews activity at one-third of that interval because a running guest command does not
reset Daytona's idle timer. Closing the machine stops renewal before deleting
the sandbox. If the client process exits without cleanup, renewal stops and idle cleanup
reclaims the sandbox. `auto_stop_interval=0` disables idle stop and renewal.
`ttl_minutes` remains the hard wall-clock cap, defaulting to 360 minutes,
including while activity is renewed.
Pass `labels={"owner": "agent-eval", "lane": "eval", "job": "swe-rebench"}`
to attach ownership metadata to every sandbox the factory creates. Use a job
prefix and lane, without personal information or credentials. The default
labels contain only `owner=shellbox`; callers supply the lane and job labels.
The factory accepts `DaytonaNetworkPolicy` with `block_all`, `unrestricted`,
`network_allow_list`, or `domain_allow_list`. The two allow-list modes require a
`value`. Without a policy override, the factory uses `MachineSpec.network`.
`MachineSpec.startup_timeout` bounds image preparation and sandbox creation. Without
this value, the factory uses its `create_timeout` setting.
Daytona gives stdout and stderr collection a shared 60-second deadline after a
command exits. `DaytonaMachineFactory(output_collection_timeout=...)` changes
this deadline without changing the command timeout or cleanup timeouts. Harbor's
Daytona environment accepts the same `output_collection_timeout` setting.
The pinned Daytona SDK does not upload local `ADD` inputs. Dockerfiles with `ADD`
require a prebuilt registry image.
Other factories do not apply `MachineSpec.startup_timeout`. Callers enforce the
deadline for the complete create operation.

```python
from shellbox.backends.qemu.image import load_qemu_bundle
from shellbox.machine import Command, MachineSpec
from shellbox.backends.qemu.machine import QemuMachineFactory

machine = await QemuMachineFactory().create(MachineSpec(source=load_qemu_bundle(bundle_path)))
try:
    result = await machine.run(Command(("/bin/sh", "-c", "echo hello")))
finally:
    await machine.close()
```

For Docker, use `DockerMachineFactory()` and `DockerImage("image:tag")` in the same `MachineSpec`. This form requires a locally available image. Backends reject unsupported source types; QEMU also rejects guest networking.

`MachineSpec` accepts `cpus`, `memory_mb`, `storage_mb`, and `gpus`. Docker passes
these settings to its resource flags. Storage limits require a Docker storage
driver that supports `--storage-opt size`, and GPU allocation requires a GPU
runtime. Daytona applies CPU, memory, storage, and GPU settings through its SDK.
It rounds memory and storage up to whole GiB. Iris applies CPU, memory, and storage settings. QEMU applies CPU and
memory settings. Backends reject resource overrides that they cannot apply.

LOCAL can enforce declared storage with
`LocalMachineFactory(disk_root=Path("/cache/shellbox-disks"))`.
The factory verifies ext4 loop mounts, OverlayFS, ENOSPC and cleanup before its
first disk allocation. Verification runs off the event loop and is shared by
worker process, mount namespace and root directory identity in a bounded cache.
The filesystem charges writable image layers, scratch, workdirs, `/dev/shm` and
command spools. Shared read-only image layers remain outside it.

SkyRL requires an absolute `disk_root`. Its launcher provisions `e2fsprogs` when
absent from the pinned task image, requiring package-mirror access during startup.
Direct LOCAL workers install these host tools before creating a factory.

The worker requires Linux 5.8 or newer, root in a privileged pod,
`/dev/loop-control`, permission to create block device nodes, available loop devices, `mkfs.ext4`, `mount`, `umount`, `losetup`, `findmnt` and `umoci`. The
root must be root-owned mode 0700 in a private or slave mount namespace.
The guardian creates a missing node for the kernel-selected free minor, attaches
its own image atomically with autoclear enabled, and retains a device reference
until the mount succeeds. It never overwrites an existing node. Run
`lib/shellbox/tests/test_local_disk.py` explicitly with `-m manual` in that worker
to verify overflow, neighbor isolation, recovery and cleanup.

Guest commands that exhaust storage return their exit status and captured output.
A command that cannot allocate its spool or stdin raises `MachineDiskFull`, which
Harbor translates to `SandboxDiskFullError` for both LOCAL and Daytona. A full
disk does not latch a permanent failure: pipe-backed sessions can remove files,
and later commands can run once space is available. Quiescence and teardown also
remain available after exhaustion.

`Command.user` selects the execution user for one command. Docker accepts a
username or UID string. An omitted user retains the image's user. ShellSim and
QEMU accept only root overrides. Iris rejects user overrides. Daytona starts its
control process as root and uses `su` for other execution users. Numeric UIDs
require `getent` and a matching guest account.
The local backend runs other users only from a root process.
Daytona framework commands use `/usr/sbin:/usr/bin:/sbin:/bin` for `PATH`.
Agent commands retain the image's `PATH` and explicit task or command overrides.
The command deadline includes account and `su` capability probes.
Failed command-file cleanup closes the sandbox and preserves a primary command error.
An empty Docker `MachineSpec.workdir` retains the image's working directory.
Docker, gVisor, Iris and local machines create a nonempty `workdir` when the machine starts, so
commands can run there even when the image lacks it.

Docker and gVisor use the Docker daemon directly. `RegistryImage` pulls an image only when it is missing from Docker's cache. `DockerfileSource` builds with Docker's layer cache. `DockerImage` requires an image already present in the daemon. Private registry credentials come from Docker's login configuration.

QEMU needs image layers as files to construct its guest filesystem. It uses Skopeo to copy registry images into an OCI directory, then `umoci` to unpack them. Skopeo must be on `PATH`. To prepare a QEMU registry image, use a standard image reference without `https://`:

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
).create(spec)
```

The QEMU factory uses `ImageCache` to prepare each source once per process. Preparation selects `linux/amd64`, copies it to a local OCI layout and caches the layout by its selected manifest digest. Registry references remain snapshots within one process. For a private registry, pass a Skopeo `authfile` path to the factory. The default Skopeo policy accepts unsigned images over verified TLS; pass a Skopeo policy file path as `policy` for signature verification.

For QEMU, `DockerfileSource(context=..., dockerfile=...)` builds with a local Docker daemon and exports the result through Skopeo into the OCI cache. The cache fingerprints the Dockerfile and every path in its build context on each call. An unchanged context builds once per Python process; a changed context triggers a new build. Pin `FROM` by digest for repeatable builds. `QemuMachineFactory` accepts `PreparedImage` and needs `QemuAssets` and a `bundle_cache` to construct a guest bundle. `tests/manual/check_prepared_image.py` compares command output in Docker and QEMU for one registry image. The QEMU bundle cache key includes the prepared image digest, disk size, and caller-supplied `runtime_id`; change `runtime_id` when QEMU, kernel, BusyBox, firmware, or libraries change.

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

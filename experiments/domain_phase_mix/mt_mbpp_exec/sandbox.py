# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compile and run one MT-MBPP program in a disposable, network-free Docker container.

``run_program(language, source)`` writes ``source`` to a fixed file name in ``/sandbox`` of a fresh container, compiles
it when the language has a compile step, runs it with an empty stdin (immediate EOF), and removes the container. A
program passes if and only if every step exits 0 within its limit; ``timeout`` bounds the run step and
``compile_timeout`` the compile step. Build the image (native arm64, about 8 GB) once:

    cd experiments/domain_phase_mix/mt_mbpp_exec/sandbox
    docker build -t mt-mbpp-sandbox:$(shasum -a 256 Dockerfile | cut -c1-12) .

``IMAGE`` is derived from the Dockerfile's hash, so after an edit to the Dockerfile every run fails until the image is
rebuilt. ``/opt/sandbox/versions.txt`` in the image lists the resolved toolchain versions.

Isolation follows ``grade_table9_accuracy.sandbox_python``: no network, read-only root filesystem, every capability
dropped, no-new-privileges, uid 65534, no container logs, and ``docker rm -f`` in a ``finally``. Compilers need more
than that sandbox's 256 MB, so each container gets 2 GB of memory without swap, 1 CPU, 256 processes and threads, no
core dumps, and two 512 MB tmpfs mounts: ``/sandbox`` (working directory, exec so compiled binaries run) and ``/tmp``
(noexec; compiler scratch and captured output). Files in either mount count against the 2 GB.

Steps per language, run by bash in ``/sandbox`` (exact strings in ``LANGUAGES``):

- bash: ``bash main.sh`` (bash 5.3 with GNU coreutils, bc, gawk, jq and rev).
- c: ``gcc -std=gnu17 main.c -lm`` (GCC 15.2, -O0, so ``assert`` stays on).
- cpp: ``g++ -std=gnu++20 main.cpp`` (GCC 15.2, -O0; ``<bits/stdc++.h>`` is precompiled for exactly these flags).
- csharp: Roslyn ``csc`` from the .NET 10.0.112 SDK with ``-main:TestMain`` (the test class is the entry point even
  when the solution has its own ``Main``), ``DEBUG`` and ``TRACE`` defined so a failed ``Debug.Assert`` aborts, the
  implicit usings of ``dotnet new console``, and the .NET 10 reference assemblies; ``dotnet Program.dll`` runs it.
- go: ``go tool compile`` then ``go tool link`` (Go 1.27.1) against the standard library prebuilt into GOROOT; a plain
  ``go build`` would recompile the standard library into an empty build cache on every run.
- haskell: ``ghc -O0 Main.hs`` (GHC 9.10.3; -O would disable ``Control.Exception.assert``), with regex-tdfa,
  regex-posix, regex-compat, split, vector and unordered-containers installed.
- java: ``javac Main.java`` then ``java -ea Main`` (OpenJDK 25; ``-ea`` because assert statements are off by default).
- javascript: ``node main.js`` (Node 22.22.1; CommonJS, with ESM syntax detected automatically).
- matlab: ``octave-cli --norc --no-history --quiet main.m`` (GNU Octave 11.1), run as a script; a script that defines
  functions must not start with one (lead with ``1;``).
- php: ``php main.php`` (PHP 8.5) with ``zend.assertions=1``, because Ubuntu's php.ini compiles ``assert`` out, and
  errors on stderr. The source may open with ``<?php``.
- python: ``python3 -I main.py`` (Python 3.14).
- r: ``Rscript --vanilla main.R`` (R 4.5.2).
- ruby: ``ruby main.rb`` (Ruby 3.3).
- rust: ``rustc --edition 2021 main.rs`` (rustc 1.93.1), a debug build so overflow checks and ``debug_assert!`` are on;
  the regex, num-bigint, num-complex, num-integer and num-traits crates are available as ``--extern`` crates.
- scala: the Scala 3.9.0 compiler on JDK 25, then ``java testMain``: the tests are ``@main def testMain(): Unit``,
  named explicitly so another ``@main`` or ``App`` in the solution cannot take over. Top-level ``def`` is allowed.
- swift: ``swiftc -Onone main.swift`` (Swift 6.4; -O would remove ``assert``), with a clang module cache prebuilt for
  Foundation, Glibc, Dispatch, RegexBuilder and Synchronization and copied to a writable path, so other imports build.
- typescript: esbuild 0.28.2 strips the types (no type checking) into CommonJS, then ``node main.js`` runs it.

javac and the Scala compiler run with C1 only, SerialGC and a JDK AOT cache trained when the image is built, which
together cut a small file's compile from about 0.5 s to 0.16 s (javac) and 1.6 s to 0.9 s (Scala). The cache only
applies under the JVM flags it was trained with, so the Dockerfile's training commands repeat those in ``LANGUAGES``.
"""

import hashlib
import json
import subprocess
import uuid
from pathlib import Path

DOCKERFILE = Path(__file__).parent / "sandbox" / "Dockerfile"
IMAGE = "mt-mbpp-sandbox:" + hashlib.sha256(DOCKERFILE.read_bytes()).hexdigest()[:12]
TAIL_BYTES = 2048
DOCKER_SLACK = 30  # seconds allowed for container start, stop, and inspection beyond the step limits

# language -> (source file, compile command or "" for none, run command)
LANGUAGES = {
    "bash": ("main.sh", "", "bash main.sh"),
    "c": ("main.c", "gcc -std=gnu17 -o main main.c -lm", "./main"),
    "cpp": ("main.cpp", "g++ -std=gnu++20 -o main main.cpp", "./main"),
    "csharp": (
        "Program.cs",
        "csc -nologo -noconfig -nostdlib -langversion:latest -main:TestMain '-define:DEBUG;TRACE' -debug-"
        " @/opt/sandbox/csharp/references.rsp -out:Program.dll /opt/sandbox/csharp/GlobalUsings.cs Program.cs"
        " && cp /opt/sandbox/csharp/Program.runtimeconfig.json .",
        "dotnet Program.dll",
    ),
    "go": (
        "main.go",
        "go tool compile -p main -complete -importcfg /opt/sandbox/go.importcfg -pack -o main.a main.go"
        " && go tool link -importcfg /opt/sandbox/go.importcfg -o main main.a",
        "./main",
    ),
    "haskell": ("Main.hs", "ghc -O0 -v0 -outputdir /tmp/ghc -o main Main.hs", "./main"),
    "java": (
        "Main.java",
        "javac -J-XX:TieredStopAtLevel=1 -J-XX:+UseSerialGC -J-XX:-UsePerfData -J-XX:AOTCache=/opt/sandbox/javac.aot"
        " -encoding UTF-8 Main.java",
        "java -ea -XX:-UsePerfData -cp . Main",
    ),
    "javascript": ("main.js", "", "node main.js"),
    "matlab": ("main.m", "", "octave-cli --norc --no-history --quiet --no-window-system main.m"),
    "php": (
        "main.php",
        "",
        "php -d zend.assertions=1 -d assert.exception=1 -d display_errors=stderr -d log_errors=0 main.php",
    ),
    "python": ("main.py", "", "python3 -I main.py"),
    "r": ("main.R", "", "Rscript --vanilla main.R"),
    "ruby": ("main.rb", "", "ruby main.rb"),
    "rust": ("main.rs", "rustc --edition 2021 @/opt/sandbox/rust/args -o main main.rs", "./main"),
    "scala": (
        "Main.scala",
        "java -XX:TieredStopAtLevel=1 -XX:+UseSerialGC -XX:-UsePerfData -XX:AOTCache=/opt/sandbox/scalac.aot"
        " -cp /opt/scala3/lib/scala.jar -Dscala.expandjavacp=true -Dscala.usejavacp=true dotty.tools.dotc.Main"
        " -color:never -d . Main.scala",
        'java -XX:-UsePerfData -cp ".:$SCALA_LIBRARY" testMain',
    ),
    "swift": (
        "main.swift",
        "cp -R /opt/sandbox/swift-module-cache /tmp/swift-module-cache"
        " && swiftc -Onone -module-cache-path /tmp/swift-module-cache -o main main.swift",
        "./main",
    ),
    "typescript": (
        "main.ts",
        "esbuild main.ts --outfile=main.js --format=cjs --platform=node --target=node22 --log-level=warning",
        "node main.js",
    ),
}

# Runs inside the container as `bash -c DRIVER driver FILE COMPILE RUN COMPILE_LIMIT RUN_LIMIT`, with the source on
# stdin. Each step gets an empty stdin and a hard wall-clock limit; its output goes to files so a flood cannot reach
# the host. The first stdout line is the status, followed by the failing (or last) step's stdout tail; its stderr tail
# goes to stderr. The driver exits 0 only when every step exited 0; it exits without a status line only when it could
# not stage the source (status 3) or when a signal killed it.
DRIVER = f"""
cd /sandbox && cat > "$1" || exit 3
compile_ms=0 run_ms=0
step() {{
  local start=${{EPOCHREALTIME/./}}
  # The outer redirection only hides this shell's own "Killed"/"Aborted" job notices.
  {{ timeout -s KILL "$1" bash -c "$2" </dev/null >/tmp/stdout 2>/tmp/stderr; }} 2>/dev/null
  code=$?
  ms=$(( (${{EPOCHREALTIME/./}} - start) / 1000 ))
  timed_out=$(( code == 137 && ms >= $1 * 1000 ))
  # Kill whatever the step left behind (a fork bomb would starve the report); the driver is PID 1, so it is spared.
  kill -KILL -1 2>/dev/null
}}
report() {{
  echo "status phase=$1 exit=$code timeout=$timed_out compile_ms=$compile_ms run_ms=$run_ms"
  tail -c {TAIL_BYTES} /tmp/stdout
  tail -c {TAIL_BYTES} /tmp/stderr >&2
}}
if [ -n "$2" ]; then
  step "$4" "$2"
  compile_ms=$ms
  if [ "$code" -ne 0 ]; then report compile; exit 1; fi
fi
step "$5" "$3"
run_ms=$ms
report run
[ "$code" -eq 0 ]
"""


def container_command(name: str, language: str, timeout: int, compile_timeout: int) -> list[str]:
    filename, compile_step, run_step = LANGUAGES[language]
    return [
        "docker",
        "create",
        "--name",
        name,
        "--network",
        "none",
        "--read-only",
        "--log-driver",
        "none",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "256",
        "--memory",
        "2g",
        "--memory-swap",
        "2g",
        "--cpus",
        "1",
        "--ulimit",
        "core=0",
        "--user",
        "65534:65534",
        "--tmpfs",
        "/sandbox:rw,exec,nosuid,nodev,size=512m,mode=1777",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,nodev,size=512m,mode=1777",
        "--workdir",
        "/sandbox",
        "-i",
        IMAGE,
        "bash",
        "-c",
        DRIVER,
        "driver",
        filename,
        compile_step,
        run_step,
        str(compile_timeout),
        str(timeout),
    ]


def run_program(language: str, source: str, *, timeout: int = 10, compile_timeout: int = 60) -> dict:
    """Compile (if needed) and run ``source`` as a ``language`` program; raise if the sandbox itself fails.

    Returns ``passed``, ``phase`` (``compile`` when that step failed, else ``run``), ``exit_code`` (None on timeout),
    ``timeout``, ``stdout_tail`` and ``stderr_tail`` (the last 2 KB of that step's output), and ``compile_seconds``
    and ``run_seconds`` as measured inside the container.
    """
    name = "mt-mbpp-" + uuid.uuid4().hex
    try:
        subprocess.run(
            container_command(name, language, timeout, compile_timeout),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=True,
            timeout=DOCKER_SLACK,
        )
        try:
            completed = subprocess.run(
                ["docker", "start", "-ai", name],
                input=source.encode(),
                capture_output=True,
                timeout=compile_timeout + timeout + DOCKER_SLACK,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f"Sandbox {name} outlived its step limits") from error
        state = json.loads(
            subprocess.check_output(["docker", "inspect", "--format", "{{json .State}}", name], timeout=DOCKER_SLACK)
        )
        if state["Error"] or state["StartedAt"].startswith("0001-"):
            raise RuntimeError(f"Sandbox did not start: {state}")
        if state["Status"] != "exited":
            raise RuntimeError(f"Sandbox did not finish: {state}")
        stdout = completed.stdout.decode(errors="replace")
        stderr_tail = completed.stderr.decode(errors="replace")[-TAIL_BYTES:]
        status, _, stdout_tail = stdout.partition("\n")
        exit_status = state["ExitCode"]
        if not status.startswith("status "):
            if exit_status < 128:
                raise RuntimeError(f"Sandbox driver failed with exit status {exit_status}: {stderr_tail}")
            # A signal killed the driver before its report, e.g. the OOM killer, which only the program can provoke.
            return {
                "passed": False,
                "phase": "run",
                "exit_code": exit_status,
                "timeout": False,
                "stdout_tail": stdout[-TAIL_BYTES:],
                "stderr_tail": stderr_tail,
                "compile_seconds": None,
                "run_seconds": None,
            }
        fields = dict(field.split("=", 1) for field in status.split()[1:])
        timed_out = fields["timeout"] == "1"
        return {
            "passed": exit_status == 0 and fields["exit"] == "0",
            "phase": fields["phase"],
            "exit_code": None if timed_out else int(fields["exit"]),
            "timeout": timed_out,
            "stdout_tail": stdout_tail[-TAIL_BYTES:],
            "stderr_tail": stderr_tail,
            "compile_seconds": int(fields["compile_ms"]) / 1000,
            "run_seconds": int(fields["run_ms"]) / 1000,
        }
    finally:
        # The exact UUID names only this invocation's sandbox, including one that outlived its limits.
        subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

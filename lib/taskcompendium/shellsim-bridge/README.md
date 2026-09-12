# ShellSim bridge

This binary owns one persistent ShellSim interpreter and in-memory filesystem per
process. Cargo.lock and Cargo.toml pin the simulator and transitive dependencies.
Build with Rust's stable toolchain:

```sh
cargo build --locked --manifest-path lib/taskcompendium/shellsim-bridge/Cargo.toml
cargo test --locked --manifest-path lib/taskcompendium/shellsim-bridge/Cargo.toml
```

Pass `target/debug/taskcompendium-shellsim` within this directory to
`ShellSimSession`, or install the binary with `cargo install --locked --path
lib/taskcompendium/shellsim-bridge`. The Python session enforces a wall deadline
for each operation and kills the process on timeout or transport failure. Closing
the session discards its state.

The bridge accepts JSONL on stdin and emits one JSON object per request on stdout.
Responses have `{"ok":true,"result":...}` or `{"ok":false,"error":"..."}`.
The first request must initialize the cumulative trial resource budgets:

```json
{"op":"init","limits":{"cpu":10000000,"memory":67108864,"disk":67108864,"output":4194304}}
```

Subsequent requests use `exec` (command, optional base64 stdin, optional cwd and
env), `read` (path), `write` (path and base64 data), `mkdir` (path, recursive),
`list` (path, immediate names), `walk` (path, absolute recursive paths), `stat`
(path, is_dir/is_file), or `close`. `exec` returns base64 stdout/stderr,
return_code, stop_reason, and cumulative resource usage. Reads and writes retain
arbitrary binary content. Unknown commands are ShellSim errors; they never fall
through to a host shell.

Requests are limited to 16 MiB including the newline and responses to 32 MiB.
Individual reads are limited to 16 MiB before base64 encoding. VFS writes obey
the simulator's disk budget, including metadata. Shell changes to cwd, variables,
and functions persist between actions. An explicit cwd or env also persists.
Fuel/output exhaustion terminates simulated execution for the rest of the trial;
files remain readable for result collection. The upstream Linux seccomp backstop
is enabled when supported; it is not a host-filesystem sandbox. Filesystem
isolation comes from the simulator's VFS rather than host command execution.

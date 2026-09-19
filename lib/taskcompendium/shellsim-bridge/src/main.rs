//! One isolated, metered ShellSim environment per JSONL connection. No host shell dispatch.
use base64::{engine::general_purpose::STANDARD, Engine};
use serde::Deserialize;
use serde_json::{json, Value};
use shellsim::{vfs, Interp, Limits};
use std::collections::BTreeMap;
use std::io::{self, BufRead, Read, Write};

const MAX_REQUEST_BYTES: u64 = 16 * 1024 * 1024;
const MAX_RESPONSE_BYTES: usize = 32 * 1024 * 1024;

#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
enum Request {
    Init {
        limits: Limits,
    },
    Exec {
        command: String,
        #[serde(default)]
        stdin: String,
        cwd: Option<String>,
        #[serde(default)]
        env: BTreeMap<String, String>,
    },
    Read {
        path: String,
    },
    Write {
        path: String,
        data: String,
    },
    Mkdir {
        path: String,
    },
    List {
        path: String,
    },
    Walk {
        path: String,
    },
    Stat {
        path: String,
    },
    Close,
}

fn handle(state: &mut Option<Interp>, request: Request) -> Result<Value, String> {
    if let Request::Init { limits } = request {
        if state.is_some() {
            return Err("session already initialized".into());
        }
        if [limits.cpu, limits.memory, limits.disk, limits.output].contains(&0) {
            return Err("limits must be positive".into());
        }
        *state = Some(Interp::with_limits(limits));
        return Ok(json!({"revision": "5674a9492c35ffe390a0d23b49c0a340b12beb30"}));
    }
    let sim = state.as_mut().ok_or("session not initialized")?;
    let result = match request {
        Request::Exec {
            command,
            stdin,
            cwd,
            env,
        } => {
            let input = STANDARD.decode(stdin).map_err(|e| e.to_string())?;
            if let Some(cwd) = cwd {
                let abs = vfs::resolve_against(&sim.cwd, &cwd);
                if !sim.vfs.is_dir("/", &abs) {
                    return Err(format!("not a directory: {abs}"));
                }
                sim.cwd = abs.clone();
                sim.set_var("PWD", abs);
            }
            for (key, value) in env {
                sim.set_var(&key, value);
                sim.export(&key);
            }
            let (outcome, stdout, stderr) = sim.run_script_capture_with_stdin(&command, &input);
            json!({"stdout": STANDARD.encode(stdout), "stderr": STANDARD.encode(stderr),
                "return_code": outcome.exit_status, "stop_reason": outcome.stop_reason,
                "usage": outcome.usage})
        }
        Request::Read { path } => {
            let bytes = sim
                .vfs
                .read_limited(&sim.cwd, &path, (MAX_RESPONSE_BYTES / 2) as usize)
                .map_err(|e| e.to_string())?;
            json!({"data": STANDARD.encode(bytes)})
        }
        Request::Write { path, data } => {
            let bytes = STANDARD.decode(data).map_err(|e| e.to_string())?;
            let cwd = sim.cwd.clone();
            sim.vfs
                .write(&cwd, &path, &bytes, 0o644)
                .map_err(|e| e.to_string())?;
            json!({})
        }
        Request::Mkdir { path } => {
            let cwd = sim.cwd.clone();
            sim.vfs.mkdir_all(&cwd, &path).map_err(|e| e.to_string())?;
            json!({})
        }
        Request::List { path } => {
            json!({"paths": sim.vfs.list_dir(&sim.cwd, &path).map_err(|e| e.to_string())?})
        }
        Request::Walk { path } => {
            let abs = vfs::resolve_against(&sim.cwd, &path);
            let real = sim.vfs.realpath(&abs, true).map_err(|e| e.to_string())?;
            sim.vfs
                .metadata("/", &real, true)
                .map_err(|e| e.to_string())?;
            json!({"paths": sim.vfs.walk(&real)})
        }
        Request::Stat { path } => {
            json!({"is_dir": sim.vfs.is_dir(&sim.cwd, &path), "is_file": sim.vfs.is_file(&sim.cwd, &path)})
        }
        Request::Close => json!({}),
        Request::Init { .. } => unreachable!(),
    };
    Ok(result)
}

fn serve(reader: impl BufRead, mut writer: impl Write) -> io::Result<()> {
    let mut reader = reader;
    let mut state = None;
    loop {
        let mut line = Vec::new();
        let count = reader
            .by_ref()
            .take(MAX_REQUEST_BYTES + 1)
            .read_until(b'\n', &mut line)?;
        if count == 0 {
            return Ok(());
        }
        if count as u64 > MAX_REQUEST_BYTES {
            writeln!(
                writer,
                "{{\"ok\":false,\"error\":\"request exceeds byte limit\"}}"
            )?;
            writer.flush()?;
            return Ok(());
        }
        let parsed = serde_json::from_slice::<Request>(&line);
        let close = matches!(parsed, Ok(Request::Close));
        let response = match parsed
            .map_err(|e| e.to_string())
            .and_then(|r| handle(&mut state, r))
        {
            Ok(result) => json!({"ok": true, "result": result}),
            Err(error) => json!({"ok": false, "error": error}),
        };
        let serialized = serde_json::to_vec(&response)?;
        if serialized.len() > MAX_RESPONSE_BYTES {
            writeln!(
                writer,
                "{{\"ok\":false,\"error\":\"response exceeds byte limit\"}}"
            )?;
        } else {
            writer.write_all(&serialized)?;
            writer.write_all(b"\n")?;
        }
        writer.flush()?;
        if close {
            return Ok(());
        }
    }
}

fn main() -> io::Result<()> {
    shellsim::sandbox::apply();
    serve(io::stdin().lock(), io::stdout().lock())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_persists_and_host_commands_do_not_execute() {
        let requests = concat!(
            "{\"op\":\"init\",\"limits\":{\"cpu\":10000,\"memory\":1048576,\"disk\":1048576,\"output\":4096}}\n",
            "{\"op\":\"exec\",\"command\":\"echo saved > /work/value\"}\n",
            "{\"op\":\"read\",\"path\":\"/work/value\"}\n",
            "{\"op\":\"exec\",\"command\":\"/usr/bin/osascript -e nonsense\"}\n"
        );
        let mut output = Vec::new();
        serve(requests.as_bytes(), &mut output).unwrap();
        let responses: Vec<Value> = output
            .split(|b| *b == b'\n')
            .filter(|s| !s.is_empty())
            .map(|line| serde_json::from_slice(line).unwrap())
            .collect();
        assert_eq!(responses[2]["result"]["data"], STANDARD.encode("saved\n"));
        assert_eq!(responses[3]["result"]["return_code"], 127);
    }
}

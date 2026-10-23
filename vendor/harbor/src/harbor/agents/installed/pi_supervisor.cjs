// Pi tools can start new sessions. Stop the process tree, including those tools,
// before returning control to Harbor's verifier.
const fs = require("node:fs");
const { spawn } = require("node:child_process");

const [control, budget, command] = process.argv.slice(2);
fs.mkdirSync(control, { recursive: true });
const cancelled = () => fs.existsSync(`${control}/cancel`);
let child;
let timer;
let stopping = false;

function signal(pid, name) {
  try {
    process.kill(pid, name);
    return true;
  } catch (error) {
    if (error.code !== "ESRCH") throw error;
    return false;
  }
}

function stopTree(pid) {
  // Stop each parent before enumerating its children so it cannot fork or exit
  // while we walk the tree. Detached children still have their original PPID.
  if (!signal(pid, "SIGSTOP")) return;
  for (const entry of fs.readdirSync("/proc")) {
    if (!/^\d+$/.test(entry)) continue;
    let stat;
    try {
      stat = fs.readFileSync(`/proc/${entry}/stat`, "utf8");
    } catch (error) {
      if (error.code === "ENOENT" || error.code === "ESRCH") continue;
      throw error;
    }
    const fields = stat.slice(stat.lastIndexOf(")") + 2).split(" ");
    if (Number(fields[1]) === pid) stopTree(Number(entry));
  }
  signal(pid, "SIGKILL");
}

function finish(code) {
  clearTimeout(timer);
  fs.rmSync(control, { recursive: true, force: true });
  process.exit(code);
}

function stop(code) {
  if (stopping) return;
  stopping = true;
  if (!child) return finish(code);
  try {
    stopTree(child.pid);
  } catch (error) {
    console.error("HARBOR_PI_CLEANUP_FAILED");
    console.error(error.message);
    // Retain the control directory as evidence that cleanup did not complete.
    process.exit(1);
  }
  child.once("exit", () => finish(code));
}

process.on("SIGTERM", () => stop(143));
process.on("SIGINT", () => stop(130));
// A cancellation arriving before startup leaves a marker. Check on both sides
// of publishing the PID so cancellation cannot miss a not-yet-started command.
if (cancelled()) finish(143);
fs.writeFileSync(`${control}/pid`, `${process.pid}\n`);
if (cancelled()) finish(143);
child = spawn("bash", ["-o", "pipefail", "-c", command], {
  stdio: "inherit",
});
child.on("error", (error) => {
  console.error(error.message);
  finish(1);
});
child.on("exit", (code, signalName) => {
  if (!stopping) {
    const signals = require("node:os").constants.signals;
    finish(code ?? 128 + signals[signalName]);
  }
});
if (budget !== "none") {
  timer = setTimeout(() => {
    console.error("HARBOR_PI_AGENT_TIMEOUT");
    stop(124);
  }, Number(budget) * 1000);
}

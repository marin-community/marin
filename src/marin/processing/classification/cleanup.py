import ray
import os
import time
import signal

ACCEL_PREFIXES = ("/dev/accel", "/dev/vfio")


def _descendants_of(root_pid: int) -> set[int]:
    # Build PPID -> [PID...] table from /proc and walk it.
    ppid_children = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        try:
            with open(f"/proc/{pid}/status") as f:
                lines = f.read().splitlines()
            kv = dict(line.split(":\t", 1) for line in lines if line.count(":\t") == 1)
            ppid = int(kv.get("PPid", "0"))
        except Exception:
            continue
        ppid_children.setdefault(ppid, []).append(pid)

    out, stack = set(), [root_pid]
    while stack:
        p = stack.pop()
        for c in ppid_children.get(p, []):
            if c not in out:
                out.add(c)
                stack.append(c)
    return out


def _holds_accel_fd(pid: int) -> bool:
    fd_dir = f"/proc/{pid}/fd"
    try:
        for fd in os.listdir(fd_dir):
            path = os.path.join(fd_dir, fd)
            try:
                target = os.readlink(path)
            except FileNotFoundError:
                continue
            except PermissionError:
                # Try to stat as fallback
                try:
                    _ = os.stat(path)
                    # device nodes will show up as character devices sometimes
                except Exception:
                    continue
                continue
            if any(target.startswith(pref) for pref in ACCEL_PREFIXES):
                return True
    except FileNotFoundError:
        return False
    return False


def _terminate(pid: int, timeout_s: float = 5.0) -> bool:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    # Wait a bit
    end = time.time() + timeout_s
    while time.time() < end:
        if not os.path.exists(f"/proc/{pid}"):
            return True
        time.sleep(0.1)
    # Escalate
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return True
    # Final confirm
    for _ in range(30):
        if not os.path.exists(f"/proc/{pid}"):
            return True
        time.sleep(0.1)
    return False


@ray.remote(num_cpus=0.1)
class Reaper:
    def kill_if_holding_accel(self, root_pid: int, include_children: bool = True):
        victims = {root_pid}
        if include_children:
            victims |= _descendants_of(root_pid)
        holding = [p for p in sorted(victims) if _holds_accel_fd(p)]
        killed, failed = [], []
        for p in holding:
            print(f"Found holding process: {p}")
            ok = _terminate(p)
            (killed if ok else failed).append(p)

        results = {
            "node_id": ray.get_runtime_context().get_node_id(),
            "root_pid": root_pid,
            "checked": sorted(victims),
            "holding": holding,
            "killed": killed,
            "failed": failed,
        }
        print(results)
        return results

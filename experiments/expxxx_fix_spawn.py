import ray
from vllm import LLM, SamplingParams
import os
import time
import signal
from ray.util.queue import Queue

ACCEL_PREFIXES = ("/dev/accel", "/dev/tpu", "/dev/nvidia")


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
            "checked": sorted(victims),
            "holding": holding,
            "killed": killed,
            "failed": failed,
        }
        print(results)
        return results


@ray.remote(max_calls=1)
def test_llm(queue: Queue):
    info = {
        "node_id": ray.get_runtime_context().get_node_id(),
        "pid": os.getpid(),
    }
    queue.put(info)
    llm = LLM(
        model="/opt/gcsfuse_mount/models/meta-llama--Llama-3-2-3B-Instruct--0cb88a4",
        max_model_len=1024,
        tensor_parallel_size=1,
        enforce_eager=True,
    )
    outputs = llm.generate(
        "Write a 1024 character long story about a cat driving into the forest.",
        sampling_params=SamplingParams(temperature=0.0, max_tokens=1024),
    )

    return outputs


@ray.remote(resources={"TPU-v4-8-head": 1})
def test_llms_on_same_node():
    node_id = ray.get_runtime_context().get_node_id()

    # Comment out for now while testing Reaper
    # _kill_processes_by_regex_if_no_vllm("test_llm")

    # Same Node
    futures = []
    queue = Queue()
    for _ in range(4):
        futures.append(
            test_llm.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(node_id, soft=False),
                resources={"TPU": 1},
            ).remote(queue)
        )

    for _ in range(4):
        info = queue.get()
        reaper = Reaper.options(
            scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(node_id, soft=False)
        ).remote()
        futures.append(reaper.kill_if_holding_accel.remote(info["pid"], include_children=True))

    ray.get(futures)
    # Same node, should error but DOES NOT
    # ray.get(test_llm.options(scheduling_strategy=ray.util.scheduling_strategies.
    # NodeAffinitySchedulingStrategy(node_id, soft=False), resources={"TPU": 4}).remote())


if __name__ == "__main__":
    ray.get(test_llms_on_same_node.remote())

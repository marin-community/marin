"""
test_ray_cluster.py

Simple debugging script that attempts to run 4096 tasks on our Ray cluster, verifying that all nodes can successfully
read from GCS (specifically `gs://marin-data/scratch/siddk/hello-ray.txt`).

Run with:
  - [Local] python tests/test_ray_cluster.py
  - [Ray] ray job submit --no-wait --address=http://127.0.0.1:8265 --working-dir . -- python tests/test_ray_cluster.py
        => Assumes that `ray dashboard infra/marin-cluster.yaml` running in a separate terminal (port forwarding)!
"""

import socket
import time

import fsspec
import ray

# === Constants ===
GCS_DEBUG_FILE_PATH = "gs://marin-data/scratch/siddk/hello-ray.txt"
N_TASKS = 4096


@ray.remote
def test_ray_cluster() -> dict[str, str]:
    """Read from `GCS_DEBUG_FILE_PATH` and return {"ip": <Worker IP>, "content": <GCS_FILE_CONTENT>}."""
    with fsspec.open(GCS_DEBUG_FILE_PATH, "r") as f:
        content = f.read()

    # Sleep to force schedule on multiple nodes
    time.sleep(0.1)

    return {"ip": socket.gethostbyname(socket.gethostname()), "content": content.strip()}


def main() -> None:
    
    print(f"[*] Launching {N_TASKS} Verification Tasks on Ray Cluster!")

    # Print Cluster Information
    print(f"[*] Cluster Statistics :: {len(ray.nodes())} nodes w/ {ray.cluster_resources().get('CPU', 0)} total CPUs")

    # Invoke Tasks (call .remote() --> return *promises* -- a list of references)
    print(f"[*] Invoking {N_TASKS} Verification Tasks...")
    output_refs = [test_ray_cluster.remote() for _ in range(N_TASKS)]

    # Resolve references (actually get return result from `test_ray_cluster()`)
    print("[*] Getting Job Results...")
    outputs = ray.get(output_refs)

    # Run Verification
    unique_ips = set()
    for output_dict in outputs:
        assert output_dict["content"].strip() == "Hello World!", f"Unexpected output `{output_dict['content'] = }"
        unique_ips.add(output_dict["ip"])

    print(f"[*] Job Successfully Executed over {len(unique_ips)} Unique IPs: {unique_ips}!")


if __name__ == "__main__":
    ray.init()
    main()

import os
from pathlib import Path


def is_local_ray_cluster():
    address = os.environ.get("RAY_ADDRESS")
    if isinstance(address, str):
        address = address.strip()
    cluster_file = Path("/tmp/ray/ray_current_cluster")
    return (address is None and not cluster_file.exists()) or address == "local"

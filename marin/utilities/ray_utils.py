import os
from pathlib import Path


def is_local_ray_cluster():
    address = os.environ.get("RAY_ADDRESS")
    if isinstance(address, str):
        address = address.strip()
    cluster_file = Path("/tmp/ray/ray_current_cluster")
    print(f"RAY_ADDRESS: '{address}', cluster_file exists: {cluster_file.exists()}")
    print(f"{not address}, {cluster_file.exists()}, {address == 'local'}")
    ret = (not address and not cluster_file.exists()) or address == "local"
    print(f"RAY_ADDRESS: '{address}', ret {ret}")
    return ret

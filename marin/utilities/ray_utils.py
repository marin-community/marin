import os


def is_local_ray_cluster():
    address = os.environ.get("RAY_ADDRESS")
    return address is None or address == "local"

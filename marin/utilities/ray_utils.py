import os


def is_local_ray_cluster():
    return os.environ.get("RAY_JOB_ID") is None

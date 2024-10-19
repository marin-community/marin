import os


def is_local_ray_cluster():
    job_id = os.environ.get("RAY_JOB_ID")
    return job_id is None or job_id == "ffffffff"

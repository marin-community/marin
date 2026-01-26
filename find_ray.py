import subprocess
import os
from ray.job_submission import JobSubmissionClient

RAY_ADDRESS = "http://10.164.0.176:8265" # http://10.130.0.38:8265
client = JobSubmissionClient(RAY_ADDRESS)

jobs = client.list_jobs()
vlm_jobs = [job for job in jobs if
            (job.job_id and "VLM" in job.job_id.upper()) or
            (job.entrypoint and "VLM" in job.entrypoint.upper())]

# 只显示 RUNNING 的
running_jobs = [job for job in vlm_jobs if job.status == "RUNNING"]
print(f"Found {len(running_jobs)} RUNNING VLM jobs:\n")
running_jobs=running_jobs
for job in running_jobs:
    # 尝试获取 submission_id，这是停止 job 需要的
    # 从 runtime_env 提取 EXP_NAME 环境变量
    job_name = 'N/A'
    if job.runtime_env and isinstance(job.runtime_env, dict):
        env_vars = job.runtime_env.get('env_vars', {})
        if isinstance(env_vars, dict):
            job_name = env_vars.get('EXP_NAME', 'N/A')
    print(f"job_id: {job.job_id}, name: {job_name}, submission_id: {job.submission_id}, Status: {job.status}")
# assert 1==2
# 停止所有 RUNNING jobs
print("\n--- Stopping jobs ---")
env = os.environ.copy()
env["RAY_ADDRESS"] = RAY_ADDRESS

for job in running_jobs:
    # 优先使用 submission_id，否则用 job_id
    stop_id = job.submission_id or job.job_id
    job_name = 'N/A'
    if job.runtime_env and isinstance(job.runtime_env, dict):
        env_vars = job.runtime_env.get('env_vars', {})
        if isinstance(env_vars, dict):
            job_name = env_vars.get('EXP_NAME', 'N/A')
    if stop_id:
        print(f"Stopping {stop_id} (name: {job_name})...")
        confirm = input(f"  Are you sure you want to stop {stop_id} (name: {job_name})? (y/n): ")
        if confirm.lower() != 'y':
            print(f"  Skipped.")
            continue
        try:
            result = subprocess.run(
                ["ray", "job", "stop", stop_id],
                env=env,
                capture_output=True,
                text=True,
                timeout=30
            )
            print(f"  stdout: {result.stdout.strip()}")
            if result.stderr:
                print(f"  stderr: {result.stderr.strip()}")
        except Exception as e:
            print(f"  Error: {e}")
    else:
        print(f"Cannot stop job with no ID (entrypoint: {job.entrypoint})")

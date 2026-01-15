# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
A wrapper for submitting and monitoring Marin jobs on SLURM clusters.

Usage:
    python slurm_run.py [options] your_command args

Examples:
    # Basic usage
    python slurm_run.py python train.py --dataset imagenet

    # Custom SLURM configuration
    python slurm_run.py --slurm time 24:00:00 --slurm mem 128G python train.py

    # Environment variables
    python slurm_run.py -e WANDB_ENTITY my_entity python train.py

    # Custom job name and no monitoring
    python slurm_run.py --job_name training_run --no_wait python train.py

    # Dry run to preview the generated SLURM script
    python slurm_run.py --dry_run python train.py

Arguments:
    cmd                       The command to run on the SLURM cluster

Options:
    --no_wait                 Submit the job and return immediately without waiting for completion
    --dry_run                 Print the generated SLURM script without submitting
    --venv_path PATH          Path to the virtual environment to activate (default: .venv)
    --env_vars, -e KEY VALUE  Set environment variables for the job
    --slurm, -s KEY VALUE     Set SLURM options, overriding defaults
    --job_name NAME           Set a custom name for the SLURM job
"""

import argparse
import logging
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time

from huggingface_hub import HfFolder

import wandb

# Setup logger
logger = logging.getLogger("slurm")

# Default configuration
DEFAULT_ENV_VARS = {
    "PYTHONUNBUFFERED": "1",
}

DEFAULT_PIP_DEPS = []

# SLURM specific defaults
DEFAULT_SLURM_ARGS = {
    "job-name": "marin_run",
    "output": "logs/marin-%j.out",
    "error": "logs/marin-%j.err",
    "time": "48:00:00",
    "mem": "200G",
    "gres": "gpu:1",
    "account": "nlp",
    "partition": "sc-loprio",
    "constraint": "[40G|48G|80G|141G]",
    "nodes": "1",
    "ntasks-per-node": "1",
    "cpus-per-task": "64",
}


def generate_pythonpath(base_dir="submodules"):
    paths = []
    if not os.path.exists(base_dir):
        logger.warning(f"Base directory {base_dir} does not exist.")
        return ""
    for submodule in os.listdir(base_dir):
        submodule_path = os.path.join(base_dir, submodule)
        if os.path.isdir(submodule_path):
            paths.append(submodule_path)
            src_path = os.path.join(submodule_path, "src")
            if os.path.isdir(src_path):
                paths.append(src_path)
    return ":".join(paths)


def create_sbatch_script(
    command: str,
    env_vars: dict[str, str],
    slurm_args: dict[str, str],
    job_name: str | None = None,
    venv_path: str = ".venv",
) -> str:
    if not job_name:
        job_name = f"job_{int(time.time())}"
    script = "#!/bin/bash\n\n"
    if "job-name" not in slurm_args:
        script += f"#SBATCH --job-name={job_name}\n"
    for key, value in slurm_args.items():
        script += f"#SBATCH --{key}={value}\n"
    script += "\n"
    for key, value in env_vars.items():
        script += f'export {key}="{value}"\n'
    current_dir = os.getcwd()
    wandb_api = wandb.Api()
    if "WANDB_ENTITY" not in env_vars:
        script += f"export WANDB_ENTITY={wandb_api.default_entity}\n"
    if "HF_TOKEN" not in env_vars:
        script += f"export HF_TOKEN={HfFolder.get_token()}\n"
    script += f"\n# Set working directory\ncd {current_dir}\n"
    script += "\n# Activate virtual environment\n"
    script += f"if [ -f {venv_path}/bin/activate ]; then\n"
    script += f"  source {venv_path}/bin/activate\n"
    script += "else\n"
    script += f"  echo 'Warning: Virtual environment {venv_path} not found. Continuing without activation.'\n"
    script += "fi\n\n"
    script += "# Run the command\n"
    script += f"srun {command}\n"
    return script


def submit_slurm_job(sbatch_script: str, no_wait: bool, slurm_args: dict[str, str]) -> str:
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".sh") as temp_file:
        temp_file.write(sbatch_script)
        script_path = temp_file.name
    try:
        logger.info(f"Submitting job with script: {script_path}")
        submit_cmd = ["sbatch", script_path]
        try:
            result = subprocess.run(submit_cmd, capture_output=True, text=True, check=True)
            # Process successful result
            print("Job submitted successfully!")
            print(f"Output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            # The command failed (non-zero return code)
            print(f"Command failed with return code {e.returncode}")
            print(f"Error output: {e.stderr}")
            print(f"Standard output: {e.stdout}")
            raise e
        job_id_match = re.search(r"Submitted batch job (\d+)", result.stdout)
        if not job_id_match:
            logger.error(f"Failed to parse job ID from output: {result.stdout}")
            return ""
        job_id = job_id_match.group(1)
        logger.info(f"Job submitted with ID: {job_id}")
        if no_wait:
            return job_id

        # Get the output log file path
        output_log_path = slurm_args.get("output", "slurm-%j.out").replace("%j", job_id).replace("%A", job_id)
        error_log_path = slurm_args.get("error", "slurm-%j.err").replace("%j", job_id).replace("%A", job_id)

        # Also replace array job placeholder with the main job ID if it exists
        output_log_path = output_log_path.replace("%a", "0")
        error_log_path = error_log_path.replace("%a", "0")

        logger.info(f"Job output will be written to: {output_log_path}")
        logger.info(f"Job errors will be written to: {error_log_path}")

        # Wait for the log file to be created
        while not os.path.exists(output_log_path):
            logger.info(f"Waiting for log file to be created: {output_log_path}")
            time.sleep(2)

            # Check if the job is still in the queue
            check_cmd = ["squeue", "-j", job_id, "-h"]
            check_result = subprocess.run(check_cmd, capture_output=True, text=True)
            if not check_result.stdout.strip():
                logger.warning(f"Job {job_id} completed or failed before log file was created")
                break

        # Start tailing the log file
        if os.path.exists(error_log_path):
            logger.info(f"Tailing log file: {error_log_path}")
            tail_process = subprocess.Popen(["tail", "-f", error_log_path])

            # Monitor job until completion
            try:
                while True:
                    # Check if job is still running
                    check_cmd = ["squeue", "-j", job_id, "-h"]
                    check_result = subprocess.run(check_cmd, capture_output=True, text=True)
                    if not check_result.stdout.strip():
                        logger.info(f"Job {job_id} completed")
                        break

                    # Wait before checking again
                    time.sleep(5)

                # Wait a few more seconds to catch any remaining output
                time.sleep(3)

            finally:
                # Terminate the tail process
                tail_process.terminate()

            # Output final job status
            output_cmd = ["sacct", "-j", job_id, "--format=JobID,JobName,State,ExitCode,Elapsed"]
            subprocess.run(output_cmd)
        else:
            logger.warning(f"Log file {output_log_path} was never created")
            # Monitor job until completion without tailing
            while True:
                check_cmd = ["squeue", "-j", job_id, "-h"]
                check_result = subprocess.run(check_cmd, capture_output=True, text=True)
                if not check_result.stdout.strip():
                    logger.info(f"Job {job_id} completed")
                    break
                time.sleep(5)

            # Output final job status
            output_cmd = ["sacct", "-j", job_id, "--format=JobID,JobName,State,ExitCode,Elapsed"]
            subprocess.run(output_cmd)

        return job_id
    finally:
        os.unlink(script_path)


def parse_slurm_args(args_list: list[list[str]]) -> dict[str, str]:
    slurm_args = DEFAULT_SLURM_ARGS.copy()
    if not args_list:
        return slurm_args
    for item in args_list:
        if len(item) != 2:
            logger.error(f"Invalid SLURM argument format: {item}. Expected 'KEY VALUE' format.")
            sys.exit(1)
        key, value = item
        slurm_args[key] = value
    return slurm_args


def main():
    parser = argparse.ArgumentParser(description="Submit SLURM jobs using the command-line.")
    parser.add_argument("--no_wait", action="store_true", help="Do not wait for the job to finish.")
    parser.add_argument("--dry_run", action="store_true", help="Print the SLURM script without submitting the job.")
    parser.add_argument("--venv_path", type=str, default=".venv", help="Path to the virtual environment to activate.")
    parser.add_argument(
        "--env_vars", "-e", action="append", nargs="+", metavar=("KEY", "VALUE"), help="Set environment variables."
    )
    parser.add_argument(
        "--slurm", "-s", action="append", nargs=2, metavar=("OPTION", "VALUE"), help="Set SLURM options."
    )
    parser.add_argument("--job_name", type=str, help="Set the name of the SLURM job")
    parser.add_argument("cmd", help="The command to run in the SLURM cluster.", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    full_cmd = " ".join(shlex.quote(arg) for arg in args.cmd).strip()
    if not full_cmd:
        logger.error("No command provided.")
        sys.exit(1)
    if full_cmd.startswith("--"):
        full_cmd = full_cmd[2:].strip()
    env_vars = DEFAULT_ENV_VARS.copy()
    if args.env_vars:
        for item in args.env_vars:
            if len(item) > 2:
                logger.error(f"Too many values for environment variable: {' '.join(item)}")
                sys.exit(1)
            elif len(item) == 1:
                if "=" in item[0]:
                    logger.error("Invalid format. Use -e KEY VALUE, not -e KEY=VALUE")
                    sys.exit(1)
                env_vars[item[0]] = ""
            else:
                if "=" in item[0]:
                    logger.error("Invalid format. Use -e KEY VALUE, not -e KEY=VALUE")
                    sys.exit(1)
                env_vars[item[0]] = item[1]
    existing_path = env_vars.get("PYTHONPATH", "")
    new_path = generate_pythonpath()
    env_vars["PYTHONPATH"] = f"{new_path}:{existing_path}" if existing_path else new_path
    slurm_args = parse_slurm_args(args.slurm)
    sbatch_script = create_sbatch_script(
        command=full_cmd,
        env_vars=env_vars,
        slurm_args=slurm_args,
        job_name=args.job_name,
        venv_path=args.venv_path,
    )
    if args.dry_run:
        print("#" * 60)
        print("# DRY RUN MODE: The following script would be submitted")
        print("#" * 60)
        print(sbatch_script)
        return
    logger.info("Generated SLURM script:")
    for i, line in enumerate(sbatch_script.split("\n")):
        logger.info(f"{i + 1}: {line}")
    job_id = submit_slurm_job(sbatch_script, args.no_wait, slurm_args)
    if job_id:
        logger.info(f"Job submitted successfully with ID: {job_id}")
        logger.info(f"To check job status: squeue -j {job_id}")
        logger.info(f"To see detailed job info: sacct -j {job_id} --format=JobID,JobName,State,ExitCode,Elapsed")
    else:
        logger.error("Failed to submit job")
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()

import os
import tempfile
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

import draccus
import levanter.infra.cli_helpers as cli
import levanter.infra.docker
import mergedeep
import yaml
from levanter.infra import docker
from ray.dashboard.modules.job.common import JobStatus
from ray.dashboard.modules.job.sdk import JobSubmissionClient

zone_to_bucket = {
    "us-central2-b": "marin-us-central2",
    "us-west4-a": "marin-us-west4",
    "eu-west4-b": "marin-eu-west4",
}


def construct_levanter_config(
    base_config: dict,
    model_config: Optional[dict],
    data_config: dict,
    cache_dir: str,
    bucket: str,
    id: str,
    exp_name: str,
    name: str,
    tags: list[str],
):
    config = deepcopy(base_config)

    if model_config is not None:
        config["model"] = model_config

    base_data = config["data"]

    mergedeep.merge(base_data, data_config)
    base_data["cache_dir"] = cache_dir

    trainer_config = config["trainer"]
    trainer_config["id"] = id
    trainer_config["tracker"]["name"] = name
    trainer_config["tracker"]["tags"] = list(trainer_config["tracker"].get("tags", [])) + list(tags) + [exp_name]
    trainer_config["checkpointer"]["base_path"] = f"gs://{bucket}/checkpoints/{exp_name}/{id}/"
    config["hf_save_path"] = f"gs://{bucket}/checkpoints/{exp_name}/{id}/hf/"
    config["hf_save_steps"] = 200

    return config


def _get_data_config(
    base_data: dict,
    data_name: Optional[str],
    dataset_path: Optional[str],
    data_config_path: Optional[str],
    tokenizer: str,
) -> dict:
    """
    We support a few different kinds of data configurations. One option is a YAML file that specifies a data mixture,
    following Levanter's data mixture config. Another option is a string that specifies a root directory for a dataset.

    For the former, we merge the data config into the base's data config, with the following change:
        - All weights for datasets not in the new config are set to 0, meaning they are only used for evaluation.

    For the latter, we add a new dataset to the mixture with the specified root directory, grabbing
    all jsonl.gz files recursively. We then set the weight of that dataset to 1, and the weight of all other datasets
    to 0, meaning they are only used for evaluation.
    """
    assert (data_name is None) != (data_config_path is None)
    assert (data_name is None) == (dataset_path is None)

    ret_data = deepcopy(base_data)
    ret_data["tokenizer"] = tokenizer

    if data_config_path is not None:
        data_config_path = yaml.load(open(data_config_path), Loader=yaml.SafeLoader)
        mergedeep.merge(ret_data, data_config_path)

        for key in ret_data["train_weights"]:
            if key not in data_config_path["train_weights"]:
                ret_data["train_weights"][key] = 0
    else:
        # TODO: this isn't really right, but it doesn't matter if you pre-run tokenization
        # I don't love this.
        ret_data["configs"][data_name] = {"train_urls": [dataset_path]}
        for key in ret_data["train_weights"]:
            ret_data["train_weights"][key] = 0

        ret_data["train_weights"][data_name] = 1

    return ret_data


@dataclass
class LaunchConfig:
    experiment: str
    cache_dir: str
    """Tokenizer cache dir"""

    address: Optional[str] = None
    """The address of the Ray dashboard. If not set, RAY_ADDRESS will be used."""


    dataset_name: Optional[str] = None
    """This should be the name of the dataset you tokenized in the tokenization step. Either (this and dataset_path) or dataset_config must be provided."""
    dataset_path: Optional[str] = None
    """This should be the path to a directory containing a dataset. Either (this and dataset_name) or dataset_config must be provided."""
    dataset_config: Optional[str] = None
    """This should be a path to a YAML file that specifies a Levanter data configuration. Either this or dataset_name must be provided."""

    model_config: Optional[str] = None
    """The model config to use. If not provided, the default model config will be used."""

    # TODO: change to llama 3
    tokenizer: str = "meta-llama/Llama-2-7b-hf"

    tpu_type: str = "v5litepod-256"
    project: Optional[str] = None
    """The GCP project to use. If not provided, the default project will be used."""
    zone: Optional[str] = None
    """The GCP zone to use. If not provided, the default zone will be used."""
    base_config: str = "config/training/standard_run.yaml"
    """The base Levanter config to use."""
    run_name: Optional[str] = None
    """The name of the run. If not provided, it will be the experiment name followed by the run_id."""
    run_id: Optional[str] = None
    """The id of the run. If not provided, a random run_id will be used."""
    foreground: bool = False
    """If True, the job will be launched in the foreground."""
    env: dict[str, str] = field(default_factory=dict)
    """Environment variables to set in the container."""
    tags: list[str] = field(default_factory=list)
    """Tags to add to the wandb run."""

    retries: int = 10
    """The number of retries in case of failure."""


@draccus.wrap()
def main(args: LaunchConfig):
    default_config = cli.load_config()

    project = args.project
    if project is None:
        project = default_config.get("project")
    if project is None:
        project = cli.gcloud_config()["project"]
    if project is None:
        raise ValueError("No project provided and no default project set.")

    zone = args.zone
    if zone is None:
        zone = cli.get_default_zone()
    if zone is None:
        raise ValueError("No zone provided and no default zone set.")

    dataset_name = args.dataset_name
    dataset_path = args.dataset_path
    dataset_config = args.dataset_config

    if dataset_name is None and dataset_config is None:
        raise ValueError("Either dataset_name or dataset_config must be provided.")
    if dataset_name is not None and dataset_config is not None:
        raise ValueError("Only one of dataset_name and dataset_config can be provided.")
    if (dataset_name is None) != (dataset_path is None):
        raise ValueError("Either both dataset_name and dataset_path must be provided, or neither.")

    run_id = args.run_id
    if run_id is None:
        run_id = cli.default_run_id()

    run_name = args.run_name
    if run_name is None:
        run_name = f"{args.experiment}-{run_id}"

    try:
        bucket = zone_to_bucket[zone]
    except KeyError:
        raise ValueError(f"Unknown zone {zone}")

    base_config = yaml.load(open(args.base_config), Loader=yaml.SafeLoader)
    if args.model_config is not None:
        model_config = yaml.load(open(args.model_config), Loader=yaml.SafeLoader)
    else:
        model_config = None

    run_config = construct_levanter_config(
        base_config=base_config,
        model_config=model_config,
        data_config=_get_data_config(base_config["data"], dataset_name, dataset_path, dataset_config, args.tokenizer),
        cache_dir=args.cache_dir,
        bucket=bucket,
        id=run_id,
        exp_name=args.experiment,
        name=run_name,
        tags=args.tags,
    )

    with tempfile.NamedTemporaryFile(
        prefix=f"{args.experiment}-{run_id}", suffix=".yaml", dir=".", delete=True, encoding="utf-8", mode="w"
    ) as config_file:
        yaml.dump(run_config, config_file, default_flow_style=False)
        config_file.flush()
        config_path = config_file.name
        # docker requires a relative path (to pwd)
        config_path = os.path.relpath(config_path)

        image_name = f"marin-{args.experiment}"
        # make an image tag based on the unix timestamp to ensure we always pull the latest image
        tag = int(time.time())

        build_args = {
            "CONFIG_FILE": config_path,
        }

        local_id = docker.build_docker(
            "docker/levanter/Dockerfile.incremental", image_name=image_name, tag=tag, build_args=build_args
        )

        region = zone.rsplit("-", 1)[0]

        full_image_id = levanter.infra.docker.push_to_gcp(
            local_id=local_id,
            project_id=project,
            region=region,
            repository="marin",
        )

        # Construct the command
        cmd = [
            "python",
            "-m",
            "levanter.main.train_lm",
            f"--config_path={config_path}",
        ]

        env = deepcopy(args.env)

        default_env = default_config.get("env")
        if default_env is not None:
            mergedeep.merge(env, default_env)

        if env.get("WANDB_API_KEY") is None:
            key = os.environ.get("WANDB_API_KEY")
            if key is not None:
                env["WANDB_API_KEY"] = key
            else:
                raise ValueError("WANDB_API_KEY must be set in the environment. Please add it to your .config.")

        env["GIT_COMMIT"] = cli.get_git_commit()
        env["RUN_ID"] = run_id
        env["WANDB_DOCKER"] = full_image_id

        # TODO: copypaste from levanter...
        from levanter.infra.ray_tpu import RunOnPodConfig

        config = RunOnPodConfig(
            image_id=full_image_id,
            command=cmd,
            tpu_type=args.tpu_type,
            env=env,
            name="levanter",
            retries=args.retries
        )

        with tempfile.NamedTemporaryFile(suffix=".yaml", prefix=f"launch-{run_id}-", dir=".") as f:
            yaml = draccus.dump(config)
            f.write(yaml.encode("utf-8"))
            f.flush()

            f_name = os.path.relpath(f.name)
            print(f"Submitting job with config path {f_name}")

            address = args.address or os.environ.get("RAY_ADDRESS")
            client = JobSubmissionClient(address)
            job_id = _make_unique_job_id(client, run_id)
            job_id = client.submit_job(
                entrypoint=f"python src/levanter/infra/ray_tpu.py --config_path {f_name}",
                runtime_env={"working_dir": "./"},
                job_id=job_id,
            )

            print(
                f"""
        -------------------------------------------------------
        Job '{job_id}' submitted successfully
        -------------------------------------------------------

        Next steps
          Query the logs of the job:
            ray job logs {job_id}
          Query the status of the job:
            ray job status {job_id}
          Request the job to be stopped:
            ray job stop {job_id}
        
       Assuming all went well, you should see a wandb run named {run_name} with id {run_id} in the wandb dashboard.
       That is likely to be:
             https://wandb.ai/stanford-mercury/marin/runs/{run_id}
        """
            )

        if args.foreground:

            async def tail_job(job_id):
                async for line in client.tail_job_logs(job_id):  # type: ignore
                    print(line, end="")

                    status = client.get_job_status(job_id)
                    if status in {JobStatus.FAILED, JobStatus.SUCCEEDED, JobStatus.STOPPED}:
                        break

            print("Tailing job logs")
            wait_until_status(
                client, job_id, {JobStatus.RUNNING, JobStatus.FAILED, JobStatus.SUCCEEDED, JobStatus.STOPPED}
            )
            import asyncio

            asyncio.run(tail_job(job_id))



def wait_until_status(client, job_id, status_to_wait_for, timeout_seconds=5):
    start = time.time()
    status = client.get_job_status(job_id)
    while status not in status_to_wait_for and time.time() - start <= timeout_seconds:
        status = client.get_job_status(job_id)
        if status in status_to_wait_for:
            break
        time.sleep(1)

    return status


# try to make the job id be the same as the run id, but if it already exists, just make it unique
def _make_unique_job_id(client, run_id):
    job_id = run_id
    try:
        while client.get_job_status(job_id) is not None:
            job_id = f"{run_id}-{time.time_ns()}"
    except Exception as e:  # noqa
        if "does not exist" in str(e):
            pass
        else:
            raise
    return job_id


if __name__ == "__main__":
    main()

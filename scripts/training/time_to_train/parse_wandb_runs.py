import pytz
from datetime import datetime

import pandas as pd
import wandb


WANDB_ENTITY = "stanford-mercury"
WANDB_PROJECT = "marin"
WANDB_PATH = f"{WANDB_ENTITY}/{WANDB_PROJECT}"
TIMEOUT = 3600  # seconds


def convert_to_local_time(utc_str: str) -> str:
    # Parse the UTC string to a datetime object
    utc_dt = datetime.strptime(utc_str, "%Y-%m-%dT%H:%M:%S")

    # Set the timezone to UTC
    utc_dt = utc_dt.replace(tzinfo=pytz.UTC)

    # Convert to Pacific Time
    pacific_tz = pytz.timezone("America/Los_Angeles")
    pacific_dt = utc_dt.astimezone(pacific_tz)

    # Format the result as a string (optional)
    pacific_str = pacific_dt.strftime("%Y-%m-%d %H:%M:%S %Z")
    return pacific_str


def get_ts_diff(utc_str1: str, utc_str2: str) -> float:
    # Parse the UTC string to a datetime object
    utc_dt1 = datetime.strptime(utc_str1, "%Y-%m-%dT%H:%M:%S")
    utc_dt2 = datetime.strptime(utc_str2, "%Y-%m-%dT%H:%M:%S")
    diff = utc_dt2 - utc_dt1
    diff_seconds = diff.total_seconds()
    diff_hrs = diff_seconds / 3600.0
    return diff_hrs


TARGET_METRIC_KEYS = [
    "throughput/mfu",
]

TARGET_SUMMARY_KEYS = [
    "parameter_count",
    "num_hosts",
    "global_step",
    "throughput/examples_per_second",
    "num_devices",
]


def parse_run(run: wandb.apis.public.Run) -> dict:
    runtime = run.summary["_runtime"] / 3600.0  # hours
    create_time = convert_to_local_time(run.createdAt)
    heartbeat_time = convert_to_local_time(run.heartbeatAt)
    # get difference between create time and heartbeat time
    total_time = get_ts_diff(run.createdAt, run.heartbeatAt)
    configs = run.config
    tokenizer = configs["data"]["tokenizer"]
    train_batch_size = configs["trainer"]["train_batch_size"]
    history = run.history(keys=TARGET_METRIC_KEYS)
    if not history.empty and "throughput/mfu" in history:
        mfu = history["throughput/mfu"].mean()
    else:
        mfu = "N/A"
    summary = run.summary
    examples_per_second = summary["throughput/examples_per_second"]
    parameters = summary["parameter_count"]
    global_step = summary["global_step"]
    training_time = global_step * train_batch_size / examples_per_second / 3600.0  # hours
    data = {
        "run_name": run.name,
        "runtime": runtime,
        "total_job_time": total_time,
        "training_time": training_time,
        "create_time": create_time,
        "heartbeat_time": heartbeat_time,
        "tokenizer": tokenizer,
        "train_batch_size": train_batch_size,
        "examples_per_second": examples_per_second,
        "global_step": global_step,
        "parameters": summary["parameter_count"],
        "num_devices": summary["num_devices"],
        "mfu": mfu,
    }
    return data


def get_runs(name_prefix="time-to-train"):
    api = wandb.Api(timeout=TIMEOUT)
    runs = api.runs(
        path=WANDB_PATH,
        filters={"display_name": {"$regex": f"^{name_prefix}.*"}},
    )
    output_data = []
    for run in runs:
        try:
            data = parse_run(run)
            output_data.append(data)
        except Exception as e:
            print(f"Unable to parse run {run.name} due to error: {e}")
    df = pd.DataFrame(output_data)
    output_csv = f"wandb_runs_{name_prefix}.csv"
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} runs to {output_csv}")


if __name__ == "__main__":
    get_runs()

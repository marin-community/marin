import os
import pytz
import wandb

import pandas as pd
from datetime import datetime

WANDB_ENTITY = os.getenv("WANDB_ENTITY", "stanford-mercury")
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


def check_create_time(create_time: str, start_date: str = None, end_date: str = None) -> bool:
    """Check if the create time is within the start and end date"""
    # Custom parsing of the create_time string
    date_part, time_part, tz_part = create_time.rsplit(maxsplit=2)
    create_dt = datetime.strptime(f"{date_part} {time_part}", "%Y-%m-%d %H:%M:%S")

    if start_date is None and end_date is None:
        return True
    if start_date is not None:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        if start_dt > create_dt:
            return False
    if end_date is not None:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        end_dt = end_dt.replace(hour=23, minute=59, second=59)
        if end_dt < create_dt:
            return False
    return True


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


def parse_run(run: wandb.apis.public.Run, start_date: str = None, end_date: str = None) -> dict:
    runtime = run.summary["_runtime"] / 3600.0  # hours
    create_time = convert_to_local_time(run.createdAt)
    if not check_create_time(create_time, start_date=start_date, end_date=end_date):
        return
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


def get_runs(name_prefix=None, start_date: str = "2024-07-25"):  # name_prefix="time-to-train"
    api = wandb.Api(timeout=TIMEOUT)
    if name_prefix is None:
        runs = api.runs(path=WANDB_PATH)
    else:
        runs = api.runs(
            path=WANDB_PATH,
            filters={"display_name": {"$regex": f"^{name_prefix}.*"}},
        )
    output_data = []
    for run in runs:
        try:
            data = parse_run(run, start_date=start_date)
            if data:
                output_data.append(data)
        except Exception as e:
            print(f"Unable to parse run {run.name} due to error: {e}")
    df = pd.DataFrame(output_data)
    name_prefix = name_prefix or "all"
    output_csv = f"wandb_runs_{name_prefix}_{start_date}.csv"
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} runs to {output_csv}")


if __name__ == "__main__":
    get_runs()

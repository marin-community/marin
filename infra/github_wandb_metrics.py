# This script is used to log weekly metrics to W&B
import wandb
from datetime import datetime

# Function to log weekly data and update the table
def log_weekly_data(data, week_id, project_name, id, table_key="Ray Restart Events"):

    try:
        table = wandb.use_artifact(f"run-{id}-RayRestartEvents:latest").get("Ray Restart Events")
    except Exception as e:
        table = wandb.Table(columns=["Week", "Instance ID", "Action", "Timestamp", "Zone", "User"])

    # Add new rows for this week's data
    for event in data['Ray restart events']:
        table.add_data(
            week_id,
            event['instance_id'],
            event['action'],
            event['timestamp'],
            event['zone'],
            event['user']
        )

    seen = set()
    unique_data = [item for item in table.data if tuple(item) not in seen and not seen.add(tuple(item))]

    # Reinitialize the table with its updated data to ensure compatibility
    table = wandb.Table(columns=table.columns, data=unique_data)

    # Log the updated table
    wandb.log({table_key: table})

    # Log summary metrics for the week
    wandb.log({
        f"Week ID": week_id,
        f"Workflow Times - Lint and Format Check": data['Workflow Times per workflow']["Lint and Format Check"],
        f"Workflow Times - Quickstart": data['Workflow Times per workflow']["Quickstart"],
        f"Workflow Times - Run unit tests": data['Workflow Times per workflow']["Run unit tests"],
        f"Closed Issues with Experiments Label": data['Closed Issues with label experiments'],
        f"Ray Restarts": data['Number of Ray cluster restart'],
        f"Total Runs": data['num_runs'],
        f"Total GFLOPS Across Runs": data['total_gflops_across_runs'],
        f"Total Petaflops Across Runs": data['total_petaflops_across_runs']
    })


    model_sizes = []
    run_ids = []
    # Log best C4 EN BPB metrics
    for model_size, details in data['best_c4_en_bpb'].items():
        model_sizes.append(model_size)
        run_ids.append(details['run_id'])
        wandb.log({
            f"Model: {model_size} - Eval BPB": details['run_metrics']['eval/paloma/c4_en/bpb'],
            f"Model: {model_size} - GFLOPS": details['run_metrics']['throughput/total_gflops'],
            f"Model: {model_size} - Runtime": details['run_metrics']['_runtime'],
            f"Model: {model_size} - Parameters": details['run_metrics']['parameter_count'],
            f"Week ID": week_id
        })


    table_key = "Wandb Run Ids of best runs"

    try:
        table = wandb.use_artifact(f"run-{id}-WandbRunIdsofbestruns:latest").get(table_key)
    except Exception as e:
        keys = ["week_id"] + model_sizes
        table = wandb.Table(columns=keys)


    row = [week_id] + run_ids
    table.add_data(*row)
    seen = set()
    unique_data = [item for item in table.data if tuple(item) not in seen and not seen.add(tuple(item))]

    # Reinitialize the table with its updated data to ensure compatibility
    table = wandb.Table(columns=table.columns, data=unique_data)
    wandb.log({table_key: table})


def log_data_to_wandb(data):
    # Generate the week ID
    week_id = int(datetime.now().strftime("%W"))  # e.g., "02"
    project_name = "marin-monitoring"
    id = "weekly-metrics-final-final"
    wandb.init(project=project_name, id=id, resume="allow")
    # Log the data for this week
    log_weekly_data(
        data=data,
        week_id=week_id,
        project_name=project_name,
        id=id
    )
    wandb.finish()

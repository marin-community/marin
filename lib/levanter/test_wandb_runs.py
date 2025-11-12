#!/usr/bin/env python
"""Test script to fetch and print all wandb runs matching 'comma_150m'."""

import wandb
import re
import csv
from tqdm import tqdm

# Initialize API
api = wandb.Api()

# Get runs from project (entity defaults to your logged-in user)
entity = "marin-community"
project = "marin"

print(f"Fetching runs from {entity}/{project} with 'comma_150m' in name...")
# Filter on server side using wandb API filters
# Date format: "YYYY-MM-DD" or "YYYY-MM-DDTHH:MM:SS"
min_date = "2025-10-11"  # Change this to filter by date
matching_runs = api.runs(
    f"{entity}/{project}",
    filters={
        "display_name": {"$regex": "comma_150m"},
        "created_at": {"$gte": min_date}  # Greater than or equal to min_date
    }
)

print(f"\nFound {len(matching_runs)} runs matching 'comma_150m'")

# Parse run names to extract token_set (M) and epoch count
data = []
pattern = r"comma_150m_(\d+M).*?(\d+)epoch"

for run in tqdm(matching_runs, desc="Processing runs", unit="run"):
    match = re.search(pattern, run.name)
    if match:
        token_set = match.group(1)  # e.g., "1M", "10M", "100M"
        epochs = int(match.group(2))

        # Use summary to get final values (much faster than scan_history)
        # The HTTPSummary object can be corrupted, so we need to access it carefully
        try:
            # Try to get the internal dict
            if hasattr(run.summary, '_json_dict'):
                import json
                json_dict = run.summary._json_dict
                # If it's a string, try to parse it as JSON
                if isinstance(json_dict, str):
                    summary = json.loads(json_dict)
                    tqdm.write(f"✓ Parsed JSON string for {run.name}")
                elif isinstance(json_dict, dict):
                    summary = json_dict
                    tqdm.write(f"✓ Used dict directly for {run.name}")
                else:
                    # Last resort: iterate through items
                    summary = {k: v for k, v in run.summary.items()}
                    tqdm.write(f"✓ Iterated items for {run.name}")
            else:
                summary = {k: v for k, v in run.summary.items()}
                tqdm.write(f"✓ Iterated items (no _json_dict) for {run.name}")
        except Exception as e:
            print(f"Warning: Could not parse summary for {run.name}: {e}")
            print(f"Falling back to scan_history...")
            # Fall back to the old scan_history method for this run
            mean_pz_values = []
            num_documents_value = None
            num_windows_value = None
            mean_pz_col_name = None

            target_col = 'pz_eval/total/mean_pz'
            for row in run.scan_history():
                if target_col in row and row[target_col] is not None:
                    mean_pz_col_name = target_col
                    mean_pz_values.append(row[target_col])

                for key in row.keys():
                    if 'num_documents' in key and row[key] is not None:
                        num_documents_value = row[key]
                    if 'num_windows' in key and row[key] is not None:
                        num_windows_value = row[key]

            if mean_pz_col_name is None:
                tqdm.write(f"⚠ Skipping {run.name} - no '{target_col}' metric found in history")
                continue

            if len(mean_pz_values) == 0:
                raise ValueError(f"No mean_pz values found for run {run.name}")

            final_mean_pz = mean_pz_values[-1]
            final_num_documents = num_documents_value
            final_num_windows = num_windows_value
            mean_pz_col = mean_pz_col_name

            data.append({
                "token_set": token_set,
                "epochs": epochs,
                "run_name": run.name,
                "run_id": run.id,
                "state": run.state,
                "created": run.created_at,
                "final_mean_pz": final_mean_pz,
                "num_documents": final_num_documents,
                "num_windows": final_num_windows,
                "mean_pz_column": mean_pz_col
            })
            continue

        # Find the specific mean_pz column we want: 'pz_eval/total/mean_pz'
        target_col = 'pz_eval/total/mean_pz'
        mean_pz_col_name = None
        final_mean_pz = None

        if target_col in summary:
            mean_pz_col_name = target_col
            final_mean_pz = summary[target_col]

        if mean_pz_col_name is None or final_mean_pz is None:
            pz_cols = [k for k in summary.keys() if 'pz' in k.lower()]
            tqdm.write(f"⚠ Skipping {run.name} - no '{target_col}' metric found. Available pz metrics: {pz_cols[:3]}")
            continue

        # Get other final values
        final_num_documents = None
        final_num_windows = None
        for key in summary.keys():
            if 'num_documents' in key and summary[key] is not None:
                final_num_documents = summary[key]
            if 'num_windows' in key and summary[key] is not None:
                final_num_windows = summary[key]

        mean_pz_col = mean_pz_col_name

        data.append({
            "token_set": token_set,
            "epochs": epochs,
            "run_name": run.name,
            "run_id": run.id,
            "state": run.state,
            "created": run.created_at,
            "final_mean_pz": final_mean_pz,
            "num_documents": final_num_documents,
            "num_windows": final_num_windows,
            "mean_pz_column": mean_pz_col
        })

# Sort by token_set and epochs
data.sort(key=lambda x: (int(x["token_set"].rstrip("M")), x["epochs"]))

# Detect duplicates (same token_set and epochs)
seen = {}
for row in data:
    key = (row["token_set"], row["epochs"])
    if key in seen:
        row["duplicate"] = "Yes"
        # Mark the first occurrence as duplicate too
        for prev_row in data:
            if prev_row["token_set"] == row["token_set"] and prev_row["epochs"] == row["epochs"]:
                prev_row["duplicate"] = "Yes"
    else:
        row["duplicate"] = "No"
        seen[key] = True

# Write to CSV
output_file = "comma_150m_runs.csv"
with open(output_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["token_set", "epochs", "duplicate", "final_mean_pz", "num_documents", "num_windows", "run_name", "run_id", "state", "created", "mean_pz_column"])
    writer.writeheader()
    writer.writerows(data)

print(f"\nWrote {len(data)} runs to {output_file}")

# Print summary grouped by token_set
print("\nSummary by token set:")
current_set = None
for row in data:
    if row["token_set"] != current_set:
        current_set = row["token_set"]
        print(f"\n{current_set}:")
    print(f"  {row['epochs']} epochs: {row['run_name']}")

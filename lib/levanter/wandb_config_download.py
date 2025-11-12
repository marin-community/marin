#!/usr/bin/env python
"""Download and save configs from specific wandb runs."""

import wandb
import json
from pathlib import Path

# Initialize API
api = wandb.Api()

# Target runs
entity = "marin-community"
project = "marin"
run_names = [
    'comma_1b_10M_single_1000epoch_central2_20251019_001204-601c34',
    'llama_1b_comma_10M_single_1epoch_central2'
]

print(f"Fetching configs from {entity}/{project}...")

for run_name in run_names:
    print(f"\nProcessing run: {run_name}")

    try:
        # Fetch all runs and filter by name
        runs = api.runs(
            f"{entity}/{project}",
            filters={"display_name": run_name}
        )

        if len(runs) == 0:
            print(f"  ⚠ No run found with name: {run_name}")
            continue

        run = runs[0]
        print(f"  ✓ Found run: {run.id}")

        # Get config - it's stored as a JSON string, need to parse it
        config_raw = run.config

        # Try to parse if it's a string
        if isinstance(config_raw, str):
            config = json.loads(config_raw)
        else:
            config = config_raw

        # Save to JSON file
        output_file = f"{run_name}_config.json"
        with open(output_file, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"  ✓ Saved config to: {output_file}")

        # Print summary info about the config structure
        if isinstance(config, dict):
            print(f"  Config top-level keys: {list(config.keys())[:10]}...")
            # Try to show some useful nested info
            if 'data' in config and 'value' in config.get('data', {}):
                data_config = config['data']['value']
                if 'tokenizer' in data_config:
                    print(f"  Tokenizer: {data_config['tokenizer']}")
            if 'model' in config and 'value' in config.get('model', {}):
                model_config = config['model']['value']
                if 'num_layers' in model_config:
                    print(f"  Model layers: {model_config['num_layers']}, hidden_dim: {model_config.get('hidden_dim', 'N/A')}")
        else:
            print(f"  Config type: {type(config)}")

    except Exception as e:
        print(f"  ✗ Error processing {run_name}: {e}")

print("\nDone!")

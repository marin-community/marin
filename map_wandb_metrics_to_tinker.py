import wandb
import os

# Configuration
SOURCE_RUN_PATH = "marin-community/marin_post_training/runs/math500--20251121-081529"
METRIC_MAPPINGS = {
    "train/rewards": "env/all/reward/total",
    "train/format_rewards": "env/all/format",
    "train/correct_rewards": "env/all/correct",
    "train/output_len": "env/all/ac_tokens_per_turn",

    "test/tinker_math/rewards": "test/env/all/reward/total",
    "test/tinker_math/format_rewards": "test/env/all/format",
    "test/tinker_math/correct_rewards": "test/env/all/correct",
    "test/tinker_math/output_len": "test/env/all/ac_tokens_per_turn",

    "reinforce_loss": "optim/loss",
    "learning_rate": "optim/lr",
}

def main():
    api = wandb.Api()
    
    print(f"Fetching source run: {SOURCE_RUN_PATH}")
    try:
        source_run = api.run(SOURCE_RUN_PATH)
    except wandb.errors.CommError as e:
        print(f"Error accessing run: {e}")
        return

    new_run_name = f"{source_run.name}_tinker_format"
    project = source_run.project
    entity = source_run.entity
    
    print(f"Creating new run: {entity}/{project}/{new_run_name}")
    
    # Initialize new run
    new_run = wandb.init(
        project=project,
        entity=entity,
        name=new_run_name,
        config=source_run.config,
        reinit=True
    )

    print("Iterating through history and logging to new run...")
    
    # Scan history is more efficient for large runs than history()
    history = source_run.scan_history()
    
    step_count = 0
    for row in history:
        new_row = {}
        
        # Copy system metrics and other unmapped fields if desired, 
        # or just specific mapped ones. The request implies renaming specific metrics.
        # We should probably keep other metrics as is? 
        # The prompt says "duplicate a wandb run but change the name of the following metrics".
        # This implies keeping others.
        
        for key, value in row.items():
            if key in METRIC_MAPPINGS:
                new_key = METRIC_MAPPINGS[key]
                new_row[new_key] = value
            else:
                # Keep original metric
                new_row[key] = value
        
        # Ensure step/timestamp alignment if needed, though wandb.log increments step by default.
        # If we want to preserve exact steps from the original run:
        if '_step' in row:
             # wandb.log(..., step=row['_step'])
             # But scan_history rows might not have _step if it's internal. 
             # Usually it's better to let wandb handle steps or explicitly pass them if they are in the row.
             pass
             
        # Log to new run
        # Note: If the original run had sparse logging (different metrics at different steps),
        # this loop will replay them.
        # We explicitly use the step from the original run if available to maintain alignment.
        
        log_kwargs = {}
        if '_step' in row:
            log_kwargs['step'] = row['_step']
            
        new_run.log(new_row, **log_kwargs)
        step_count += 1
        
        if step_count % 100 == 0:
            print(f"Processed {step_count} steps...", end='\r')

    print(f"\nFinished processing {step_count} steps.")
    new_run.finish()
    print(f"New run created successfully: {new_run.url}")

if __name__ == "__main__":
    main()

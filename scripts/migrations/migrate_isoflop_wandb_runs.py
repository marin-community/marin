#!/usr/bin/env python3
# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Migrate WandB isoflop runs to match migrated checkpoint paths.

After scripts/migrations/migrate_isoflop_checkpoints.sh strips the 6-char hash
suffix from checkpoint paths (e.g., 'isoflop-1e+19-d2048-nemo-abc123' becomes
'isoflop-1e+19-d2048-nemo'), this script copies the corresponding WandB runs
to have matching names without the hash suffix.

This enables eval_metrics_reader.py to find WandB runs by checkpoint name
without needing complex override mappings.
"""

import argparse
import logging
import re
import sys

try:
    import wandb
except ImportError:
    print("Error: wandb package not installed. Install with: pip install wandb")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def copy_wandb_run(
    api: wandb.Api,
    source_run: wandb.apis.public.Run,
    new_name: str,
    entity: str,
    project: str,
    dry_run: bool = True,
) -> bool:
    """
    Copy a WandB run to a new run with a different name.

    Args:
        api: WandB API instance
        source_run: The source run to copy
        new_name: The new name for the copied run
        entity: WandB entity
        project: WandB project
        dry_run: If True, don't actually create the run

    Returns:
        True if successful (or would be successful in dry run)
    """
    if dry_run:
        logger.info(f"  [DRY RUN] Would copy {source_run.name} -> {new_name}")
        return True

    try:
        # Initialize a new run with the clean name
        new_run = wandb.init(
            entity=entity,
            project=project,
            name=new_name,
            id=new_name,  # Use name as ID to make it deterministic
            resume="never",
            config=dict(source_run.config),
            tags=list(source_run.tags),
        )

        # Copy summary metrics
        summary = dict(source_run.summary)
        for key, value in summary.items():
            new_run.summary[key] = value

        logger.info(f"  Created new run: {new_name}")
        new_run.finish()
        return True

    except Exception as e:
        logger.error(f"  Failed to copy run {source_run.name}: {e}")
        return False


def migrate_isoflop_wandb_runs(
    entity_project: str,
    run_name_filter: str | None = None,
    dry_run: bool = True,
) -> None:
    """
    Migrate WandB isoflop runs by copying them without hash suffixes.

    Args:
        entity_project: WandB entity/project (format: 'entity/project')
        run_name_filter: Optional filter to only process specific runs
        dry_run: If True, don't actually create runs
    """
    if "/" not in entity_project:
        raise ValueError(f"Invalid entity_project format: {entity_project}. Expected 'entity/project'")

    entity, project = entity_project.split("/", 1)
    api = wandb.Api()

    logger.info(f"Querying WandB for isoflop runs in {entity_project}...")

    # Query for isoflop runs with hash suffixes
    filters = {
        "displayName": {"$regex": "isoflop"},
        "state": "finished",
    }

    runs = api.runs(entity_project, filters=filters)

    migrated_count = 0
    skipped_count = 0
    error_count = 0

    for run in runs:
        display_name = run.displayName

        # Check if this run has a hash suffix
        if not re.search(r"-[0-9a-fA-F]{6}$", display_name):
            logger.debug(f"Skipping {display_name} (no hash suffix)")
            skipped_count += 1
            continue

        # Strip the hash to get the clean name
        clean_name = re.sub(r"-[0-9a-fA-F]{6}$", "", display_name)

        # Apply filter if specified
        if run_name_filter and run_name_filter not in clean_name:
            logger.debug(f"Skipping {display_name} (doesn't match filter)")
            skipped_count += 1
            continue

        # Check if a run with the clean name already exists
        try:
            api.run(f"{entity_project}/{clean_name}")
            logger.info(f"Skipping {display_name} -> {clean_name} (already exists)")
            skipped_count += 1
            continue
        except wandb.errors.CommError:
            # Run doesn't exist, we can create it
            pass

        logger.info(f"Processing: {display_name} -> {clean_name}")

        if copy_wandb_run(api, run, clean_name, entity, project, dry_run):
            migrated_count += 1
        else:
            error_count += 1

    logger.info("\n" + "=" * 60)
    logger.info("Migration Summary:")
    logger.info(f"  Migrated: {migrated_count}")
    logger.info(f"  Skipped:  {skipped_count}")
    logger.info(f"  Errors:   {error_count}")

    if dry_run:
        logger.info("\nDry run complete. Run with --execute to perform the migration.")


def main():
    parser = argparse.ArgumentParser(
        description="Migrate WandB isoflop runs to match migrated checkpoint paths",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (default)
  python migrate_isoflop_wandb_runs.py marin-community/marin

  # Execute the migration
  python migrate_isoflop_wandb_runs.py marin-community/marin --execute

  # Filter to specific runs
  python migrate_isoflop_wandb_runs.py marin-community/marin --filter nemo --execute
        """,
    )

    parser.add_argument(
        "entity_project",
        help="WandB entity/project (format: entity/project)",
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually perform the migration (default is dry run)",
    )

    parser.add_argument(
        "--filter",
        help="Only process runs whose clean name contains this string",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        migrate_isoflop_wandb_runs(
            entity_project=args.entity_project,
            run_name_filter=args.filter,
            dry_run=not args.execute,
        )
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

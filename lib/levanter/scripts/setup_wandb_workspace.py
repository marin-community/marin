# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import argparse
import os

import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws

import wandb  # For wandb.Api()


def main():
    parser = argparse.ArgumentParser(description="Create or update a WandB workspace.")
    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help="WandB entity (user or team name). Optional, defaults to the logged-in user's default entity.",
    )
    parser.add_argument(
        "--project", type=str, default="levanter", help="WandB project name. Optional, defaults to 'levanter'."
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default="levanter-default",
        help="Name of the workspace to create or update. Optional, defaults to 'levanter-default'.",
    )
    parser.add_argument("--base_url", type=str, help="Base URL for WandB (for self-hosted instances)")

    args = parser.parse_args()

    if args.base_url:
        os.environ["WANDB_BASE_URL"] = args.base_url

    # Define workspace sections and panels
    main_section = ws.Section(
        name="main",
        panels=[
            wr.LinePlot(
                x="throughput/total_tokens",
                y=["train/loss"],
                title="log(train/loss) vs log(tokens)",
                log_x=True,
                log_y=True,
            ),
            wr.LinePlot(
                y=["train/loss"],
                title="log(train/loss) vs log(steps)",
                log_x=True,
                log_y=True,
            ),
            wr.LinePlot(
                x="throughput/total_tokens",
                y=["eval/bpb"],
                title="log(eval/bpb) vs log(tokens)",
                log_x=True,
                log_y=True,
            ),
            wr.BarPlot(metrics=["throughput/mfu"], title="Max MFU"),  # Assuming this shows max or latest value
        ],
    )

    workspace_settings = ws.WorkspaceSettings(group_by_prefix="first")

    # Try to load existing workspace or create a new one
    workspace = None
    resolved_entity = args.entity

    try:
        # Initialize wandb.Api to determine default entity if args.entity is None
        # This also helps verify API key setup early.
        try:
            api = wandb.Api(overrides={"base_url": args.base_url} if args.base_url else {})
            if not resolved_entity:
                resolved_entity = api.default_entity
                if not resolved_entity:
                    print(
                        "Warning: WandB entity not specified and could not determine default entity (user might not be"
                        " logged in)."
                    )
                    # Proceeding with entity=None, creation might still work if project is unique enough or API handles it.
                    # Loading by URL will likely fail if entity is required in the path and is None.
                else:
                    print(f"Using default WandB entity: {resolved_entity}")
            else:
                print(f"Using specified WandB entity: {resolved_entity}")

        except wandb.errors.Error as e:  # Catch wandb specific errors for API initialization
            print(f"Error initializing WandB API (ensure you are logged in and API key is set): {e}")
            # Fallback to trying with entity=None if args.entity was None, or the specified one.
            # This path might be problematic for from_url if entity is truly needed.
            resolved_entity = args.entity  # Stick to original if API init fails
            if not resolved_entity:
                resolved_entity = wandb.Api().default_entity
                print(f"Falling back to default entity: {resolved_entity}")

        # Construct the workspace URL for from_url attempt
        # Only attempt from_url if we have a resolved_entity, as it's part of the URL path
        if resolved_entity:
            web_url_base = args.base_url or "https://wandb.ai"
            workspace_url_path = f"{resolved_entity}/{args.project}/workspaces/{args.workspace}"
            web_url = f"{web_url_base}/{workspace_url_path}"

            try:
                print(f"Attempting to load workspace from URL: {web_url}")
                workspace = ws.Workspace.from_url(web_url)
                print(
                    f"Found existing workspace: '{args.workspace}' (Entity: {resolved_entity}, Project:"
                    f" {args.project})"
                )
                workspace.sections = [main_section]
                workspace.settings = workspace_settings
                print(f"Updating existing workspace: '{args.workspace}'")
            except Exception as e:
                print(
                    f"Workspace '{args.workspace}' not found via URL (or error: {e}). Will attempt to create a new"
                    " one."
                )
                # Fall through to creation block
                workspace = None  # Ensure workspace is None to trigger creation
        else:
            print("Skipping attempt to load workspace by URL as entity is not resolved. Will proceed to creation.")

        if workspace is None:  # If not loaded, create new
            print(
                f"Creating new workspace: '{args.workspace}' (Entity: {args.entity or 'default'}, Project:"
                f" {args.project})"
            )
            # Pass args.entity (which can be None) to Workspace constructor.
            # The wandb_workspaces library or underlying wandb client should handle None as default entity.

            workspace = ws.Workspace(
                entity=resolved_entity,
                project=args.project,
                name=args.workspace,
                sections=[main_section],
                settings=workspace_settings,
            )

        workspace.save()

        final_entity = workspace.entity  # Get entity from workspace object after save
        final_project = workspace.project
        final_name = workspace.name

        print(f"Workspace '{final_name}' (Entity: {final_entity}, Project: {final_project}) saved successfully.")
        print(f"View it at: {workspace.url}")

    except wandb.errors.Error as e:  # Catch Wandb specific errors
        print(f"A WandB specific error occurred: {e}")
        print("Please ensure your WandB API key is configured (e.g., via 'wandb login' or WANDB_API_KEY).")
        print(
            f"Also verify that the project '{args.project}' exists under entity"
            f" '{resolved_entity or 'your default entity'}' and that you have necessary permissions."
        )
    except Exception as e:
        print(f"An unexpected error occurred during workspace setup: {e}")


if __name__ == "__main__":
    main()

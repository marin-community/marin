#!/usr/bin/env -S uv run --script
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["wandb-workspaces>=0.1.12"]
# ///
"""Define the W&B mini report the status page embeds.

The Training panel (web/src/components/WandbPanel.tsx) iframes a small
two-panel report that mirrors the headline charts of the hero report
"67B-A2B MoE on 10T tokens". This script is the source of truth for that
mini report: rerun it (WANDB_API_KEY with marin-community write access
required) to recreate the report or push definition changes. There is no
automatic sync with the parent report — if the parent's headline panels
change, update the definition here and rerun.

The runset is a run-name search rather than pinned run ids, so new hero
resumes (which keep the "sw2k_v4_2048_muon" fragment in their name) show
up on the status page without any edits.
"""

import sys

import wandb_workspaces.reports.v2 as wr

ENTITY = "marin-community"
PROJECT = "marin_moe"
# Stable report id; the slug before it follows the title but W&B resolves
# any slug with the right id suffix. Must match REPORT_EMBED_URL in
# web/src/components/WandbPanel.tsx.
REPORT_URL = (
    "https://wandb.ai/marin-community/marin_moe/reports/"
    "67B-A2B-MoE-on-10T:-status-page-hero-charts--VmlldzoxNzQzMTI4MQ=="
)
# Uniquely matches the muon hero lineage: the d512 muon experiments lack
# "sw2k" and the d2560 test/rmsadam runs lack "muon".
RUNSET_QUERY = "sw2k_v4_2048_muon"
X_AXIS = "throughput/total_tokens"


def report_definition() -> wr.Report:
    return wr.Report(
        entity=ENTITY,
        project=PROJECT,
        title="67B-A2B MoE on 10T: status-page hero charts",
        description=(
            "Two-panel embed for the Marin infra status page "
            "(infra/status-page, defined by wandb_report.py). Runset "
            f"self-updates via search query '{RUNSET_QUERY}', so new hero "
            "resumes appear automatically. Layout changes here change the "
            "status page."
        ),
        width="fluid",
        blocks=[
            wr.PanelGrid(
                runsets=[
                    wr.Runset(
                        entity=ENTITY,
                        project=PROJECT,
                        name="10T muon hero runs",
                        query=RUNSET_QUERY,
                    )
                ],
                panels=[
                    wr.LinePlot(
                        title="train/cross_entropy_loss",
                        x=X_AXIS,
                        y=["train/cross_entropy_loss"],
                        # Cap like the parent report: the warmup spike would
                        # otherwise squash the interesting part of the curve.
                        range_y=(None, 2.0),
                        layout=wr.Layout(x=0, y=0, w=12, h=8),
                    ),
                    wr.LinePlot(
                        title="eval/paloma/macro_loss",
                        x=X_AXIS,
                        y=["eval/paloma/macro_loss"],
                        layout=wr.Layout(x=12, y=0, w=12, h=8),
                    ),
                ],
            )
        ],
    )


def main() -> None:
    definition = report_definition()
    if "--create" in sys.argv:
        definition.save()
        print(f"created: {definition.url}")
        print("update REPORT_URL here and REPORT_EMBED_URL in WandbPanel.tsx")
        return
    report = wr.Report.from_url(REPORT_URL)
    report.title = definition.title
    report.description = definition.description
    report.width = definition.width
    report.blocks = definition.blocks
    report.save()
    print(f"updated: {report.url}")


if __name__ == "__main__":
    main()

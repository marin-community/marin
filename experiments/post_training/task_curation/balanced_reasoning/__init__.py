# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Difficulty-budgeted balanced reasoning traces for SFT.

The lane bands verified Atlas tasks by per-task solve rate, generates budget-capped
reasoning traces on the relay through the taskforge effect-driven stack, verifies each
trace with the task's own grader, keeps the shortest correct trace per task, and routes
tasks the generator cannot solve to a graceful-resolution lane. See the experiment's
DESIGN.md for the full design.
"""

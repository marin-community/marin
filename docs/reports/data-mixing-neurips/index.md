# Data Mixing NeurIPS Working Notes

## TL;DR

- This directory is the paper-facing home for data-mixing findings we may want to cite later.
- Chronological debugging should stay in experiment logs and debug notes.
- Once a finding matters for slides or a paper, promote it here in a claim-centric format.

## Purpose

This directory is meant to reduce drift between:

- raw experiment artifacts,
- chronological debug logs,
- slide text,
- and eventual paper claims.

The goal is not to duplicate every experiment note. The goal is to keep a small set of stable, paper-relevant writeups that answer:

1. what claim we think is true,
2. how strong the evidence is,
3. what artifacts back it up,
4. and what caveats still matter.

## How To Use This Directory

Each note here should be organized around a claim or methodological question, not around the order we discovered things.

Each note should include:

- `Status`: exploratory, replicated, stable, or blocked
- `Claim`
- `Evidence`
- `Caveats`
- `Artifact paths`
- `Reproduction command`

Chronological details, dead ends, and step-by-step debugging should stay in the originating debug log. The paper note should summarize the conclusion and link back to the raw log when needed.

## Current Notes

| Note | Status | Main takeaway |
| --- | --- | --- |
| [Olmix implementation review](./olmix.md) | Stable enough for paper background | The released Olmix code uses an exact convex proposer, but matching that proposer in Marin does not materially change our subset-fit Olmix optima; the instability is in the fit, not the final solve. |

## Recommended Workflow

Use three layers.

1. Raw layer: experiment logs, debug logs, local analysis notebooks, ad hoc CSVs
2. Claim layer: short notes in this directory that summarize findings we may cite
3. Paper layer: the actual draft, figures, and tables

In practice:

1. Do the exploratory work where it is easiest.
2. When a claim starts affecting slides or decisions, add a note here.
3. When a figure or table becomes important, record its source path and reproduction command in the note.
4. Only move a claim into the paper once it has a note here or an equally stable writeup elsewhere.

## What I Recommend Next

If this directory becomes the canonical paper-note home, the next useful additions are:

- a GRP note covering the currently best validated form and ablations,
- a scaling note covering what we can and cannot identify from the current 60M/300M panel,
- and a metric-provenance note describing how eval and lm-eval metrics are sourced and backfilled.

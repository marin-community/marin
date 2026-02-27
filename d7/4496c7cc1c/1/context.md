# Session Context

## User Prompts

### Prompt 1

Implement the following plan:

# Re-merge vendored lib/harbor from upstream

## Context

The vendored `lib/harbor/` was synced from upstream in commit `37cf3cbf0`, but that commit was a manual (non-git-merge) copy that missed changes — notably the mini-swe-agent PyPI install fix. We need to undo it and redo a proper merge that gets all latest upstream changes while preserving the Python 3.11 compatibility patches.

**Key facts:**
- `37cf3cbf0` is HEAD, only touches `lib/harbor/`, safe to reset...


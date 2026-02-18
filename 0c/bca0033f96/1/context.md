# Session Context

## User Prompts

### Prompt 1

The @.github/workflows/claude-review.yml can't post PR review comments, in the logs I see:

```
Parameters:

{
  "command": "gh pr review 2879 --comment -b \"Review: This PR makes zephyr use region-local marin-tmp scratch buckets (3-day TTL) for chunk storage instead of writing under the main MARIN_PREFIX path. Fixes 2628. Three findings: (1) __post_init__ loses MARIN_PREFIX fallback - before, ZephyrContext fell back to MARIN_PREFIX/tmp/zephyr; now it only tries _get_temp_chunk_prefix() then /tm...


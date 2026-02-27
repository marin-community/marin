# Session Context

## User Prompts

### Prompt 1

in the @.github/workflows/claude-review.yml add trigger on labelled event

### Prompt 2

ah, ok. when labeled happens, I get error in GH:

```
Action failed with error: track_progress for pull_request events is only supported for actions: opened, synchronize, ready_for_review, reopened. Current action: labeled
```

### Prompt 3

ok, with the current workflow file, should the review trigger on every new PR?

### Prompt 4

ok, now add a trigger that is based on a comment. If a comment in PR contains "claude review this", run the review workflow

### Prompt 5

I want comments on PRs only

### Prompt 6

ok, now remove the label based trigger

### Prompt 7

ok, now all pass the comment to the prompt as extra instructions


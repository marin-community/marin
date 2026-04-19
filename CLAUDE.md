@AGENTS.md

## Secrets handling: `.env` file

There is a file at `.env` in the worktree root that, when sourced in a
shell, exports `OPENAI_API_KEY` and possibly other credentials into the
environment.

**You MUST NEVER read, print, cat, grep, head, tail, less, open, Read,
or otherwise inspect the contents of this file.** Do not include `.env`
as an argument to any tool that would reveal its contents. Do not pipe
it through anything. Do not copy it anywhere. Do not stat its size in a
way that leaks information. Do not try to diff it or ask git anything
about its contents. Even accidentally surfacing the key through a tool
result is unacceptable — tool results go into conversation history.

**The ONLY permitted operation is `source .env` inside a Bash tool
invocation**, and only when you immediately need the environment
variable for a subsequent command in the same invocation. Examples of
acceptable usage:

```bash
source .env && python /tmp/some_script.py
source .env && python experiments/posttrain/run_bloom_judge.py --judge-model openai/gpt-4.1 ...
```

Never:

```bash
cat .env                       # NO
head .env                      # NO
grep OPENAI .env               # NO
source .env && env | grep OPENAI_API_KEY  # NO — this prints the key
echo $OPENAI_API_KEY           # NO — never echo the variable
```

If a command would produce output that includes the key (e.g. an error
trace from a failed API call that OpenAI's SDK echoes back), do not
include that output in your response or commit it anywhere. Redact it
to `<REDACTED>` before surfacing it to the user, and note that
redaction was performed.

If you need to verify the env var was loaded, the only permitted check
is:

```bash
source .env && python -c 'import os; print("OPENAI_API_KEY set:", "OPENAI_API_KEY" in os.environ)'
```

— which prints a boolean, not the value.

This rule applies to the `.env` file and to any similarly-named
credentials file anywhere in this repo. When in doubt, don't.
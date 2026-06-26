# Post fork changes merge protocol

For clarity purposes, we want:
- The head of the main branch of both `vllm` and `tpu-inference` forks should always point to
the same commits as the ones pinned in Marin's `tool.uv.sources` section of the root `pyproject.toml`.
- We want the fork commit history to be linear: A few custom commits overlayed on top of the upstream commits (i.e. no merge commits).

Consequently, we must force push the tip of each fork, each time the main branch of the marin repo is updated to point to 
different fork commits. Refer to `.agents/skills/refresh-tpu-vllm-forks/SKILL.md` for additional context.  

Here is the protocol (identical for both forks):
- Backup the current head of the main branch into a branch called `main-backup/YYYYMMDD/SLUG`. Skip if already exists (for the same commit)
- Make the main branch of the fork post to the same sha as pinned by the marin reqo. This requires a force push.

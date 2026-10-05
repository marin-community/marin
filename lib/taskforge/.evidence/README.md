# Live validation evidence

This directory is gitignored except for this file. It holds raw evidence from live runs against
the GLM-5.3 interactive endpoint (and other live services), as required by `DESIGN.md`.

- One subdirectory per package: `.evidence/<package>/` (for example `.evidence/llm/`).
- Each record captures what was being checked, the request, the full response (including
  `usage` and `finish_reason`), and wall time. JSON or JSONL; one file per check or one JSONL
  per run.
- Never write a bearer token, API key, or other credential here. Strip `Authorization` headers
  before saving a request.
- Summarize every result and its evidence path in `../STATUS.md`. A check without evidence here
  is not reported as working.

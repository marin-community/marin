# Curriculum-blind task generation prompt

Use a separate `gpt-5.6-sol` high-reasoning context. Do not attach the curriculum, rubric, curriculum probes,
reviews, or fit results.

```text
Generate exactly [N = max(24, 2 * guidepost_count)] diverse, concrete tasks that a capable practitioner could
reasonably be asked to perform in the subject below. You are sampling the subject, not designing or
reverse-engineering a curriculum.

SUBJECT
[subject ID, name, guideposts, and optional pre-curriculum domain brief with sources and hash]

REQUIREMENTS
- Do not ask for a taxonomy, syllabus, curriculum, or discussion of these guideposts.
- Cover every guidepost with at least two tasks. Use the remaining tasks for cross-guidepost work, boundary cases, and
  underrepresented central operations.
- Vary central operations, artifacts, contexts, and difficulty. Include entry, representative, and boundary tasks.
- Each instruction must be self-contained. Supply the data, code, measurements, source excerpts, legal text, or
  interface facts needed to act. Do not rely on an unspecified file, hidden source, or private verifier.
- Keep each task genuinely in the named subject. Narrative subject matter alone is insufficient.
- Avoid near-duplicates and simple surface rewrites. Do not include answers or solution sketches.
- Output strict JSON using the blind-task contract. `guidepost_basis`, `operation_family`, and `difficulty_intent`
  document how you sampled; they will be hidden from the fit judge.
```

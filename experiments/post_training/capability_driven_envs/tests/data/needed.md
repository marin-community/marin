---
id: "d28.research.search/3"
source: {kind: capability, ref: "d28.research.search", hash: "2e40a1355f7c664edd3e7cff4acdf319f528a14702e34d6988ba6f2fdefbac0d"}
grounding: unverified
null_reason: null
environment: container
solver_machine: null
verification: composite
research_need: needed
research_reason: "Realism depends on actual federal statutes, regulations, and DOL guidance text, citations, and current status, which builders cannot derive from the capability record."
research:
  - {kind: web, purpose: "Current official text and amendment history of the FLSA internship/trainee provisions in 29 C.F.R. Part 531 (exact section numbers unverified), including any 2023-2026 rulemaking and its as-of status, from eCFR and govinfo; desired: dated version history and current text; fallback: govinfo annual-edition XML or PDF of the CFR part. Public domain as US federal regulation; verify."}
  - {kind: web, purpose: "Current official text of the relevant FLSA sections (expected 29 U.S.C. §§ 203, 206, 212, 216; unverified) with amendment notes from the Office of Law Revision Counsel or govinfo; desired: exact subsection text on employee definitions and minimum wage; fallback: govinfo Statutes at Large extracts. Public domain; verify."}
  - {kind: web, purpose: "Citations, dates, and current status of DOL opinion letters and field assistance bulletins on unpaid interns (including the primary-beneficiary adoption and any withdrawn letters) from dol.gov WHD pages; desired: at least three on-point letters with a determinate current status as of the corpus freeze date; fallback: public web-archive copies of the official pages. Public domain as US government work; verify."}
  - {kind: web, purpose: "Current and superseded versions of WHD Fact Sheet #71 and related internship fact sheets from dol.gov; desired: both the pre-2018 six-factor version and the current version to seed the currency trap; fallback: archived editions. Public domain; verify."}
  - {kind: web, purpose: "License and redistribution check for federal statutes, regulations, and DOL guidance in the task container, covering 17 U.S.C. § 105 treatment of US government works and any DOL website terms; desired: confirmation that verbatim redistribution is permitted; fallback: if terms are ambiguous, substitute a topic whose official text is mirrored on govinfo/eCFR under plain public-domain terms."}
  - {kind: web, purpose: "Contamination search confirming the synthetic 'marketing fellowship' scenario and the assembled corpus do not replicate an existing published benchmark, course exercise, or public write-up of unpaid-internship research; desired: no substantial match; fallback: alter the fact pattern and corpus selection until it is distinct."}
build:
  - "corpus-harvester": "Script plus manifest that fetch and checksum the public-domain federal corpus into /corpus; checked by a manifest validator verifying file counts, checksums, and provenance fields."
  - "sidecar-metadata": "Per-document JSON sidecars carrying doc type, citation, date, amendment history, and status fields; checked by schema validation and cross-check against the research findings."
  - "container-image": "Dockerfile and reset/seed scripts installing ripgrep, jq, and python3 with /corpus read-only and /work writable; checked by a smoke test that boots the solver path."
  - "task-brief": "Solver-facing facts, as-of date, deliverable specification, and README; checked by a leakage review against the answer key."
  - "code-verifier": "Private executable checks for citation targets, status correctness, log re-execution, and citation resolution, with the answer key; checked against gold and negative controls."
  - "judge-rubric": "Ten anchored binary judge criteria with required evidence quotes; checked by calibration on gold and deliberately flawed records."
  - "eval-harness": "Orchestrator computing 0.5 x code-pass-fraction + 0.5 x rubric-score with the merits gate; checked by an end-to-end dry run over all controls."
resources: ["container/Dockerfile", "corpus/manifest.json", "corpus/**", "grader/code_checks.py", "grader/answer_key.json", "grader/rubric.yaml", "eval/run_eval.py", "task/brief.md"]
---
## Task

The solver plays a research associate at a law firm. A supervising attorney needs an authority research record — not a merits memo — on the following facts, to support a later memorandum on whether participants in a client's program are employees under the Fair Labor Standards Act:

- The client, Northbeam Collective, Inc. (synthetic identity), is a for-profit digital-markaging agency.
- It runs a 'LaunchPad Marketing Fellowship': a 12-week, unpaid program with a fixed 30-hour weekly schedule.
- Fellows produce client-facing social-media campaigns and deliverables under a fellowship manager; the work is hands-on production work for the agency's clients.
- The company describes the program as educational and asserts the fellows are 'not employees.'
- The research must be current as of the corpus freeze date recorded in `/corpus/MANIFEST.json` (the date the shipped corpus was assembled; the task is graded against that corpus, not against the live internet).

**Visible inputs (in the container):**

- `/corpus/` — a real federal administrative corpus of plain-text files, each with a `.meta.json` sidecar, containing: the relevant FLSA sections of the United States Code; the applicable CFR part covering interns/trainees, including its primary-factor test and amendment history; several Department of Labor (WHD) opinion letters and/or field assistance bulletins on internships, including at least one that has been withdrawn; the current and a superseded version of the key WHD internship fact sheet; and related Federal Register material (e.g., withdrawal notices, rulemaking documents). Roughly 40–80 files. All status and currency information (dates, amendment history, withdrawal status) is present either in sidecar metadata or in corpus full text — nothing needed for the task is withheld.
- `/corpus/MANIFEST.json` — file list, checksums, freeze date.
- `/work/` — empty writable directory.
- Installed tools: `ripgrep`, `jq`, `python3`, standard coreutils, and a shell. No network access.

**Deliverables (written into `/work/`):**

1. `research_log.md` — a reproducible log of the search process: the issue decomposition, the vocabulary translation performed (e.g., 'fellowship' to 'intern'/'trainee'), every search command issued with its recorded result count, and a one-line rationale for each query refinement. Logged commands must be re-runnable from `/work` against the shipped corpus.
2. `authority_record.md` — the authority-ranked research record: a ranked list of authorities with explicit hierarchy labels (statute / regulation / non-binding guidance), a status/currency annotation for each (including any withdrawal, supersession, or amendment), a one-to-three-sentence relevance explanation tying the authority to the facts, an exclusions section giving the reason each consulted-but-excluded document was dropped, documentation of any conflicts or transitions in the governing framework, and a coverage-gaps section stating what the corpus cannot resolve and what source would resolve it.

**Constraints:**

- Cite only documents present in the corpus; every citation must resolve to a corpus file.
- The record must not state or recommend a conclusion about whether the company violated the FLSA or is liable — the merits analysis is out of scope and is a critical gate.
- Currency and status claims must be validated against corpus evidence (sidecars or full text), not asserted from memory.
- The log must be reproducible: re-running the logged commands in a fresh copy of the container must yield the recorded result counts.

**Why it is difficult:** the most keyword-salient documents for terms like 'intern', 'unpaid', and 'training' are the non-binding fact sheets and an old opinion letter; one guidance item has been withdrawn (recorded in sidecar metadata and in a withdrawal notice in the corpus full text, but not flagged in the document text itself); the fact sheet exists in a superseded version that still matches 'six factors' queries; and the facts use lay vocabulary ('fellowship', 'program') rather than the corpus's statutory and regulatory terms. The solver must translate facts into legal terms, iterate searches, apply hierarchy (statute over regulation over guidance), validate currency and withdrawal status, and rank accordingly — prominence and recency must not stand in for authority.

**How it exercises the capability:** this is the full contract of `d28.research.search` on a real administrative corpus: fact-to-issue/term translation, executing and refining searches, validating currency/treatment/hierarchy, and documenting conflicts and coverage gaps — without performing the excluded merits analysis, without patent/prior-art search, and with keyword frequency explicitly a trap rather than a signal.

## Realism and workflow

**Who and why:** a junior associate or senior paralegal at a firm preparing the research foundation for a wage-and-hour memorandum. The supervising attorney wants the authorities identified, validated, and ranked before anyone writes the analysis — exactly the division of labor the capability record encodes (the merits memorandum is excluded from this capability).

**Workflow reproduced:** the standard legal-research loop — decompose the question into issues (employee status; trainee/intern exclusion; minimum-wage obligation), translate facts into the corpus's controlled vocabulary, run and refine filtered searches over the available materials, validate each retrieved authority's currency and treatment (here: sidecar metadata, Federal Register material, withdrawal notices), rank by hierarchy and relevance, and record unresolved gaps. The container reproduces the tooling realistically: `ripgrep` and `jq` over files with structured metadata are a faithful stand-in for a flat-file research database, and the scale (dozens of documents, thousands of lines) makes manual reading alone impractical, forcing genuine search.

**Environment fit (unchanged from the plan):** the corpus is real public-domain federal text at a scale and with real tooling (ripgrep, jq, python3) that requires a container; a shellsim fixture could not carry authentic statutory and regulatory text with authentic amendment history, and a reasoning-only prompt could not exercise real command execution and log reproducibility. Container is therefore kept as planned.

**Verification fit (unchanged from the plan):** composite, because the deliverable has a determinate core (which statute and regulation sections control; which guidance is on-point and its true status; whether the log re-executes) that private code can check exactly, and an open-ended shell (relevance explanations, hierarchy reasoning, gap statements) that only an anchored rubric can score. Neither alone is sufficient: code alone would miss quality; a rubric alone would miss fabrications and status errors.

**Solver-visible vs. evaluator-only:** the solver sees the corpus, sidecars, manifest, README, and tools. The evaluator holds the answer key (expected controlling sections, expected on-point guidance, expected statuses), the code checks, the judge rubric, and the control records. The evaluator re-executes the solver's logged commands in an identically seeded fresh container; the solver never sees any of this.

## Research plan

Research is **needed**: the task's realism and its determinate answer key depend on actual federal statutory text, regulatory text and amendment history, and DOL guidance citations and withdrawal statuses — none of which a builder can author or derive from the capability record. All citations below are **unverified** until the builder confirms them; the corpus must not be assembled until each entry is resolved.

Already known from general knowledge (all unverified until confirmed): the FLSA's employee definitions and minimum-wage requirement sit in Title 29 of the U.S. Code; the intern/trainee framework lives in a CFR part whose internship provisions embody a 'primary beneficiary' style test adopted by DOL after judicial developments in the Second Circuit; WHD Fact Sheet #71 exists in a pre-2018 six-factor version and a later version; DOL has engaged in internship-related rulemaking in recent years and has periodically withdrawn opinion letters. The exact section numbers, dates, and current statuses are precisely what the research must pin down.

- **research[0]** — Query eCFR (ecfr.gov) for the current text of the internship/trainee provisions of 29 C.F.R. Part 531, and govinfo for the part's amendment history and any 2023–2026 rulemaking documents and their current status. Purpose: establish the controlling regulation text, its amendment timeline, and whether the regulatory test was revised as of the freeze date — this drives the answer key's C2 check and the currency trap. License: US federal regulations are public domain (subject to research[4]); verify. Fallback: use govinfo annual-edition XML/PDF of the CFR part if eCFR is unusable. Abandon-if: the current status of the regulatory test cannot be determined determinately as of a chosen freeze date (e.g., unresolved vacatur or an in-effect-but-contested rule), because the answer key would be ambiguous.
- **research[1]** — Query the Office of Law Revision Counsel (uscode.house.gov) or govinfo for current official text of the FLSA sections relevant to employee status and minimum wage (expected 29 U.S.C. §§ 203, 206, and possibly 212 and 216; unverified). Purpose: ship authentic statutory text and fix the expected controlling statute sections for the C1 check. License: public domain; verify. Fallback: govinfo Statutes at Large extracts. Abandon-if: none expected; statutory text is stable.
- **research[2]** — Query dol.gov (WHD opinion letters and field assistance bulletin pages) for internship-related opinion letters and bulletins, their dates, and current status, including any withdrawn items. Purpose: select at least three on-point letters, at least one withdrawn, with determinate status as of the freeze date — this drives the C3/C4 checks and the withdrawal trap. License: US government works, expected public domain; verify per research[4]. Fallback: public web-archive copies of the official pages. Abandon-if: no on-point letter with verifiable status can be obtained under redistribution-permitted terms.
- **research[3]** — Query dol.gov for the current version of WHD Fact Sheet #71 and any superseded versions. Purpose: ship both the current fact sheet and the superseded six-factor version as the keyword-salient currency trap. License: public domain; verify. Fallback: archived editions via public web archives. Abandon-if: no superseded version retrievable — then the trap is replaced by an older opinion letter applying the superseded framework (research[2] must then include one).
- **research[4]** — Check 17 U.S.C. § 105 and any DOL website terms of use. Purpose: confirm verbatim redistribution of statutes, regulations, and DOL guidance inside the task container is permitted. Fallback: if DOL site terms are ambiguous, restrict the corpus to materials mirrored on govinfo/eCFR under plain public-domain terms and link fact sheets by description only (not acceptable for the trap documents — see risks). Abandon-if: redistribution is barred for a material portion of the corpus and no licensed mirror exists.
- **research[5]** — Contamination search: search for existing unpaid-internship research exercises, benchmarks, or course assignments matching the 'marketing fellowship' fact pattern and this corpus composition. Purpose: ensure the task is not a copy or near-copy of a public evaluation item. Fallback: alter the synthetic fact pattern (industry, program name, schedule details) and corpus selection until distinct. The fact pattern is freshly generated with synthetic identities, so contamination risk is low; this entry confirms it.

## Build plan

Build proceeds as a dependency graph of sessions, each ending in a handoff artifact with an acceptance check. No session starts before its research dependencies are resolved and recorded in a provenance file (`corpus/PROVENANCE.json`, listing per document: source URL, retrieval date, checksum, license basis, and the research entry that authorized it).

**S1 — Corpus harvest (depends on research[0]–[4]).** Artifact: `corpus-harvester` script + `/corpus` raw files + `corpus/manifest.json`. Steps: fetch the USC sections, the CFR part (current text plus prior-version material needed for the amendment history), the selected opinion letters/FABs (including the withdrawn one and its withdrawal notice), both fact-sheet versions, and any Federal Register rulemaking/withdrawal documents; store each as UTF-8 plain text with a stable filename (`us-29usc-203.txt`, `cfr-29-531-<sect>.txt`, `dol-whd-<letter-id>.txt`, etc.); record checksums. Acceptance: manifest validator confirms every expected file exists, checksums match, no zero-byte or mojibake files, and every file has a provenance entry. If research[0] shows a recent rulemaking changed the test, include both pre- and post-rule regulatory text so the amendment history is in-corpus evidence.

**S2 — Metadata and status authoring (depends on S1).** Artifact: `sidecar-metadata` — one `.meta.json` per document with fields: `doc_type` (statute | regulation | opinion_letter | field_assistance_bulletin | fact_sheet | federal_register), `citation`, `date_issued`, `effective_or_current_as_of`, `amendment_history` (array of {date, description}), `status` (current | superseded | withdrawn | proposed), `withdrawal_evidence` (filename of the corpus document evidencing a withdrawal, when applicable). Sidecars carry factual status only — never which documents the answer key expects. Acceptance: JSON-schema validation of every sidecar; a cross-check script confirms each status value is supported by a corpus document or a research finding recorded in PROVENANCE; specifically, the withdrawn letter's sidecar and the superseded fact sheet's sidecar carry their statuses.

**S3 — Container image (depends on S2).** Artifact: `container-image` — Dockerfile (base: a slim Debian/Ubuntu image; packages: ripgrep, jq, python3, coreutils) plus `entrypoint` and `reset.sh`. Layout: `/corpus` mounted read-only, `/work` writable and empty, `/task/README.md` and `/task/brief.md` visible. Runtime: no network (`--network none`). Reset strategy: `reset.sh` re-copies the pristine corpus and clears `/work`; each solver session starts from a fresh container so state never leaks between attempts. Isolation: the solver's shell runs as a non-root user; only `/work` is writable. Resource estimates (to be measured during S7): image ~400–600 MB; corpus ~10–30 MB across 40–80 files; expected solver session 30–60 minutes; expected evaluation run under 10 minutes. Acceptance: smoke test boots the container, runs `rg --version`, `jq --version`, `python3 --version`, lists `/corpus`, confirms `/work` is writable and `/corpus` is not, and confirms network egress fails.

**S4 — Task brief (depends on S3).** Artifact: `task-brief` — `/task/brief.md` (the facts, deliverable spec, and constraints as in the Task section above) and `/task/README.md` (tool inventory, corpus layout, sidecar schema, citation format guidance, and the explicit statement that status evidence is in sidecars and corpus text). Acceptance: a leakage review confirms neither file names the expected controlling sections, the expected guidance, or any answer-key content; a readability pass confirms a competent solver can complete the task from these files alone.

**S5 — Code verifier and answer key (depends on S2).** Artifact: `code-verifier` — `grader/code_checks.py` plus `grader/answer_key.json` (expected statute sections, expected CFR sections, expected on-point guidance identifiers, expected statuses). Checks C1–C7 as specified in the Grader design. Acceptance: a gold model record passes all seven; each negative control fails exactly the checks it is designed to fail; the verifier is deterministic across three runs.

**S6 — Judge rubric (depends on S5).** Artifact: `judge-rubric` — `grader/rubric.yaml` with the ten anchored binary criteria R1–R10, each with the evidence quote the judge must extract from the record before scoring. Acceptance: calibration — the gold record scores 10/10; two deliberately flawed records (one missing gap documentation, one with hierarchy errors) score the intended criteria; inter-run judge agreement on the control set is at least 0.8 (measured; if an LLM judge is used, this is measured over repeated runs with fixed prompts).

**S7 — Evaluation harness and dry run (depends on S4, S5, S6).** Artifact: `eval-harness` — `eval/run_eval.py`, which copies the solver's `/work` outputs into an evaluator-side fresh container, runs C1–C7 (re-executing logged commands with a per-command timeout in the no-network container), runs the rubric, applies the merits gate, and computes the composite reward. Acceptance: end-to-end dry run over the gold and all negative controls produces the expected rewards from the Grader design section; measured resource usage is recorded and fits the S3 estimates (or the estimates are corrected).

## Grader design and controls

**Observable success condition:** the solver's two files in `/work` identify the correct controlling statute and regulation sections, cite at least one on-point guidance item with correct status, exclude withdrawn/superseded material from the controlling tier, carry a re-executable log, and provide hierarchy-grounded, fact-tied relevance explanations with documented conflicts and gaps — all without a merits conclusion.

**Code checks (private, deterministic; `grader/code_checks.py`):**

- C1 — `authority_record.md` cites the controlling statute sections from the answer key (expected: the FLSA employee-definition and minimum-wage sections, per research[1]). Normalization: case-insensitive, whitespace-collapsed; accepts `29 U.S.C. § 203`, `29 USC 203`, `29 U.S.C. §§ 203, 206` forms.
- C2 — cites the controlling CFR section(s) of the internship provisions (per research[0]), as current as of the freeze date.
- C3 — cites at least one on-point opinion letter or field assistance bulletin with its correct status.
- C4 — no withdrawn or superseded guidance item is cited as current or controlling (a withdrawn letter correctly flagged as withdrawn does not fail this).
- C5 — log reproducibility: every logged command is re-executed from `/work` in a fresh identical container with a timeout; a command passes if it runs and its recorded result count matches the re-executed count; commands the log honestly marks as failed (with a reason) are exempt from the count match. C5 passes if ≥90% of commands pass and at least five commands are logged.
- C6 — a ranked list is present with a status annotation on every entry, and no superseded or withdrawn document appears in the controlling tier.
- C7 — citation resolution: every citation in `authority_record.md` resolves to a corpus file (fabricated-authority check).

Code-pass-fraction = (C1..C7 passed) / 7.

**Judge rubric (ten equal-weight binary criteria; each requires a verbatim evidence quote from the record):**

- R1 — issue decomposition maps the facts to the employee-status / trainee-exclusion / minimum-wage questions.
- R2 — vocabulary translation is documented (e.g., 'fellowship' → 'intern'/'trainee').
- R3 — hierarchy reasoning is explicit: statute ranked above regulation above non-binding guidance, with a stated reason.
- R4 — the operative regulatory test is correctly identified as of the freeze date (per research[0]).
- R5 — relevance explanations tie at least three facts (for-profit employer, fixed schedule, hands-on production work) to specific authorities.
- R6 — the transition/conflict between the superseded framework (e.g., the six-factor test) and the current framework is documented.
- R7 — the withdrawal status of any cited guidance is documented with corpus evidence.
- R8 — at least one material coverage gap is stated with what source would resolve it (e.g., absence of controlling judicial interpretation in this administrative corpus; no DOL guidance specific to marketing fellowships).
- R9 — exclusions are recorded with reasons (non-binding fact sheets, superseded versions).
- R10 — the log shows at least three query refinements, each with a stated rationale.

Rubric-score = (R1..R10 met) / 10.

**Aggregation formula:** `reward = 0.5 × code-pass-fraction + 0.5 × rubric-score`. **Critical gate:** if the record states or recommends a conclusion about whether the company violated the FLSA or is liable (merits analysis), reward = 0 regardless of everything else. The gate is evaluated by a dedicated check (an LLM judge prompted only on the gate question, with an appeal-style evidence quote) plus a keyword screen for liability-recommendation phrasing; only the judge's evidence-anchored determination triggers the gate, the keyword screen only flags for review.

**Controls and expected rewards:**

- **Gold control** (a complete model record built by the builder): code 7/7, rubric 10/10 → reward 1.0.
- **Partial-credit control** (correct statute and regulation, but cites the withdrawn letter as current, omits gap documentation, has only basic refinements): fails C3 and C4 → 5/7 ≈ 0.714; meets R1–R5 and R10 → 6/10 = 0.6; reward = 0.5×0.714 + 0.5×0.6 ≈ 0.657. This shows partial credit is reachable and distinguishable.
- **Frequency-trap control** (ranks the superseded fact sheet and the withdrawn letter as controlling, keyword-driven log): fails C2, C3, C4, C6 → 3/7 ≈ 0.429; meets only R1, R2 → 0.2; reward ≈ 0.314.
- **Merits control** (gold-quality record plus a concluding paragraph that the company 'is liable' / 'owes minimum wages'): gate → reward 0.
- **Fabrication control** (gold-quality record plus one invented citation, e.g., a nonexistent CFR section): fails C7 → ≤ 6/7 code and loses R-rubric points where the fabricated citation was the evidence; expected reward ≈ 0.5–0.6, never 0 from fabrication alone — fabrication is punished but not conflated with the merits gate.

**Anti-shortcut controls:** C5 defeats fabricated logs (counts are re-executed, not trusted); C7 defeats invented authorities; R-criteria require verbatim evidence quotes, so boilerplate cannot satisfy them; the answer key and rubric live only in the evaluator; corpus sidecars contain status facts the solver legitimately needs but never indicate which documents are expected. **Prompt-injection controls:** the corpus is genuine public-domain federal text; the only builder-authored content is sidecar metadata and the brief, which contain no instructions; the grader never executes solver-authored code — logged commands are re-run read-only in an isolated no-network container with per-command timeouts, so a malicious log cannot touch the evaluator.

**Privacy of hidden evidence:** the answer key, code checks, rubric, controls, and calibration records exist only in the evaluator-side repository and are never mounted into the solver container.

## Risks and null conditions

- **Risk: regulatory flux.** The internship regulatory test may have been revised or its status contested by recent rulemaking or litigation, making 'the current test as of the freeze date' ambiguous. Mitigation: research[0] pins the text and status at a documented freeze date, and the corpus ships the amendment history as in-corpus evidence. Abandon-if: no freeze date yields a determinate current test (e.g., an active vacatur with unclear effect) — then pivot to a different, stable CFR/USC topic under the same capability contract (e.g., another DOL wage-hour area with clear guidance hierarchy), or null the slot.
- **Risk: guidance unavailability or unverifiable status.** DOL has reorganized its guidance pages; opinion letters may move or disappear. Mitigation: research[2] with web-archive fallback; PROVENANCE records the source. Abandon-if: no on-point guidance with verifiable status and redistribution rights can be assembled.
- **Risk: licensing.** Federal works are generally public domain, but site terms must be checked. Mitigation: research[4]. Abandon-if: redistribution is barred for material corpus components and no licensed mirror exists.
- **Risk: answer leakage through sidecars.** Status metadata is solver-visible by design. Mitigation: sidecars carry only factual status fields; the S4 leakage review checks no answer-key content appears in any solver-visible file; which documents are 'expected' is never marked anywhere in the container.
- **Risk: judge variance.** Mitigation: binary anchored criteria with mandatory evidence quotes and S6 calibration; if LLM-judged, fixed prompts and measured agreement. Abandon-if: calibration agreement on the control set falls below 0.8 and cannot be recovered by tightening anchors.
- **Risk: log re-execution flakiness** (state-dependent commands, timing). Mitigation: fresh identical container, exact command re-run, per-command timeout, honest-failure exemption in C5. Abandon-if: the gold control shows nondeterministic counts across three runs.
- **Risk: task too easy** (corpus small enough to read exhaustively, defeating the search component). Mitigation: corpus sized (40–80 files, full statutory and regulatory text) so exhaustive reading is impractical; R10 and C5 require genuine query iteration. Abandon-if: dry-run timing shows a read-everything strategy completes within the intended session budget with full marks — then expand the corpus with additional on-topic and adjacent-part documents.

**Data provenance, license, and split:** every corpus document is a US federal government work sourced from official sites (research[0]–[3]), recorded in `corpus/PROVENANCE.json` with URL, retrieval date, checksum, and license basis confirmed by research[4]. This is a single task with no item-level train/eval split; the source group (all federal administrative materials) and the derivation group (sidecars, answer key, rubric — all derived from the same corpus) stay together in the single evaluation instance, so no split leakage is possible. Contamination is controlled by construction: the fact pattern is freshly generated with a synthetic company identity, and research[5] confirms the scenario and corpus combination do not replicate an existing public benchmark or exercise; if a substantial match is found, the fact pattern is regenerated before the task ships.


# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""FACET-Terminal-Tasks-6k at its contract-repaired prepared-image pin.

The public upstream publishes 6,020 terminal tasks; the prepared-image pin converted
them to 6,010 canonical TaskSpecs whose environments pin the
``ghcr.io/marin-community/atlas-facet`` runtime images by digest. The prepared pin
bound no SHARED integrity contracts, so the versioned source repair
(``facet-terminal-contract-v1``, coordinator-approved 2026-11-24) republished every
TaskSpec with the RootDeliveryContract derived from its own pinned private verifier
files; that repaired pin is the declared conversion input. The census, controls
re-derivation, lexical scan, SAMPLE review and oracle-filtered admission all measure
it, and ``import_taskcompendium_prebuilt`` admits the pinned tasks without repeating
conversion. The oracle-filtered admission selects the sealed private
``open-athena/facet-terminal-tasks-6k-oracle-filtered-20261124-contract-v1`` release.
"""

from finestore.rl.curation import SourceQualityResult, Verdict
from finestore.rl.status_taxonomy import DataRestriction
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.prebuilt import PrebuiltCohort, import_taskcompendium_prebuilt
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, SourceOrigin, TaskFormat

UPSTREAM = "FACET-Terminal/FACET-Terminal-Tasks-6k"
UPSTREAM_REVISION = "b2d02645932e3989c8332a41e57d0f0855de7002"
PREPARED_REPO = "open-athena/facet-terminal-tasks-6k-prepared-images-20261124-contract-v1"
PREPARED_REVISION = "71fe5ec64454b53350ec9a8d281ce39627ab67db"
CONVERTER_VERSION = "2026.11.24.facet-contract-repair.1"
COUNT = 6010
SOURCE_ID = "FACET-Terminal:FACET-Terminal/FACET-Terminal-Tasks-6k"
REPAIR_PROTOCOL = "facet-terminal-contract-v1"
SAMPLE_QC_CAMPAIGN = (
    "s3://marin-us-east-08a/marin/rl-data/atlas/ingest-a/2026.11.20.1/facet-repaired-sample-qc-v1/"
    "campaigns/6c99131ae92746618c6b5edfd8f83f27/1a227946a0b8be06cced.json"
)
SAMPLE_QC_TRIAGE = (
    "s3://marin-us-east-02a/marin/rl-data/atlas/ingest-a/2026.11.25.1/facet-repaired-sample-qc-triage-v1/triage.json"
)
SAMPLE_QC_TRIAGE_SHA256 = "5298093c0c15bf728c8f3034c26e801ee352d835f2c7dda78bac12d6d86ce358"
SAMPLE_QUALITY_RESULT = SourceQualityResult(
    reviewed=99,
    issue_count=36,
    issues=[
        ""
        "facet-terminal-facet-terminal-facet-terminal-tasks-6k-16bb442b652a4e7117defa21c6f8d77ccf8f2c46aab08b7a5dc69da8c9c088a2:"
        "The task is largely coherent and answerable from the fixtures, but the audit manifest spec has a"
        'concrete mismatch with the oracle. Public spec: `soulprint_status` is "the authentication status'
        'extracted from the Soulprint token response (e.g., active, expired, revoked)". The oracle'
        '(solution/solve.sh) instead returns the verification level: get_soulprint_status returns "gold"'
        '(verification_level) when verified, or "unverified"/"unknown". Neither the public examples'
        '("active"/"expired"/"revoked") nor the oracle\'s "gold" aligns; an actor following the public'
        'contract (e.g., "active" or "verified") would diverge from the reference. Similarly,'
        '`authenticated_principal` is specified as "the principal identity string (e.g., a DID or user ID)"'
        'while the oracle returns display_name "Aurora Official Publisher" — a plausible but not uniquely'
        "determined choice. The report sections are free-text and depend on an unseen grader's tolerance; no"
        "grader semantics are visible to confirm these ambiguities are tolerated.",
        ""
        "facet-terminal-facet-terminal-facet-terminal-tasks-6k-21a477f10ef3fc786207ab880eaf7cc91a3af71ed8745c800e69b145689e20fa:"
        'The task asks to extract "every city code that appears in any value within configuration.json", but'
        'configuration.json\'s target_cities is a list of objects with a "code" key (NYC, LON, TKY, XYZ, PAR,'
        'ABC) — clear enough. However the public task never defines what counts as a "city code" appearing'
        'in string values (e.g., "NYC" as substring vs. structured "code" fields), and the oracle\'s'
        "regex-based extraction ([A-Z]{2,} matched against catalog) is only one plausible interpretation; a"
        "solver could reasonably extract only the structured codes and produce a different checked_codes"
        "set. Additionally, the oracle hardcodes assumptions not derivable from public inputs (e.g.,"
        "placeholder token format {PROJECT_TITLE}, email composition details), though these are inferable"
        "from the provided fixture files. The core ambiguity around code-extraction semantics makes the"
        "expected output under-specified relative to a single canonical answer.",
        ""
        "facet-terminal-facet-terminal-facet-terminal-tasks-6k-228160381d738071ac9dcf7749a2ac0bde0607bdada307c1151016e2cf15414a:"
        "The task is largely coherent: fixtures are consistent (watchlist_screener present and active in"
        "skills_config; account tier premium, balance 12.75 >= 5.00, auto_recharge enabled — so capability"
        "match true; workspace dirs match session features nasdaq-watchlist-refresh, nyse-sector-scan,"
        "otc-penny-stock-scan, daily-summary-export). However, tests/test_state.py's"
        "test_required_top_level_keys_and_order asserts `list(self.data.keys()) == expected_keys`, i.e., the"
        "exact top-level key ORDER in binding_audit.json. The public instructions never require a key order"
        "— any valid JSON object with the required keys should pass. This exact-order requirement rejects"
        "otherwise-correct answers and is a concrete rubric mismatch between the public contract and the"
        "private grader.",
        ""
        "facet-terminal-facet-terminal-facet-terminal-tasks-6k-355705f1510b19915d6e64391401708ca5c8cd911af319cd1c020510c1274f05:"
        "The public spec is mostly well-defined, but two derivation rules are under-specified. (1)"
        "background_job_status_summary.status_counts: the logs carry level INFO/WARN/ERROR and free-text"
        "messages; the spec only says \"a map of status strings (e.g., 'completed','failed','running')\"."
        'Mapping is not unique: does the system-health line (no job_id) count as a job? Is "Retrying with'
        'CPU fallback" running? Is the scheduler summary line a job? The oracle (solve.sh) uses ad-hoc'
        'heuristics ("complete"→completed, "error/out of memory"→failed, "starting/retrying"→running, else'
        '"info"), so multiple defensible answers yield different status_counts/total_jobs. (2)'
        'overall_readiness: no criteria define when status is "ready" vs "not_ready" (fixtures contain a'
        "CUDA OOM error and an over-80% budget alert), leaving the key value judgment-dependent. Private"
        "grader (test_state.py) not previewed, so reference agreement is unknown; if it requires exact match"
        "to the oracle's heuristic mapping, valid alternative derivations would be rejected.",
        ""
        "facet-terminal-facet-terminal-facet-terminal-tasks-6k-432ca2bb21f361250a094581611c2689c63cf930b8b7f94b23382b000aba54f0:"
        "The public task is largely coherent, but the newsletter filtering rule is materially ambiguous: it"
        'says "a case-insensitive substring match for a keyword" obtained from feed_filter_keyword in'
        "fixture_manifest.json, yet the actual fixture_manifest.json (provided) contains no"
        "feed_filter_keyword or feed_date_threshold fields — only feed_urls.json has filter_metadata with a"
        "date_threshold and a keywords list. So the keyword is not actually obtainable from the manifest as"
        "instructed, and the rule doesn't specify how to handle multiple keywords. The hidden tests likely"
        "check specific filtered items, but the referenced manifest fields don't exist, making the correct"
        "subset ambiguous. Additionally, the openapi_lint_issues and workflow_alignment fields are"
        "subjective/free-form, but the task does acknowledge this. The missing feed_filter_keyword field is"
        "a concrete defect.",
        ""
        "facet-terminal-facet-terminal-facet-terminal-tasks-6k-4f5030f100be1ae1feb19e1ebd60892f6682282d42f064918e6a1fdd4d809b6c:"
        "Two material concerns. (1) Internal contradiction: the public instructions require each channel's"
        '`full_source` to embed the complete source JSON ("No fields present in any source file may be'
        'omitted"), yet also state "The output must not include any external URLs." The source fixtures'
        'contain URLs (e.g., bearblog_confirmation.json has "https://nexusinsights.bearblog.dev/...",'
        'rednote_publication_confirmation.json has "https://rednote.com/p/..."). A solver cannot tell'
        "whether embedded URLs violate the URL prohibition or whether that clause only bars URLs absent from"
        "sources; the two readings yield different valid outputs. (2) The oracle (solution/solve.sh)"
        "extracts identifiers via a fixed key set"
        "{'post_id','id','note_id','batch_id','slug','video_id','request_id'} plus limited nested traversal,"
        'which omits fields that plausibly "serve as unique identifiers" per the public contract — e.g.,'
        "'user_id'/'account_id'/'handle' in tiktok's account object and rednote's account object,"
        "'comment_id' in rednote_note_detail's comment_thread, and 'trace_id'. The public spec says \"All"
        "such fields must be included,\" so the oracle's output may fail that stated requirement; whether the"
        "actual grader (test_state.py, not fully reviewable here) enforces the oracle's narrower extraction"
        "or the broader public wording cannot be statically resolved. The rest of the task (paths, merge"
        'precedence, stock_context parsing of "Change: +$1.23 (2.96%)" into absolute=1.23/percentage=2.96,'
        "key sets) is coherent and consistent with the supplied fixtures.",
        ""
        "facet-terminal-facet-terminal-facet-terminal-tasks-6k-6832263a322eb1c3ec2df5d11f5c53c22f3e8c78a8edbd3ab8da21586169697c:"
        "The oracle (solve.sh) is consistent with the public contract: it builds all 11 required top-level"
        "keys in order, embeds full fixture contents, uses the relative path"
        '"output/notification_chime.mp3", extracts default_phone from manychat_config.json, and its'
        "self-checks align with the actual fixture structures (e.g., wechat_delivery_receipt.json nests"
        'message_id under a "delivery" key, matching the oracle\'s nested lookup). However, the public'
        'instruction contains an internal contradiction: it requires fixture content to be "preserved'
        'exactly," yet also states the dashboard "must not contain absolute paths, network URLs, or'
        'references to files not present in the fixture set." The fixtures themselves contain network URLs'
        "(e.g., avatar_url fields), absolute paths (wechat_bridge_status.log contains"
        '"/etc/wechat-bridge/config.yaml"), and references to files not in the fixture set'
        '(elevenlabs_dryrun_video_cost.json references "input/first_frame_pudong.png", which is not in the'
        "fixture manifest). An actor cannot satisfy both constraints simultaneously without clarification on"
        "whether the prohibition applies to dashboard-authored path references or to embedded fixture"
        "content. The actual verifier (test_state.py) is truncated in the preview, so it cannot be confirmed"
        "whether it enforces this contradictory constraint strictly or leniently.",
        ""
        "facet-terminal-facet-terminal-facet-terminal-tasks-6k-6b4507d027e039881c00de5fdc14370eda0c6b79223fbd5630011f6ebf2651e8:"
        '1) Missing context: the task requires `freshness.within_five_minutes` computed against "the'
        'reference timestamp found in /task_file/input/fixture_manifest.json", but the supplied'
        "fixture_manifest.json contains no timestamp field (only generated_by, fixtures, empty_directories)."
        "The reference script silently falls back to alerts.json's captured_at, contradicting the stated"
        'source. 2) Rubric mismatch: the public spec says estimated_savings_range should be "derive[d] from'
        'the length distribution fields present in the snapshot", but the oracle takes it from'
        "optimization_projection.estimated_savings_eur, a different field. 3) Ambiguity: `suggested_action`"
        'is required to be "a short, human-readable suggestion" with only loose content constraints, and'
        'merged_weather "enrichment" is described as "all localized enrichment fields" while the oracle'
        "includes pollen and air_quality too — the public wording is ambiguous about whether"
        "non-localization fields belong. These make the grading criteria underdetermined relative to the"
        "stated contract.",
        ""
        "facet-terminal-facet-terminal-facet-terminal-tasks-6k-71d0264ed191c099d342a1ba623a0843287e2b0c8f753bf325ce1f7cc25c3b95:"
        "The public instruction references data structures that do not exist in the actual fixture files,"
        'creating multiple concrete defects: (1) `content_package` instructs copying a "content_package"'
        "object from fixture_manifest.json, but no such key exists in that file. (2) `expert_analytics`"
        'instructs deriving from an "analysts" array with "rating" and "confidence" fields, but the actual'
        'guru_analyst_ratings.json has a "stocks" array with aggregated ratings and no "confidence" field —'
        "the solve script itself acknowledges this mismatch and builds pseudo-analysts as a fallback. (3)"
        '`pricing_snapshot` instructs grouping based on a "category" field present in each entry, but no'
        'such field exists; the actual file has nested "precious_metals" and "industrial_metals" groups, and'
        'the instruction\'s "futures" group name doesn\'t match the actual "industrial_metals" group. (4)'
        '`candlestick_table` lists "period" and "volume" as original fields, but the source uses "id" and'
        '"vol". (5) `fixture_inventory` references a "fixture_inventory" array, but the manifest has a'
        '"fixtures" array. (6) `table_indices.csv` is described as "maps logical table names to index'
        'positions" but actually contains stock market index data (S&P 500, NASDAQ, etc.). (7) `visual_aid`'
        "hardcodes size_bytes as 1560 but the solve script computes the actual file size which may differ."
        "(8) `asset_ownership` is ambiguous about which collection to select when multiple exist. The solve"
        "script's own comments acknowledge these mismatches (\"The actual file doesn't have a 'category'"
        "field\", \"The preview shows 'stocks' array, not 'analysts'\"), confirming the instruction is"
        "materially inconsistent with the actual data.",
        ""
        "facet-terminal-facet-terminal-facet-terminal-tasks-6k-75791b82f64b9298ef4d451146db3edcbc80e04f03ee9b53ba5f0e98d39eb41b:"
        "Two concrete defects. (1) Ambiguity: billing_log.json contains TWO ¥3 (amount 3.00, JPY) records —"
        'bill_001 (2025-10-14T11:30:00Z) and bill_003 (2025-10-15T07:45:00Z). The instruction says "the'
        'exact billing record showing the ¥3 charge" without disambiguating, so a valid report quoting'
        "bill_003 is equally correct; the oracle silently picks the first match, making the grading key"
        "underdetermined. (2) Rubric mismatch / reference conflict: the instruction requires quoting the"
        'record "exactly as it appears in the source", but the oracle re-serializes the entry via'
        'json.dumps(entry, indent=2), which does not reproduce the source bytes (e.g. "amount": 3.00 becomes'
        "3.0, nested formatting differs from the original file). Similarly, section 6 requires the text"
        '"quoted verbatim", yet the oracle emits it unquoted. The public contract and the oracle\'s literal'
        "output therefore diverge on exact-quoting requirements.",
        ""
        "facet-terminal-facet-terminal-facet-terminal-tasks-6k-762f80f7fb114424b536d08df618f97ffd4daa84e301733fc954658e8865ac8f:"
        "The public contract references fields that do not exist in the fixtures. (1) The Matching Clipboard"
        "Entries section says the sync window is determined by `last_sync_start` and `last_sync_end` fields"
        "in feishu_sync_status.json, but the actual fixture contains no such fields (it has last_full_sync,"
        "events, etc.). (2) The trip_expenses.csv section says the input has columns `date, category,"
        "amount, running_total, alert` and requires preserving existing `alert` values, but the actual input"
        "CSV header is `date,category,description,amount,currency,payment_method,running_total` — no `alert`"
        "column exists, so the instruction to not modify pre-existing alert values is unsatisfiable as"
        "stated. (3) The private oracle (solve.sh) resolves the sync window differently — by taking min/max"
        "of the events' start/end timestamps — which conflicts with the public"
        "`last_sync_start`/`last_sync_end` definition; it also writes an empty `alert` for pre-existing"
        "rows, a choice not specified by the public instructions. (4) The inventory fixture has no `flagged`"
        "field on any item, so both security outputs would be empty, but the public instructions imply"
        "flagged items exist. These are concrete contradictions between the public task and its"
        "fixtures/reference.",
        ""
        "facet-terminal-facet-terminal-facet-terminal-tasks-6k-7db8bf07e6c4ac833e4d87fe1220d00c712bfe176fd1bb87a13710733c727396:"
        "1) Edge count conflict: instruction says check against blueprint's `expected_edges`, but blueprint"
        "uses `expected_edge_count: 7` while actual XML has 8 edges. Oracle hardcodes `expected_edges = 8`"
        "with comment admitting it overrides the blueprint value to force PASS. A literal solver would"
        "report FAIL (8≠7), contradicting the oracle's PASS. 2) virtual_tryon_manifest check references"
        "top-level keys `source_person`, `target_garment`, `generated_outputs` that don't exist in the"
        "actual manifest (it uses `results[].role`). Oracle checks roles in `results` array instead. 3)"
        "diagram_blueprint.json and pdf_snapshot_info.json are listed in Inputs but have no validation rules"
        "specified (only 12 of 14 non-manifest files have numbered rules). 4) Oracle's"
        "`check_dash_for_missing()` is a no-op dead-code function. These create concrete grading mismatches"
        "between instruction-literal answers and the oracle's expected output.",
        ""
        "facet-terminal-facet-terminal-facet-terminal-tasks-6k-7f1104deb521f276a2b4588ac175de35f0b56d7785ca18c82e1d33138e12c19f:"
        "Concrete defects: (1) Media manifest spec is internally inconsistent and conflicts with fixtures:"
        'it requires remote_url from cloud-urls.json "field image_url"/"field video_url", but that file has'
        "no such fields (only entries[].media_url); it requires quality_restoration/watermark_removal values"
        "from edits-log.json, but that file contains only an operations list with no such values; it"
        'requires "edits_applied: the entire edits array", but the file has applied_edits (array of'
        'objects), not an "edits" array. (2) The oracle (solution/solve.sh) does not match the public'
        "contract: it sets edits_applied to the operations array of the first applied_edits entry, and sets"
        "quality_restoration/watermark_removal to booleans derived by string-matching operation names (e.g."
        'quality_restoration=true) — not "values from edits-log.json". It also extracts the video URL via'
        'heuristic ("dubbed"/"video" substring) rather than a documented field, and picks httpbin origin'
        'from data.origin, while the instruction says field "origin" without specifying nesting; these'
        "ambiguities leave multiple defensible answers. (3) messages.jsonl field mapping: instruction lists"
        'sender/timestamp/content/message_type "extracted and mapped from the original JSONL record" but'
        "does not state the source field names (sender_id, timestamp, content, message_type) — answerable by"
        "inspection, minor. (4) The oracle's video_url heuristic matches the second entry"
        "(media-gen.example.com/...video-background-loop.mp4) which is a background-loop source, arguably"
        'not the "final dubbed video" URL; the intended value is under-specified.',
        ""
        "facet-terminal-facet-terminal-facet-terminal-tasks-6k-8706a710855d15eea2893da7612ff449fe718faf840ed7b3c0c47742601141ea:"
        "The public contract and the oracle (solution/solve.sh) disagree on table structure. Public spec:"
        "Secret Scan table has exactly 5 columns (File Path, Line Number, Rule ID, Severity, Description);"
        'the oracle builds a 6-column table adding a "Finding ID" column. Public spec: Root Cause table must'
        '"preserve exact IDs", but the diagnostic JSON contains no finding IDs — the oracle fabricates'
        'DIAG-001..DIAG-00N IDs and invents severity "info" values for warnings, which are not in the'
        'source. Similarly, the required "overall risk level" for the secret scan does not exist in the'
        "source JSON; the oracle computes it heuristically. Also the oracle derives the incident date from"
        "fixture_manifest fields with a fallback to the diagnostic timestamp, while the actual"
        'fixture_manifest.json content is not previewed, so the "snapshot date from fixture_manifest"'
        "requirement may be unverifiable. These are concrete public-vs-reference mismatches that could cause"
        "a compliant 5-column answer to be rejected or the oracle to fail the public contract.",
        ""
        "facet-terminal-facet-terminal-facet-terminal-tasks-6k-8db519b75e2282ad15c94db552d1b74af60e25b40d88c18428925c8ce8c6a863:"
        "The public task is coherent overall, but the JSON deliverable has material ambiguity. The template"
        "schema's payload keys (greeting, date, meal_suggestion, preparation_constraints, habit_adherence,"
        "error_state, media_processing) do not include figma_highlights, desktop_screenshot_path, or"
        "connected_components, yet the instructions require those sections and simultaneously forbid"
        '"additional top-level keys beyond those in the template." The instruction never states where these'
        "three sections must live (inside payload, alongside payload, etc.), and the template schema (e.g.,"
        "meal_suggestion.name, preparation_constraints.available_ingredients) conflicts with the required"
        '"Source-to-Output Mapping" (e.g., meal name/description/metadata, ingredients array with'
        "available/unavailable entries). Multiple structurally valid outputs satisfy the prose but not the"
        "schema, and vice versa. The archived solve.sh visibly wrestles with this same unresolved conflict"
        "in its own comments. This is a concrete ambiguity in the public contract, not merely difficulty.",
        ""
        "facet-terminal-facet-terminal-facet-terminal-tasks-6k-9809661959e723f4c6d418cb4f807527feef8ba265b1a766d14824082dc9b9be:"
        'Sub-task 1 instructions reference fields that do not exist in the actual data: (1) "field'
        '`global_style_tokens`" in manifest.json — the manifest actually has `style_tokens`; (2) "Increment'
        'the `version` field" — manifest slide entries have `revision`, not `version`; (3) "field'
        '`revision_prompt`" in sl-003_metadata.json — the metadata has `notes`, no `revision_prompt` field.'
        "The oracle script handles these by using the actual field names (e.g., `manifest['style_tokens']`,"
        "`updated_slide['revision']`, fallback `metadata.get('revision_prompt', metadata.get('notes',''))`),"
        "but an agent following the public instructions literally would look for nonexistent fields."
        'Sub-task 3 has a genuine ambiguity: the instruction says "For each instrument defined in'
        'okx_watchlist.json" (which defines 4 instruments including DOGE with enabled:false), but the oracle'
        'only processes enabled instruments (3). The cache note ("DOGE disabled in watchlist") suggests'
        "exclusion is intended, but the public instruction does not specify this, and DOGE has data in the"
        "snapshot. This creates a mismatch risk between agent output and the oracle's expected report/cache"
        "content.",
        ""
        "facet-terminal-facet-terminal-facet-terminal-tasks-6k-9bb57fdf06a7182ca59a26a93845644dbc73ec055c3da2e119e66e7299a4a719:"
        "The incident pattern summary requirements cannot be satisfied from the supplied source file. The"
        'task mandates (1) an "Incident Frequency by Type" table derived by grouping all incident entries in'
        "incident_log.json by `type`, and (2) a ranked list of the three most frequent values of the"
        "`root_cause` field. However, every incident object in /task_file/input/incident_log.json contains"
        "only the keys id, severity, service, port, timestamp, message, latency_ms, resolved,"
        "resolution_notes — there is no `type` field and no `root_cause` field anywhere in the file"
        "(metadata keys are unrelated). The closest analogues (`severity`, `resolution_notes`) are different"
        "fields, so a solver cannot derive the required tables from the specified source, and any plausible"
        "substitution is unverifiable against the stated contract. This is a concrete"
        "missing-context/wrong-reference defect in the public task, independent of the grader. Other"
        "deliverables (design_dna.json, port report, audit log, tasks.md update) are well-specified against"
        "the provided fixtures, but the incident summary defect is material to a required output.",
        ""
        "facet-terminal-facet-terminal-facet-terminal-tasks-6k-9c31f9fe4b70a91f2ee6ac2ce4aa4998870e10b177ad4f2a6502e60f26e97bce:"
        "The public contract requires values 'copied exactly from the source fixtures', but the actual"
        "fixture contents are not visible in this review (all input fixture previews are empty). More"
        "importantly, the private test_state.py uses a fallback-chain strategy (e.g."
        "comp.get('canonical_company_name') or comp.get('company_name') or comp.get('name') or '') that is"
        "NOT documented in the public instructions. The public task says 'The resolved canonical company"
        "name from the disambiguation result' without specifying the exact JSON key. An agent following the"
        "public instructions literally could pick a different field than the hidden test expects, and the"
        "hidden test's fallback order is not derivable from the public contract. This is a material"
        "ambiguity: the public task does not specify which JSON key within each source fixture maps to each"
        "output key, yet the private grader enforces a specific (undocumented) fallback order. A"
        "correct-by-public-spec answer could fail the private test.",
        ""
        "facet-terminal-facet-terminal-facet-terminal-tasks-6k-a52cdbe9cb3803e46b45aae5d969c98ec514c54890fffb601d13e688d3b8a1ca:"
        "The task is largely coherent, but there is a concrete missing-context defect: the public"
        "instructions require a PowerPoint template at /task_file/input/template.pptx, yet the fixture"
        "manifest lists no such file among the visible starting fixtures, and the environment's"
        "build_scripts include a generate_template.py that is not shown to have been run into the task_file"
        "directory. The task's own instructions acknowledge this (\"If this file is not present, the task"
        'cannot produce the output presentation"), which means the presentation deliverable is contingent on'
        "a template file whose presence cannot be confirmed from the provided evidence. All other"
        "deliverables (consolidation report, sheet export copy) are well-specified with clear"
        "source-to-output mappings and exact formatting requirements, and the input fixtures referenced"
        "(market data, smart light status, unread emails, skill inventory) are all present in the manifest.",
        ""
        "facet-terminal-facet-terminal-facet-terminal-tasks-6k-a6e01168ed56d592a566385226b6d1a6d7f6567c82552e8ecc90bded89d8902c:"
        "The task is broadly coherent and fixtures are visible, but the public instructions repeatedly"
        'reference fields that do not exist in the actual fixtures: (1) "use the `request` portion of'
        'webhook_dooray.json ... the `payload` field within the `request` object" — none of the three'
        "webhook fixtures contain a `request` or `payload` key; they contain `payload_template` and"
        "`last_response`. The oracle (solve.sh) resolves this by using `payload_template` for both"
        "configured_state and payload_sent, an interpretation not stated in the public contract, so a"
        'literal reading of the instructions cannot be satisfied. (2) "the speaker assigned to the left'
        'channel in voice_mapping.json" — voice_mapping.json contains no channel assignment; channel'
        'assignment only exists in podcast_session.json hosts (`assigned_channel`). (3) "must match the'
        '`speaker` field values in podcast_session.json" — the session uses `speaker_id` and `name`, not'
        "`speaker`. These mismatches between the stated contract, the visible fixtures, and the oracle's"
        "interpretation create genuine ambiguity about the expected manifest structure (e.g., what counts as"
        '"configured_state" vs "payload_sent" when both map to payload_template). The grader (test_state.py)'
        "is truncated in the preview, so agreement between the public contract and the private grader cannot"
        "be confirmed.",
        ""
        "facet-terminal-facet-terminal-facet-terminal-tasks-6k-b1348560e938c7b98a1b20b3b4d67c288a94fc89ae40054e254d1f78f5e7acec:"
        "Task is largely coherent: clear deliverables, complete fixtures, and tests that check headings, URL"
        "whitelist, tables, and manifest structure. One concrete mismatch: test_report_no_unexpected_urls"
        "builds allowed = {billing_url, video_url} only, while the public instruction for the video metadata"
        'table says to extract "any asset URL or reference" — the success response also contains'
        'thumbnail_url. A compliant report that includes the thumbnail URL (permitted by "any asset URL")'
        'would fail the URL whitelist check. Also, the Xiaomi wording ("include power, voltage, current,'
        'temperature if present; otherwise include all available properties") is slightly ambiguous since'
        "some are present and some are not, but the fixture makes all-available a defensible reading."
        "Reference solve.sh is preview-truncated, so full agreement is unverifiable, but visible portions"
        "align with the public contract.",
        ""
        "facet-terminal-facet-terminal-facet-terminal-tasks-6k-b2558aee55de4b0b0fdb0c069e97d7d4d7c529ff8daa9a35f898a3a32b3e928c:"
        "The public contract is largely coherent, but `vector_store_point_count` is materially ambiguous."
        'The instruction says "the total number of vector points implied by the indexing event log ... must'
        'be extracted from the log," yet the log offers several candidate values: L2 point_count 4521 (the'
        'log\'s collection is knowledge_base_l2, and the Qdrant email says "New point count: 4521"), the sum'
        "of branch point_counts (4521+12890+8932=26343), or ingestion_summary.total_vectors_ingested 89. The"
        "oracle solve.sh is truncated mid-deliberation over exactly this choice, so I cannot confirm the"
        "reference picks one defensible interpretation; a solver choosing 4521 vs 26343 vs 89 has no public"
        "text disambiguating. Other requirements (record fields, six per-source keys, image reference,"
        "dedup) are well specified and checkable by the hidden test_state.py.",
        ""
        "facet-terminal-facet-terminal-facet-terminal-tasks-6k-b59be0b700450d72a006cd1a0be3f6a1cdb8c3dc31ce65986c573dd4890c8bee:"
        'The public contract specifies the manifest must contain "exactly the following top-level keys" but'
        "never mandates their ORDER. test_state.py::test_top_level_keys_exact asserts `list(manifest.keys())"
        "== REQUIRED_KEYS` (assets, project_version, preview_url, preview_expires_at_utc,"
        "deploy_key_verified, messaging_summary), rejecting a semantically identical manifest with keys in a"
        "different order. This is stricter than the published contract. Minor ambiguity: the instructions"
        'say "Read the `version` field" but the receipt stores it as `task.project_version`; the tests'
        "resolve this the same way, so it is not a conflict. Other checks (assets set, preview_url pattern,"
        "expiry ≥24h, deploy_key_verified, messaging_summary) match the public spec.",
        ""
        "facet-terminal-facet-terminal-facet-terminal-tasks-6k-b9c053aaef855888cd55cca1481803f97133ac88d2c079bec2345b1e1e3e4b6b:"
        "The private tests and oracle are internally consistent with the fixtures, but the public"
        "instructions systematically cite field names that do not exist in the fixtures. Instructions:"
        "`companyName`, `currentPrice`, `dollarChange`, `percentChange`; actual stock_quote.json uses"
        "`company_name`, `price`, `change_dollar`, `change_percent`. Instructions:"
        "`tweetId`/`oauthAuthorizationUrl`; fixture nests `tweet_id` under `result`. Instructions:"
        "`mp3OutputPath`; fixture uses `output_mp3_path` under `transcription`. Credit values are nested"
        'under `balance`, not top-level as implied. Additionally, the instruction demands "exactly one data'
        'row" from a fixture containing three quotes (AAPL/MSFT/GOOGL) with no rule for selecting which one;'
        "the hidden tests hard-code `quotes[0]` (AAPL), so any defensible alternative choice (e.g., first"
        'alphabetically, or MSFT) fails verification. The manifest\'s "focus on relevant entries" hint does'
        "not resolve this. The core task is otherwise answerable and the reference (tests/oracle) agrees"
        "with the fixture data.",
        ""
        "facet-terminal-facet-terminal-facet-terminal-tasks-6k-bca28cceddc0f3eaeb8fff914e8c81b0b8b0af0483244705b90fee02008677cb:"
        "The public contract for conclusion.md contains an internal conflict. It demands the conclusion"
        'confirm "No pending items remain" and "The topic lifecycle is closed," while simultaneously'
        "requiring alignment with voicemaster_transcript.txt. That transcript's final status explicitly"
        'states: "The only outstanding item is the TX-2025-006 follow-up. Once resolved, the lifecycle can'
        'be closed." So the transcript says an outstanding item remains and the lifecycle is NOT yet closed'
        '— directly contradicting the mandated conclusion content. The supplied oracle itself hedges ("No'
        "pending items remain that would block closure... outstanding item, TX-2025-006... does not prevent"
        'lifecycle closure"), showing the reference cannot literally satisfy the stated "No pending items'
        'remain / lifecycle is closed" requirement while matching the transcript. A solver cannot satisfy'
        "both constraints. The rest of the task (three deliverable files, fixture-driven tasks.md) is"
        "coherent and well-specified.",
        ""
        "facet-terminal-facet-terminal-facet-terminal-tasks-6k-bf0c6c33acd547efe33bfc09eb8805557873976217519132803b84ba92d1f5f6:"
        '1) Ambiguity: the pipeline summary requires `target_entity_id` for "the primary target entity," but'
        "home_assistant_entities.json has 8 entities with no primary/target flag; the oracle itself comments"
        "\"the task says 'target entity', but does not define which entity is target\" and picks"
        "`switch.media_render_node` via heuristic. Any of several entities is defensible. 2) Wrong"
        "reference/contradiction: instructions say the artifact paths (`zip_archive_path`,"
        '`journal_digest_path`) are "as recorded in /task_file/input/fixture_manifest.json," but'
        "fixture_manifest.json contains no such paths; the oracle falls back to journal_template.json. The"
        "source citation is factually wrong for the supplied fixtures. 3) Segmentation manifest ambiguity:"
        'instructions say to derive segment_count, naming pattern and duration "from the relevant command'
        'entry in ffmpeg_command_catalog.json," but the catalog has four profiles with different patterns'
        "(segment_%04d.mp4 vs archive_segment_%04d.mp4) and durations (5.0 vs 10.0), and"
        "media_generation_response.json independently carries a segment_plan (24 segments, 5.0s,"
        "segment_%04d.mp4, profile libx264-fast). The oracle mixes sources (count from response,"
        "pattern/duration from the libx264-fast profile), while other defensible readings (e.g., veryslow"
        "profile, or all values from response) yield different valid outputs. These unresolved ambiguities"
        "mean alternative valid answers would be rejected.",
        ""
        "facet-terminal-facet-terminal-facet-terminal-tasks-6k-bf15e434177020f64a4f11f293109f0386e7fe632b5fca85a38aaa7bf58c0fdf:"
        'Public spec for figma_structure_summary says "count of immediate child frames", but the reference'
        "oracle (solution/solve.sh, build_figma_summary) counts nodes of type FRAME *or* COMPONENT. Page 2 -"
        "Product Grid Components has 4 children all of type COMPONENT, so the oracle yields 4 while a"
        "literal reading (count type FRAME) yields 0; a spec-following answer would be rejected or mismatch"
        'the reference. Additionally, the spec\'s research_reference requires a "publication date" while the'
        "source field is published_date — the oracle's key choice is plausible but the public wording leaves"
        "the exact key/name underdetermined. No other concrete defects found: deliverable paths, top-level"
        "keys, semver bump rule, and content_readiness booleans are coherently specified and consistent with"
        "the fixtures shown.",
        ""
        "facet-terminal-facet-terminal-facet-terminal-tasks-6k-c08797c7351c971c40f031193013ad857e873a0338f1518bd6ae430bd2036ddd:"
        "The task has concrete contradictions between the public instructions and the ASA fixture: the"
        "public task mandates `platform: openeuler` and path `/task_file/drafts/content_draft.md`, while the"
        "ASA requires `platform: internal-review` and filename pattern `readiness-draft-YYYY-MM-DD.md`. The"
        "actor cannot satisfy both simultaneously, creating genuine ambiguity about the correct content"
        "draft and gate checklist. Additionally, the oracle's own gate checklist evidence is factually"
        'incorrect: it mentions "model Alpha-7B from leaderboard.json" but the oracle\'s draft references the'
        "actual top model (Epsilon-70B-Disabled, final_score ≈0.97), not Alpha-7B. The leaderboard inclusion"
        "of the disabled/incomplete Epsilon-70B-Disabled model is also ambiguous — the public spec does not"
        "clarify whether deprecated models should be excluded, and different reasonable interpretations"
        "produce different leaderboards.",
        ""
        "facet-terminal-facet-terminal-facet-terminal-tasks-6k-c386f801186dfa04a02c2ec85ec550ad7ce35826fd7c5635bedba5b65277949e:"
        "Public contract only requires all top-level keys to be present (no ordering requirement), but"
        "hidden test test_required_top_level_keys asserts `actual_keys == required_keys` — an exact"
        "key-order match. A manifest with the correct keys in a different order would be rejected by the"
        "grader despite satisfying the public instructions. The oracle (solve.sh) happens to emit keys in"
        "the required order, so the reference agrees with the test, but the public contract is stricter than"
        "disclosed. The fixture_manifest.yaml 'quirks' (filter to active actions, filter successes) conflict"
        "with the public contract's explicit 'every action identifier' and 'one entry per memory store'"
        "requirements, but the public instructions are unambiguous, so these read as distractors rather than"
        "defects. Otherwise the task is coherent, all inputs are present, and the oracle's field extractions"
        "match the fixtures.",
        ""
        "facet-terminal-facet-terminal-facet-terminal-tasks-6k-d0d95260943e3368ea4270cc3238274ff5cbb717b1ce19ff54f3511f3921713b:"
        "The task is largely coherent: 10 deliverables with clear source-to-output mappings, and the visible"
        "grader tests (enriched record structure, card-field extraction, is_new/store consistency) align"
        "with the public contract. However, output 5 has a material ambiguity: the card's birthday is \"农历"
        '八月十五 午时" with no year, so a unique Gregorian date is not determined — the actor must guess a year'
        "(and the Ziwei chart in output 10 inherits the same missing-year assumption). Additionally, the"
        "note-version semantics are underspecified: note_version_history.json contains five notes with"
        "versions 3,2,1,1,1, and the public text does not pin down whether total_versions counts all notes"
        '(6) or versions (4), nor what "history from oldest to newest" contains given non-monotonic version'
        "numbers. The encryption contract is loose (any algorithm string + ciphertext) but that is"
        "permissibly underdetermined rather than contradictory. The oracle (solution/solve.sh) and"
        "test_state.py are truncated, so full reference agreement cannot be confirmed; the visible portions"
        "show the oracle hardcodes plausible values (Fernet with SHA-256-derived key labeled as such)"
        "consistent with the loose public spec.",
        ""
        "facet-terminal-facet-terminal-facet-terminal-tasks-6k-d3456f3b38045362568a575a6dfb66c87523520b7e44d4c563ad748069d10a4b:"
        'The task requires "the total count of distinct findings" and "the total number of results available'
        'in the file," but the fixture data contains internal contradictions that make these counts'
        "ambiguous. Specifically: (1) secret_scan_results.txt lists 13 individually numbered findings"
        '(FINDING 1 through FINDING 13) but explicitly states "Scan completed with 21 total findings" and'
        "shows a summary of CRITICAL:2, HIGH:5, MEDIUM:3, LOW:11 (summing to 21). An actor could reasonably"
        "report either 13 (count of detailed findings) or 21 (the file's self-reported total). (2)"
        "wikipedia_search_results.json contains 8 entries in the results array, but search_metadata says"
        '"returned: 20" and total_hits: 42. The task asks for "the total number of results available in the'
        'file" — an actor could report 8 (array length) or 20 (metadata "returned" value). The oracle'
        "(solve.sh) resolves these by counting parsed findings (13) and array length (8), but the public"
        "instructions do not clarify which interpretation is expected, creating genuine ambiguity that could"
        "cause a faithful actor to produce a different answer than the reference.",
        ""
        "facet-terminal-facet-terminal-facet-terminal-tasks-6k-d9d55cbae9ec83d7a15ea859ebc51d574b317c50ae055fd1f9d6a7ac249b9de7:"
        '1) Public instructions reference `"request_manifest": true` as a config.json key, but actual'
        'config.json has `"output_manifest": true` — key name mismatch between public contract and actual'
        'fixture. 2) Public instructions say to filter trend products by their "region field," but no'
        "per-product region field exists in echotik data (only file-level `market_region`). The oracle falls"
        "back to file-level value. 3) Public instructions describe fuzzy brand/company name matching for"
        "market data chart, but oracle uses `brand_ref` field matching — different approaches that could"
        "produce different matched stock sets. These are material concerns but don't fundamentally break the"
        "task.",
        ""
        "facet-terminal-facet-terminal-facet-terminal-tasks-6k-dc12f7ca9b079f5e97a6cf5d363fe962c8d798ba1c4cda2958a8ccd3f64eb7ed:"
        "Two material issues. (1) session_state: the public contract says browser_snapshot contains outcome"
        'and session_state "taken from the snapshot result", and the global constraint says strings must be'
        "copied exactly with no transformation. But no fixture field is named session_state; the oracle"
        'instead fabricates a derived value: session_metadata["browser"] + " " + browser_version ("chromium'
        '122.0.6261.94"). This concatenation is not derivable from the public instructions — an actor'
        'copying a single source field (e.g., session_id, or "chromium", or browser_version alone) produces'
        "equally defensible answers that the reference would reject. (2) cheapest_detail_url: the"
        'instruction says results are "sorted by ascending price" and asks for "the detailUrl of the first'
        "result in the ascending-price list\", but the fixture's results array is NOT price-sorted — flt-011"
        "(F9 222, expired, $156) appears last while flt-001 ($178.50) is first. The oracle acknowledges this"
        "and picks the literal first element (flt-001). An actor who sorts by price gets flt-011 ($156), an"
        'equally valid reading of "ascending-price list"; the manifest\'s noted quirks (expired fare below'
        "valid cheapest) make both readings defensible, so the reference's single witness is"
        'under-determined by the public text. Remaining derivations (weflow REACHABLE, flyai "Overall:'
        'OPERATIONAL", provider PASS checks, download sizes 2147483648 vs 4294967296, qdrant delta 3, habit'
        "name/frequency, spotify counts) check out against visible fixtures.",
        ""
        "facet-terminal-facet-terminal-facet-terminal-tasks-6k-f09a218625ef1369f293ce4142c3f715068b8e34589908b12eb40b0b5bec77a7:"
        "The task's structure and most mappings are coherent, but several required fields are"
        'under-specified relative to the actual fixtures. (1) clipboard_match_results requires entry_id "as'
        'present in the log", yet clipboard entries contain no id field — only timestamp, content_type,'
        "preview, length. The oracle uses timestamp as a stand-in id, which the public contract does not"
        'define. (2) query_terms must be "the original query terms extracted from the OCR result", but the'
        "OCR extracted_text is a dashboard summary with no explicit query terms; the oracle invents a"
        "stop-word-filtered word/ticker extraction heuristic that is not derivable from public instructions."
        '(3) review_completed and audit_completed require booleans "true only when the report indicates the'
        'review/audit is complete", but neither the safety report nor the accessibility report contains an'
        "explicit completion indicator; the oracle derives these from presence of review_comments and"
        "summary fields. (4) accessibility_summary classification is ambiguous: impacts are"
        'serious/moderate/minor with zero "critical" issues, while the spec only defines "critical" and'
        '"warning" buckets; the oracle\'s choice (non-critical → warnings) is one of several defensible'
        "readings. Each defect allows valid alternative answers that a strict grader could reject. The"
        "oracle script is also truncated in the preview, so full reference agreement cannot be confirmed.",
        ""
        "facet-terminal-facet-terminal-facet-terminal-tasks-6k-f73b9473fc8a8b18da54d9cc7c1ba43b238816bf705225daf27f7d7cf4a9c365:"
        'The summary-file spec contains a concrete internal contradiction. Early in the instructions: "If no'
        "blocking failures exist, the conclusion must state that the environment is ready and the list must"
        'be empty." Later, in the readiness_summary.txt section: "If `overall_ready` is `true`, the list'
        'must contain exactly the line `- None`." An empty list and a list containing exactly "- None" are'
        "mutually exclusive, so a solver cannot determine the required format for the no-blocking-failures"
        "case, and an exact-match grader will reject one of the two valid readings. The oracle"
        "(solution/solve.sh) is truncated in the available evidence, so it cannot confirm which reading the"
        "reference follows (reference agreement unknown). All other public requirements (manifest keys,"
        "field mappings, overall_ready aggregation conditions) are internally consistent and answerable from"
        "the provided fixtures.",
        ""
        "facet-terminal-facet-terminal-facet-terminal-tasks-6k-f9c404a3b42ef63a47f014cd521f5591fecdbf08677e2a05cc281aa01d2f40f0:"
        "The public instructions require coupon expiry to be determined \"by comparing each coupon's"
        '`expiry_date` field against the current date", but the fixture'
        "`/task_file/input/fixtures/mcdonalds_coupons.json` contains no `expiry_date` field — coupons have"
        "`valid_from`/`valid_until` and a `status` field. This mismatch makes the intended derivation"
        'ambiguous (valid_until vs status=="expired" yield different expired_count values, and the "current'
        'date" is runtime-dependent, so all coupons may be expired). The rest of the task is coherent:'
        "snapshots, trigger rules, config, and fixtures are all present and mutually consistent, and the"
        "required outputs are well specified. The oracle (solution/solve.sh) is truncated in preview, so"
        "full reference agreement cannot be confirmed.",
    ],
    repair_recommendation=(
        "Inspect all task-level issue evidence and qualify repairs with renewed QA. Resolve unavailable and uncertain"
        "observations before source-wide conclusions."
    ),
    unavailable_count=1,
    uncertain_count=0,
    sample_count=100,
    issue_rate=0.36363636363636365,
    verdict=Verdict.SOME_ISSUES,
)


def sources() -> list[RlDataSource[PrebuiltCohort]]:
    return [
        RlDataSource(
            pipeline=import_taskcompendium_prebuilt,
            config=PrebuiltCohort(
                intended_use=IntendedUse.TRAIN,
                name="facet-terminal-tasks-6k",
                version=CONVERTER_VERSION,
                repo=PREPARED_REPO,
                revision=PREPARED_REVISION,
                expected_rows=COUNT,
            ),
            info=SourceInfo(
                data_restriction=DataRestriction.OPEN,
                origin=SourceOrigin.FACET_TERMINAL,
                upstream_dataset=UPSTREAM,
                upstream_name="FACET-Terminal-Tasks-6k",
                upstream_url=f"https://huggingface.co/datasets/{UPSTREAM}/tree/{UPSTREAM_REVISION}",
                build_path=(
                    "Prepared native-image pin with the versioned SHARED-contract repair "
                    f"({REPAIR_PROTOCOL}): RootDeliveryContracts derived from each package's pinned "
                    "private verifier files"
                ),
                family="terminal",
                tags=("license:apache-2.0", "facet-terminal", "terminal", "pristine-image", "root-delivery"),
                task_format=TaskFormat.TASKSPEC,
                count=COUNT,
                quality_result=SAMPLE_QUALITY_RESULT,
                notes=(
                    "ADMIRATION state (coordinator ruling 2026-11-25 08:12 EST): admitted. The repaired chain "
                    "completed: the canonical controls (5,961 of 6,010 at the oracle-filtered private release "
                    "open-athena/facet-terminal-tasks-6k-oracle-filtered-20261124-contract-v1 @ 34b8ec2975df6836"
                    "9ad6e4bbd04a1af475a0a2ed; 49 typed oracle failures), the wire-4 refilter over the final "
                    "4-receipt chain, the filtered-pin census (lexical 0 contaminated, within-source 0 "
                    "duplicates), and the qualification binding at the filtered pin. The SAMPLE QC (V4, seed "
                    "0/size 100) reviewed 99 with 1 unavailable (<=5% allowed) and returned SOME_ISSUES "
                    "(36 issues, 36.4%). QC never rejects a source by itself: the source is admitted as-is with "
                    "its weakness documented - the sealed triage is "
                    + SAMPLE_QC_TRIAGE
                    + " (sha "
                    + SAMPLE_QC_TRIAGE_SHA256
                    + ", from the sealed QC campaign "
                    + SAMPLE_QC_CAMPAIGN
                    + "). The "
                    "repair campaign is a follow-up, not an admission blocker: the source-wide detector scan "
                    "(fixture-field mismatches + missing referenced paths) records the repair-candidate class "
                    "sizes. Coordinator admission decision 2026-11-25 08:12 EST."
                ),
            ),
        )
    ]

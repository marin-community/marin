# defensive — detector prompt

## AGENTS.md anchor

§ Error Handling — "Let exceptions propagate by default. Only catch to add
meaningful context and re-raise, or to intentionally alter control flow.
NEVER swallow exceptions unless specifically requested. Assert liberally;
prefer `raise ValueError` over silent fallbacks." § LLM-Generated Code
Pitfalls — over-protective try/except and defensive None checks. Companion
detector: `config-explicitness` owns "where does this knob come from"; this
detector owns "what happens when the value is missing/wrong."

## What to look for

Flag code that silently masks bugs with fallbacks, overly broad exception
handlers, or insufficient guards at system boundaries. Look for absent input
validation, fallback identifiers that collide under edge cases, and
`try/except` blocks that hide rather than propagate errors. This detector
owns the "silently trimming" / "fail-fast" family of findings.

## Anchor examples

1. **Fallback identifiers that collide silently** (yonromai):
   - Code shape: `fallback_id = f"{basename}::{idx}"` without stable path
   - Problem: When multiple data sources contain files with the same name,
     fallback IDs collide silently, merging distinct records into one
     `eval_id`. The contract promised per-record attribution; this breaks it
     invisibly.

2. **Try/except fallback instead of fail-fast** (wmoss):
   - Code shape: `try` clause with fallback path instead of fail
   - Problem: "have generally subscribed to a fail-fast mentality... having
     this fallback is more likely to confuse someone." Silent fallbacks
     obscure whether code is handling a real case or a bug.

3. **Validation bypass** (yonromai):
   - Code shape: `request.sql` executed with only a pooled connection guard
     (read-snapshot mode)
   - Problem: A pooled read-only snapshot is not a sufficient guard;
     `PRAGMA query_only = OFF` executes first, flips connection state, then
     fails later. An explicit allowlist was removed, leaving the gap. (If
     the finding is auth/injection-shaped, route it to `/security-review`
     instead of this detector.)

4. **Compression fallback gap** (yonromai):
   - Code shape: `_SEND_COMPRESSION = ZstdCompression()` with only
     `accept_compression` for responses, no request fallback
   - Problem: Gzip-only downstream servers reject zstd requests as
     unsupported. Setting request compression without end-to-end
     compatibility is a silent failure under deployment variance.

5. **Exception swallowing → return None / default** (rjpower):
   - Code shape: `try: json.loads(...) except: return None` or
     `with suppress(Exception):` or "silently trimming hides correctness
     bugs."
   - Problem: Caller code proceeds without knowing the input was malformed,
     making wrong decisions on `None`. The fix is `raise ValueError(...)`
     with context, or an explicit opt-in flag (e.g. `max_rows`) that the
     caller must set.

6. **Guard ordering** (rjpower):
   - Code shape: Defensive `isinstance` guard *after* error handling instead
     of at boundary
   - Problem: Bare `except` or catch-all clauses need guards; an
     `isinstance(payload, dict)` should gate `.get()` calls, not follow them
     in the except clause.

## False-positive guidance

- **Real system boundaries** (network, filesystem, JSON deserialization):
  `try/except` with logging and graceful degradation is acceptable if the
  fallback is explicit and tested.
- **Type narrowing**: `isinstance` guards when genuinely mixing types (e.g.,
  union parameters) are not defensive; guard before dereferencing.
- **Documented compatibility shims** (e.g., imports from `_src` due to
  version pinning): if the fallback is explained and inevitable, allow it
  with a comment linking the issue.
- **Schema evolution**: optional fields with `.get(key, default)` are
  acceptable; collision-prone fallback *identifiers* are not.
- **Retries with backoff**: intentional exponential backoff for transient
  failures is not defensive; bare `time.sleep` loops or silent unlimited
  retries are.

## Suggested confidence floor

High confidence: fallback identifiers that can silently collide; absent input
validation at RPC/file boundaries; overly broad exception handlers
(`except Exception`) with no re-raise or explicit alternative. Lower
confidence: reasonable system-boundary guards if the fallback is tested and
documented. Security-flavored findings (auth bypass, injection, secrets) are
out of scope — route to `/security-review`.

# defensive — detector prompt

## What to look for

Flag code that silently masks bugs with fallbacks, overly broad exception handlers, or insufficient guards at system boundaries. Look for absent input validation, fallback identifiers that collide under edge cases, and try/except blocks that hide rather than propagate errors.

## Anchor examples

1. **yonromai** on fallback eval IDs:
   - Code shape: `fallback_id = f"{basename}::{idx}"` without stable path
   - Problem: When multiple data sources contain files with the same name, fallback IDs collide silently, merging distinct records into one `eval_id`. The contract promised per-record attribution; this breaks it invisibly.

2. **wmoss** on fail-fast:
   - Code shape: `try` clause with fallback path instead of fail
   - Problem: Reviewer: "have generally subscribed to a fail-fast mentality... having this fallback is more likely to confuse someone." Silent fallbacks obscure whether code is handling a real case or a bug.

3. **yonromai** on SQL validation bypass:
   - Code shape: `request.sql` executed with only a pooled connection guard (read-snapshot mode)
   - Problem: A pooled read-only snapshot is not a sufficient guard; `PRAGMA query_only = OFF` executes first, flips connection state, then fails later. The old explicit allowlist was removed, leaving a security gap.

4. **yonromai** on compression fallback gap:
   - Code shape: `_SEND_COMPRESSION = ZstdCompression()` with only `accept_compression` for responses, no request fallback
   - Problem: Gzip-only downstream servers reject zstd requests as unsupported. Setting request compression without end-to-end compatibility is a silent failure under deployment variance.

5. **rjpower** on exception swallowing:
   - Code shape: `try: json.loads(...) except: return None` or `with suppress(Exception):`
   - Problem: Reviewer: "silently trimming hides correctness bugs." Caller code can proceed without knowing the input was malformed, making wrong decisions on `None`.

6. **rjpower** on guard ordering:
   - Code shape: Defensive `isinstance` guard *after* error handling instead of at boundary
   - Problem: Bare `except` or catch-all clauses need guards; an `isinstance(payload, dict)` should gate `.get()` calls, not follow them in the except clause.

## False-positive guidance

- **Real system boundaries** (network, filesystem, JSON deserialization): `try/except` with logging and graceful degradation is acceptable if the fallback is explicit and tested.
- **Type narrowing**: `isinstance` guards when genuinely mixing types (e.g., union parameters) are not defensive; guard before dereferencing.
- **Documented compatibility shims** (e.g., imports from `_src` due to version pinning): if the fallback is explained and inevitable, allow it with a comment linking the issue.
- **Schema evolution**: optional fields with `.get(key, default)` are acceptable; collision-prone fallback *identifiers* are not.
- **Retries with backoff**: intentional exponential backoff for transient failures is not defensive; bare `time.sleep` loops or silent unlimited retries are.

## Suggested confidence floor

High confidence (P1): fallback identifiers that can silently collide; absent input validation at RPC/file boundaries; overly broad exception handlers (`except Exception`) with no re-raise or explicit alternative. Lower confidence: reasonable system-boundary guards if the fallback is tested and documented.

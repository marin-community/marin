//! Pure log-read logic for `LogService::fetch_logs`.
//!
//! No I/O: the predicate builders produce a SQL fragment the query engine runs
//! over the sealed `log` segments, and the shaper turns result rows into
//! `LogEntry`s.
//!
//! Match-scope semantics (the load-bearing contract):
//! - EXACT: `key = <literal source> AND seq > cursor`. The source is a literal;
//!   regex metacharacters are never reinterpreted. `attempt_id` comes from the
//!   one `exact_key`.
//! - PREFIX: `seq > cursor` plus `prefix(key, <source>)`. Empty source ->
//!   `invalid_argument` ("source is required"). `attempt_id` is per-row.
//! - REGEX: `seq > cursor` plus `prefix(key, <leading literal>)` and, for a
//!   non-pure-prefix pattern, `regexp_matches(key, <pattern>)`. `attempt_id` is
//!   per-row.
//!
//! These builders emit the `prefix()` / `regexp_matches()` predicates as the
//! user's intent. The `prefix()` UDF alone is opaque to statistics pruning, so
//! the half-open key range that makes a prefix tail fast (`key >= P AND
//! key < succ(P)`) is added by the [`crate::query::optimizer::PrefixRangeRewrite`]
//! analyzer rule at plan time — one place, applied to both this path and the
//! generic Query API. The rule is tautology-preserving, so emitting `prefix()`
//! here is correct on its own; the range is purely a pruning optimization.

use buffa::Enumeration;

use crate::proto::finelog::logging::{LogLevel, MatchScope};

/// Characters that mark the end of a regex's leading literal prefix.
const REGEX_METACHARS: &[char] = &[
    '.', '*', '+', '?', '[', ']', '(', ')', '{', '}', '^', '$', '|', '\\',
];

/// Convert a level name (e.g. `"INFO"`) to the LogLevel enum, returning
/// `LOG_LEVEL_UNKNOWN` for `None`/empty/unrecognized. Case-insensitive.
pub fn str_to_log_level(level_name: &str) -> LogLevel {
    match level_name.to_ascii_uppercase().as_str() {
        "DEBUG" => LogLevel::LOG_LEVEL_DEBUG,
        "INFO" => LogLevel::LOG_LEVEL_INFO,
        "WARNING" => LogLevel::LOG_LEVEL_WARNING,
        "ERROR" => LogLevel::LOG_LEVEL_ERROR,
        "CRITICAL" => LogLevel::LOG_LEVEL_CRITICAL,
        _ => LogLevel::LOG_LEVEL_UNKNOWN,
    }
}

/// Best-effort attempt-id from a structured key: keys ending `...:<int>` carry
/// the int (e.g. `/user/job/0:3` -> 3); else 0.
pub fn parse_attempt_id(key: &str) -> i32 {
    let Some((_, suffix)) = key.rsplit_once(':') else {
        return 0;
    };
    if suffix.is_empty() {
        return 0;
    }
    suffix.parse::<i32>().unwrap_or(0)
}

/// The *guaranteed* literal prefix of a regex `pattern`: the run before the
/// first metacharacter, minus its final char when that metacharacter is a
/// quantifier that permits zero repetitions.
///
/// `*`, `?`, and `{…}` are postfix quantifiers binding to the *preceding* char
/// and allowing it to be absent, so that char is not guaranteed and is dropped
/// (`a?b` -> ``, `ab*c` -> `a`, `ab{0,3}c` -> `a`). `+` requires at least one
/// repetition, so its char is kept (`/a+b` -> `/a`); the remaining
/// metacharacters (`.`, `[`, `(`, `$`, `|`, `\`, …) do not bind to the char
/// before them, so the run stands (`/job/.*` -> `/job/`). `{` is treated
/// conservatively as zero-allowing without parsing the count — dropping a char
/// only forfeits pruning, never matches.
pub fn regex_literal_prefix(pattern: &str) -> &str {
    let Some((idx, meta)) = pattern
        .char_indices()
        .find(|(_, c)| REGEX_METACHARS.contains(c))
    else {
        return pattern;
    };
    let literal = &pattern[..idx];
    if !matches!(meta, '*' | '?' | '{') {
        return literal;
    }
    // The quantifier makes the char it binds to optional; drop it.
    match literal.char_indices().next_back() {
        Some((last, _)) => &literal[..last],
        None => literal,
    }
}

/// Escape a string literal for embedding in single-quoted SQL: double any
/// embedded single quote (`'` -> `''`). The literal carries verbatim through the
/// query engine — no regex/LIKE reinterpretation.
fn sql_literal(s: &str) -> String {
    format!("'{}'", s.replace('\'', "''"))
}

/// Push the `prefix(key, <prefix>)` predicate for keys starting with `prefix`.
///
/// This is the exact (and sole) residual; the statistics-prunable half-open key
/// range it implies is synthesized by
/// [`crate::query::optimizer::PrefixRangeRewrite`] at plan time, so it is not
/// hand-built here.
fn push_key_prefix_predicates(where_parts: &mut Vec<String>, prefix: &str) {
    where_parts.push(format!("prefix(key, {})", sql_literal(prefix)));
}

/// The WHERE-clause predicates for a FetchLogs read, plus how to shape the
/// result rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogPredicates {
    /// SQL boolean predicates ANDed together to form the WHERE clause.
    pub where_parts: Vec<String>,
    /// Whether `key` is included in the SELECT (needed for per-row attempt_id on
    /// multi-key scopes).
    pub include_key: bool,
    /// For EXACT scope, the single literal key whose `attempt_id` every row
    /// inherits; `None` for PREFIX/REGEX (attempt_id is per-row).
    pub exact_key: Option<String>,
}

/// Build the scope predicates for a FetchLogs read.
///
/// `source` is interpreted per `match_scope` (UNSPECIFIED must already have been
/// mapped to REGEX by the caller). `cursor` is EXCLUSIVE (`seq > cursor`).
/// Returns `Err` only for an empty PREFIX source.
pub fn build_log_predicates(
    source: &str,
    cursor: i64,
    match_scope: MatchScope,
) -> Result<LogPredicates, String> {
    match match_scope {
        MatchScope::MATCH_SCOPE_EXACT => Ok(LogPredicates {
            where_parts: vec![
                format!("key = {}", sql_literal(source)),
                format!("seq > {cursor}"),
            ],
            include_key: false,
            exact_key: Some(source.to_string()),
        }),
        MatchScope::MATCH_SCOPE_PREFIX => {
            if source.is_empty() {
                // An empty prefix would match every key in the store. Almost
                // always a caller bug; fail fast instead of paging everything.
                return Err("FetchLogs source is required for MATCH_SCOPE_PREFIX".to_string());
            }
            let mut where_parts = vec![format!("seq > {cursor}")];
            push_key_prefix_predicates(&mut where_parts, source);
            Ok(LogPredicates {
                where_parts,
                include_key: true,
                exact_key: None,
            })
        }
        MatchScope::MATCH_SCOPE_REGEX => {
            let literal_prefix = regex_literal_prefix(source);
            let suffix = &source[literal_prefix.len()..];
            // `^literal$`, `^literal`, `^literal.*` all reduce to the literal
            // prefix alone; any other suffix still needs regexp_matches.
            let is_pure_prefix = suffix == ".*" || suffix.is_empty();
            let mut where_parts = vec![format!("seq > {cursor}")];
            if !literal_prefix.is_empty() {
                push_key_prefix_predicates(&mut where_parts, literal_prefix);
            }
            if !is_pure_prefix {
                where_parts.push(format!("regexp_matches(key, {})", sql_literal(source)));
            }
            Ok(LogPredicates {
                where_parts,
                include_key: true,
                exact_key: None,
            })
        }
        // The wire UNSPECIFIED is mapped to REGEX before reaching here; any
        // other unknown value is a caller error.
        other => Err(format!("unknown match_scope: {other:?}")),
    }
}

/// Append the common filters (since_ms / substring / min_level) to `where_parts`.
///
/// - `since_ms > 0` -> `epoch_ms > since_ms`.
/// - non-empty `substring` -> `contains(data, <literal>)` (literal, not LIKE).
/// - `min_level > 0` -> `(level = 0 OR level >= min_level)` (UNKNOWN passthrough).
pub fn add_common_filters(
    where_parts: &mut Vec<String>,
    since_ms: i64,
    substring: &str,
    min_level: LogLevel,
) {
    if since_ms > 0 {
        where_parts.push(format!("epoch_ms > {since_ms}"));
    }
    if !substring.is_empty() {
        where_parts.push(format!("contains(data, {})", sql_literal(substring)));
    }
    let min_level_int = min_level.to_i32();
    if min_level_int > 0 {
        where_parts.push(format!("(level = 0 OR level >= {min_level_int})"));
    }
}

/// Append the origin-cluster filter to `where_parts`.
///
/// The `cluster` column carries the writer-supplied origin of each push, so
/// filtering on it namespaces a global finelog's logs by origin cluster. A
/// non-empty `cluster` restricts to rows whose column equals it exactly; empty
/// adds no predicate (all origins — the local single-cluster read behavior).
/// Segments written before the column existed store NULL, which
/// `cluster = <literal>` excludes — correct, since those rows predate federation.
pub fn add_cluster_filter(where_parts: &mut Vec<String>, cluster: &str) {
    if !cluster.is_empty() {
        where_parts.push(format!("cluster = {}", sql_literal(cluster)));
    }
}

/// One result row from the log read. `key` is `Some` only when the scope
/// included it in the SELECT.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogRow {
    pub seq: i64,
    pub key: Option<String>,
    pub source: String,
    pub data: String,
    pub epoch_ms: i64,
    pub level: i32,
}

/// The shaped result: entry fields + the cursor for the next poll.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShapedEntry {
    pub source: String,
    pub data: String,
    pub epoch_ms: i64,
    pub level: i32,
    pub attempt_id: i32,
    /// Populated on multi-key (PREFIX/REGEX) scopes; `None` for EXACT.
    pub key: Option<String>,
}

/// The shaped log-read result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShapedResult {
    pub entries: Vec<ShapedEntry>,
    pub cursor: i64,
}

/// Shape result `rows` into entries + cursor.
///
/// When `tail && max_lines > 0` the rows arrive in `seq DESC` order and are
/// reversed to ascending; the cursor is `max(seq)` over the returned rows, or
/// `default_cursor` (the request cursor) when empty. `attempt_id` is per-row
/// from `key` for multi-key scopes, or from `exact_key` for EXACT.
pub fn shape_log_read_result(
    mut rows: Vec<LogRow>,
    tail: bool,
    max_lines: i32,
    default_cursor: i64,
    include_key: bool,
    exact_key: Option<&str>,
) -> ShapedResult {
    if tail && max_lines > 0 {
        rows.reverse();
    }
    if rows.is_empty() {
        return ShapedResult {
            entries: Vec::new(),
            cursor: default_cursor,
        };
    }
    let max_seq = rows.iter().map(|r| r.seq).max().unwrap_or(default_cursor);
    let exact_attempt = exact_key.map(parse_attempt_id).unwrap_or(0);
    let entries = rows
        .into_iter()
        .map(|r| {
            let attempt_id = if include_key {
                r.key.as_deref().map(parse_attempt_id).unwrap_or(0)
            } else {
                exact_attempt
            };
            ShapedEntry {
                source: r.source,
                data: r.data,
                epoch_ms: r.epoch_ms,
                level: r.level,
                attempt_id,
                key: if include_key { r.key } else { None },
            }
        })
        .collect();
    ShapedResult {
        entries,
        cursor: max_seq,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_attempt_id_boundaries() {
        assert_eq!(parse_attempt_id("/user/job/0:3"), 3);
        assert_eq!(parse_attempt_id("/k"), 0);
        assert_eq!(parse_attempt_id("/k:"), 0);
        assert_eq!(parse_attempt_id("/k:abc"), 0);
        assert_eq!(parse_attempt_id("/a:1:2"), 2); // rsplit on the LAST ':'
    }

    #[test]
    fn str_to_log_level_mapping() {
        assert_eq!(str_to_log_level("DEBUG"), LogLevel::LOG_LEVEL_DEBUG);
        assert_eq!(str_to_log_level("info"), LogLevel::LOG_LEVEL_INFO); // case-insensitive
        assert_eq!(str_to_log_level("WARNING"), LogLevel::LOG_LEVEL_WARNING);
        assert_eq!(str_to_log_level("ERROR"), LogLevel::LOG_LEVEL_ERROR);
        assert_eq!(str_to_log_level("CRITICAL"), LogLevel::LOG_LEVEL_CRITICAL);
        assert_eq!(str_to_log_level(""), LogLevel::LOG_LEVEL_UNKNOWN);
        assert_eq!(str_to_log_level("bogus"), LogLevel::LOG_LEVEL_UNKNOWN);
    }

    #[test]
    fn regex_literal_prefix_on_metachars() {
        assert_eq!(regex_literal_prefix("/job/test/.*"), "/job/test/");
        assert_eq!(regex_literal_prefix("/job/test/0"), "/job/test/0"); // no metachar
        assert_eq!(regex_literal_prefix("^/job"), ""); // leading ^ is a metachar
        assert_eq!(regex_literal_prefix("/a+b"), "/a"); // `+` requires the char -> kept
                                                        // A zero-allowing quantifier (`*`/`?`/`{…}`) makes the char it binds to
                                                        // optional, so it is dropped from the guaranteed prefix.
        assert_eq!(regex_literal_prefix("a?b"), ""); // `a` optional
        assert_eq!(regex_literal_prefix("a*b"), ""); // `a` optional
        assert_eq!(regex_literal_prefix("ab?c"), "a"); // `b` optional, `a` kept
        assert_eq!(regex_literal_prefix("ab*c"), "a");
        assert_eq!(regex_literal_prefix("ab{0,3}c"), "a"); // counted -> conservative drop
    }

    #[test]
    fn exact_scope_is_literal_equality() {
        let p =
            build_log_predicates("/job/curation-9e+20", 0, MatchScope::MATCH_SCOPE_EXACT).unwrap();
        assert_eq!(
            p.where_parts,
            vec![
                "key = '/job/curation-9e+20'".to_string(),
                "seq > 0".to_string()
            ]
        );
        assert!(!p.include_key);
        assert_eq!(p.exact_key.as_deref(), Some("/job/curation-9e+20"));
    }

    #[test]
    fn exact_scope_escapes_single_quote() {
        let p = build_log_predicates("/job/o'brien", 5, MatchScope::MATCH_SCOPE_EXACT).unwrap();
        assert_eq!(p.where_parts[0], "key = '/job/o''brien'");
        assert_eq!(p.where_parts[1], "seq > 5");
    }

    #[test]
    fn prefix_scope_emits_prefix_residual() {
        // PREFIX emits seq filter + the prefix() residual; the prunable key range
        // is added later by the PrefixRangeRewrite analyzer rule, not here.
        let p = build_log_predicates("/job/test/0:", 7, MatchScope::MATCH_SCOPE_PREFIX).unwrap();
        assert_eq!(
            p.where_parts,
            vec![
                "seq > 7".to_string(),
                "prefix(key, '/job/test/0:')".to_string(),
            ]
        );
        assert!(p.include_key);
        assert_eq!(p.exact_key, None);
    }

    #[test]
    fn empty_prefix_source_rejected() {
        let err = build_log_predicates("", 0, MatchScope::MATCH_SCOPE_PREFIX).unwrap_err();
        assert!(err.contains("source is required"));
    }

    #[test]
    fn regex_pure_prefix_prunes_without_regexp_matches() {
        // `/job/test/.*` is a pure-prefix: just the prefix() residual, no regexp.
        // The key range it implies is added by the analyzer rule.
        let p = build_log_predicates("/job/test/.*", 0, MatchScope::MATCH_SCOPE_REGEX).unwrap();
        assert_eq!(
            p.where_parts,
            vec![
                "seq > 0".to_string(),
                "prefix(key, '/job/test/')".to_string(),
            ]
        );
        assert!(p.include_key);
    }

    #[test]
    fn regex_non_pure_prefix_adds_regexp_matches() {
        // A regex with an interior metachar pattern needs regexp_matches too, on
        // top of the leading-literal prefix() residual (whose range the rule adds).
        let p = build_log_predicates("/job/(a|b)/0", 2, MatchScope::MATCH_SCOPE_REGEX).unwrap();
        assert_eq!(
            p.where_parts,
            vec![
                "seq > 2".to_string(),
                "prefix(key, '/job/')".to_string(),
                "regexp_matches(key, '/job/(a|b)/0')".to_string(),
            ]
        );
    }

    #[test]
    fn regex_no_literal_prefix_only_regexp() {
        // Leading metachar -> no prefix prune, just regexp_matches.
        let p = build_log_predicates(".*foo", 0, MatchScope::MATCH_SCOPE_REGEX).unwrap();
        assert_eq!(
            p.where_parts,
            vec![
                "seq > 0".to_string(),
                "regexp_matches(key, '.*foo')".to_string()
            ]
        );
    }

    #[test]
    fn common_filters_since_substring_minlevel() {
        let mut wp = vec!["seq > 0".to_string()];
        add_common_filters(&mut wp, 1000, "100%", LogLevel::LOG_LEVEL_WARNING);
        assert_eq!(
            wp,
            vec![
                "seq > 0".to_string(),
                "epoch_ms > 1000".to_string(),
                "contains(data, '100%')".to_string(),
                "(level = 0 OR level >= 3)".to_string(),
            ]
        );
    }

    #[test]
    fn cluster_filter_appends_escaped_equality_or_nothing() {
        let mut wp = vec!["seq > 0".to_string()];
        add_cluster_filter(&mut wp, "alpha");
        assert_eq!(
            wp,
            vec!["seq > 0".to_string(), "cluster = 'alpha'".to_string()]
        );
        // Empty cluster adds no predicate (unfiltered / local read).
        let mut wp = vec!["seq > 0".to_string()];
        add_cluster_filter(&mut wp, "");
        assert_eq!(wp, vec!["seq > 0".to_string()]);
        // A single quote in the cluster name is SQL-escaped.
        let mut wp: Vec<String> = vec![];
        add_cluster_filter(&mut wp, "o'brien");
        assert_eq!(wp, vec!["cluster = 'o''brien'".to_string()]);
    }

    #[test]
    fn common_filters_min_level_unknown_passthrough() {
        // min_level UNKNOWN (0) adds no level filter (UNKNOWN means "no minimum").
        let mut wp = vec!["seq > 0".to_string()];
        add_common_filters(&mut wp, 0, "", LogLevel::LOG_LEVEL_UNKNOWN);
        assert_eq!(wp, vec!["seq > 0".to_string()]);
    }

    fn row(seq: i64, key: &str, data: &str) -> LogRow {
        LogRow {
            seq,
            key: Some(key.to_string()),
            source: "stdout".to_string(),
            data: data.to_string(),
            epoch_ms: seq,
            level: 2,
        }
    }

    #[test]
    fn shape_tail_reverses_and_sets_cursor_max_seq() {
        // tail rows arrive seq DESC; shaping reverses to ascending.
        let rows = vec![row(5, "/k", "e"), row(4, "/k", "d"), row(3, "/k", "c")];
        let shaped = shape_log_read_result(rows, true, 3, 0, true, None);
        assert_eq!(
            shaped
                .entries
                .iter()
                .map(|e| e.data.as_str())
                .collect::<Vec<_>>(),
            vec!["c", "d", "e"]
        );
        assert_eq!(shaped.cursor, 5);
    }

    #[test]
    fn shape_empty_falls_back_to_default_cursor() {
        let shaped = shape_log_read_result(vec![], false, 0, 42, false, Some("/k"));
        assert!(shaped.entries.is_empty());
        assert_eq!(shaped.cursor, 42);
    }

    #[test]
    fn shape_exact_attempt_id_from_exact_key() {
        // EXACT: include_key=false; attempt_id from the single exact_key.
        let mut r = row(1, "/job/0:3", "x");
        r.key = None; // EXACT does not select key
        let shaped = shape_log_read_result(vec![r], false, 0, 0, false, Some("/job/0:3"));
        assert_eq!(shaped.entries[0].attempt_id, 3);
        assert_eq!(shaped.entries[0].key, None);
        assert_eq!(shaped.cursor, 1);
    }

    #[test]
    fn shape_multi_key_attempt_id_per_row() {
        let rows = vec![row(1, "/job/a/0:0", "a"), row(2, "/job/a/1:5", "b")];
        let shaped = shape_log_read_result(rows, false, 0, 0, true, None);
        assert_eq!(shaped.entries[0].attempt_id, 0);
        assert_eq!(shaped.entries[1].attempt_id, 5);
        assert_eq!(shaped.entries[0].key.as_deref(), Some("/job/a/0:0"));
        assert_eq!(shaped.cursor, 2);
    }
}

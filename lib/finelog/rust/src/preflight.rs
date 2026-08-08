//! Decide, without a store or a network, whether this image would register.
//!
//! A finelog image registers `log` (`Store::new`) and `telemetry_v1` (the
//! telemetry router's startup task) for itself. Both go through
//! `RegisterTable`, which merges the requested schema against whatever the
//! catalog holds. A merge the catalog rejects wedges that namespace for as long
//! as the image is deployed: the server listens, `/health` is green, and every
//! write to the namespace fails.
//!
//! [`check`] takes the registered side as JSON and runs it through the same
//! [`merge_schemas`] the server calls.
//!
//! Scope: only the two server-owned namespaces. A namespace a client registers
//! (`iris.worker`, zephyr's tables) is reported as out of scope.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use serde::Deserialize;

use crate::errors::StatsError;
use crate::server::telemetry::{telemetry_schema, TELEMETRY_NAMESPACE};
use crate::store::schema::{
    merge_schemas, schema_from_json, stored_form, with_implicit_seq, Schema,
};
use crate::store::store::{log_registered_schema, LOG_NAMESPACE_NAME};

/// The namespaces this image registers for itself, in the store form
/// `RegisterTable` merges.
pub fn server_owned_schemas() -> Vec<(&'static str, Schema)> {
    vec![
        (LOG_NAMESPACE_NAME, stored_form(log_registered_schema())),
        (TELEMETRY_NAMESPACE, stored_form(telemetry_schema())),
    ]
}

/// The registered side of the decision: what a deployment's catalog holds.
///
/// Written by `safe_deploy` from `ListNamespaces` against a live server, and
/// checked in per deployment as the deploy golden. Each schema is in the catalog
/// JSON sidecar form (see `schema_from_json`). Metadata keys the writer adds —
/// which deployment, when, from what — are ignored here.
#[derive(Debug, Deserialize)]
pub struct RegisteredSchemas {
    /// Registered schema per namespace name.
    pub namespaces: BTreeMap<String, String>,
}

impl RegisteredSchemas {
    /// Parse the document. Each namespace's schema is a nested JSON object, kept
    /// as raw text here and decoded per namespace so one unparseable namespace
    /// names itself instead of failing the whole document.
    pub fn parse(text: &str) -> Result<RegisteredSchemas, StatsError> {
        #[derive(Deserialize)]
        struct Document {
            #[serde(default)]
            namespaces: BTreeMap<String, serde_json::Value>,
        }
        let document: Document = serde_json::from_str(text)
            .map_err(|e| StatsError::SchemaValidation(format!("registered schema JSON: {e}")))?;
        Ok(RegisteredSchemas {
            namespaces: document
                .namespaces
                .into_iter()
                .map(|(name, value)| (name, value.to_string()))
                .collect(),
        })
    }
}

/// What registering this image's schema would do to a namespace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    /// The namespace is not registered yet; this image creates it.
    Creates,
    /// The registered schema already covers the request; nothing changes.
    Unchanged,
    /// Registration succeeds and rewrites the stored schema. The consequences
    /// are listed per namespace: added columns, newly enabled indexes, and
    /// superseded covering projections, whose derived Parquet the index
    /// backfill rebuilds.
    Evolves,
    /// The merge is rejected. `RegisterTable` fails for as long as this image
    /// is deployed, and every write to the namespace fails with it.
    Fails,
}

impl Decision {
    fn label(self) -> &'static str {
        match self {
            Decision::Creates => "CREATES",
            Decision::Unchanged => "UNCHANGED",
            Decision::Evolves => "EVOLVES",
            Decision::Fails => "FAILS",
        }
    }
}

/// The decision for one server-owned namespace.
#[derive(Debug, Clone)]
pub struct NamespaceOutcome {
    pub namespace: String,
    pub decision: Decision,
    /// The consequences of an `Evolves`, or the merge error for a `Fails`. Empty
    /// for the other two.
    pub notes: Vec<String>,
}

/// The whole decision for one deployment's catalog.
#[derive(Debug, Clone)]
pub struct Report {
    pub outcomes: Vec<NamespaceOutcome>,
    /// Registered namespaces this image does not own, so does not decide for.
    pub out_of_scope: Vec<String>,
}

impl Report {
    /// Whether every server-owned namespace would register.
    pub fn passes(&self) -> bool {
        !self
            .outcomes
            .iter()
            .any(|outcome| outcome.decision == Decision::Fails)
    }
}

/// Decide each server-owned namespace against `registered`.
///
/// A namespace absent from `registered` is a fresh registration, which always
/// succeeds; a namespace present but not server-owned is out of scope.
pub fn check(registered: &RegisteredSchemas) -> Result<Report, StatsError> {
    let owned = server_owned_schemas();
    let mut outcomes = Vec::with_capacity(owned.len());
    for (namespace, requested) in &owned {
        outcomes.push(decide(
            namespace,
            requested,
            registered.namespaces.get(*namespace),
        )?);
    }
    let out_of_scope = registered
        .namespaces
        .keys()
        .filter(|name| !owned.iter().any(|(owned_name, _)| owned_name == name))
        .cloned()
        .collect();
    Ok(Report {
        outcomes,
        out_of_scope,
    })
}

fn decide(
    namespace: &str,
    requested: &Schema,
    registered_json: Option<&String>,
) -> Result<NamespaceOutcome, StatsError> {
    let Some(registered_json) = registered_json else {
        return Ok(NamespaceOutcome {
            namespace: namespace.to_string(),
            decision: Decision::Creates,
            notes: Vec::new(),
        });
    };
    // `GetTableSchema` strips the server-assigned `seq` column on the way out,
    // so restore it before merging: `merge_schemas` sees the catalog's store
    // form, and the requested side already carries it. A document that kept
    // `seq` (a catalog row read directly) is unaffected — the helper is a no-op
    // when the column is present.
    let registered = with_implicit_seq(
        schema_from_json(registered_json)
            .map_err(|e| StatsError::SchemaValidation(format!("namespace {namespace:?}: {e}")))?,
    );
    match merge_schemas(&registered, requested) {
        Err(error) => Ok(NamespaceOutcome {
            namespace: namespace.to_string(),
            decision: Decision::Fails,
            notes: vec![error.to_string()],
        }),
        Ok(merged) if merged == registered => Ok(NamespaceOutcome {
            namespace: namespace.to_string(),
            decision: Decision::Unchanged,
            notes: Vec::new(),
        }),
        Ok(merged) => Ok(NamespaceOutcome {
            namespace: namespace.to_string(),
            decision: Decision::Evolves,
            notes: evolution_notes(&registered, &merged),
        }),
    }
}

/// Describe what an accepted merge changes, so an `EVOLVES` says what it costs.
fn evolution_notes(registered: &Schema, merged: &Schema) -> Vec<String> {
    let mut notes = Vec::new();
    for column in &merged.columns {
        match registered.column(&column.name) {
            None => notes.push(format!(
                "adds column {:?} ({})",
                column.name,
                if column.nullable {
                    "nullable"
                } else {
                    "non-nullable"
                }
            )),
            Some(existing) => {
                if column.index.trigram && !existing.index.trigram {
                    notes.push(format!("enables the trigram index on {:?}", column.name));
                }
                if column.index.value_counts && !existing.index.value_counts {
                    notes.push(format!("enables value counts on {:?}", column.name));
                }
                let added: Vec<&String> = column
                    .index
                    .exact_values
                    .iter()
                    .filter(|value| !existing.index.exact_values.contains(value))
                    .collect();
                if !added.is_empty() {
                    notes.push(format!(
                        "adds exact values on {:?}: {}",
                        column.name,
                        added
                            .iter()
                            .map(|value| value.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ));
                }
            }
        }
    }
    for projection in &merged.projections {
        match registered
            .projections
            .iter()
            .find(|existing| existing.name == projection.name)
        {
            None => notes.push(format!("adds covering projection {:?}", projection.name)),
            Some(existing) if existing != projection => notes.push(format!(
                "supersedes covering projection {:?}; the index backfill rebuilds its Parquet, \
                 and covered segments keep serving the registered definition until it does",
                projection.name
            )),
            Some(_) => {}
        }
    }
    for config in &merged.grouped_extrema {
        if !registered.grouped_extrema.contains(config) {
            notes.push(format!(
                "adds grouped extrema on {:?} by {:?}",
                config.filter_column, config.json_key
            ));
        }
    }
    notes
}

/// Render the report as the pre-flight's operator-facing output.
pub fn render(report: &Report) -> String {
    let mut out = String::new();
    for outcome in &report.outcomes {
        let _ = writeln!(out, "{:<9} {}", outcome.decision.label(), outcome.namespace);
        for note in &outcome.notes {
            let _ = writeln!(out, "            {note}");
        }
    }
    if report.out_of_scope.is_empty() {
        out.push_str("no other namespaces registered\n");
    } else {
        let _ = writeln!(
            out,
            "not checked (registered by clients, not by this image): {}",
            report.out_of_scope.join(", ")
        );
    }
    let _ = writeln!(
        out,
        "{}",
        if report.passes() {
            "PASS — every server-owned namespace registers"
        } else {
            "FAIL — a server-owned namespace would not register; ingest into it would stop"
        }
    );
    out
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::proto::finelog::stats::ColumnType;
    use crate::store::schema::{schema_to_json, Column};

    /// Build a document from schemas already in store form.
    fn document(entries: &[(&str, &Schema)]) -> RegisteredSchemas {
        let namespaces = entries
            .iter()
            .map(|(name, schema)| (name.to_string(), schema_to_json(schema)))
            .collect::<Vec<_>>();
        let body = namespaces
            .iter()
            .map(|(name, json)| format!("{}: {json}", serde_json::to_string(name).unwrap()))
            .collect::<Vec<_>>()
            .join(",");
        RegisteredSchemas::parse(&format!("{{\"namespaces\":{{{body}}}}}")).unwrap()
    }

    fn owned(namespace: &str) -> Schema {
        server_owned_schemas()
            .into_iter()
            .find(|(name, _)| *name == namespace)
            .map(|(_, schema)| schema)
            .expect("server-owned namespace")
    }

    fn outcome<'a>(report: &'a Report, namespace: &str) -> &'a NamespaceOutcome {
        report
            .outcomes
            .iter()
            .find(|outcome| outcome.namespace == namespace)
            .expect("namespace decided")
    }

    /// The registered side as `GetTableSchema` returns it: store form minus the
    /// server-assigned `seq` column.
    fn as_wire(schema: &Schema) -> Schema {
        let mut wire = schema.clone();
        wire.columns.retain(|column| column.name != "seq");
        wire
    }

    #[test]
    fn own_schemas_register_against_their_own_registered_form() {
        let report = check(&document(&[
            (LOG_NAMESPACE_NAME, &as_wire(&owned(LOG_NAMESPACE_NAME))),
            (TELEMETRY_NAMESPACE, &as_wire(&owned(TELEMETRY_NAMESPACE))),
        ]))
        .unwrap();

        assert!(report.passes());
        for outcome in &report.outcomes {
            assert_eq!(outcome.decision, Decision::Unchanged, "{outcome:?}");
        }
    }

    #[test]
    fn an_unregistered_namespace_is_a_fresh_creation() {
        let report = check(&RegisteredSchemas::parse("{\"namespaces\":{}}").unwrap()).unwrap();

        assert!(report.passes());
        assert_eq!(
            outcome(&report, TELEMETRY_NAMESPACE).decision,
            Decision::Creates
        );
    }

    #[test]
    fn a_type_change_fails_and_names_the_column() {
        let mut registered = as_wire(&owned(TELEMETRY_NAMESPACE));
        let value = registered
            .columns
            .iter_mut()
            .find(|column| column.name == "value")
            .unwrap();
        value.r#type = ColumnType::COLUMN_TYPE_STRING;
        let report = check(&document(&[(TELEMETRY_NAMESPACE, &registered)])).unwrap();

        assert!(!report.passes());
        let decided = outcome(&report, TELEMETRY_NAMESPACE);
        assert_eq!(decided.decision, Decision::Fails);
        assert!(
            decided.notes.iter().any(|note| note.contains("\"value\"")),
            "{decided:?}"
        );
        assert!(render(&report).contains("ingest into it would stop"));
    }

    #[test]
    fn a_redefined_projection_evolves_and_reports_the_rebuild() {
        // A registered projection name whose definition this image has changed.
        // It supersedes rather than conflicts, so the pre-flight passes — and
        // says what the acceptance costs.
        let mut registered = as_wire(&owned(TELEMETRY_NAMESPACE));
        let projection = registered
            .projections
            .iter_mut()
            .find(|projection| projection.name == "training-status")
            .unwrap();
        projection.columns.retain(|column| column != "value");
        let report = check(&document(&[(TELEMETRY_NAMESPACE, &registered)])).unwrap();

        assert!(report.passes());
        let decided = outcome(&report, TELEMETRY_NAMESPACE);
        assert_eq!(decided.decision, Decision::Evolves);
        assert!(
            decided
                .notes
                .iter()
                .any(|note| note.contains("supersedes covering projection \"training-status\"")),
            "{decided:?}"
        );
    }

    #[test]
    fn a_registered_column_this_image_dropped_is_kept_without_a_note() {
        // Removing a column from the image's schema is not a change the catalog
        // applies: the registered column stays. The pre-flight must not report a
        // change that does not happen.
        let mut registered = as_wire(&owned(LOG_NAMESPACE_NAME));
        registered
            .columns
            .push(Column::new("retired", ColumnType::COLUMN_TYPE_STRING, true));
        let report = check(&document(&[(LOG_NAMESPACE_NAME, &registered)])).unwrap();

        let decided = outcome(&report, LOG_NAMESPACE_NAME);
        assert_eq!(decided.decision, Decision::Unchanged);
        assert!(decided.notes.is_empty(), "{decided:?}");
    }

    #[test]
    fn a_new_column_and_index_are_reported_as_the_evolution() {
        let mut registered = as_wire(&owned(TELEMETRY_NAMESPACE));
        registered.columns.retain(|column| column.name != "unit");
        let name = registered
            .columns
            .iter_mut()
            .find(|column| column.name == "name")
            .unwrap();
        name.index.trigram = false;
        let report = check(&document(&[(TELEMETRY_NAMESPACE, &registered)])).unwrap();

        let decided = outcome(&report, TELEMETRY_NAMESPACE);
        assert_eq!(decided.decision, Decision::Evolves);
        assert!(
            decided
                .notes
                .iter()
                .any(|note| note == "adds column \"unit\" (nullable)"),
            "{decided:?}"
        );
        assert!(
            decided
                .notes
                .iter()
                .any(|note| note == "enables the trigram index on \"name\""),
            "{decided:?}"
        );
    }

    #[test]
    fn client_owned_namespaces_are_reported_as_out_of_scope() {
        let client = Schema::new(
            vec![Column::new(
                "timestamp_ms",
                ColumnType::COLUMN_TYPE_INT64,
                false,
            )],
            "timestamp_ms",
        );
        let report = check(&document(&[("iris.worker", &client)])).unwrap();

        assert_eq!(report.out_of_scope, vec!["iris.worker".to_string()]);
        assert!(render(&report).contains("registered by clients, not by this image"));
    }

    #[test]
    fn a_namespace_with_an_unreadable_schema_names_itself() {
        let error = check(
            &RegisteredSchemas::parse(
                "{\"namespaces\":{\"telemetry_v1\":{\"key_column\":\"t\",\"columns\":[{\"name\":\"t\",\"type\":\"nope\",\"nullable\":false}]}}}",
            )
            .unwrap(),
        )
        .unwrap_err();

        assert!(error.to_string().contains("telemetry_v1"), "{error}");
    }
}

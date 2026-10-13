//! Logical and physical layout policies for complete ingestion batches.

use std::collections::HashMap;

use arrow::record_batch::RecordBatch;

use crate::errors::StatsError;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum IngestionBatchSource<'a> {
    /// A namespace selected by the writer at the ingestion boundary.
    Declared(&'a str),
    /// A namespace already attached to persisted or forwarded rows.
    Stored(&'a str),
}

impl<'a> IngestionBatchSource<'a> {
    pub(crate) fn namespace(self) -> &'a str {
        match self {
            Self::Declared(namespace) | Self::Stored(namespace) => namespace,
        }
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct IngestionDestination {
    pub logical_namespace: String,
}

#[derive(Debug)]
pub(crate) struct RoutedIngestionBatch {
    pub destination: IngestionDestination,
    pub batch: RecordBatch,
}

pub(crate) trait IngestionPolicy: Sync {
    /// Partition a complete batch into logical destinations.
    fn route_batch(
        &self,
        source: IngestionBatchSource<'_>,
        batch: &RecordBatch,
        state: &mut IngestionState,
    ) -> Result<Vec<RoutedIngestionBatch>, StatsError>;

    /// Index state needed to route an immutable migration independently of
    /// the source files' physical row order.
    fn index_migration_batch(
        &self,
        _source: IngestionBatchSource<'_>,
        _batch: &RecordBatch,
        _state: &mut IngestionState,
    ) -> Result<(), StatsError> {
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct StepCursor {
    pub timestamp_ms: i64,
    pub order: i64,
    pub step: i64,
}

#[derive(Debug, Default)]
pub(crate) struct IngestionState {
    levanter_steps: HashMap<(String, i64), StepCursor>,
    levanter_step_history: HashMap<(String, i64), Vec<StepCursor>>,
}

impl IngestionState {
    pub(crate) fn update_levanter_step(
        &mut self,
        execution_uid: String,
        process_index: i64,
        candidate: StepCursor,
    ) {
        self.levanter_steps
            .entry((execution_uid, process_index))
            .and_modify(|current| {
                if cursor_position(candidate) >= cursor_position(*current) {
                    *current = candidate;
                }
            })
            .or_insert(candidate);
    }

    pub(crate) fn index_levanter_step(
        &mut self,
        execution_uid: String,
        process_index: i64,
        cursor: StepCursor,
    ) {
        self.levanter_step_history
            .entry((execution_uid, process_index))
            .or_default()
            .push(cursor);
    }

    pub(crate) fn finish_migration_index(&mut self) {
        for history in self.levanter_step_history.values_mut() {
            history.sort_unstable_by_key(|cursor| cursor_position(*cursor));
            history.dedup_by_key(|cursor| cursor_position(*cursor));
        }
    }

    pub(crate) fn levanter_step_at(
        &self,
        execution_uid: &str,
        process_index: i64,
        timestamp_ms: i64,
        order: i64,
    ) -> Option<i64> {
        let key = (execution_uid.to_string(), process_index);
        let position = (timestamp_ms, order);
        let historical = self.levanter_step_history.get(&key).and_then(|history| {
            let index = history.partition_point(|cursor| cursor_position(*cursor) <= position);
            index.checked_sub(1).map(|index| history[index])
        });
        let current = self
            .levanter_steps
            .get(&key)
            .copied()
            .filter(|cursor| cursor_position(*cursor) <= position);
        historical
            .into_iter()
            .chain(current)
            .max_by_key(|cursor| cursor_position(*cursor))
            .map(|cursor| cursor.step)
    }
}

fn cursor_position(cursor: StepCursor) -> (i64, i64) {
    (cursor.timestamp_ms, cursor.order)
}

#[derive(Debug)]
pub(crate) struct IdentityIngestionPolicy;

pub(crate) const IDENTITY_INGESTION_POLICY: IdentityIngestionPolicy = IdentityIngestionPolicy;

impl IngestionPolicy for IdentityIngestionPolicy {
    fn route_batch(
        &self,
        source: IngestionBatchSource<'_>,
        batch: &RecordBatch,
        _state: &mut IngestionState,
    ) -> Result<Vec<RoutedIngestionBatch>, StatsError> {
        let namespace = source.namespace().to_string();
        Ok(vec![RoutedIngestionBatch {
            destination: IngestionDestination {
                logical_namespace: namespace,
            },
            batch: batch.clone(),
        }])
    }
}

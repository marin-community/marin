use super::*;

use arrow::array::{Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};

use crate::indices::cache::DEFAULT_INDEX_CACHE_MB;
use crate::proto::finelog::stats::ColumnType;
use crate::store::policy::StoragePolicy;
use crate::store::schema::{Column, Schema};
use crate::store::segment::write_segment_to_dir_with_max_row_group_rows;
use crate::store::ServeMode;

fn source_fixture(
    segments: &[&[i64]],
    payload_bytes: usize,
    foreign_sequences: &[i64],
) -> (Arc<Store>, PathBuf) {
    let dir = crate::test_support::unique_dir("native_forward_source");
    let table_dir = dir.join("events");
    std::fs::create_dir_all(&table_dir).unwrap();
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("seq", DataType::Int64, false),
        Field::new("id", DataType::Utf8, false),
        Field::new("cluster", DataType::Utf8, true),
    ]));
    for sequences in segments {
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int64Array::from(sequences.to_vec())),
                Arc::new(StringArray::from(
                    sequences
                        .iter()
                        .map(|seq| format!("{seq}:{}", "x".repeat(payload_bytes)))
                        .collect::<Vec<_>>(),
                )),
                Arc::new(StringArray::from(
                    sequences
                        .iter()
                        .map(|seq| foreign_sequences.contains(seq).then_some("other-cluster"))
                        .collect::<Vec<_>>(),
                )),
            ],
        )
        .unwrap();
        write_segment_to_dir_with_max_row_group_rows(
            &table_dir,
            1,
            *sequences.iter().min().unwrap(),
            &batch,
            2,
        )
        .unwrap();
    }
    let store = Arc::new(
        Store::new(
            Some(dir.clone()),
            "".into(),
            DEFAULT_INDEX_CACHE_MB,
            ServeMode::Shadow,
        )
        .unwrap(),
    );
    // Reopen real Parquet with a newer additive schema, as recovery does.
    tokio::task::block_in_place(|| {
        store.register_table(
            "events",
            Schema::new(
                vec![
                    Column::new("id", ColumnType::COLUMN_TYPE_STRING, false),
                    Column::new("note", ColumnType::COLUMN_TYPE_STRING, true),
                ],
                "id",
            ),
            StoragePolicy::default(),
        )
    })
    .unwrap();
    (store, dir)
}

fn budget(rows: usize, bytes: usize) -> ReadBudget {
    ReadBudget {
        rows,
        bytes,
        segments: 128,
        timeout: Duration::from_secs(10),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reads_ordered_prefixes_across_reordered_overlapping_and_evolved_segments() {
    for segments in [
        vec![&[1, 2, 3][..], &[4, 5, 6][..]],
        vec![&[5, 1, 3][..], &[4, 6, 2][..]],
    ] {
        let (store, dir) = source_fixture(&segments, 8, &[2, 6]);
        let mut cursor = 0;
        let mut received = Vec::new();
        while cursor < 6 {
            let read = read_forwarding(Arc::clone(&store), "events", cursor, 6, budget(4, 1 << 20))
                .await
                .unwrap();
            assert!(read.scanned_through > cursor);
            for batch in &read.batches {
                let seq = batch
                    .column_by_name("seq")
                    .unwrap()
                    .as_primitive::<Int64Type>();
                let ids = batch.column_by_name("id").unwrap().as_string::<i32>();
                assert_eq!(
                    batch.column_by_name("note").unwrap().null_count(),
                    batch.num_rows()
                );
                received.extend(
                    (0..batch.num_rows()).map(|row| (seq.value(row), ids.value(row).to_owned())),
                );
            }
            cursor = read.scanned_through;
        }
        assert_eq!(
            received,
            [1, 3, 4, 5].map(|seq| (seq, format!("{seq}:xxxxxxxx")))
        );
        store.shutdown(Duration::from_secs(1)).await;
        std::fs::remove_dir_all(dir).unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_byte_limited_prefix_resumes_without_losing_lower_rows_in_later_segments() {
    let (store, dir) = source_fixture(&[&[5, 1, 3], &[4, 6, 2]], 16 << 10, &[]);
    let bytes = 32 << 10;
    let mut cursor = 0;
    let mut received: Vec<i64> = Vec::new();
    let mut turns = 0;
    while cursor < 6 {
        let read = read_forwarding(Arc::clone(&store), "events", cursor, 6, budget(16, bytes))
            .await
            .unwrap();
        assert!(read.scanned_through > cursor);
        assert!(
            read.batches
                .iter()
                .map(RecordBatch::get_array_memory_size)
                .sum::<usize>()
                <= bytes
        );
        for batch in &read.batches {
            received.extend(
                batch
                    .column_by_name("seq")
                    .unwrap()
                    .as_primitive::<Int64Type>()
                    .values()
                    .iter()
                    .copied(),
            );
        }
        cursor = read.scanned_through;
        turns += 1;
    }
    assert_eq!(received, vec![1, 2, 3, 4, 5, 6]);
    assert!(
        turns > 1,
        "wide rows must split source progress into bounded prefixes"
    );
    store.shutdown(Duration::from_secs(1)).await;
    std::fs::remove_dir_all(dir).unwrap();
}

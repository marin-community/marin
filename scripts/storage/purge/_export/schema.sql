CREATE SEQUENCE delete_rules_id_seq INCREMENT BY 1 MINVALUE 1 MAXVALUE 9223372036854775807 START 2 NO CYCLE;;
CREATE SEQUENCE protect_rules_id_seq INCREMENT BY 1 MINVALUE 1 MAXVALUE 9223372036854775807 START 7905 NO CYCLE;;
CREATE TABLE cache_meta("key" VARCHAR PRIMARY KEY, "value" VARCHAR NOT NULL);;
CREATE TABLE delete_rules(id INTEGER DEFAULT(nextval('delete_rules_id_seq')) PRIMARY KEY, pattern VARCHAR NOT NULL, storage_class VARCHAR, description VARCHAR, created_at VARCHAR NOT NULL, UNIQUE(pattern, storage_class));;
CREATE TABLE delete_rule_costs(rule_id INTEGER, bucket VARCHAR, storage_class_id INTEGER, object_count INTEGER NOT NULL, total_bytes BIGINT NOT NULL, monthly_cost_usd FLOAT NOT NULL, PRIMARY KEY(rule_id, storage_class_id, bucket));;
CREATE TABLE protect_rules(id INTEGER DEFAULT(nextval('protect_rules_id_seq')) PRIMARY KEY, bucket VARCHAR NOT NULL, pattern VARCHAR NOT NULL, pattern_type VARCHAR NOT NULL, owners VARCHAR, reasons VARCHAR, sources VARCHAR, UNIQUE(bucket, pattern));;
CREATE TABLE scanned_prefixes(bucket VARCHAR, prefix VARCHAR, object_count INTEGER NOT NULL, scanned_at VARCHAR NOT NULL, PRIMARY KEY(bucket, prefix));;
CREATE TABLE split_cache(cache_key VARCHAR PRIMARY KEY, entries_json VARCHAR NOT NULL, updated_at VARCHAR NOT NULL);;
CREATE TABLE step_markers(action_id VARCHAR PRIMARY KEY, completed_at VARCHAR NOT NULL, dry_run BOOLEAN NOT NULL, input_fingerprint VARCHAR NOT NULL, extra_json VARCHAR);;
CREATE TABLE storage_classes(id INTEGER PRIMARY KEY, "name" VARCHAR NOT NULL UNIQUE, price_per_gib_month_us FLOAT NOT NULL, price_per_gib_month_eu FLOAT NOT NULL);;
CREATE TABLE rule_costs(rule_id INTEGER, bucket VARCHAR, storage_class_id INTEGER, object_count INTEGER NOT NULL, total_bytes BIGINT NOT NULL, monthly_cost_usd FLOAT NOT NULL, FOREIGN KEY (rule_id) REFERENCES protect_rules(id), FOREIGN KEY (storage_class_id) REFERENCES storage_classes(id), PRIMARY KEY(rule_id, storage_class_id, bucket));;
CREATE VIEW dir_summary AS SELECT * FROM read_parquet('/Users/power/.codex/worktrees/4b24/marin/scripts/storage/purge/dir_summary_parquet/dir_summary_*.parquet', (union_by_name = CAST('t' AS BOOLEAN)), (hive_partitioning = CAST('f' AS BOOLEAN)));;
CREATE VIEW objects AS SELECT * FROM read_parquet('/Users/power/.codex/worktrees/4b24/marin/scripts/storage/purge/objects_parquet/objects_*.parquet', (union_by_name = CAST('t' AS BOOLEAN)), (hive_partitioning = CAST('f' AS BOOLEAN)));;


COPY cache_meta FROM 'scripts/storage/purge/_export/cache_meta.csv' (FORMAT 'csv', force_not_null ('value', 'key'), quote '"', delimiter ',', header 1);
COPY delete_rules FROM 'scripts/storage/purge/_export/delete_rules.csv' (FORMAT 'csv', force_not_null ('pattern', 'created_at', 'id'), quote '"', delimiter ',', header 1);
COPY delete_rule_costs FROM 'scripts/storage/purge/_export/delete_rule_costs.csv' (FORMAT 'csv', force_not_null ('rule_id', 'bucket', 'storage_class_id', 'object_count', 'total_bytes', 'monthly_cost_usd'), quote '"', delimiter ',', header 1);
COPY protect_rules FROM 'scripts/storage/purge/_export/protect_rules.csv' (FORMAT 'csv', force_not_null ('bucket', 'pattern', 'pattern_type', 'id'), quote '"', delimiter ',', header 1);
COPY scanned_prefixes FROM 'scripts/storage/purge/_export/scanned_prefixes.csv' (FORMAT 'csv', force_not_null ('bucket', 'prefix', 'object_count', 'scanned_at'), quote '"', delimiter ',', header 1);
COPY split_cache FROM 'scripts/storage/purge/_export/split_cache.csv' (FORMAT 'csv', force_not_null ('entries_json', 'updated_at', 'cache_key'), quote '"', delimiter ',', header 1);
COPY step_markers FROM 'scripts/storage/purge/_export/step_markers.csv' (FORMAT 'csv', force_not_null ('completed_at', 'dry_run', 'input_fingerprint', 'action_id'), quote '"', delimiter ',', header 1);
COPY storage_classes FROM 'scripts/storage/purge/_export/storage_classes.csv' (FORMAT 'csv', force_not_null ('name', 'price_per_gib_month_us', 'price_per_gib_month_eu', 'id'), quote '"', delimiter ',', header 1);
COPY rule_costs FROM 'scripts/storage/purge/_export/rule_costs.csv' (FORMAT 'csv', force_not_null ('rule_id', 'bucket', 'storage_class_id', 'object_count', 'total_bytes', 'monthly_cost_usd'), quote '"', delimiter ',', header 1);

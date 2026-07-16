# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Normalize raw downloaded data into the datakit standard Parquet format using DataFusion."""

import logging
import os
import re
import time
from enum import StrEnum
from pathlib import Path
from typing import Any

import dupekit
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from datafusion import SessionConfig, SessionContext, udf
from iris.cli.connect import open_iris_client
from marin.datakit import partition_filename
from marin.execution.step_spec import StepSpec
from pydantic import BaseModel
from rigging.filesystem import StoragePath, prefix_join, url_to_fs
from smallquery.deploy import (
    run_query,
    scheduler_endpoint,
    submit_executors,
    submit_scheduler,
    wait_for_executors,
)

logger = logging.getLogger(__name__)

DEFAULT_MAX_WHITESPACE_RUN_CHARS = 128
COMPACTED_WHITESPACE_COUNTER = "datakit_normalize_compacted_whitespace"

compacted_count = 0


class DedupMode(StrEnum):
    NONE = "none"
    EXACT = "exact"


class NormalizedData(BaseModel):
    version: str = "v1"
    main_output_dir: str
    dup_output_dir: str
    counters: dict[str, int | float]


def generate_id(text: str) -> str:
    return format(dupekit.hash_xxh3_128(text.encode("utf-8")), "032x")


def _discover_files(
    input_path: str,
    file_extensions: tuple[str, ...] | None = None,
) -> list[str]:
    extensions = file_extensions or (".jsonl", ".jsonl.gz", ".json", ".parquet")
    fs, resolved = url_to_fs(input_path)
    protocol = input_path.split("://")[0] if "://" in input_path else ""

    def _full_path(p: str) -> str:
        return f"{protocol}://{p}" if protocol else p

    discovered: list[str] = []
    for root, _dirs, files in fs.walk(resolved):
        rel_root = os.path.relpath(root, resolved)
        parts = [] if rel_root == "." else rel_root.split(os.sep)
        if any(p.startswith(".") for p in parts):
            continue
        for fname in files:
            if fname.startswith("."):
                continue
            if not fname.endswith(extensions):
                continue
            discovered.append(_full_path(os.path.join(root, fname)))

    discovered.sort()
    return discovered


def _compute_total_bytes(file_paths: list[str]) -> int:
    return sum(StoragePath(path).size() for path in file_paths)


def normalize_to_parquet(
    *,
    input_path: str,
    output_path: str,
    text_field: str = "text",
    id_field: str = "id",
    target_partition_bytes: int = 256 * 1024 * 1024,
    max_whitespace_run_chars: int = DEFAULT_MAX_WHITESPACE_RUN_CHARS,
    worker_resources: Any = None,
    max_workers: int = 1024,
    file_extensions: tuple[str, ...] | None = None,
    dedup_mode: DedupMode = DedupMode.EXACT,
    bare: bool = False,
) -> NormalizedData:
    global compacted_count
    compacted_count = 0

    files = _discover_files(input_path, file_extensions=file_extensions)
    if not files:
        raise FileNotFoundError(f"No data files found under {input_path}")

    total_bytes = _compute_total_bytes(files)
    num_shards = max(1, total_bytes // target_partition_bytes)

    logger.info(
        "Normalizing (DataFusion) %s → %s: %d files, %d bytes, %d shards",
        input_path,
        output_path,
        len(files),
        total_bytes,
        num_shards,
    )

    # Local path resolution for DataFusion if protocol is missing
    resolved_files = [
        f if f.startswith(("gs://", "s3://")) else (f.split("://")[-1] if "://" in f else f) for f in files
    ]

    # In order to validate columns and get text_field_type, read the schema of the first file using PyArrow
    first_file = resolved_files[0]
    is_json = first_file.endswith((".jsonl", ".jsonl.gz", ".json", ".json.gz"))
    if is_json:
        dataset = ds.dataset(first_file, format="json")
    elif first_file.endswith(".parquet"):
        dataset = ds.dataset(first_file, format="parquet")
    else:
        raise ValueError(f"Unsupported file type: {first_file}")

    input_schema = dataset.schema
    schema_fields = input_schema.names
    if text_field not in schema_fields:
        raise ValueError(f"Text field {text_field!r} not found in input schema: {schema_fields}")
    text_field_type = input_schema.field(text_field).type

    is_binary = pa.types.is_binary(text_field_type) or pa.types.is_fixed_size_binary(text_field_type)
    if is_binary:
        text_field_expr = f"CAST({text_field} AS BYTEA)"
        text_field_type = pa.binary()
    else:
        text_field_expr = text_field

    if input_path.startswith(("gs://", "s3://")):
        # 1. Distributed execution on remote Iris/Ballista cluster
        cluster = os.environ.get("MARIN_CLUSTER", "marin")
        region = os.environ.get("MARIN_REGION", "us-central2")
        image = "ghcr.io/marin-community/smallquery-ballista:54.0.0"

        # Determine resource requirements from worker_resources
        sch_cpu = 2.0
        sch_mem = "4g"
        exe_cpu = 4.0
        exe_mem = "8g"
        if worker_resources:
            if hasattr(worker_resources, "cpu") and worker_resources.cpu:
                exe_cpu = float(worker_resources.cpu)
            if hasattr(worker_resources, "ram") and worker_resources.ram:
                exe_mem = str(worker_resources.ram)

        ts = int(time.time())
        scheduler_name = f"ballista-scheduler-{ts}"
        executors_name = f"ballista-executors-{ts}"

        # Setup output directories
        main_output_dir = prefix_join(output_path, "outputs/main")
        dup_output_dir = prefix_join(output_path, "outputs/dups")

        # Build column projection string
        has_source_id = id_field in schema_fields
        select_exprs = []
        limit = max_whitespace_run_chars
        replacement_str = " " * limit
        compacted_text_expr = (
            f"regexp_replace({text_field_expr}, '[ \\t\\xa0]{{{limit + 1},}}', '{replacement_str}', 'g')"
        )

        select_exprs.append(f"lower(md5({compacted_text_expr})) AS id")
        select_exprs.append(f"{compacted_text_expr} AS text")
        if has_source_id:
            select_exprs.append(f"{id_field} AS source_id")
        if not bare:
            for field in schema_fields:
                if field == id_field:
                    continue
                if field == text_field and text_field != "text":
                    continue
                if field in ("id", "text", "source_id"):
                    continue
                select_exprs.append(field)

        cols_str = ", ".join(select_exprs)

        projected_names = ["id", "text"]
        if has_source_id:
            projected_names.append("source_id")
        if not bare:
            for field in schema_fields:
                if field == id_field:
                    continue
                if field == text_field and text_field != "text":
                    continue
                if field in ("id", "text", "source_id"):
                    continue
                projected_names.append(field)

        cols_without_aliases = ", ".join(projected_names)

        # Build the dynamic SQL queries for COPY TO
        is_json = files[0].endswith((".jsonl", ".jsonl.gz", ".json", ".json.gz"))
        is_gzip = files[0].endswith(".gz")
        fmt_str = "JSON" if is_json else "PARQUET"
        comp_str = "OPTIONS (format.compression gzip)" if is_gzip else ""

        location_path = input_path
        if not input_path.endswith("/") and not input_path.endswith(
            (".json", ".jsonl", ".parquet", ".gz", ".jsonl.gz", ".json.gz")
        ):
            location_path = input_path + "/"

        sql_lines = []
        sql_lines.append(f"CREATE EXTERNAL TABLE input_table STORED AS {fmt_str} {comp_str} LOCATION '{location_path}';")

        if dedup_mode == DedupMode.EXACT:
            for i in range(num_shards):
                # Main result COPY
                main_query = f"""
                COPY (
                    SELECT {cols_without_aliases} FROM (
                        SELECT *, ROW_NUMBER() OVER (PARTITION BY id ORDER BY id) as rn
                        FROM (
                            SELECT {cols_str} FROM input_table
                            WHERE {text_field_expr} IS NOT NULL
                              AND regexp_replace({text_field_expr}, '[ \t\xa0\n\r\f]+', '', 'g') != ''
                        )
                    )
                    WHERE rn = 1
                      AND (ascii(substr(id, 1, 1))
                           + ascii(substr(id, 2, 1))
                           + ascii(substr(id, 3, 1))) % {num_shards} = {i}
                    ORDER BY id
                ) TO '{main_output_dir}/part-{i:05d}-of-{num_shards:05d}.parquet' STORED AS PARQUET;
                """
                sql_lines.append(main_query)

                # Dup result COPY
                dup_query = f"""
                COPY (
                    SELECT {cols_without_aliases} FROM (
                        SELECT *, ROW_NUMBER() OVER (PARTITION BY id ORDER BY id) as rn
                        FROM (
                            SELECT {cols_str} FROM input_table
                            WHERE {text_field_expr} IS NOT NULL
                              AND regexp_replace({text_field_expr}, '[ \t\xa0\n\r\f]+', '', 'g') != ''
                        )
                    )
                    WHERE rn > 1
                      AND (ascii(substr(id, 1, 1))
                           + ascii(substr(id, 2, 1))
                           + ascii(substr(id, 3, 1))) % {num_shards} = {i}
                    ORDER BY id
                ) TO '{dup_output_dir}/part-{i:05d}-of-{num_shards:05d}.parquet' STORED AS PARQUET;
                """
                sql_lines.append(dup_query)
        else:
            for i in range(num_shards):
                main_query = f"""
                COPY (
                    SELECT {cols_without_aliases} FROM (
                        SELECT {cols_str} FROM input_table
                        WHERE {text_field_expr} IS NOT NULL
                          AND regexp_replace({text_field_expr}, '[ \t\xa0\n\r\f]+', '', 'g') != ''
                    )
                    WHERE (ascii(substr(id, 1, 1))
                           + ascii(substr(id, 2, 1))
                           + ascii(substr(id, 3, 1))) % {num_shards} = {i}
                    ORDER BY id
                ) TO '{main_output_dir}/part-{i:05d}-of-{num_shards:05d}.parquet' STORED AS PARQUET;
                """
                sql_lines.append(main_query)

        # Metadata counts
        metadata_dir = prefix_join(output_path, "metadata")
        sql_lines.append(
            f"""
        COPY (
            SELECT COUNT(*) as total_in FROM input_table
        ) TO '{metadata_dir}/total_in.parquet' STORED AS PARQUET;
        """
        )

        sql_lines.append(
            f"""
        COPY (
            SELECT COUNT(*) as empty_text_filtered FROM input_table
            WHERE NOT ({text_field_expr} IS NOT NULL
              AND regexp_replace({text_field_expr}, '[ \t\xa0\n\r\f]+', '', 'g') != '')
        ) TO '{metadata_dir}/empty_text_filtered.parquet' STORED AS PARQUET;
        """
        )

        if dedup_mode == DedupMode.EXACT:
            sql_lines.append(
                f"""
            COPY (
                SELECT COUNT(*) as unique_records_out FROM (
                    SELECT *, ROW_NUMBER() OVER (PARTITION BY id ORDER BY id) as rn
                    FROM (
                        SELECT lower(md5({compacted_text_expr})) AS id, {text_field_expr} FROM input_table
                        WHERE {text_field_expr} IS NOT NULL
                          AND regexp_replace({text_field_expr}, '[ \t\xa0\n\r\f]+', '', 'g') != ''
                    )
                )
                WHERE rn = 1
            ) TO '{metadata_dir}/unique_records_out.parquet' STORED AS PARQUET;
            """
            )

            sql_lines.append(
                f"""
            COPY (
                SELECT COUNT(*) as duplicate_records_out FROM (
                    SELECT *, ROW_NUMBER() OVER (PARTITION BY id ORDER BY id) as rn
                    FROM (
                        SELECT lower(md5({compacted_text_expr})) AS id, {text_field_expr} FROM input_table
                        WHERE {text_field_expr} IS NOT NULL
                          AND regexp_replace({text_field_expr}, '[ \t\xa0\n\r\f]+', '', 'g') != ''
                    )
                )
                WHERE rn > 1
            ) TO '{metadata_dir}/duplicate_records_out.parquet' STORED AS PARQUET;
            """
            )
        else:
            sql_lines.append(
                f"""
            COPY (
                SELECT COUNT(*) as unique_records_out FROM (
                    SELECT {text_field_expr} FROM input_table
                    WHERE {text_field_expr} IS NOT NULL
                      AND regexp_replace({text_field_expr}, '[ \t\xa0\n\r\f]+', '', 'g') != ''
                )
            ) TO '{metadata_dir}/unique_records_out.parquet' STORED AS PARQUET;
            """
            )

        create_table_sql = sql_lines[0]
        copy_queries = sql_lines[1:]

        logger.info("Standing up distributed Ballista cluster on Iris (%d workers)...", max_workers)
        with open_iris_client(cluster_name=cluster, workspace=Path.cwd()) as client:
            scheduler_job = submit_scheduler(
                client,
                image=image,
                region=region,
                cpu=sch_cpu,
                memory=sch_mem,
                name=scheduler_name,
            )
            executor_job = None
            try:
                endpoint = scheduler_endpoint(scheduler_job)
                executor_job = submit_executors(
                    client,
                    image=image,
                    region=region,
                    scheduler=endpoint,
                    num_workers=max_workers,
                    cpu=exe_cpu,
                    memory=exe_mem,
                    name=executors_name,
                )
                wait_for_executors(executor_job, max_workers)

                logger.info("Ballista cluster ready! Running %d distributed queries in parallel...", len(copy_queries))
                client_jobs = []
                for idx, query in enumerate(copy_queries):
                    query_sql = f"{create_table_sql}\n{query}"
                    client_job = run_query(
                        client,
                        image=image,
                        region=region,
                        scheduler=endpoint,
                        sql=query_sql,
                        name=f"ballista-query-{ts}-{idx}",
                    )
                    client_jobs.append(client_job)

                for client_job in client_jobs:
                    client_job.wait(timeout=3600.0, stream_logs=False, raise_on_failure=True)
                logger.info("Distributed Ballista queries completed successfully!")
            finally:
                logger.info("Tearing down Ballista cluster...")
                if executor_job is not None:
                    executor_job.terminate()
                scheduler_job.terminate()

        # Collect metadata counts locally via PyArrow
        logger.info("Collecting output metadata counts...")
        fs, _ = url_to_fs(output_path)

        def read_count(url: str, col_name: str) -> int:
            try:
                if fs.exists(url):
                    m_files = fs.glob(f"{url}/*.parquet")
                    if m_files:
                        proto = "gs://" if url.startswith("gs://") else "s3://"
                        full_paths = [f"{proto}{f}" if not f.startswith(("gs://", "s3://")) else f for f in m_files]
                        tbl = pq.read_table(full_paths, filesystem=fs)
                        return int(tbl.column(col_name)[0].as_py())
            except Exception as e:
                logger.warning("Failed to read metadata count from %s: %s", url, e)
            return 0

        total_in = read_count(f"{metadata_dir}/total_in.parquet", "total_in")
        empty_text_filtered = read_count(f"{metadata_dir}/empty_text_filtered.parquet", "empty_text_filtered")
        unique_records_out = read_count(f"{metadata_dir}/unique_records_out.parquet", "unique_records_out")
        duplicate_records_out = read_count(f"{metadata_dir}/duplicate_records_out.parquet", "duplicate_records_out")

        counters_dict = {
            "zephyr/records_in": total_in,
            "normalize/empty_text_filtered": empty_text_filtered,
            "normalize/unique_records_out": unique_records_out,
            "normalize/duplicate_records_out": duplicate_records_out,
            COMPACTED_WHITESPACE_COUNTER: 0,
        }

        return NormalizedData(
            main_output_dir=main_output_dir,
            dup_output_dir=dup_output_dir,
            counters=counters_dict,
        )

    # 2. Local execution (local files / dry-run)
    config = SessionConfig().with_target_partitions(max_workers)
    ctx = SessionContext(config)

    df = None
    for file in resolved_files:
        if file.endswith((".jsonl", ".jsonl.gz", ".json", ".json.gz")):
            ext = "." + file.split("/")[-1].split(".", 1)[-1]
            compression = "gzip" if file.endswith(".gz") else None
            file_df = ctx.read_json(file, file_extension=ext, file_compression_type=compression)
        elif file.endswith(".parquet"):
            file_df = ctx.read_parquet(file)
        else:
            raise ValueError(f"Unsupported file type: {file}")

        if df is None:
            df = file_df
        else:
            df = df.union(file_df)

    # Register UDFs
    def is_valid_text_impl(arr: pa.Array) -> pa.Array:
        vals = arr.to_pylist()
        results = []
        for val in vals:
            if val is None:
                results.append(False)
                continue
            if isinstance(val, bytes):
                text = val.decode("utf-8", errors="replace")
            else:
                text = str(val)
            results.append(text.strip() != "")
        return pa.array(results, type=pa.bool_())

    is_valid_text_udf = udf(
        is_valid_text_impl,
        input_fields=[text_field_type],
        return_field=pa.bool_(),
        volatility="immutable",
        name="is_valid_text",
    )

    def normalize_and_compact_text_impl(arr: pa.Array, max_whitespace_arr: pa.Array) -> pa.Array:
        global compacted_count
        texts = arr.to_pylist()
        limits = max_whitespace_arr.to_pylist()
        results = []
        regex_cache = {}
        for text, limit_val in zip(texts, limits, strict=True):
            if text is None:
                results.append("")
                continue
            if isinstance(text, bytes):
                t = text.decode("utf-8", errors="replace")
            else:
                t = str(text)

            limit = int(limit_val)
            if limit not in regex_cache:
                regex_cache[limit] = re.compile(r"\s{" + str(limit + 1) + r",}")
            pattern = regex_cache[limit]

            compacted, count = pattern.subn(lambda m, limit=limit: m.group(0)[:limit], t)
            if count > 0:
                compacted_count += 1
            results.append(compacted)
        return pa.array(results, type=pa.string())

    normalize_and_compact_text_udf = udf(
        normalize_and_compact_text_impl,
        input_fields=[text_field_type, pa.int64()],
        return_field=pa.string(),
        volatility="immutable",
        name="normalize_and_compact_text",
    )

    def generate_id_impl(arr: pa.Array) -> pa.Array:
        texts = arr.to_pylist()
        results = []
        for text in texts:
            if text is None:
                results.append("")
                continue
            results.append(format(dupekit.hash_xxh3_128(text.encode("utf-8")), "032x"))
        return pa.array(results, type=pa.string())

    generate_id_udf = udf(
        generate_id_impl,
        input_fields=[pa.string()],
        return_field=pa.string(),
        volatility="immutable",
        name="generate_id",
    )

    def get_partition_impl(arr: pa.Array, shards_arr: pa.Array) -> pa.Array:
        ids = arr.to_pylist()
        shards_list = shards_arr.to_pylist()
        results = []
        for id_val, shards in zip(ids, shards_list, strict=True):
            if id_val is None:
                results.append(0)
                continue
            results.append(int(id_val[:8], 16) % int(shards))
        return pa.array(results, type=pa.int64())

    get_partition_udf = udf(
        get_partition_impl,
        input_fields=[pa.string(), pa.int64()],
        return_field=pa.int64(),
        volatility="immutable",
        name="get_partition",
    )

    ctx.register_udf(is_valid_text_udf)
    ctx.register_udf(normalize_and_compact_text_udf)
    ctx.register_udf(generate_id_udf)
    ctx.register_udf(get_partition_udf)

    ctx.register_table("input_table", df)

    # 1. Total records in
    total_in = ctx.sql("SELECT COUNT(*) FROM input_table").collect()[0][0][0].as_py()

    # 2. Empty text filtered records
    empty_text_filtered = (
        ctx.sql(f"SELECT COUNT(*) FROM input_table WHERE NOT is_valid_text({text_field_expr})")
        .collect()[0][0][0]
        .as_py()
    )

    if total_in > 0 and empty_text_filtered == total_in:
        raise ValueError(
            f"All {total_in} records were filtered out due to missing/empty text. "
            f"Your data is either invalid or you have selected the wrong column, "
            f"current column: {text_field!r}"
        )

    # Build Projection
    has_source_id = id_field in schema_fields
    select_exprs = []
    select_exprs.append(f"generate_id(normalize_and_compact_text({text_field_expr}, {max_whitespace_run_chars})) AS id")
    select_exprs.append(f"normalize_and_compact_text({text_field_expr}, {max_whitespace_run_chars}) AS text")

    if has_source_id:
        select_exprs.append(f"{id_field} AS source_id")

    if not bare:
        for field in schema_fields:
            if field == id_field:
                continue
            if field == text_field and text_field != "text":
                continue
            if field in ("id", "text", "source_id"):
                continue
            select_exprs.append(field)

    projection_str = ", ".join(select_exprs)
    projected_df = ctx.sql(
        f"""
        SELECT {projection_str}
        FROM input_table
        WHERE is_valid_text({text_field_expr})
    """
    )
    ctx.register_table("projected_table", projected_df)

    projected_cols = [f.name for f in projected_df.schema()]
    cols_str = ", ".join(projected_cols)

    # Setup outputs directories
    main_output_dir = prefix_join(output_path, "outputs/main")
    dup_output_dir = prefix_join(output_path, "outputs/dups")

    local_main_dir = main_output_dir.split("://")[-1] if "://" in main_output_dir else main_output_dir
    local_dup_dir = dup_output_dir.split("://")[-1] if "://" in dup_output_dir else dup_output_dir
    os.makedirs(local_main_dir, exist_ok=True)
    os.makedirs(local_dup_dir, exist_ok=True)

    unique_records_out = 0
    duplicate_records_out = 0

    if dedup_mode == DedupMode.EXACT:
        main_sql = f"""
            SELECT {cols_str}, get_partition(id, {num_shards}) as part_id FROM (
                SELECT {cols_str}, ROW_NUMBER() OVER (PARTITION BY id ORDER BY id) as rn
                FROM projected_table
            )
            WHERE rn = 1
        """
        dup_sql = f"""
            SELECT {cols_str}, get_partition(id, {num_shards}) as part_id FROM (
                SELECT {cols_str}, ROW_NUMBER() OVER (PARTITION BY id ORDER BY id) as rn
                FROM projected_table
            )
            WHERE rn > 1
        """
        main_df = ctx.sql(main_sql)
        dup_df = ctx.sql(dup_sql)

        ctx.register_table("main_results", main_df)
        ctx.register_table("dup_results", dup_df)

        for i in range(num_shards):
            filename = partition_filename(i, num_shards)

            part_main_df = ctx.sql(f"SELECT {cols_str} FROM main_results WHERE part_id = {i} ORDER BY id")
            main_table = part_main_df.to_arrow_table()
            if main_table.num_rows > 0:
                unique_records_out += main_table.num_rows
                pq.write_table(main_table, os.path.join(local_main_dir, filename))

            part_dup_df = ctx.sql(f"SELECT {cols_str} FROM dup_results WHERE part_id = {i} ORDER BY id")
            dup_table = part_dup_df.to_arrow_table()
            if dup_table.num_rows > 0:
                duplicate_records_out += dup_table.num_rows
                pq.write_table(dup_table, os.path.join(local_dup_dir, filename))

    else:
        main_sql = f"SELECT {cols_str}, get_partition(id, {num_shards}) as part_id FROM projected_table"
        main_df = ctx.sql(main_sql)
        ctx.register_table("main_results", main_df)

        for i in range(num_shards):
            filename = partition_filename(i, num_shards)
            part_main_df = ctx.sql(f"SELECT {cols_str} FROM main_results WHERE part_id = {i} ORDER BY id")
            main_table = part_main_df.to_arrow_table()
            if main_table.num_rows > 0:
                unique_records_out += main_table.num_rows
                pq.write_table(main_table, os.path.join(local_main_dir, filename))

    counters_dict = {
        "zephyr/records_in": total_in,
        "normalize/empty_text_filtered": empty_text_filtered,
        "normalize/unique_records_out": unique_records_out,
        "normalize/duplicate_records_out": duplicate_records_out,
        COMPACTED_WHITESPACE_COUNTER: compacted_count,
    }

    return NormalizedData(
        main_output_dir=main_output_dir,
        dup_output_dir=dup_output_dir,
        counters=counters_dict,
    )


def normalize_step(
    *,
    name: str,
    download: StepSpec,
    text_field: str = "text",
    id_field: str = "id",
    target_partition_bytes: int = 256 * 1024 * 1024,
    max_whitespace_run_chars: int = DEFAULT_MAX_WHITESPACE_RUN_CHARS,
    worker_resources: Any = None,
    max_workers: int = 1024,
    output_path_prefix: str | None = None,
    override_output_path: str | None = None,
    relative_input_path: str | None = None,
    file_extensions: tuple[str, ...] | None = None,
    dedup_mode: DedupMode = DedupMode.EXACT,
    bare: bool = False,
) -> StepSpec:
    if relative_input_path:
        resolved_input = prefix_join(download.output_path, relative_input_path)
    else:
        resolved_input = download.output_path

    hash_attrs: dict[str, Any] = {
        "text_field": text_field,
        "id_field": id_field,
        "target_partition_bytes": target_partition_bytes,
        "max_whitespace_run_chars": max_whitespace_run_chars,
        "relative_input_path": relative_input_path,
        "file_extensions": file_extensions,
        "dedup_mode": dedup_mode,
    }
    if bare:
        hash_attrs["bare"] = bare

    return StepSpec(
        name=name,
        fn=lambda output_path: normalize_to_parquet(
            input_path=resolved_input,
            output_path=output_path,
            text_field=text_field,
            id_field=id_field,
            target_partition_bytes=target_partition_bytes,
            max_whitespace_run_chars=max_whitespace_run_chars,
            worker_resources=worker_resources,
            max_workers=max_workers,
            file_extensions=file_extensions,
            dedup_mode=dedup_mode,
            bare=bare,
        ),
        deps=[download],
        hash_attrs=hash_attrs,
        output_path_prefix=output_path_prefix,
        override_output_path=override_output_path,
    )

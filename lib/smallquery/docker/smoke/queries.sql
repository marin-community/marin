-- Smoke queries: a scan, a high-cardinality GROUP BY (forces a shuffle across
-- executors), and a join. Expected over the generated fixture (100000 rows, 1000 grps):
--   scan_count = 100000
--   first three groups each = 100
--   join_count = 100000
CREATE EXTERNAL TABLE nums STORED AS PARQUET LOCATION '/data/nums.parquet';
CREATE EXTERNAL TABLE dims STORED AS PARQUET LOCATION '/data/dims.parquet';

SELECT count(*) AS scan_count FROM nums;

SELECT grp, count(*) AS c FROM nums GROUP BY grp ORDER BY grp LIMIT 3;

SELECT count(*) AS join_count FROM nums JOIN dims ON nums.grp = dims.grp;

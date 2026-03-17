DROP TABLE IF EXISTS healthcare_curated_db.control_watermark;

-- (in shell) clear old table files if needed:
-- aws s3 rm s3://healthcare-data-lake-gj/control/watermark/ --recursive

CREATE TABLE healthcare_curated_db.control_watermark
WITH (
  format = 'PARQUET',
  external_location = 's3://healthcare-data-lake-gj/curated/control_watermark/',
  parquet_compression = 'SNAPPY',
  partitioned_by = ARRAY['ingest_dt']
) AS
WITH m AS (
  SELECT
    ingest_dt,
    run_ts_utc,
    -- if run_ts_utc is ISO8601 like 2026-03-15T05:34:29.935139+00:00
    -- this yields timestamp WITHOUT time zone (Hive compatible)
    cast(from_iso8601_timestamp(run_ts_utc) as timestamp) AS run_ts,
    cardinality(files) AS uploaded,
    cardinality(skipped_unchanged) AS skipped_unchanged,
    cardinality(errors) AS errors,
    cast('s3://healthcare-data-lake-gj/control/manifests/latest/manifest.json' as varchar) AS manifest_s3_key
  FROM healthcare_catalog_db.manifest_latest
)
SELECT
  run_ts,                 -- timestamp (no tz)
  run_ts_utc,             -- keep the original string too (handy for debugging)
  cast(status as varchar) as status,
  manifest_s3_key,
  CAST(
    ROW(
      uploaded,
      skipped_unchanged,
      errors,
      (uploaded + skipped_unchanged + errors)
    )
    AS ROW(
      uploaded BIGINT,
      skipped_unchanged BIGINT,
      errors BIGINT,
      total_files_seen BIGINT
    )
  ) AS counts,
  ingest_dt               -- MUST be last (partition column)
FROM (
  SELECT
    *,
    CASE WHEN errors = 0 THEN 'ok' ELSE 'partial_failure' END AS status
  FROM m
);
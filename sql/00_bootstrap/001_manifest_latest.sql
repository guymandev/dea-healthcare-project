DROP TABLE IF EXISTS healthcare_catalog_db.manifest_latest;

CREATE EXTERNAL TABLE healthcare_catalog_db.manifest_latest (
  ingest_dt string,
  run_ts_utc string,
  source string,
  bucket string,
  raw_prefix string,
  files array<
    struct<
      filename:string,
      dataset_key:string,
      s3_key:string,
      size_bytes:bigint,
      sha256:string,
      drive_file_id:string
    >
  >,
  skipped_unchanged array<
    struct<
      filename:string,
      sha256:string,
      size_bytes:bigint
    >
  >,
  errors array<
    struct<
      filename:string,
      dataset_key:string,
      drive_file_id:string,
      error:string
    >
  >
)
ROW FORMAT SERDE 'org.openx.data.jsonserde.JsonSerDe'
WITH SERDEPROPERTIES (
  'ignore.malformed.json' = 'true'
)
LOCATION 's3://healthcare-data-lake-gj/control/manifests/latest/';
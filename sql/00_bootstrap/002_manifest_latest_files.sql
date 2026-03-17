DROP TABLE IF EXISTS healthcare_catalog_db.manifest_latest_files;

CREATE EXTERNAL TABLE healthcare_catalog_db.manifest_latest_files (
  ingest_dt        string,
  run_ts_utc       string,
  source           string,
  bucket           string,
  raw_prefix       string,

  filename         string,
  dataset_key      string,
  s3_key           string,
  size_bytes       bigint,
  sha256           string,
  drive_file_id    string
)
ROW FORMAT SERDE 'org.openx.data.jsonserde.JsonSerDe'
WITH SERDEPROPERTIES (
  'ignore.malformed.json' = 'true'
)
LOCATION 's3://healthcare-data-lake-gj/control/manifests/latest_files/';
DROP TABLE IF EXISTS healthcare_curated_db.fact_state_covid_vax_avgs;

-- S3_PREFIX: s3://healthcare-data-lake-gj/curated/fact_state_covid_vax_avgs/

CREATE TABLE healthcare_curated_db.fact_state_covid_vax_avgs
WITH (
  format = 'PARQUET',
  external_location = 's3://healthcare-data-lake-gj/curated/fact_state_covid_vax_avgs/',
  parquet_compression = 'SNAPPY',
  partitioned_by = ARRAY['ingest_dt']
) AS
SELECT
  upper(trim(regexp_replace(state_raw, '"', ''))) AS state,

  try_cast(regexp_replace(pct_residents_up_to_date_raw, '"', '') AS double)
    AS pct_residents_w_up_to_date_vaxes,

  try_cast(regexp_replace(pct_staff_up_to_date_raw, '"', '') AS double)
    AS pct_staff_w_up_to_date_vaxes,

  try(
    cast(
      date_parse(regexp_replace(vax_data_last_updated_raw, '"', ''), '%m/%d/%Y')
      as date
    )
  ) AS date_vax_data_last_updated,

  ingest_dt
FROM healthcare_catalog_db.raw_nh_covidvaxaverages_20241027_fixed
WHERE ingest_dt = '{{INGEST_DT:nh_covidvaxaverages_20241027}}'
  AND state_raw IS NOT NULL
  AND trim(state_raw) <> '';
DROP TABLE IF EXISTS healthcare_curated_db.fact_facility_covid;

-- S3_PREFIX: s3://healthcare-data-lake-gj/curated/fact_facility_covid/

CREATE TABLE healthcare_curated_db.fact_facility_covid
WITH (
  format = 'PARQUET',
  external_location = 's3://healthcare-data-lake-gj/curated/fact_facility_covid/',
  parquet_compression = 'SNAPPY',
  partitioned_by = ARRAY['ingest_dt']
) AS
WITH base AS (
  SELECT
    -- CCN normalized to 6-char string
    lpad(regexp_replace(trim(cast(cms_certification_number_ccn as varchar)), '"', ''), 6, '0') AS ccn_provnum,

    trim(cast(state as varchar)) AS state,

    trim(cast(pct_residents_up_to_date_vaxes as varchar)) AS pct_residents_w_up_to_date_vaxes,
    trim(cast(pct_staff_up_to_date_vaxes as varchar))     AS pct_staff_w_up_to_date_vaxes,

    -- date can vary by file; try a couple common formats
    coalesce(
      try(cast(date_parse(trim(cast(vax_data_last_update as varchar)), '%Y-%m-%d') as date)),
      try(cast(date_parse(trim(cast(vax_data_last_update as varchar)), '%m/%d/%Y') as date))
    ) AS vax_data_last_update,

    ingest_dt
  FROM healthcare_catalog_db.raw_nh_covidvaxprovider_20241027_fixed
  WHERE ingest_dt = '{{INGEST_DT:nh_covidvaxprovider_20241027}}'
  AND cms_certification_number_ccn IS NOT NULL
)
SELECT
  ccn_provnum,
  state,
  pct_residents_w_up_to_date_vaxes,
  pct_staff_w_up_to_date_vaxes,
  vax_data_last_update,
  ingest_dt
FROM base
WHERE ccn_provnum IS NOT NULL;
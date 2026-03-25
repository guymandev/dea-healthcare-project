DROP TABLE IF EXISTS healthcare_curated_db.fact_survey_event;

-- aws s3 rm s3://healthcare-data-lake-gj/curated/fact_survey_event/ --recursive

CREATE TABLE healthcare_curated_db.fact_survey_event
WITH (
  format='PARQUET',
  external_location='s3://healthcare-data-lake-gj/curated/fact_survey_event/',
  parquet_compression='SNAPPY',
  partitioned_by = ARRAY['ingest_dt']
) AS
WITH base AS (
  SELECT
    lpad(regexp_replace(trim(cast(cms_certification_number_ccn as varchar)), '"', ''), 6, '0') AS ccn_provnum,
    try(cast(date_parse(trim(cast(survey_date as varchar)), '%Y-%m-%d') as date)) AS survey_date,
    trim(cast(type_of_survey as varchar)) AS survey_type,
    try_cast(survey_cycle as integer) AS survey_cycle,
    ingest_dt
  FROM healthcare_catalog_db.raw_nh_surveydates_oct2024_fixed
  WHERE ingest_dt='{{INGEST_DT}}'
    AND cms_certification_number_ccn IS NOT NULL
)
SELECT
  b.ccn_provnum,
  b.survey_date,
  d.date_id,
  b.survey_type,
  b.survey_cycle,
  b.ingest_dt
FROM base b
JOIN healthcare_curated_db.dim_date d
  ON d.date = b.survey_date
WHERE b.survey_date IS NOT NULL;
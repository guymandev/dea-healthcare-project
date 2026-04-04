-- S3_PREFIX: s3://healthcare-data-lake-gj/curated/dim_measure/

DROP TABLE IF EXISTS healthcare_curated_db.dim_measure;

CREATE TABLE healthcare_curated_db.dim_measure
WITH (
  format = 'PARQUET',
  external_location = 's3://healthcare-data-lake-gj/curated/dim_measure/',
  parquet_compression = 'SNAPPY',
  partitioned_by = ARRAY['ingest_dt']
) AS

WITH all_measures AS (

  -- Claims (has descriptions)
  SELECT
    nullif(regexp_replace(trim(measure_code), '"', ''), '') as measure_code,
    nullif(trim(measure_description), '') as measure_description
  FROM healthcare_catalog_db.raw_nh_qualitymsr_claims_oct2024_fixed
  WHERE ingest_dt = '{{INGEST_DT:nh_qualitymsr_claims_oct2024}}'

  UNION ALL

  -- MDS (has descriptions)
  SELECT
    nullif(regexp_replace(trim(measure_code), '"', ''), '') as measure_code,
    nullif(trim(measure_description), '') as measure_description
  FROM healthcare_catalog_db.raw_nh_qualitymsr_mds_oct2024_fixed
  WHERE ingest_dt = '{{INGEST_DT:nh_qualitymsr_mds_oct2024}}'

  UNION ALL

  -- SNF QRP Provider Data (codes, usually no description column)
  SELECT
    nullif(regexp_replace(trim(measure_code), '"', ''), '') as measure_code,
    cast(null as varchar) as measure_description
  FROM healthcare_catalog_db.raw_skilled_nursing_facility_quality_reporting_program_provider_data_oct2024_fixed
  WHERE ingest_dt = '{{INGEST_DT:skilled_nursing_facility_quality_reporting_program_provider_data_oct2024}}'

  UNION ALL

  -- Swing bed fixed (codes, no descriptions)
  SELECT
    nullif(regexp_replace(trim(measure_code), '"', ''), '') as measure_code,
    cast(null as varchar) as measure_description
  FROM healthcare_catalog_db.raw_swing_bed_snf_data_oct2024_fixed
  WHERE ingest_dt = '{{INGEST_DT:swing_bed_snf_data_oct2024}}'
),

dedup AS (
  SELECT
    measure_code,
    max_by(measure_description, length(coalesce(measure_description,''))) as measure_description
  FROM all_measures
  WHERE measure_code is not null
    AND measure_code <> ''
  GROUP BY 1
)

SELECT
  measure_code,
  measure_description,
  CAST('{{MANIFEST_RUN_INGEST_DT}}' AS varchar) as ingest_dt
FROM dedup;
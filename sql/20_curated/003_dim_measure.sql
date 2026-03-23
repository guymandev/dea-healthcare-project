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
  WHERE ingest_dt = '{{INGEST_DT}}'

  UNION ALL

  -- MDS (has descriptions)
  SELECT
    nullif(regexp_replace(trim(measure_code), '"', ''), '') as measure_code,
    nullif(trim(measure_description), '') as measure_description
  FROM healthcare_catalog_db.raw_nh_qualitymsr_mds_oct2024_fixed
  WHERE ingest_dt = '{{INGEST_DT}}'

  UNION ALL

  -- SNF QRP Provider Data (codes, usually no description column)
  SELECT
    nullif(regexp_replace(trim(measure_code), '"', ''), '') as measure_code,
    cast(null as varchar) as measure_description
  FROM healthcare_catalog_db.raw_snf_qrp_provider_data_oct2024_fixed
  WHERE ingest_dt = '{{INGEST_DT}}'

  UNION ALL

  -- Swing bed fixed (codes, no descriptions)
  SELECT
    nullif(regexp_replace(trim(measure_code), '"', ''), '') as measure_code,
    cast(null as varchar) as measure_description
  FROM healthcare_catalog_db.raw_swing_bed_snf_data_oct2024_fixed
  WHERE ingest_dt = '{{INGEST_DT}}'
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
  CAST('{{INGEST_DT}}' AS varchar) as ingest_dt
FROM dedup;
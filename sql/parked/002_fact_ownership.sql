-- S3_PREFIX: s3://healthcare-data-lake-gj/curated/fact_ownership/

DROP TABLE IF EXISTS healthcare_curated_db.fact_ownership;

CREATE TABLE healthcare_curated_db.fact_ownership
WITH (
  format = 'PARQUET',
  external_location = 's3://healthcare-data-lake-gj/curated/fact_ownership/',
  parquet_compression = 'SNAPPY',
  partitioned_by = ARRAY['ingest_dt']
) AS

WITH base AS (
  SELECT
    lpad(
      regexp_replace(trim(cast(cms_certification_number_ccn AS varchar)), '"', ''),
      6, '0'
    ) AS ccn_provnum,

    trim(cast(owner_name AS varchar)) AS owner_name,
    trim(cast(owner_type AS varchar)) AS owner_type,
    trim(cast(ownership_percentage AS varchar)) AS ownership_pct,
    trim(cast(role_played_by_owner_or_manager_in_facility AS varchar)) AS ownership_role,

    regexp_extract(
      trim(cast(association_date AS varchar)),
      '([0-9]{1,2}/[0-9]{1,2}/[0-9]{4})',
      1
    ) AS assoc_date_match,

    ingest_dt
  FROM healthcare_catalog_db.raw_nh_ownership_oct2024_fixed
  WHERE ingest_dt = '{{INGEST_DT}}'
    AND cms_certification_number_ccn IS NOT NULL
    AND association_date IS NOT NULL
    AND owner_name IS NOT NULL
    AND owner_type IS NOT NULL
    AND ownership_percentage IS NOT NULL
    AND role_played_by_owner_or_manager_in_facility IS NOT NULL
),

parsed AS (
  SELECT
    b.*,
    try(cast(date_parse(assoc_date_match, '%m/%d/%Y') AS date)) AS association_dt
  FROM base b
  WHERE assoc_date_match IS NOT NULL
    AND assoc_date_match <> ''
),

with_dim AS (
  SELECT
    p.*,
    d.date_id
  FROM parsed p
  JOIN healthcare_curated_db.dim_date d
    ON d.date = p.association_dt
)

SELECT
  ccn_provnum,
  association_dt AS association_date,
  date_id,
  owner_name,
  owner_type,
  ownership_pct,
  ownership_role,
  ingest_dt
FROM with_dim;
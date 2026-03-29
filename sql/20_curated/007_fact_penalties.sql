DROP TABLE IF EXISTS healthcare_curated_db.fact_penalties;

-- S3_PREFIX: s3://healthcare-data-lake-gj/curated/fact_penalties/

CREATE TABLE healthcare_curated_db.fact_penalties
WITH (
  format = 'PARQUET',
  external_location = 's3://healthcare-data-lake-gj/curated/fact_penalties/',
  parquet_compression = 'SNAPPY',
  partitioned_by = ARRAY['ingest_dt']
) AS

WITH base AS (
  SELECT
    -- CCN normalized to 6-char string, remove quotes
    lpad(regexp_replace(trim(cast("cms certification number (ccn)" as varchar)), '"', ''), 6, '0') AS ccn_provnum,

    -- parse penalty_date: try MM/DD/YYYY first, then YYYY-MM-DD
    coalesce(
      try(cast(date_parse(regexp_replace(trim(cast("penalty date" as varchar)), '"', ''), '%m/%d/%Y') as date)),
      try(cast(date_parse(regexp_replace(trim(cast("penalty date" as varchar)), '"', ''), '%Y-%m-%d') as date))
    ) AS penalty_date,

    cast(trim(cast("penalty type" as varchar)) as varchar) AS penalty_type,

    -- fine amount as double
    try_cast(regexp_replace(trim(cast("fine amount" as varchar)), '"', '') as double) AS penalty_amt,

    -- parse payment denial start date (can be blank)
    coalesce(
      try(cast(date_parse(regexp_replace(trim(cast("payment denial start date" as varchar)), '"', ''), '%m/%d/%Y') as date)),
      try(cast(date_parse(regexp_replace(trim(cast("payment denial start date" as varchar)), '"', ''), '%Y-%m-%d') as date))
    ) AS penalty_start_dt,

    try_cast("payment denial length in days" as integer) AS penalty_denial_length,

    ingest_dt
  FROM healthcare_catalog_db.raw_nh_penalties_oct2024
  WHERE ingest_dt = '{{INGEST_DT}}' 
),

with_date AS (
  SELECT
    b.*,
    d.date_id
  FROM base b
  JOIN healthcare_curated_db.dim_date d
    ON d.date = b.penalty_date
)

SELECT
  ccn_provnum,
  penalty_date,
  date_id,
  penalty_type,
  penalty_amt,
  penalty_start_dt,
  penalty_denial_length,
  ingest_dt
FROM with_date
WHERE ccn_provnum IS NOT NULL
  AND penalty_date IS NOT NULL
  AND penalty_type IS NOT NULL
  AND penalty_amt IS NOT NULL;
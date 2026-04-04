DROP TABLE IF EXISTS healthcare_curated_db.fact_staffing_daily;

-- S3_PREFIX: s3://healthcare-data-lake-gj/curated/fact_staffing_daily/

CREATE TABLE healthcare_curated_db.fact_staffing_daily
WITH (
  format = 'PARQUET',
  external_location = 's3://healthcare-data-lake-gj/curated/fact_staffing_daily/',
  parquet_compression = 'SNAPPY',
  partitioned_by = ARRAY['ingest_dt']
) AS
WITH base AS (
  SELECT
    cast(ingest_dt as varchar) as ingest_dt,
    -- unify CCN/PROVNUM as string, preserve leading zeros (6 chars)
    lpad(regexp_replace(trim(cast(provnum AS varchar)), '"', ''), 6, '0') AS ccn_provnum,

    cast(provname AS varchar) AS provider_name,
    cast(city AS varchar)     AS provider_city,
    cast(state AS varchar)    AS provider_state,
    cast(county_name AS varchar) AS county_name,
    cast(county_fips AS varchar) AS county_fips,
    cast(cy_qtr AS varchar)      AS cy_qtr,

    -- parse WorkDate safely
    try(cast(date_parse(trim(cast(workdate AS varchar)), '%Y%m%d') AS date)) AS work_date,

    try_cast(mdscensus AS integer) AS mdscensus,

    -- hours: safe numeric parsing
    try_cast(hrs_rndon AS double)        AS hrs_rndon,
    try_cast(hrs_rndon_emp AS double)    AS hrs_rndon_emp,
    try_cast(hrs_rndon_ctr AS double)    AS hrs_rndon_ctr,

    try_cast(hrs_rnadmin AS double)      AS hrs_rnadmin,
    try_cast(hrs_rnadmin_emp AS double)  AS hrs_rnadmin_emp,
    try_cast(hrs_rnadmin_ctr AS double)  AS hrs_rnadmin_ctr,

    try_cast(hrs_rn AS double)           AS hrs_rn,
    try_cast(hrs_rn_emp AS double)       AS hrs_rn_emp,
    try_cast(hrs_rn_ctr AS double)       AS hrs_rn_ctr,

    try_cast(hrs_lpnadmin AS double)     AS hrs_lpnadmin,
    try_cast(hrs_lpnadmin_emp AS double) AS hrs_lpnadmin_emp,
    try_cast(hrs_lpnadmin_ctr AS double) AS hrs_lpnadmin_ctr,

    try_cast(hrs_lpn AS double)          AS hrs_lpn,
    try_cast(hrs_lpn_emp AS double)      AS hrs_lpn_emp,
    try_cast(hrs_lpn_ctr AS double)      AS hrs_lpn_ctr,

    try_cast(hrs_cna AS double)          AS hrs_cna,
    try_cast(hrs_cna_emp AS double)      AS hrs_cna_emp,
    try_cast(hrs_cna_ctr AS double)      AS hrs_cna_ctr,

    try_cast(hrs_natrn AS double)        AS hrs_natrn,
    try_cast(hrs_natrn_emp AS double)    AS hrs_natrn_emp,
    try_cast(hrs_natrn_ctr AS double)    AS hrs_natrn_ctr,

    try_cast(hrs_medaide AS double)      AS hrs_medaide,
    try_cast(hrs_medaide_emp AS double)  AS hrs_medaide_emp,
    try_cast(hrs_medaide_ctr AS double)  AS hrs_medaide_ctr

  FROM healthcare_catalog_db.raw_pbj_daily_nurse_staffing_q2_2024_fixed
  WHERE ingest_dt = '{{INGEST_DT:pbj_daily_nurse_staffing_q2_2024}}'   
    AND workdate IS NOT NULL
    AND length(trim(cast(workdate AS varchar))) = 8
    AND regexp_like(trim(cast(workdate AS varchar)), '^[0-9]{8}$')
),
with_dim AS (
  SELECT
    b.*,
    d.date_id,
    d.year,
    d.month_nbr
  FROM base b
  JOIN healthcare_curated_db.dim_date d
    ON d.date = b.work_date
)
SELECT
  ccn_provnum,
  work_date AS workdate,
  date_id,

  county_fips,
  mdscensus,

  hrs_rndon,
  hrs_rndon_emp,
  hrs_rndon_ctr,

  hrs_rnadmin,
  hrs_rnadmin_emp,
  hrs_rnadmin_ctr,

  hrs_rn,
  hrs_rn_emp,
  hrs_rn_ctr,

  hrs_lpnadmin,
  hrs_lpnadmin_emp,
  hrs_lpnadmin_ctr,

  hrs_lpn,
  hrs_lpn_emp,
  hrs_lpn_ctr,

  hrs_cna,
  hrs_cna_emp,
  hrs_cna_ctr,

  hrs_natrn,
  hrs_natrn_emp,
  hrs_natrn_ctr,

  hrs_medaide,
  hrs_medaide_emp,
  hrs_medaide_ctr,

  -- partition columns must be included in SELECT when using partitioned_by
  ingest_dt

FROM with_dim;
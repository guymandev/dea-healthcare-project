DROP TABLE IF EXISTS healthcare_curated_db.dim_date;

-- S3_PREFIX: s3://healthcare-data-lake-gj/curated/dim_date/

CREATE TABLE healthcare_curated_db.dim_date
WITH (
  format='PARQUET',
  external_location='s3://healthcare-data-lake-gj/curated/dim_date/',
  parquet_compression='SNAPPY'
) AS
WITH params AS (
  SELECT
    date '1940-01-01' AS start_dt,
    date '2026-12-31' AS end_dt
),
nums AS (
  SELECT n
  FROM params
  CROSS JOIN UNNEST(
    sequence(
      0,
      date_diff('day', start_dt, end_dt)
    )
  ) AS t(n)
),
dates AS (
  SELECT date_add('day', n, (SELECT start_dt FROM params)) AS d
  FROM nums
)
SELECT
  cast(date_format(d, '%Y%m%d') AS integer) AS date_id,
  d AS date,
  cast(date_format(d, '%d') AS varchar) AS day_nbr,
  cast(date_format(d, '%m') AS varchar) AS month_nbr,
  cast(date_format(d, '%Y') AS varchar) AS year,
  cast(quarter(d) AS varchar) AS quarter
FROM dates;
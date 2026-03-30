DROP TABLE IF EXISTS healthcare_curated_db.fact_quality_measure_score;

-- S3_PREFIX: s3://healthcare-data-lake-gj/curated/fact_quality_measure_score/

CREATE TABLE healthcare_curated_db.fact_quality_measure_score
WITH (
  format = 'PARQUET',
  external_location = 's3://healthcare-data-lake-gj/curated/fact_quality_measure_score/',
  parquet_compression = 'SNAPPY',
  partitioned_by = ARRAY['ingest_dt']
) AS

WITH claims AS (
  SELECT
    lpad(regexp_replace(trim(cast(cms_certification_number_ccn as varchar)), '"', ''), 6, '0') AS ccn_provnum,
    trim(cast(measure_code as varchar))   AS measure_code,
    trim(cast(measure_period as varchar)) AS measure_period,

    -- Parse measure_period when it looks like YYYYMMDD-YYYYMMDD
    CASE
      WHEN regexp_like(trim(cast(measure_period as varchar)), '^[0-9]{8}-[0-9]{8}$')
      THEN try(cast(date_parse(substr(trim(cast(measure_period as varchar)), 1, 8), '%Y%m%d') AS date))
      ELSE NULL
    END AS measure_period_start_date,

    CASE
      WHEN regexp_like(trim(cast(measure_period as varchar)), '^[0-9]{8}-[0-9]{8}$')
      THEN try(cast(date_parse(substr(trim(cast(measure_period as varchar)), 10, 8), '%Y%m%d') AS date))
      ELSE NULL
    END AS measure_period_end_date,

    cast(try_cast(adjusted_score as double) as varchar) AS score,
    cast(footnote_for_score as varchar) AS footnote,

    CAST(NULL AS date)    AS start_date,
    CAST(NULL AS date)    AS end_date,
    CAST(NULL AS varchar) AS measure_date_range,

    cast(resident_type as varchar) AS resident_type,

    CAST(NULL AS double)  AS q1_measure_score,
    CAST(NULL AS varchar) AS q1_measure_score_footnote,
    CAST(NULL AS double)  AS q2_measure_score,
    CAST(NULL AS varchar) AS q2_measure_score_footnote,
    CAST(NULL AS double)  AS q3_measure_score,
    CAST(NULL AS varchar) AS q3_measure_score_footnote,
    CAST(NULL AS double)  AS q4_measure_score,
    CAST(NULL AS varchar) AS q4_measure_score_footnote,

    CAST(NULL AS double)  AS four_quarter_avg_score,
    CAST(NULL AS varchar) AS four_quarter_avg_score_footnote,

    CASE
      WHEN upper(trim(cast(used_in_quality_measure_five_star_rating as varchar))) IN ('Y','YES','TRUE','T','1') THEN true
      WHEN upper(trim(cast(used_in_quality_measure_five_star_rating as varchar))) IN ('N','NO','FALSE','F','0') THEN false
      ELSE NULL
    END AS used_in_quality_measure_five_star_rating,

    try_cast(adjusted_score as double) AS adjusted_score,
    try_cast(observed_score as double) AS observed_score,
    try_cast(expected_score as double) AS expected_score,

    ingest_dt
  FROM healthcare_catalog_db.raw_nh_qualitymsr_claims_oct2024_fixed
  WHERE ingest_dt = '{{INGEST_DT}}' 
    AND cms_certification_number_ccn IS NOT NULL
    AND measure_code IS NOT NULL
    AND measure_period IS NOT NULL
),

mds AS (
  SELECT
    lpad(regexp_replace(trim(cast(cms_certification_number_ccn as varchar)), '"', ''), 6, '0') AS ccn_provnum,
    trim(cast(measure_code as varchar))   AS measure_code,
    trim(cast(measure_period as varchar)) AS measure_period,

    CASE
      WHEN regexp_like(trim(cast(measure_period as varchar)), '^[0-9]{8}-[0-9]{8}$')
      THEN try(cast(date_parse(substr(trim(cast(measure_period as varchar)), 1, 8), '%Y%m%d') AS date))
      ELSE NULL
    END AS measure_period_start_date,

    CASE
      WHEN regexp_like(trim(cast(measure_period as varchar)), '^[0-9]{8}-[0-9]{8}$')
      THEN try(cast(date_parse(substr(trim(cast(measure_period as varchar)), 10, 8), '%Y%m%d') AS date))
      ELSE NULL
    END AS measure_period_end_date,

    cast(try_cast(four_quarter_average_score as double) as varchar) AS score,
    cast(footnote_for_four_quarter_average_score as varchar) AS footnote,

    CAST(NULL AS date)    AS start_date,
    CAST(NULL AS date)    AS end_date,
    CAST(NULL AS varchar) AS measure_date_range,

    cast(resident_type as varchar) AS resident_type,

    try_cast(q1_measure_score as double) AS q1_measure_score,
    cast(footnote_for_q1_measure_score as varchar) AS q1_measure_score_footnote,

    try_cast(q2_measure_score as double) AS q2_measure_score,
    cast(footnote_for_q2_measure_score as varchar) AS q2_measure_score_footnote,

    try_cast(q3_measure_score as double) AS q3_measure_score,
    cast(footnote_for_q3_measure_score as varchar) AS q3_measure_score_footnote,

    try_cast(q4_measure_score as double) AS q4_measure_score,
    cast(footnote_for_q4_measure_score as varchar) AS q4_measure_score_footnote,

    try_cast(four_quarter_average_score as double) AS four_quarter_avg_score,
    cast(footnote_for_four_quarter_average_score as varchar) AS four_quarter_avg_score_footnote,

    CASE
      WHEN upper(trim(cast(used_in_quality_measure_five_star_rating as varchar))) IN ('Y','YES','TRUE','T','1') THEN true
      WHEN upper(trim(cast(used_in_quality_measure_five_star_rating as varchar))) IN ('N','NO','FALSE','F','0') THEN false
      ELSE NULL
    END AS used_in_quality_measure_five_star_rating,

    CAST(NULL AS double) AS adjusted_score,
    CAST(NULL AS double) AS observed_score,
    CAST(NULL AS double) AS expected_score,

    ingest_dt
  FROM healthcare_catalog_db.raw_nh_qualitymsr_mds_oct2024_fixed
  WHERE ingest_dt = '{{INGEST_DT}}' 
    AND cms_certification_number_ccn IS NOT NULL
    AND measure_code IS NOT NULL
    AND measure_period IS NOT NULL
)

SELECT * FROM claims
UNION ALL
SELECT * FROM mds;
SELECT coalesce(count(*), 0) AS dupes
FROM (
  SELECT
    ccn_provnum,
    measure_code,
    measure_period
  FROM healthcare_curated_db.fact_quality_measure_score
  WHERE ingest_dt = '{{INGEST_DT}}'
  GROUP BY ccn_provnum, measure_code, measure_period
  HAVING count(*) > 1
) d;
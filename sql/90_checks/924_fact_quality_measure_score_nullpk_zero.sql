SELECT coalesce(
  sum(
    CASE
      WHEN ccn_provnum IS NULL
        OR measure_code IS NULL
        OR trim(measure_code) = ''
        OR measure_period IS NULL
        OR trim(measure_period) = ''
      THEN 1
      ELSE 0
    END
  ),
  0
) AS null_pk_rows
FROM healthcare_curated_db.fact_quality_measure_score
WHERE ingest_dt = '{{INGEST_DT}}';
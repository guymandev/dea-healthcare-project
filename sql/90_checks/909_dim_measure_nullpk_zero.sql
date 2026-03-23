SELECT coalesce(sum(CASE WHEN measure_code IS NULL THEN 1 ELSE 0 END), 0) AS null_pk
FROM healthcare_curated_db.dim_measure
WHERE ingest_dt = '{{INGEST_DT}}';
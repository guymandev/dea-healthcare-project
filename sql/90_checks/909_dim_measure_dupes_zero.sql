SELECT coalesce(count(*) - count(distinct measure_code), 0) AS dupes
FROM healthcare_curated_db.dim_measure
WHERE ingest_dt = '{{MANIFEST_RUN_INGEST_DT}}';
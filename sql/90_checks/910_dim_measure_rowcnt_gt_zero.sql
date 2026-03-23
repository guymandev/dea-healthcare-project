SELECT count(*) AS row_cnt
FROM healthcare_curated_db.dim_measure
WHERE ingest_dt = '{{INGEST_DT}}';
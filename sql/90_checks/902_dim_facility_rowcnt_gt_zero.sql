SELECT count(*) AS row_cnt
FROM healthcare_curated_db.dim_facility
WHERE ingest_dt = '{{MANIFEST_RUN_INGEST_DT}}';
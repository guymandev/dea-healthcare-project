SELECT count(*) AS row_cnt
FROM healthcare_curated_db.fact_staffing_daily
WHERE ingest_dt = '{{INGEST_DT}}';
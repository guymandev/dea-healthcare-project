SELECT count(*) AS row_cnt
FROM healthcare_curated_db.fact_ownership
WHERE ingest_dt = '{{INGEST_DT:nh_ownership_oct2024}}';
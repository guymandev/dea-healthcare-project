SELECT count(*) AS row_cnt
FROM healthcare_catalog_db.raw_nh_ownership_oct2024_fixed
WHERE ingest_dt = '{{INGEST_DT}}';
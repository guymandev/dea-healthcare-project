SELECT count(*) AS row_cnt
FROM healthcare_curated_db.fact_staffing_daily
WHERE ingest_dt = '{{INGEST_DT:pbj_daily_nurse_staffing_q2_2024}}';
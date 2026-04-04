SELECT count(*) AS row_cnt
FROM healthcare_curated_db.fact_quality_measure_score
WHERE ingest_dt = '{{INGEST_DT:nh_qualitymsr_claims_oct2024}}';